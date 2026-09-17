from __future__ import annotations

from typing import Any, Sequence

import torch

ORDERING_PART_TOKEN_TYPES = tuple(f"part{idx}" for idx in range(1, 12))
INTERVENTION_TOKEN_TYPES = (
    "audio",
    "text",
    "events",
    "control",
    *ORDERING_PART_TOKEN_TYPES,
)
DEFAULT_MODEL_IDS = {
    "af3": "nvidia/audio-flamingo-3-hf",
    "af-next": "nvidia/audio-flamingo-next-hf",
    "moss-audio": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


def default_model_id(model_name: str) -> str:
    try:
        return DEFAULT_MODEL_IDS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"model must be one of: {', '.join(sorted(DEFAULT_MODEL_IDS))}"
        ) from exc


def model_output_slug(model_id: str) -> str:
    return (
        model_id.strip()
        .replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
    )


def build_model_interface(model_name: str, model_id: str | None = None, device: str = "auto") -> Any:
    from audio_lm_interfaces import (
        AudioFlamingo3Interface,
        AudioFlamingoNextInterface,
        MossAudioInterface,
        Qwen3OmniInterface,
    )

    model_id = model_id or default_model_id(model_name)
    if model_name == "af3":
        return AudioFlamingo3Interface(model_id=model_id, device=device)
    if model_name == "af-next":
        return AudioFlamingoNextInterface(model_id=model_id, device=device)
    if model_name == "moss-audio":
        return MossAudioInterface(model_id=model_id, device=device)
    if model_name == "qwen3-omni":
        return Qwen3OmniInterface(model_id=model_id, device=device)
    raise ValueError(f"Unsupported model: {model_name}")


class OrderingIntervention:
    """Experiment-local next-token scoring and activation swapping."""

    def __init__(self, interface: Any):
        self.interface = interface
        self.model = interface.model
        self.processor = interface.processor

    def forward_intervention(
        self,
        audio_arrays: Sequence[Any],
        alternative_audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        correct_keywords: Sequence[str],
        layer: int,
        token_type: str | Sequence[str],
        teacher_forced_texts: Sequence[str] | str | None = None,
        sampling_rates: Sequence[int] | int | None = None,
        keywords: Sequence[str] = (" before", " after"),
        top_k: int = 10,
        **forward_kwargs: Any,
    ) -> list[dict[str, Any]]:
        _validate_batch_lengths(
            audio_arrays,
            alternative_audio_arrays,
            prompts,
            correct_keywords,
        )
        token_types = self._normalize_token_types(token_type)
        token_type_label = "+".join(token_types)
        teacher_forced_texts = _batch_values(teacher_forced_texts, len(audio_arrays), default="")
        self._validate_query_args(keywords, top_k)
        for correct_keyword in correct_keywords:
            if correct_keyword not in keywords:
                raise ValueError("correct_keyword must be one of keywords")

        normal_inputs, _normal_tokens = self.interface.build_model_inputs(
            audio_arrays=audio_arrays,
            prompts=prompts,
            teacher_forced_texts=teacher_forced_texts,
            sampling_rates=sampling_rates,
        )
        swapped_inputs, _swapped_tokens = self.interface.build_model_inputs(
            audio_arrays=alternative_audio_arrays,
            prompts=prompts,
            teacher_forced_texts=teacher_forced_texts,
            sampling_rates=sampling_rates,
        )
        if normal_inputs["input_ids"].shape != swapped_inputs["input_ids"].shape:
            raise ValueError(
                "Normal and alternative inputs must produce the same token shape "
                "for activation replacement"
            )

        target_mask = self.token_mask(
            normal_inputs,
            token_types,
        )
        swapped_target_mask = self.token_mask(
            swapped_inputs,
            token_types,
        )
        if not torch.equal(target_mask.cpu(), swapped_target_mask.cpu()):
            raise ValueError(
                "Normal and alternative inputs do not have matching intervention token positions"
            )
        if not bool(target_mask.any()):
            raise ValueError(f"No {token_type_label!r} tokens found for intervention")
        if not torch.all(target_mask.any(dim=1)):
            raise ValueError(f"At least one batch element has no {token_type_label!r} tokens")

        candidate_ids = self._candidate_token_ids(keywords)
        target_layer = self.decoder_layer(layer)
        normal_outputs, _ = self._forward_with_activation_capture(
            target_layer,
            normal_inputs,
            forward_kwargs,
        )
        swapped_outputs, swapped_activations = self._forward_with_activation_capture(
            target_layer,
            swapped_inputs,
            forward_kwargs,
        )
        intervened_outputs = self._forward_with_activation_replacement(
            target_layer,
            normal_inputs,
            swapped_activations,
            target_mask,
            forward_kwargs,
        )

        normal_results = self._probability_results(
            normal_outputs.logits,
            keywords,
            candidate_ids,
            top_k,
            attention_mask=normal_inputs.get("attention_mask"),
        )
        swapped_results = self._probability_results(
            swapped_outputs.logits,
            keywords,
            candidate_ids,
            top_k,
            attention_mask=swapped_inputs.get("attention_mask"),
        )
        intervened_results = self._probability_results(
            intervened_outputs.logits,
            keywords,
            candidate_ids,
            top_k,
            attention_mask=normal_inputs.get("attention_mask"),
        )

        results = []
        for batch_idx, (
            correct_keyword,
            normal_result,
            swapped_result,
            intervened_result,
        ) in enumerate(zip(
            correct_keywords,
            normal_results,
            swapped_results,
            intervened_results,
        )):
            sample_mask = target_mask[batch_idx]
            sample_input_ids = normal_inputs["input_ids"][batch_idx]
            results.append({
                "normal_correct_probability": self._keyword_probability(
                    normal_result,
                    correct_keyword,
                ),
                "swapped_correct_probability": self._keyword_probability(
                    swapped_result,
                    correct_keyword,
                ),
                "intervened_correct_probability": self._keyword_probability(
                    intervened_result,
                    correct_keyword,
                ),
                "normal": normal_result,
                "swapped": swapped_result,
                "intervened": intervened_result,
                "intervention": {
                    "layer": int(layer),
                    "token_type": token_type_label,
                    "normal_correct_keyword": correct_keyword,
                    "token_indices": sample_mask.nonzero(as_tuple=False).flatten().tolist(),
                    "tokens": self._convert_ids_to_tokens(sample_input_ids[sample_mask].tolist()),
                },
            })
        return results

    def decoder_layer_count(self) -> int:
        layers, _name = self._decoder_layers()
        return len(layers)

    def decoder_layer(self, layer: int) -> torch.nn.Module:
        layers, name = self._decoder_layers()
        if not 0 <= layer < len(layers):
            raise ValueError(f"layer must be in [0, {len(layers) - 1}] for {name}")
        return layers[layer]

    def token_mask(self, model_inputs: Any, token_type: str | Sequence[str]) -> torch.Tensor:
        token_types = self._normalize_token_types(token_type)
        if len(token_types) > 1:
            masks = [
                self.token_mask(model_inputs, single_token_type)
                for single_token_type in token_types
            ]
            combined_mask = masks[0].clone()
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
            return combined_mask

        token_type = token_types[0]
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if token_type == "audio":
            return self._audio_token_mask(model_inputs)
        if token_type == "text":
            return self._text_token_mask(input_ids, attention_mask)
        if token_type == "events":
            return self._ordering_part_token_mask(
                input_ids,
                attention_mask,
                part_indices=(2, 7, 10),
            )
        if token_type == "control":
            return self._ordering_part_token_mask(
                input_ids,
                attention_mask,
                part_indices=(4, 5, 6),
            )
        if token_type in ORDERING_PART_TOKEN_TYPES:
            part_index = int(token_type.removeprefix("part"))
            return self._ordering_part_token_mask(
                input_ids,
                attention_mask,
                part_indices=(part_index,),
            )
        raise ValueError(f"token_type must be one of: {', '.join(INTERVENTION_TOKEN_TYPES)}")

    def _normalize_token_types(self, token_type: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(token_type, str):
            token_types = (token_type,)
        else:
            token_types = tuple(token_type)
        if not token_types:
            raise ValueError("At least one token type is required for intervention")

        invalid_token_types = [
            single_token_type
            for single_token_type in token_types
            if single_token_type not in INTERVENTION_TOKEN_TYPES
        ]
        if invalid_token_types:
            raise ValueError(
                "token_type must be one or more of: "
                f"{', '.join(INTERVENTION_TOKEN_TYPES)}; "
                f"got {', '.join(invalid_token_types)}"
            )
        return token_types

    def _audio_token_mask(self, model_inputs: Any) -> torch.Tensor:
        audio_input_mask = model_inputs.get("audio_input_mask")
        if audio_input_mask is not None:
            mask = audio_input_mask.bool()
        else:
            input_ids = model_inputs["input_ids"]
            mask = input_ids == self._audio_token_id()

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        return mask

    def _ordering_part_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        part_indices: Sequence[int],
    ) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        requested_parts = set(part_indices)

        for batch_idx in range(input_ids.shape[0]):
            sample_ids = input_ids[batch_idx]
            sample_attention_mask = (
                attention_mask[batch_idx].bool()
                if attention_mask is not None
                else torch.ones_like(sample_ids, dtype=torch.bool)
            )
            tokens = self._convert_ids_to_tokens(sample_ids.tolist())
            spans = self._ordering_part_spans(tokens)
            for part_index in requested_parts:
                start, end = spans[part_index]
                for idx in range(start, end):
                    if bool(sample_attention_mask[idx]):
                        mask[batch_idx, idx] = True

        return mask

    def _ordering_part_spans(self, tokens: Sequence[str]) -> dict[int, tuple[int, int]]:
        question_start = self._first_user_content_start(tokens)
        assistant_header_start = self._assistant_header_start(tokens)

        if question_start >= len(tokens):
            raise ValueError("Could not locate ordering question tokens")

        question_mark_index = self._find_decoded_token(tokens, "?", question_start, assistant_header_start)
        if question_mark_index is None:
            raise ValueError("Could not locate '?' in ordering question")

        does_start = self._find_decoded_token(tokens, "Does", question_start, question_mark_index)
        if does_start is None:
            raise ValueError("Could not locate 'Does' in ordering question")

        question_word_starts = self._content_word_starts(
            tokens,
            does_start,
            question_mark_index,
        )
        if len(question_word_starts) < 7:
            raise ValueError(
                "Ordering question must contain at least seven word spans: "
                "Does, query label, occur, before, or, after, reference label"
            )

        assistant_content_start = self._assistant_content_start(tokens)
        if assistant_content_start >= len(tokens):
            assistant_content_start = question_mark_index + 1

        assistant_word_starts = self._content_word_starts(
            tokens,
            assistant_content_start,
            len(tokens),
        )
        if len(assistant_word_starts) < 2:
            raise ValueError(
                "Teacher-forced ordering text must contain query label and occurs"
            )

        query_start = question_word_starts[1]
        occur_start = self._find_decoded_token(tokens, "occur", query_start, question_mark_index)
        before_start = self._find_decoded_token(tokens, "before", query_start, question_mark_index)
        or_start = self._find_decoded_token(tokens, "or", query_start, question_mark_index)
        after_start = self._find_decoded_token(tokens, "after", query_start, question_mark_index)
        if None in {occur_start, before_start, or_start, after_start}:
            raise ValueError("Could not locate ordering relation words in question")

        reference_start = after_start + 1
        query_2_start = assistant_word_starts[0]
        occurs_2_start = assistant_word_starts[-1]

        return {
            1: (does_start, does_start + 1),
            2: (query_start, occur_start),
            3: (occur_start, occur_start + 1),
            4: (before_start, before_start + 1),
            5: (or_start, or_start + 1),
            6: (after_start, after_start + 1),
            7: (reference_start, question_mark_index),
            8: (question_mark_index, question_mark_index + 1),
            9: (question_mark_index + 1, query_2_start),
            10: (query_2_start, occurs_2_start),
            11: (occurs_2_start, occurs_2_start + 1),
        }

    def _content_word_starts(
        self,
        tokens: Sequence[str],
        start: int,
        end: int,
    ) -> list[int]:
        starts = []
        previous_has_content = False

        for idx in range(start, end):
            if self._is_non_text_token(tokens[idx]):
                previous_has_content = False
                continue
            text = self._decode_token(tokens[idx])
            if not text or text.isspace():
                previous_has_content = False
                continue
            if text.startswith(" ") or not previous_has_content:
                starts.append(idx)
            previous_has_content = True

        return starts

    def _find_decoded_token(
        self,
        tokens: Sequence[str],
        text: str,
        start: int,
        end: int,
    ) -> int | None:
        for idx in range(start, end):
            if self._decode_token(tokens[idx]).strip() == text:
                return idx
        return None

    def _text_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        special_ids = self._special_token_ids()
        audio_token_id = self._maybe_audio_token_id()
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for batch_idx in range(input_ids.shape[0]):
            sample_ids = input_ids[batch_idx]
            sample_attention_mask = (
                attention_mask[batch_idx].bool()
                if attention_mask is not None
                else torch.ones_like(sample_ids, dtype=torch.bool)
            )
            tokens = self._convert_ids_to_tokens(sample_ids.tolist())
            start = self._first_user_content_start(tokens)
            skip_role_header_until = None

            for idx in range(start, len(tokens)):
                if not bool(sample_attention_mask[idx]):
                    continue
                token = tokens[idx]
                token_id = int(sample_ids[idx].item())
                if skip_role_header_until is not None:
                    if idx <= skip_role_header_until:
                        continue
                    skip_role_header_until = None
                if token == "<|im_start|>":
                    skip_role_header_until = self._next_newline_index(tokens, idx)
                    continue
                if token_id in special_ids:
                    continue
                if audio_token_id is not None and token_id == audio_token_id:
                    continue
                mask[batch_idx, idx] = True
        return mask

    def _candidate_token_ids(self, keywords: Sequence[str]) -> list[int]:
        candidate_ids = []
        for keyword in keywords:
            token_ids = self.processor.tokenizer.encode(
                keyword,
                add_special_tokens=False,
            )
            if not token_ids:
                raise ValueError(f"Keyword {keyword!r} does not tokenize to any tokens")
            if len(token_ids) != 1:
                raise ValueError(
                    f"Keyword {keyword!r} must tokenize to exactly one token, "
                    f"got {len(token_ids)} tokens: {token_ids}"
                )
            candidate_ids.append(int(token_ids[0]))
        return candidate_ids

    def _probability_results(
        self,
        logits: torch.Tensor,
        keywords: Sequence[str],
        candidate_ids: Sequence[int],
        top_k: int,
        attention_mask: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        if attention_mask is None:
            next_token_indices = torch.full(
                (logits.shape[0],),
                logits.shape[1] - 1,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            mask = attention_mask.to(logits.device).bool()
            positions = torch.arange(logits.shape[1], device=logits.device)
            next_token_indices = positions.masked_fill(~mask, 0).max(dim=1).values.long()

        results = []
        for batch_idx, next_token_idx in enumerate(next_token_indices.tolist()):
            results.append(self._probability_result_for_logits(
                logits[batch_idx, next_token_idx],
                keywords,
                candidate_ids,
                top_k,
            ))
        return results

    def _probability_result_for_logits(
        self,
        next_token_logits: torch.Tensor,
        keywords: Sequence[str],
        candidate_ids: Sequence[int],
        top_k: int,
    ) -> dict[str, Any]:
        full_probs = torch.softmax(next_token_logits, dim=-1)
        candidate_id_tensor = torch.tensor(candidate_ids, device=next_token_logits.device)
        candidate_probs = full_probs[candidate_id_tensor]
        top_k = min(top_k, full_probs.shape[-1])
        top_probs, top_ids = torch.topk(full_probs, k=top_k)

        keyword_probabilities = []
        for keyword, token_id, probability in zip(keywords, candidate_ids, candidate_probs):
            keyword_probabilities.append({
                "text": keyword,
                "token_id": int(token_id),
                "token": self._convert_ids_to_tokens(int(token_id)),
                "probability": float(probability.item()),
            })

        top_tokens = []
        for token_id, probability in zip(top_ids.tolist(), top_probs.tolist()):
            top_tokens.append({
                "text": self._decode([token_id]),
                "token_id": int(token_id),
                "token": self._convert_ids_to_tokens(int(token_id)),
                "probability": float(probability),
            })

        return {
            "keyword_probabilities": keyword_probabilities,
            "top_tokens": top_tokens,
        }

    def _keyword_probability(self, result: dict[str, Any], keyword: str) -> float:
        for item in result["keyword_probabilities"]:
            if item["text"] == keyword:
                return item["probability"]
        raise ValueError(f"Keyword {keyword!r} was not scored")

    def _validate_query_args(self, keywords: Sequence[str], top_k: int) -> None:
        if isinstance(keywords, str) or not keywords:
            raise ValueError("keywords must be a non-empty list of strings")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

    def _decoder_layers(self) -> tuple[Sequence[torch.nn.Module], str]:
        layer_containers = [
            ("thinker.model.layers", ("thinker", "model", "layers")),
            ("language_model.model.layers", ("language_model", "model", "layers")),
            ("language_model.layers", ("language_model", "layers")),
            ("alm.model.layers", ("alm", "model", "layers")),
            ("model.layers", ("model", "layers")),
            ("layers", ("layers",)),
        ]

        for name, path in layer_containers:
            module = self.model
            for attr in path:
                if not hasattr(module, attr):
                    module = None
                    break
                module = getattr(module, attr)
            if module is not None:
                return module, name

        if hasattr(self.model, "get_decoder"):
            decoder = self.model.get_decoder()
            for attr in ("model", "layers"):
                if hasattr(decoder, attr):
                    candidate = getattr(decoder, attr)
                    if attr == "model" and hasattr(candidate, "layers"):
                        candidate = candidate.layers
                    return candidate, f"decoder.{attr}"

        raise AttributeError("Could not find decoder layers on the audio language model")

    def _forward_with_activation_capture(
        self,
        target_layer: torch.nn.Module,
        model_inputs: Any,
        forward_kwargs: dict[str, Any],
    ) -> tuple[Any, torch.Tensor]:
        activation = {}

        def hook(_module: Any, _args: Any, output: Any) -> None:
            activation["value"] = self._hidden_states_from_layer_output(output).detach()

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = self.interface.forward(model_inputs, **forward_kwargs)
        finally:
            handle.remove()

        if "value" not in activation:
            raise RuntimeError("Forward hook did not capture layer activations")
        return outputs, activation["value"]

    def _forward_with_activation_replacement(
        self,
        target_layer: torch.nn.Module,
        model_inputs: Any,
        replacement_activations: torch.Tensor,
        token_mask: torch.Tensor,
        forward_kwargs: dict[str, Any],
    ) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden_states = self._hidden_states_from_layer_output(output)
            edited_hidden_states = hidden_states.clone()
            mask = token_mask.to(hidden_states.device)
            replacement = replacement_activations.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            edited_hidden_states[mask] = replacement[mask]
            return self._replace_hidden_states_in_layer_output(output, edited_hidden_states)

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                return self.interface.forward(model_inputs, **forward_kwargs)
        finally:
            handle.remove()

    def _hidden_states_from_layer_output(self, output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple):
            return output[0]
        raise TypeError(f"Unsupported layer output type: {type(output)!r}")

    def _replace_hidden_states_in_layer_output(
        self,
        output: Any,
        hidden_states: torch.Tensor,
    ) -> Any:
        if torch.is_tensor(output):
            return hidden_states
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        raise TypeError(f"Unsupported layer output type: {type(output)!r}")

    def _audio_token_id(self) -> int:
        audio_token_id = self._maybe_audio_token_id()
        if audio_token_id is None:
            raise ValueError("Could not resolve the audio token id")
        return audio_token_id

    def _maybe_audio_token_id(self) -> int | None:
        model_config = getattr(self.model, "config", None)
        nested_configs = (
            getattr(model_config, "thinker_config", None),
            getattr(getattr(self.model, "thinker", None), "config", None),
        )
        for owner in (model_config, *nested_configs, self.processor):
            audio_token_id = getattr(owner, "audio_token_id", None)
            if audio_token_id is not None:
                return int(audio_token_id)

        tokenizer = self.processor.tokenizer
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        audio_tokens = [
            getattr(self.processor, "audio_token", None),
            getattr(tokenizer, "audio_token", None),
            "<sound>",
            "<audio>",
            "<|audio|>",
        ]
        for token in audio_tokens:
            if token is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != unk_token_id:
                return int(token_id)
        return None

    def _special_token_ids(self) -> set[int]:
        return set(getattr(self.processor.tokenizer, "all_special_ids", []) or [])

    def _is_non_text_token(self, token: str) -> bool:
        token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
        if token_id in self._special_token_ids():
            return True
        audio_token_id = self._maybe_audio_token_id()
        return audio_token_id is not None and token_id == audio_token_id

    def _convert_ids_to_tokens(self, ids: int | Sequence[int]) -> Any:
        return self.processor.tokenizer.convert_ids_to_tokens(ids)

    def _decode(self, token_ids: Sequence[int]) -> str:
        decoder = getattr(self.processor.tokenizer, "decode", None)
        if decoder is None:
            decoder = getattr(self.processor, "decode")
        return decoder(token_ids)

    def _first_user_content_start(self, tokens: Sequence[str]) -> int:
        for idx in range(len(tokens) - 2):
            if (
                tokens[idx] == "<|im_start|>"
                and tokens[idx + 1] == "user"
                and self._is_newline_token(tokens[idx + 2])
            ):
                return idx + 3
        return 0

    def _assistant_header_start(self, tokens: Sequence[str]) -> int:
        for idx in range(len(tokens) - 2):
            if (
                tokens[idx] == "<|im_start|>"
                and tokens[idx + 1] == "assistant"
                and self._is_newline_token(tokens[idx + 2])
            ):
                return idx
        return len(tokens)

    def _assistant_content_start(self, tokens: Sequence[str]) -> int:
        assistant_header_start = self._assistant_header_start(tokens)
        if assistant_header_start + 2 < len(tokens):
            return assistant_header_start + 3
        return len(tokens)

    def _next_newline_index(self, tokens: Sequence[str], start: int) -> int:
        for idx in range(start + 1, len(tokens)):
            if self._is_newline_token(tokens[idx]):
                return idx
        return start

    def _is_newline_token(self, token: str) -> bool:
        if token in {"Ċ", "\n"}:
            return True
        try:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            return self._decode([token_id]) == "\n"
        except Exception:
            return False

    def _decode_token(self, token: str) -> str:
        if token == "Ċ":
            return "\n"
        if token.startswith("Ġ"):
            return " " + token[1:]
        try:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            return self._decode([token_id])
        except Exception:
            return token


def _validate_batch_lengths(*values: Sequence[Any]) -> None:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"Batch inputs must have matching lengths, got {sorted(lengths)}")


def _batch_values(values: Any, batch_size: int, default: Any) -> list[Any]:
    if values is None:
        return [default] * batch_size
    if isinstance(values, (str, bytes)):
        return [values] * batch_size
    if not hasattr(values, "__len__"):
        return [values] * batch_size
    values = list(values)
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} batch values, got {len(values)}")
    return values
