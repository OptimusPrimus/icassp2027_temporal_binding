import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "activations"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_SPLIT_SIZES = {
    "train": 2_000,
    "validation": 1_000,
    "test": 1_000,
}
DEFAULT_MODEL_IDS = {
    "af3": "nvidia/audio-flamingo-3-hf",
    "af-next": "nvidia/audio-flamingo-next-hf",
    "moss-audio": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (
    build_model_interface,
    default_model_id,
    model_output_slug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture text-token activations for random two-event 30s SyntheticSED "
            "ESC-50 yes/no prompts."
        )
    )
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_MODEL_IDS),
        default="af3",
        help="Audio language model backend.",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--length-sec", type=float, default=30.0)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_SPLIT_SIZES["train"])
    parser.add_argument("--validation-examples", type=int, default=DEFAULT_SPLIT_SIZES["validation"])
    parser.add_argument("--test-examples", type=int, default=DEFAULT_SPLIT_SIZES["test"])
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument(
        "--activation-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Storage dtype for captured activations.",
    )
    parser.add_argument(
        "--max-output-gb",
        type=float,
        default=100.0,
        help="Abort if the estimated activation tensor exceeds this size.",
    )
    parser.add_argument(
        "--allow-large-output",
        action="store_true",
        help="Write even when the estimated activation tensor exceeds --max-output-gb.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def storage_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def default_output_name(args: argparse.Namespace) -> str:
    model_name = args.model.replace("-", "_")
    return (
        f"{model_name}_synthetic_sed_esc50_2events_30s_"
        f"train{args.train_examples}_val{args.validation_examples}_test{args.test_examples}_"
        f"seed{args.random_seed}_all_decoder_text_activations.pt"
    )


def display_label(label: str) -> str:
    return str(label).replace("_", " ")


def prompt_for_query_event(query_event: dict[str, Any]) -> str:
    return f"Is there {display_label(query_event['event_label'])}?"


def sample_with_prompt(sample: dict[str, Any], rng: torch.Generator) -> dict[str, Any]:
    event_index = int(torch.randint(len(sample["events"]), (1,), generator=rng).item())
    query_event = sample["events"][event_index]
    enriched = dict(sample)
    enriched["query_event_index"] = event_index
    enriched["query_event_label"] = display_label(query_event["event_label"])
    enriched["prompt"] = prompt_for_query_event(query_event)
    return enriched


class TextSuffixActivationExtractor:
    def __init__(self, interface: Any, activation_dtype: torch.dtype):
        self.interface = interface
        self.model = interface.model
        self.processor = interface.processor
        self.activation_dtype = activation_dtype

    def capture_batch(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        sampling_rates: Sequence[int],
    ) -> dict[str, Any]:
        model_inputs, _tokens = self.interface.build_model_inputs(
            audio_arrays=audio_arrays,
            prompts=prompts,
            sampling_rates=sampling_rates,
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        token_mask, token_spans, tokens = self._text_suffix_mask(input_ids, attention_mask)

        if not torch.all(token_mask.any(dim=1)):
            missing = [
                idx
                for idx, has_tokens in enumerate(token_mask.any(dim=1).tolist())
                if not has_tokens
            ]
            raise ValueError(f"No target text tokens found for batch indices: {missing}")

        captured = self._capture_all_layer_activations(model_inputs)
        packed = self._pack_activations(captured, token_mask)
        generated_answers = self.interface.generate(
            model_inputs,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
            decode=True,
        )
        return {
            "activations": packed["activations"],
            "generated_answers": generated_answers,
            "token_mask": token_mask.detach().cpu(),
            "token_spans": token_spans,
            "tokens": tokens,
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": None if attention_mask is None else attention_mask.detach().cpu(),
            "num_layers": len(captured) - 1,
            "max_target_tokens": packed["max_target_tokens"],
            "target_lengths": packed["target_lengths"],
        }

    def _capture_all_layer_activations(self, model_inputs: Any) -> list[torch.Tensor]:
        from audio_lm_interfaces.interfaces import _decoder_layers

        layers, _name = _decoder_layers(self.model)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise TypeError("Could not capture first decoder layer input hidden states")
            captured[0] = args[0].detach()

        handles.append(layers[0].register_forward_pre_hook(pre_hook))
        for layer_index, layer in enumerate(layers):
            handles.append(
                layer.register_forward_hook(
                    self._layer_hook(layer_index + 1, captured)
                )
            )

        try:
            with torch.no_grad():
                self.interface.forward(model_inputs)
        finally:
            for handle in handles:
                handle.remove()

        expected = len(layers) + 1
        missing = [idx for idx in range(expected) if idx not in captured]
        if missing:
            raise RuntimeError(f"Missing activation captures for layer slots: {missing}")
        return [captured[idx] for idx in range(expected)]

    def _layer_hook(self, output_index: int, captured: dict[int, torch.Tensor]):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured[output_index] = self._hidden_states_from_layer_output(output).detach()

        return hook

    def _pack_activations(
        self,
        captured: Sequence[torch.Tensor],
        token_mask: torch.Tensor,
    ) -> dict[str, Any]:
        target_lengths = token_mask.sum(dim=1).detach().cpu()
        max_target_tokens = int(target_lengths.max().item())
        first = captured[0]
        packed = torch.zeros(
            len(captured),
            first.shape[0],
            max_target_tokens,
            first.shape[-1],
            dtype=self.activation_dtype,
            device="cpu",
        )

        for layer_index, layer_activations in enumerate(captured):
            layer_activations = layer_activations.detach()
            for batch_index in range(layer_activations.shape[0]):
                selected = layer_activations[batch_index][token_mask[batch_index].to(layer_activations.device)]
                packed[
                    layer_index,
                    batch_index,
                    : selected.shape[0],
                ] = selected.to(dtype=self.activation_dtype, device="cpu")

        return {
            "activations": packed,
            "max_target_tokens": max_target_tokens,
            "target_lengths": target_lengths,
        }

    def _text_suffix_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[list[str]]]:
        tokenizer = self.processor.tokenizer
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        spans = []
        tokens_by_sample = []
        audio_token_id = self._maybe_audio_token_id()
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        for batch_index, sample_ids_tensor in enumerate(input_ids):
            sample_ids = sample_ids_tensor.detach().cpu().tolist()
            sample_attention = (
                attention_mask[batch_index].bool()
                if attention_mask is not None
                else torch.ones_like(sample_ids_tensor, dtype=torch.bool)
            )
            tokens = tokenizer.convert_ids_to_tokens(sample_ids)
            tokens_by_sample.append(tokens)
            question_start = self._first_question_token_index(tokens)
            span_start = question_start
            span_end = len(tokens)
            spans.append((span_start, span_end))
            for idx in range(span_start, span_end):
                if not bool(sample_attention[idx]):
                    continue
                token_id = int(sample_ids_tensor[idx].item())
                if audio_token_id is not None and token_id == audio_token_id:
                    continue
                if token_id in special_ids and idx < question_start:
                    continue
                mask[batch_index, idx] = True
        return mask, spans, tokens_by_sample

    def _first_question_token_index(self, tokens: Sequence[str]) -> int:
        user_content_start = self._first_user_content_start(tokens)
        assistant_header_start = self._assistant_header_start(tokens)
        is_index = self._find_decoded_token(tokens, "Is", user_content_start, assistant_header_start)
        if is_index is not None:
            return is_index
        return user_content_start

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

    def _is_newline_token(self, token: str) -> bool:
        return token in {"Ċ", "\n"} or self._decode_token(token) == "\n"

    def _decode_token(self, token: str) -> str:
        tokenizer = self.processor.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(token)
        try:
            return tokenizer.decode([token_id])
        except TypeError:
            return tokenizer.decode(token_id)

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
        for token in (
            getattr(self.processor, "audio_token", None),
            getattr(tokenizer, "audio_token", None),
            "<sound>",
            "<audio>",
            "<|audio|>",
        ):
            if token is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != unk_token_id:
                return int(token_id)
        return None

    @staticmethod
    def _hidden_states_from_layer_output(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple):
            return output[0]
        raise TypeError(f"Unsupported layer output type: {type(output)!r}")


def metadata_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "dataset": sample["dataset"],
        "split": sample["split"],
        "sample_rate": sample["sample_rate"],
        "length_sec": sample["length_sec"],
        "prompt": sample["prompt"],
        "query_event_index": sample["query_event_index"],
        "query_event_label": sample["query_event_label"],
        "background_id": sample["background_id"],
        "background_path": sample["background_path"],
        "background_allowed_event_classes": sample.get("background_allowed_event_classes"),
        "event_snr_db": sample["event_snr_db"],
        "foreground_onset_samples": sample["foreground_onset_samples"],
        "events": sample["events"],
    }


def extend_records(records: dict[str, Any], batch: list[dict[str, Any]], result: dict[str, Any]) -> None:
    for batch_index, sample in enumerate(batch):
        records["activations"].append(result["activations"][:, batch_index].contiguous())
        records["generated_answers"].append(result["generated_answers"][batch_index])
        records["target_lengths"].append(result["target_lengths"][batch_index])
        records["token_indices"].append(
            result["token_mask"][batch_index].nonzero(as_tuple=False).flatten().tolist()
        )
        records["target_tokens"].append([
            result["tokens"][batch_index][idx]
            for idx in records["token_indices"][-1]
        ])
        records["all_input_tokens"].append(result["tokens"][batch_index])
        records["input_ids"].append(result["input_ids"][batch_index])
        if result["attention_mask"] is not None:
            records["attention_masks"].append(result["attention_mask"][batch_index])
        records["metadata"].append(metadata_for_sample(sample))


def estimated_tensor_gb(num_examples: int, shape: Sequence[int], dtype: torch.dtype) -> float:
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    layer_slots, _batch, max_tokens, hidden_size = shape
    return num_examples * layer_slots * max_tokens * hidden_size * bytes_per_element / (1024 ** 3)


def validate_size_guard(
    args: argparse.Namespace,
    num_examples: int,
    batch_activation_shape: Sequence[int],
    dtype: torch.dtype,
) -> None:
    estimate_gb = estimated_tensor_gb(num_examples, batch_activation_shape, dtype)
    print(f"Estimated activation tensor size: {estimate_gb:.2f} GiB")
    if estimate_gb > args.max_output_gb and not args.allow_large_output:
        raise RuntimeError(
            "Estimated activation tensor exceeds "
            f"--max-output-gb={args.max_output_gb:g}. Re-run with "
            "--allow-large-output or reduce split sizes/batch target length."
        )


def save_records(
    output_path: Path,
    records: dict[str, Any],
    args: argparse.Namespace,
    model_id: str,
    layer_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_lengths = torch.tensor([int(value) for value in records["target_lengths"]], dtype=torch.long)
    activation_shapes = [list(tensor.shape) for tensor in records["activations"]]

    payload = {
        "activations": records["activations"],
        "generated_answers": records["generated_answers"],
        "target_lengths": target_lengths,
        "activation_shapes": activation_shapes,
        "layer_names": ["decoder_input", *[f"decoder_layer_{idx}_output" for idx in range(layer_count)]],
        "token_indices": records["token_indices"],
        "target_tokens": records["target_tokens"],
        "all_input_tokens": records["all_input_tokens"],
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            records["input_ids"],
            batch_first=True,
            padding_value=0,
        ),
        "attention_masks": (
            None
            if not records["attention_masks"]
            else torch.nn.utils.rnn.pad_sequence(
                records["attention_masks"],
                batch_first=True,
                padding_value=0,
            )
        ),
        "metadata": records["metadata"],
        "config_json": json.dumps({
            "model": args.model,
            "model_id": model_id,
            "sample_rate": args.sample_rate,
            "length_sec": args.length_sec,
            "split_sizes": {
                "train": args.train_examples,
                "validation": args.validation_examples,
                "test": args.test_examples,
            },
            "num_events": 2,
            "unique_event_classes": True,
            "trim_silence": True,
            "random_seed": args.random_seed,
            "activation_dtype": args.activation_dtype,
            "generation_max_new_tokens": 10,
            "generation_decoding": "greedy",
            "activation_format": "ragged list of one tensor per sample",
            "activation_sample_shape": "[layer_slot, selected_token, hidden]",
            "activation_layer_slot_0": "decoder input before first decoder layer",
            "activation_layer_slots_1_to_n": "outputs after decoder layers 0..n-1",
        }, sort_keys=True),
    }
    torch.save(payload, output_path)


def run_extraction(args: argparse.Namespace) -> Path:
    from dataset.synthetic_sed import SyntheticSoundEventDetectionDataset

    model_id = args.model_id or default_model_id(args.model)
    split_sizes = {
        "train": args.train_examples,
        "validation": args.validation_examples,
        "test": args.test_examples,
    }
    datasets = {
        split: SyntheticSoundEventDetectionDataset(
            foreground_dataset="esc50",
            root=args.dataset_root,
            split=split,
            sample_rate=args.sample_rate,
            min_length_sec=args.length_sec,
            max_length_sec=args.length_sec,
            size=size,
            num_events=2,
            unique_event_classes=True,
            trim_silence=True,
            random_seed=args.random_seed,
            auto_download=not args.no_auto_download,
        )
        for split, size in split_sizes.items()
        if size > 0
    }
    total_examples = sum(len(dataset) for dataset in datasets.values())
    interface = build_model_interface(args.model, model_id=model_id, device=args.device)
    extractor = TextSuffixActivationExtractor(interface, storage_dtype(args.activation_dtype))
    query_rng = torch.Generator().manual_seed(args.random_seed)

    records = {
        "activations": [],
        "generated_answers": [],
        "target_lengths": [],
        "token_indices": [],
        "target_tokens": [],
        "all_input_tokens": [],
        "input_ids": [],
        "attention_masks": [],
        "metadata": [],
    }
    checked_size = False
    layer_count = None

    for split, dataset in datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=list,
        )
        for raw_batch in tqdm(dataloader, desc=f"Capturing {split} activations"):
            batch = [sample_with_prompt(sample, query_rng) for sample in raw_batch]
            result = extractor.capture_batch(
                [sample["waveform"] for sample in batch],
                [sample["prompt"] for sample in batch],
                [sample["sample_rate"] for sample in batch],
            )
            layer_count = result["num_layers"]
            if not checked_size:
                validate_size_guard(
                    args,
                    total_examples,
                    result["activations"].shape,
                    storage_dtype(args.activation_dtype),
                )
                checked_size = True
            extend_records(records, batch, result)

    if not records["activations"]:
        raise RuntimeError("No activations were captured")
    if layer_count is None:
        raise RuntimeError("Could not resolve decoder layer count")

    output_dir = Path(args.output_dir) / model_output_slug(model_id)
    output_path = output_dir / (args.output_name or default_output_name(args))
    save_records(output_path, records, args, model_id, layer_count)
    print(f"Wrote activation bundle: {output_path}")
    print(f"Final file size: {output_path.stat().st_size / (1024 ** 3):.2f} GiB")
    return output_path


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
