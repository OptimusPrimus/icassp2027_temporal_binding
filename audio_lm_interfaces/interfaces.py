from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from types import MethodType
from typing import Any, Callable

import numpy as np
import torch

REQUIRED_TRANSFORMERS_VERSIONS = {
    "af3": "5.14.1",
    "af-next": "5.14.1",
    "moss-audio": "4.57.1",
    "qwen3-omni": None,
}
BACKEND_ENVIRONMENT_FILES = {
    "af3": "environment.yml",
    "af-next": "environment.yml",
    "moss-audio": "environment-moss.yml",
    "qwen3-omni": "environment-qwen3-omni.yml",
}
CONCATENATED_MODEL_INPUT_KEYS = {
    "audio_attention_mask",
    "input_features",
    "input_features_mask",
    "num_audio_tokens",
}


class AudioFlamingo3Interface:
    """Small batched interface for AudioFlamingo3 audio-language forwards."""

    def __init__(self, model_id: str = "nvidia/audio-flamingo-3-hf", device: str = "auto"):
        ensure_transformers_version("af3")

        from transformers import (
            AudioFlamingo3ForConditionalGeneration,
            AudioFlamingo3Processor,
        )

        self.processor = AudioFlamingo3Processor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.device = self.model.device
        self.pad_token_id = _tokenizer_pad_token_id(self.processor.tokenizer)
        _set_generation_pad_token_id(self.model, self.pad_token_id)
        self._align_audio_tower_dtype()

    def build_model_inputs(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        teacher_forced_texts: Sequence[str] | str | None = None,
        sampling_rates: Sequence[int] | int | None = None,
    ) -> tuple[Any, list[list[str]]]:
        _validate_batch_lengths(audio_arrays, prompts)
        teacher_forced_texts = _batch_values(
            teacher_forced_texts,
            len(audio_arrays),
            default="",
        )
        sampling_rates = _batch_values(sampling_rates, len(audio_arrays), default=16000)
        if any(sampling_rate != 16000 for sampling_rate in sampling_rates):
            raise ValueError("AudioFlamingo3Interface only supports 16000 Hz audio")

        sample_inputs = [
            self._build_single_model_input(audio_array, prompt, teacher_forced_text)
            for audio_array, prompt, teacher_forced_text in zip(
                audio_arrays,
                prompts,
                teacher_forced_texts,
            )
        ]
        model_inputs = self._pad_model_input_batch(sample_inputs).to(self.device)
        model_inputs = self._cast_floating_model_inputs(model_inputs)
        return model_inputs, self._tokens_from_input_ids(model_inputs["input_ids"])

    def forward(self, model_inputs: Any, **forward_kwargs: Any) -> Any:
        device = next(self.model.parameters()).device
        dtype = getattr(self.model, "dtype", None)
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            with torch.autocast(device_type=device.type, dtype=dtype):
                return self.model(**model_inputs, **forward_kwargs)
        return self.model(**model_inputs, **forward_kwargs)

    @torch.no_grad()
    def generate(
        self,
        model_inputs: Any,
        max_new_tokens: int = 40,
        decode: bool = True,
        **generate_kwargs: Any,
    ) -> Any:
        input_length = model_inputs["input_ids"].shape[1]
        generation_args = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.pad_token_id,
        }
        generation_args.update(generate_kwargs)
        generated_ids = self.model.generate(**model_inputs, **generation_args)
        if not decode:
            return generated_ids
        return [
            self.processor.tokenizer.decode(
                sequence[input_length:],
                skip_special_tokens=True,
            ).strip()
            for sequence in generated_ids
        ]

    def register_forward_hook(
        self,
        hook: Callable[..., Any],
        module: torch.nn.Module | None = None,
        layer: int | None = None,
    ) -> torch.utils.hooks.RemovableHandle:
        target = _resolve_hook_target(self.model, module=module, layer=layer)
        return target.register_forward_hook(hook)

    def _build_single_model_input(
        self,
        audio_array: Any,
        prompt: str,
        teacher_forced_text: str,
    ) -> Any:
        audio_value = audio_array.cpu().numpy() if torch.is_tensor(audio_array) else audio_array
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": audio_value},
                ],
            }
        ]

        model_inputs = self.processor.apply_chat_template(
            [conversation],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "text_kwargs": {"padding": False},
                "audio_kwargs": {
                    "sampling_rate": 16000,
                    "chunk_length": 30.0,
                    "return_attention_mask": True,
                    "padding": "max_length",
                },
            },
        )
        if teacher_forced_text:
            teacher_inputs = self.processor.tokenizer(
                teacher_forced_text,
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
            )
            model_inputs["input_ids"] = torch.cat(
                [model_inputs["input_ids"], teacher_inputs.input_ids],
                dim=1,
            )
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = torch.cat(
                    [model_inputs["attention_mask"], teacher_inputs.attention_mask],
                    dim=1,
                )
        return model_inputs

    def _pad_model_input_batch(self, sample_inputs: Sequence[Any]) -> Any:
        tokenizer = self.processor.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            padded_text = tokenizer.pad(
                [
                    {
                        "input_ids": sample_input["input_ids"].squeeze(0),
                        "attention_mask": sample_input["attention_mask"].squeeze(0),
                    }
                    for sample_input in sample_inputs
                ],
                padding=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding_side

        model_inputs = dict(padded_text)
        for key in sample_inputs[0]:
            if key in model_inputs:
                continue
            values = [sample_input[key] for sample_input in sample_inputs]
            if torch.is_tensor(values[0]):
                model_inputs[key] = _combine_model_input_tensors(key, values)
            else:
                model_inputs[key] = values
        return type(sample_inputs[0])(model_inputs)

    def _tokens_from_input_ids(self, input_ids: torch.Tensor) -> list[list[str]]:
        return [
            self.processor.tokenizer.convert_ids_to_tokens(sample_ids.detach().cpu().tolist())
            for sample_ids in input_ids
        ]

    def _align_audio_tower_dtype(self) -> None:
        audio_tower = _nested_attr(self.model, ("model", "audio_tower"))
        if audio_tower is None:
            audio_tower = _nested_attr(self.model, ("audio_tower",))
        if audio_tower is not None:
            audio_tower.to(dtype=self._model_floating_dtype())

    def _model_floating_dtype(self) -> torch.dtype:
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            return model_dtype
        return next(self.model.parameters()).dtype

    def _cast_floating_model_inputs(self, model_inputs: Any) -> Any:
        model_dtype = self._model_floating_dtype()
        for key, value in list(model_inputs.items()):
            if (
                torch.is_tensor(value)
                and value.is_floating_point()
                and _should_cast_model_input(key)
            ):
                model_inputs[key] = value.to(dtype=model_dtype)
        return model_inputs


class AudioFlamingoNextInterface(AudioFlamingo3Interface):
    """Small batched interface for Audio Flamingo Next audio-language forwards."""

    def __init__(
        self,
        model_id: str = "nvidia/audio-flamingo-next-hf",
        device: str = "auto",
    ):
        ensure_transformers_version("af-next")

        from transformers import AutoProcessor

        model_class = _first_available_transformers_class(
            "MusicFlamingoForConditionalGeneration",
            "AudioFlamingoNextForConditionalGeneration",
            "AutoModelForMultimodalLM",
            "AutoModelForSeq2SeqLM",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = model_class.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.device = self.model.device
        self.pad_token_id = _tokenizer_pad_token_id(self.processor.tokenizer)
        _set_generation_pad_token_id(self.model, self.pad_token_id)
        self._align_audio_tower_dtype()

    def _build_single_model_input(
        self,
        audio_array: Any,
        prompt: str,
        teacher_forced_text: str,
    ) -> Any:
        audio_value = (
            audio_array.detach().cpu().numpy()
            if torch.is_tensor(audio_array)
            else audio_array
        )
        audio_value = np.asarray(audio_value, dtype=np.float32)
        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "audio": audio_value},
                    ],
                }
            ]
        ]

        model_inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        if teacher_forced_text:
            teacher_inputs = self.processor.tokenizer(
                teacher_forced_text,
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
            )
            model_inputs["input_ids"] = torch.cat(
                [model_inputs["input_ids"], teacher_inputs.input_ids],
                dim=1,
            )
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = torch.cat(
                    [model_inputs["attention_mask"], teacher_inputs.attention_mask],
                    dim=1,
                )
        return model_inputs


class MossAudioInterface:
    """Small batched interface for MOSS-Audio forwards."""

    def __init__(
        self,
        model_id: str = "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        device: str = "auto",
    ):
        ensure_transformers_version("moss-audio")

        try:
            from src.modeling_moss_audio import MossAudioModel
            from src.processing_moss_audio import MossAudioProcessor
        except ImportError as exc:
            raise RuntimeError(
                "MOSS-Audio inference requires the OpenMOSS MOSS-Audio package. "
                "Install it with: pip install "
                "git+https://github.com/OpenMOSS/MOSS-Audio.git"
            ) from exc

        self.processor = MossAudioProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            enable_time_marker=True,
        )
        self.model = MossAudioModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map=_device_map(device),
        )
        self.model.eval()
        self.device = self.model.device
        self.pad_token_id = _tokenizer_pad_token_id(self.processor.tokenizer)
        _set_generation_pad_token_id(self.model, self.pad_token_id)
        #self._patch_audio_encoder_chunk_batch()
        #self._patch_prepare_inputs_for_generation()

    def build_model_inputs(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        teacher_forced_texts: Sequence[str] | str | None = None,
        sampling_rates: Sequence[int] | int | None = None,
    ) -> tuple[Any, list[list[str]]]:
        _validate_batch_lengths(audio_arrays, prompts)
        teacher_forced_texts = _batch_values(
            teacher_forced_texts,
            len(audio_arrays),
            default="",
        )
        sampling_rates = _batch_values(
            sampling_rates,
            len(audio_arrays),
            default=self.processor.config.mel_sr,
        )

        encoded_inputs = []
        for audio_array, prompt, teacher_forced_text, sampling_rate in zip(
            audio_arrays,
            prompts,
            teacher_forced_texts,
            sampling_rates,
        ):
            encoded_inputs.append(
                self._build_single_model_input(
                    audio_array,
                    prompt,
                    teacher_forced_text,
                    sampling_rate,
                )
            )

        model_inputs = _pad_moss_inputs(
            encoded_inputs,
            pad_token_id=self.pad_token_id,
            padding_side="left",
        )
        model_inputs = model_inputs.to(self.device)
        if model_inputs.get("audio_data") is not None:
            model_inputs["audio_data"] = model_inputs["audio_data"].to(self.model.dtype)
        model_inputs["audio_input_mask"] = (
            model_inputs["input_ids"] == self.processor.audio_token_id
        )
        return model_inputs, self._tokens_from_input_ids(model_inputs["input_ids"])

    def forward(self, model_inputs: Any, **forward_kwargs: Any) -> Any:
        return self.model(**model_inputs, **forward_kwargs)

    @torch.no_grad()
    def generate(
        self,
        model_inputs: Any,
        max_new_tokens: int = 40,
        decode: bool = True,
        **generate_kwargs: Any,
    ) -> Any:
        input_length = model_inputs["input_ids"].shape[1]
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": self.pad_token_id,
        }
        generation_args.update(generate_kwargs)
        generated_ids = self.model.generate(**model_inputs, **generation_args)
        if not decode:
            return generated_ids
        return [
            self.processor.decode(
                sequence[input_length:],
                skip_special_tokens=True,
            ).strip()
            for sequence in generated_ids
        ]

    def register_forward_hook(
        self,
        hook: Callable[..., Any],
        module: torch.nn.Module | None = None,
        layer: int | None = None,
    ) -> torch.utils.hooks.RemovableHandle:
        target = _resolve_hook_target(self.model, module=module, layer=layer)
        return target.register_forward_hook(hook)

    def _build_single_model_input(
        self,
        audio_array: Any,
        prompt: str,
        teacher_forced_text: str,
        sampling_rate: int,
    ) -> Any:
        target_sr = self.processor.config.mel_sr
        if sampling_rate != target_sr:
            raise ValueError(f"MOSS-Audio requires {target_sr} Hz audio")

        audio_value = audio_array.detach().cpu().numpy() if torch.is_tensor(audio_array) else audio_array
        model_inputs = self.processor(
            text=prompt,
            audios=[np.asarray(audio_value, dtype=np.float32)],
            return_tensors="pt",
        )
        if teacher_forced_text:
            teacher_inputs = self.processor.tokenizer(
                teacher_forced_text,
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
            )
            model_inputs["input_ids"] = torch.cat(
                [model_inputs["input_ids"], teacher_inputs.input_ids],
                dim=1,
            )
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = torch.cat(
                    [model_inputs["attention_mask"], teacher_inputs.attention_mask],
                    dim=1,
                )
        return model_inputs

    def _tokens_from_input_ids(self, input_ids: torch.Tensor) -> list[list[str]]:
        return [
            self.processor.tokenizer.convert_ids_to_tokens(sample_ids.detach().cpu().tolist())
            for sample_ids in input_ids
        ]

    def describe_model_inputs(self, model_inputs: Any) -> list[dict[str, Any]]:
        input_ids = model_inputs["input_ids"].detach().cpu()
        attention_mask = model_inputs.get("attention_mask")
        audio_input_mask = model_inputs.get("audio_input_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.detach().cpu()
        if audio_input_mask is not None:
            audio_input_mask = audio_input_mask.detach().cpu()

        descriptions = []
        for batch_index, sample_ids in enumerate(input_ids):
            active = (
                attention_mask[batch_index].bool()
                if attention_mask is not None
                else torch.ones_like(sample_ids, dtype=torch.bool)
            )
            sample_audio_mask = (
                audio_input_mask[batch_index].bool()
                if audio_input_mask is not None
                else sample_ids == self.processor.audio_token_id
            )
            descriptions.append({
                "sequence_length": int(sample_ids.shape[0]),
                "active_tokens": int(active.sum().item()),
                "audio_tokens": int(sample_audio_mask.sum().item()),
                "audio_start_tokens": int((sample_ids == self.processor.audio_start_id).sum().item()),
                "audio_end_tokens": int((sample_ids == self.processor.audio_end_id).sum().item()),
                "audio_token_positions": sample_audio_mask.nonzero(as_tuple=False).flatten().tolist(),
                "audio_aliases": {
                    "audio_start_id": self.processor.audio_start_id,
                    "audio_token_id": self.processor.audio_token_id,
                    "audio_end_id": self.processor.audio_end_id,
                    "display_tokens": self.processor.tokenizer.convert_ids_to_tokens([
                        self.processor.audio_start_id,
                        self.processor.audio_token_id,
                        self.processor.audio_end_id,
                    ]),
                },
            })
        return descriptions

    def _patch_audio_encoder_chunk_batch(self) -> None:
        audio_encoder = getattr(self.model, "audio_encoder", None)
        if audio_encoder is None:
            return
        audio_encoder._encode_chunk_batch = MethodType(
            _moss_encode_chunk_batch_compat,
            audio_encoder,
        )

    def _patch_prepare_inputs_for_generation(self) -> None:
        self.model.prepare_inputs_for_generation = MethodType(
            _moss_prepare_inputs_for_generation_compat,
            self.model,
        )


class Qwen3OmniInterface:
    """Small batched interface for Qwen3-Omni audio-language generation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: str = "auto",
        return_audio: bool = False,
        attn_implementation: str | None = None,
    ):
        try:
            from qwen_omni_utils import process_mm_info
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except ImportError as exc:
            detail = f" Missing dependency or import: {exc.name}." if exc.name else f" Import failed: {exc}."
            raise RuntimeError(
                "Qwen3-Omni inference requires a source build of Transformers and "
                "qwen-omni-utils. Install environment-qwen3-omni.yml for this backend."
                f"{detail}"
            ) from exc

        model_kwargs = {
            "dtype": "auto",
            "device_map": _device_map(device),
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs,
        )
        if not return_audio and hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.model.eval()
        self.device = self.model.device
        self.return_audio = return_audio
        self.process_mm_info = process_mm_info
        self.pad_token_id = _tokenizer_pad_token_id(self.processor.tokenizer)
        _set_generation_pad_token_id(self.model, self.pad_token_id)

    def build_model_inputs(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        teacher_forced_texts: Sequence[str] | str | None = None,
        sampling_rates: Sequence[int] | int | None = None,
    ) -> tuple[Any, list[list[str]]]:
        _validate_batch_lengths(audio_arrays, prompts)
        teacher_forced_texts = _batch_values(
            teacher_forced_texts,
            len(audio_arrays),
            default="",
        )
        if any(teacher_forced_text for teacher_forced_text in teacher_forced_texts):
            has_teacher_forced_text = True
        else:
            has_teacher_forced_text = False
        sampling_rates = _batch_values(sampling_rates, len(audio_arrays), default=16000)
        target_sr = getattr(
            getattr(self.processor, "feature_extractor", None),
            "sampling_rate",
            16000,
        )
        if any(sampling_rate != target_sr for sampling_rate in sampling_rates):
            raise ValueError(f"Qwen3OmniInterface requires {target_sr} Hz audio")

        conversations = []
        for audio_array, prompt, _sampling_rate in zip(
            audio_arrays,
            prompts,
            sampling_rates,
        ):
            audio_value = (
                audio_array.detach().cpu().numpy()
                if torch.is_tensor(audio_array)
                else audio_array
            )
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": np.asarray(audio_value, dtype=np.float32)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ])

        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self.process_mm_info(
            conversations,
            use_audio_in_video=False,
        )
        model_inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        model_inputs = model_inputs.to(self.device)
        model_inputs = self._cast_floating_model_inputs(model_inputs)
        if has_teacher_forced_text:
            model_inputs = self._append_teacher_forced_texts(
                model_inputs,
                teacher_forced_texts,
            )
        return model_inputs, self._tokens_from_input_ids(model_inputs["input_ids"])

    def forward(self, model_inputs: Any, **forward_kwargs: Any) -> Any:
        return self.model.thinker(**model_inputs, **forward_kwargs)

    @torch.no_grad()
    def generate(
        self,
        model_inputs: Any,
        max_new_tokens: int = 40,
        decode: bool = True,
        **generate_kwargs: Any,
    ) -> Any:
        input_length = model_inputs["input_ids"].shape[1]
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_audio": self.return_audio,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": False,
        }
        generation_args.update(generate_kwargs)
        text_ids, audio = _split_qwen3_omni_generate_output(
            self.model.generate(**model_inputs, **generation_args)
        )

        if not decode:
            return text_ids, audio

        if isinstance(text_ids, str):
            texts = [text_ids]
        elif isinstance(text_ids, Sequence) and all(isinstance(text, str) for text in text_ids):
            texts = list(text_ids)
        else:
            sequences = text_ids.sequences if hasattr(text_ids, "sequences") else text_ids
            texts = self.processor.batch_decode(
                sequences[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        texts = [text.strip() for text in texts]
        if self.return_audio:
            return list(zip(audio, texts))
        return texts

    def register_forward_hook(
        self,
        hook: Callable[..., Any],
        module: torch.nn.Module | None = None,
        layer: int | None = None,
    ) -> torch.utils.hooks.RemovableHandle:
        target = _resolve_hook_target(self.model, module=module, layer=layer)
        return target.register_forward_hook(hook)

    def _tokens_from_input_ids(self, input_ids: torch.Tensor) -> list[list[str]]:
        return [
            self.processor.tokenizer.convert_ids_to_tokens(sample_ids.detach().cpu().tolist())
            for sample_ids in input_ids
        ]

    def _append_teacher_forced_texts(
        self,
        model_inputs: Any,
        teacher_forced_texts: Sequence[str],
    ) -> Any:
        teacher_inputs = self.processor.tokenizer(
            list(teacher_forced_texts),
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        teacher_input_ids = teacher_inputs.input_ids.to(model_inputs["input_ids"].device)
        teacher_attention_mask = teacher_inputs.attention_mask.to(
            model_inputs["attention_mask"].device
        )
        model_inputs["input_ids"] = torch.cat(
            [model_inputs["input_ids"], teacher_input_ids],
            dim=1,
        )
        model_inputs["attention_mask"] = torch.cat(
            [model_inputs["attention_mask"], teacher_attention_mask],
            dim=1,
        )
        return model_inputs

    def _model_floating_dtype(self) -> torch.dtype | None:
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            return model_dtype
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return None

    def _cast_floating_model_inputs(self, model_inputs: Any) -> Any:
        model_dtype = self._model_floating_dtype()
        if model_dtype is None:
            return model_inputs
        for key, value in list(model_inputs.items()):
            if (
                torch.is_tensor(value)
                and value.is_floating_point()
                and _should_cast_model_input(key)
            ):
                model_inputs[key] = value.to(dtype=model_dtype)
        return model_inputs


def ensure_transformers_version(backend: str) -> None:
    try:
        required_version = REQUIRED_TRANSFORMERS_VERSIONS[backend]
    except KeyError as exc:
        raise ValueError(
            "backend must be one of: "
            f"{', '.join(sorted(REQUIRED_TRANSFORMERS_VERSIONS))}"
        ) from exc

    try:
        installed_version = version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "transformers is not installed. Install "
            f"{BACKEND_ENVIRONMENT_FILES[backend]} before running {backend} inference."
        ) from exc

    if required_version is None:
        return

    if installed_version != required_version:
        raise RuntimeError(
            f"{backend} inference requires transformers=={required_version}; "
            f"found transformers=={installed_version}. Use "
            f"{BACKEND_ENVIRONMENT_FILES[backend]} for this backend."
        )


def _tokenizer_pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return pad_token_id

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    if eos_token_id is not None:
        return eos_token_id

    return 0


def _set_generation_pad_token_id(model: Any, pad_token_id: int) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = pad_token_id


def _split_qwen3_omni_generate_output(generated: Any) -> tuple[Any, Any]:
    if isinstance(generated, Mapping):
        text_ids = generated.get("sequences", generated)
        audio = generated.get("audio", generated.get("audios"))
        return text_ids, audio

    if isinstance(generated, tuple):
        if len(generated) == 2:
            return generated
        if len(generated) == 1:
            return generated[0], None

    return generated, None


def _first_available_transformers_class(*class_names: str) -> Any:
    import transformers

    for class_name in class_names:
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError(
        "AF-Next inference requires a generation-capable Transformers class; "
        f"none of {', '.join(class_names)} are available. Upgrade transformers."
    )


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


def _combine_model_input_tensors(key: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
    if _should_concatenate_model_input_tensors(key, values):
        return torch.cat(values, dim=0)
    return torch.stack(values, dim=0)


def _should_concatenate_model_input_tensors(
    key: str,
    values: Sequence[torch.Tensor],
) -> bool:
    if not values or values[0].ndim == 0:
        return False

    rank = values[0].ndim
    trailing_shape = values[0].shape[1:]
    if any(value.ndim != rank or value.shape[1:] != trailing_shape for value in values):
        return False

    if all(value.shape[0] == 1 for value in values):
        return True

    return key in CONCATENATED_MODEL_INPUT_KEYS


def _pad_1d_tensors(
    tensors: Sequence[torch.Tensor],
    pad_value: int,
    padding_side: str = "right",
) -> torch.Tensor:
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    max_length = max(tensor.shape[0] for tensor in tensors)
    padded = torch.full(
        (len(tensors), max_length),
        pad_value,
        dtype=tensors[0].dtype,
    )
    for index, tensor in enumerate(tensors):
        if padding_side == "left":
            padded[index, -tensor.shape[0] :] = tensor
        else:
            padded[index, : tensor.shape[0]] = tensor
    return padded


def _pad_moss_inputs(
    encoded_inputs: Sequence[Any],
    pad_token_id: int = 0,
    padding_side: str = "left",
) -> Any:
    data = {
        "input_ids": _pad_1d_tensors(
            [inputs["input_ids"].squeeze(0) for inputs in encoded_inputs],
            pad_value=pad_token_id,
            padding_side=padding_side,
        ),
        "attention_mask": _pad_1d_tensors(
            [inputs["attention_mask"].squeeze(0) for inputs in encoded_inputs],
            pad_value=0,
            padding_side=padding_side,
        ),
    }
    if encoded_inputs[0].get("audio_data") is not None:
        audio_tensors = [inputs["audio_data"].squeeze(0) for inputs in encoded_inputs]
        max_audio_length = max(tensor.shape[-1] for tensor in audio_tensors)
        audio_data = torch.zeros(
            len(audio_tensors),
            audio_tensors[0].shape[0],
            max_audio_length,
            dtype=audio_tensors[0].dtype,
        )
        for index, tensor in enumerate(audio_tensors):
            audio_data[index, :, : tensor.shape[-1]] = tensor
        data["audio_data"] = audio_data
        data["audio_data_seqlens"] = torch.cat(
            [inputs["audio_data_seqlens"] for inputs in encoded_inputs],
            dim=0,
        )
    return type(encoded_inputs[0])(data=data, tensor_type="pt")


def _moss_encode_chunk_batch_compat(
    self: torch.nn.Module,
    input_features: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if input_features.dim() == 2:
        input_features = input_features.unsqueeze(0)

    downsampled_lengths = self._compute_downsampled_length(seq_lengths)

    x = input_features.unsqueeze(1)
    x = self.gelu(self.conv1(x))
    x = self.gelu(self.conv2(x))
    x = self.gelu(self.conv3(x))

    x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
    x = self.stem_proj(x)

    max_len = int(downsampled_lengths.max().item())
    if x.size(1) > max_len:
        x = x[:, :max_len, :]

    positions = self.embed_positions(x.shape[1], x.device)
    x = x + positions.to(x.dtype)

    padding_mask = (
        torch.arange(x.size(1), device=x.device)[None, :] >= downsampled_lengths[:, None]
    )
    attention_mask = (1.0 - (~padding_mask).to(dtype=x.dtype)) * torch.finfo(x.dtype).min
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    deepstack_hidden_states = [None] * len(self.deepstack_encoder_layer_indexes)
    for layer_idx, layer in enumerate(self.layers):
        layer_output = layer(
            x,
            attention_mask,
            layer_head_mask=None,
            output_attentions=False,
        )
        x = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        capture_idx = self._deepstack_capture_map.get(layer_idx)
        if capture_idx is not None:
            deepstack_hidden_states[capture_idx] = x

    x = self.layer_norm(x)
    x = self.out_proj(x)

    ordered_deepstack_hidden_states = [
        hidden_state for hidden_state in deepstack_hidden_states if hidden_state is not None
    ]
    if not isinstance(self.out_proj, torch.nn.Identity):
        ordered_deepstack_hidden_states = [
            self.out_proj(hidden_state)
            for hidden_state in ordered_deepstack_hidden_states
        ]
    return x, ordered_deepstack_hidden_states


def _moss_prepare_inputs_for_generation_compat(
    self: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    next_sequence_length: int | None = None,
    is_first_iteration: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    position_ids = kwargs.get("position_ids", None)
    is_prefill = past_key_values is None
    if position_ids is None and attention_mask is not None:
        position_ids = _position_ids_from_attention_mask(attention_mask)

    if is_prefill:
        if next_sequence_length is not None and input_ids is not None:
            input_ids = input_ids[:, -next_sequence_length:]
            if position_ids is not None:
                position_ids = position_ids[:, -next_sequence_length:]
        audio_data = kwargs.get("audio_data", None)
        audio_input_mask = kwargs.get("audio_input_mask", None)
        audio_data_seqlens = kwargs.get("audio_data_seqlens", None)
        if audio_input_mask is not None and input_ids is not None:
            audio_input_mask = _align_sequence_mask(audio_input_mask, input_ids.shape[1])
    else:
        input_ids = input_ids[:, -1:]
        if position_ids is not None:
            position_ids = position_ids[:, -1:]
        audio_data = None
        audio_input_mask = None
        audio_data_seqlens = None

    if inputs_embeds is not None and is_prefill:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "audio_data": audio_data,
            "audio_input_mask": audio_input_mask,
            "audio_data_seqlens": audio_data_seqlens,
        }
    )
    return model_inputs


def _align_sequence_mask(mask: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if mask.shape[1] == sequence_length:
        return mask
    if mask.shape[1] > sequence_length:
        return mask[:, -sequence_length:]

    pad = torch.zeros(
        mask.shape[0],
        sequence_length - mask.shape[1],
        dtype=mask.dtype,
        device=mask.device,
    )
    return torch.cat([pad, mask], dim=1)


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def _device_map(device: str) -> str:
    if device == "auto":
        return "auto"
    return device


def _should_cast_model_input(key: str) -> bool:
    key = key.lower()
    if "mask" in key or key.startswith("is_"):
        return False
    return any(name in key for name in ("feature", "pixel", "embedding", "value"))


def _resolve_hook_target(
    model: torch.nn.Module,
    module: torch.nn.Module | None = None,
    layer: int | None = None,
) -> torch.nn.Module:
    if module is not None and layer is not None:
        raise ValueError("Pass either module or layer, not both")
    if module is not None:
        return module
    if layer is None:
        return model

    layers, name = _decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer must be in [0, {len(layers) - 1}] for {name}")
    return layers[layer]


def _decoder_layers(model: torch.nn.Module) -> tuple[Sequence[torch.nn.Module], str]:
    layer_containers = [
        ("thinker.model.layers", ("thinker", "model", "layers")),
        ("language_model.model.layers", ("language_model", "model", "layers")),
        ("language_model.layers", ("language_model", "layers")),
        ("alm.model.layers", ("alm", "model", "layers")),
        ("model.layers", ("model", "layers")),
        ("layers", ("layers",)),
    ]
    for name, path in layer_containers:
        module = _nested_attr(model, path)
        if module is not None:
            return module, name

    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        for attr in ("model", "layers"):
            if hasattr(decoder, attr):
                candidate = getattr(decoder, attr)
                if attr == "model" and hasattr(candidate, "layers"):
                    candidate = candidate.layers
                return candidate, f"decoder.{attr}"

    raise AttributeError("Could not find decoder layers on the model")


def _nested_attr(module: Any, path: Sequence[str]) -> Any | None:
    for attr in path:
        if not hasattr(module, attr):
            return None
        module = getattr(module, attr)
    return module
