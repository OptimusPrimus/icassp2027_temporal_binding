import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "activations"
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_REALDESED_ROOTS = (
    Path("/home/paul/repos/domestic_sed_dataset/data/release"),
    Path("/opt/scratch/paul/data/real_desed"),
)
DEFAULT_SYNTHETIC_SPLIT_SIZES = {
    "train": 2_000,
    "validation": 1_000,
    "test": 1_000,
}
SYNTHETIC_SPLITS = tuple(DEFAULT_SYNTHETIC_SPLIT_SIZES)
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
            "Capture all decoder text-suffix activations for one-event "
            "SyntheticSED ESC-50 examples."
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
    parser.add_argument(
        "--real-desed-root",
        default=None,
        help=(
            "RealDESED release root. Defaults to the first existing path among "
            "/home/paul/repos/domestic_sed_dataset/data/release and "
            "/opt/scratch/paul/data/real_desed."
        ),
    )
    parser.add_argument(
        "--real-desed-split",
        choices=("all", "train", "validation", "test"),
        default="all",
    )
    parser.add_argument(
        "--real-desed-examples",
        type=int,
        default=0,
        help="Optional maximum RealDESED examples to capture. Defaults to 0, so only synthetic audio is saved.",
    )
    parser.add_argument("--max-real-recording-duration-sec", type=float, default=30.0)
    parser.add_argument("--min-real-event-duration-sec", type=float, default=3.0)
    parser.add_argument("--max-real-event-duration-sec", type=float, default=7.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--length-sec", type=float, default=30.0)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_SYNTHETIC_SPLIT_SIZES["train"])
    parser.add_argument("--validation-examples", type=int, default=DEFAULT_SYNTHETIC_SPLIT_SIZES["validation"])
    parser.add_argument("--test-examples", type=int, default=DEFAULT_SYNTHETIC_SPLIT_SIZES["test"])
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
        default=50.0,
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


def display_label(label: str) -> str:
    return str(label).replace("_", " ")


def prompt_for_event(event_label: str) -> str:
    return f"Is there {display_label(event_label)}?"


def resolve_real_desed_root(root: str | Path | None) -> Path:
    if root is not None:
        path = Path(root)
        if not path.exists():
            raise FileNotFoundError(f"RealDESED root does not exist: {path}")
        return path

    for path in DEFAULT_REALDESED_ROOTS:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in DEFAULT_REALDESED_ROOTS)
    raise FileNotFoundError(f"Could not find RealDESED root. Searched: {searched}")


def default_output_name(args: argparse.Namespace) -> str:
    model_name = args.model.replace("-", "_")
    real_desed_slug = f"_realdesed{args.real_desed_examples}" if args.real_desed_examples else ""
    return (
        f"{model_name}_temporal_ids_synthetic_esc50_1event_30s_"
        f"train{args.train_examples}_val{args.validation_examples}_test{args.test_examples}_"
        f"seed{args.random_seed}{real_desed_slug}_all_decoder_text_activations.pt"
    )


def synthetic_split_size(args: argparse.Namespace, split: str) -> int:
    sizes = {
        "train": args.train_examples,
        "validation": args.validation_examples,
        "test": args.test_examples,
    }
    return sizes[split]


def build_synthetic_dataset(args: argparse.Namespace, split: str) -> Any:
    from dataset.synthetic_sed import SyntheticSoundEventDetectionDataset

    return SyntheticSoundEventDetectionDataset(
        foreground_dataset="esc50",
        root=args.dataset_root,
        split=split,
        sample_rate=args.sample_rate,
        min_length_sec=args.length_sec,
        max_length_sec=args.length_sec,
        size=synthetic_split_size(args, split),
        num_events=1,
        unique_event_classes=True,
        trim_silence=True,
        random_seed=args.random_seed,
        auto_download=not args.no_auto_download,
    )


class PromptedSyntheticOneEventDataset(Dataset):
    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.dataset[idx])
        if len(sample["events"]) != 1:
            raise ValueError(f"Expected one synthetic event, got {len(sample['events'])}")
        event = sample["events"][0]
        sample["source_group"] = "synthetic_sed_esc50"
        sample["query_event_index"] = int(event["event_index"])
        sample["query_event_label"] = display_label(event["event_label"])
        sample["prompt"] = prompt_for_event(event["event_label"])
        return sample


class FilteredRealDESEDTemporalDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None,
        splits: Sequence[str],
        sample_rate: int,
        max_recording_duration_sec: float,
        min_event_duration_sec: float,
        max_event_duration_sec: float,
        size: int,
    ):
        from dataset.real_desed import RealDESEDDataset

        root = resolve_real_desed_root(root)
        self.sample_rate = int(sample_rate)
        self.root = root
        self.examples = []
        for split in splits:
            dataset = RealDESEDDataset(
                root=root,
                split=split,
                sample_rate=sample_rate,
                include_metadata=True,
                cache_audio=True,
            )
            for example_index, example in enumerate(dataset.examples):
                duration_sec = example.get("duration_sec")
                if (
                    duration_sec is None
                    or duration_sec > max_recording_duration_sec
                ):
                    continue
                event = self._matching_event(
                    example.get("events", []),
                    min_event_duration_sec=min_event_duration_sec,
                    max_event_duration_sec=max_event_duration_sec,
                )
                if event is None:
                    continue
                self.examples.append({
                    "dataset": dataset,
                    "example_index": example_index,
                    "split": dataset.split,
                    "query_annotation": event,
                })
                if size > 0 and len(self.examples) >= size:
                    break
            if size > 0 and len(self.examples) >= size:
                break

        if not self.examples:
            raise ValueError(
                "No RealDESED examples matched the requested recording-duration, "
                "event-duration, and unique-class filters"
            )

    @staticmethod
    def _matching_event(
        events: Sequence[dict[str, Any]],
        min_event_duration_sec: float,
        max_event_duration_sec: float,
    ) -> dict[str, Any] | None:
        label_counts = Counter(event["event_label"] for event in events)
        candidates = [
            event
            for event in events
            if label_counts[event["event_label"]] == 1
            and min_event_duration_sec <= (event["offset_sec"] - event["onset_sec"]) <= max_event_duration_sec
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda event: (
                abs((event["offset_sec"] - event["onset_sec"]) - 5.0),
                event["onset_sec"],
                event["event_label"],
            ),
        )[0]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.examples[idx]
        sample = dict(record["dataset"][record["example_index"]])
        query = record["query_annotation"]
        matching_events = [
            event
            for event in sample["events"]
            if event["event_index"] == query.get("event_index")
            or (
                event["event_label"] == query["event_label"]
                and abs(event["annotation_onset_sec"] - query["onset_sec"]) < 1e-6
                and abs(event["annotation_offset_sec"] - query["offset_sec"]) < 1e-6
            )
        ]
        if not matching_events:
            raise ValueError(f"Could not locate selected RealDESED event in {sample['id']}")
        event = matching_events[0]
        sample["source_group"] = "real_desed"
        sample["query_event_index"] = int(event["event_index"])
        sample["query_event_label"] = display_label(event["event_label"])
        sample["prompt"] = prompt_for_event(event["event_label"])
        return sample


def build_datasets(args: argparse.Namespace) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {}
    for split in SYNTHETIC_SPLITS:
        if synthetic_split_size(args, split) > 0:
            datasets[f"synthetic_{split}"] = PromptedSyntheticOneEventDataset(
                build_synthetic_dataset(args, split)
            )

    if args.real_desed_examples != 0:
        real_splits = (
            ("train", "validation", "test")
            if args.real_desed_split == "all"
            else (args.real_desed_split,)
        )
        datasets["real_desed"] = FilteredRealDESEDTemporalDataset(
            root=args.real_desed_root,
            splits=real_splits,
            sample_rate=args.sample_rate,
            max_recording_duration_sec=args.max_real_recording_duration_sec,
            min_event_duration_sec=args.min_real_event_duration_sec,
            max_event_duration_sec=args.max_real_event_duration_sec,
            size=args.real_desed_examples,
        )
    return datasets


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
        return {
            "activations": packed["activations"],
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
            spans.append((question_start, len(tokens)))
            for idx in range(question_start, len(tokens)):
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
        "source_group": sample["source_group"],
        "split": sample["split"],
        "sample_rate": sample["sample_rate"],
        "length_sec": sample["length_sec"],
        "prompt": sample["prompt"],
        "query_event_index": sample["query_event_index"],
        "query_event_label": sample["query_event_label"],
        "events": sample["events"],
        "event_labels": sample.get("event_labels"),
        "filename": sample.get("filename", ""),
        "path": sample.get("path", ""),
        "background_id": sample.get("background_id", ""),
        "background_path": sample.get("background_path", ""),
        "background_allowed_event_classes": sample.get("background_allowed_event_classes"),
        "event_snr_db": sample.get("event_snr_db"),
        "foreground_onset_samples": sample.get("foreground_onset_samples"),
        "real_desed_metadata": sample.get("metadata"),
    }


def extend_records(records: dict[str, Any], batch: list[dict[str, Any]], result: dict[str, Any]) -> None:
    for batch_index, sample in enumerate(batch):
        records["activations"].append(result["activations"][:, batch_index].contiguous())
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


def validate_size_guard(
    args: argparse.Namespace,
    num_examples: int,
    batch_activation_shape: Sequence[int],
    dtype: torch.dtype,
) -> None:
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    layer_slots, _batch, max_tokens, hidden_size = batch_activation_shape
    estimate_gb = (
        num_examples
        * layer_slots
        * max_tokens
        * hidden_size
        * bytes_per_element
        / (1024 ** 3)
    )
    print(f"Estimated activation tensor size: {estimate_gb:.2f} GiB")
    if estimate_gb > args.max_output_gb and not args.allow_large_output:
        raise RuntimeError(
            "Estimated activation tensor exceeds "
            f"--max-output-gb={args.max_output_gb:g}. Re-run with "
            "--allow-large-output or reduce dataset sizes."
        )


def save_records(
    output_path: Path,
    records: dict[str, Any],
    args: argparse.Namespace,
    model_id: str,
    layer_count: int,
    dataset_lengths: dict[str, int],
    real_desed_root: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_lengths = torch.tensor([int(value) for value in records["target_lengths"]], dtype=torch.long)
    activation_shapes = [list(tensor.shape) for tensor in records["activations"]]

    payload = {
        "activations": records["activations"],
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
            "synthetic_split_sizes": {
                "train": args.train_examples,
                "validation": args.validation_examples,
                "test": args.test_examples,
            },
            "real_desed_split": args.real_desed_split,
            "real_desed_root_candidates": [str(path) for path in DEFAULT_REALDESED_ROOTS],
            "real_desed_root": None if real_desed_root is None else str(real_desed_root),
            "real_desed_examples_requested": args.real_desed_examples,
            "real_desed_max_recording_duration_sec": args.max_real_recording_duration_sec,
            "real_desed_min_event_duration_sec": args.min_real_event_duration_sec,
            "real_desed_max_event_duration_sec": args.max_real_event_duration_sec,
            "dataset_lengths": dataset_lengths,
            "synthetic_num_events": 1,
            "synthetic_trim_silence": True,
            "random_seed": args.random_seed,
            "activation_dtype": args.activation_dtype,
            "generation": None,
            "forward_pass_only": True,
            "activation_format": "ragged list of one tensor per sample",
            "activation_sample_shape": "[layer_slot, selected_token, hidden]",
            "activation_layer_slot_0": "decoder input before first decoder layer",
            "activation_layer_slots_1_to_n": "outputs after decoder layers 0..n-1",
        }, sort_keys=True),
    }
    torch.save(payload, output_path)


def run_extraction(args: argparse.Namespace) -> Path:
    if args.real_desed_examples < 0:
        raise ValueError("--real-desed-examples must be non-negative")

    model_id = args.model_id or default_model_id(args.model)
    datasets = build_datasets(args)
    dataset_lengths = {name: len(dataset) for name, dataset in datasets.items()}
    real_desed_root = getattr(datasets.get("real_desed"), "root", None)
    total_examples = sum(dataset_lengths.values())
    interface = build_model_interface(args.model, model_id=model_id, device=args.device)
    extractor = TextSuffixActivationExtractor(interface, storage_dtype(args.activation_dtype))

    records = {
        "activations": [],
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

    for name, dataset in datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=list,
        )
        for batch in tqdm(dataloader, desc=f"Capturing {name} activations"):
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
    save_records(output_path, records, args, model_id, layer_count, dataset_lengths, real_desed_root)
    print(f"Wrote activation bundle: {output_path}")
    print(f"Final file size: {output_path.stat().st_size / (1024 ** 3):.2f} GiB")
    return output_path


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
