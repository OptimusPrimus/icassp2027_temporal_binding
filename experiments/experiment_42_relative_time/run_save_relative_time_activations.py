import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
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
from experiments.experiment_41_temporal_IDs.run_save_activations import (
    TextSuffixActivationExtractor,
    display_label,
    prompt_for_event,
    storage_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture all decoder text-suffix activations for RealDESED examples "
            "with random 15-30s input windows."
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
        default=None,
        help="Maximum RealDESED examples to capture. Defaults to all eligible examples; use 0 to skip.",
    )
    parser.add_argument("--min-real-event-duration-sec", type=float, default=3.0)
    parser.add_argument("--max-real-event-duration-sec", type=float, default=7.0)
    parser.add_argument("--min-length-sec", type=float, default=15.0)
    parser.add_argument("--max-length-sec", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also capture synthetic ESC-50 examples. By default only RealDESED is captured.",
    )
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


class RelativeWindowRealDESEDDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None,
        splits: Sequence[str],
        sample_rate: int,
        min_event_duration_sec: float,
        max_event_duration_sec: float,
        min_length_sec: float,
        max_length_sec: float,
        size: int,
        random_seed: int,
    ):
        from dataset.real_desed import RealDESEDDataset

        root = resolve_real_desed_root(root)
        self.sample_rate = int(sample_rate)
        self.min_length_sec = float(min_length_sec)
        self.max_length_sec = float(max_length_sec)
        self.random_seed = int(random_seed)
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
                "No RealDESED examples matched the requested duration and unique-class filters"
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
        event = self._selected_event(sample, record["query_annotation"])
        original_num_samples = len(sample["waveform"])
        window = self._sample_window(idx, original_num_samples, event)
        window_end = window["start_samples"] + window["num_samples"]
        if window_end > original_num_samples:
            raise RuntimeError(f"Sampled window exceeds recording length for {sample['id']}")
        waveform = sample["waveform"][window["start_samples"]:window_end].astype(np.float32, copy=False)
        events = []
        for event_item in sample["events"]:
            adjusted = self._event_in_window(
                event_item,
                window["start_samples"],
                window["num_samples"],
            )
            if adjusted is None:
                continue
            adjusted["event_index"] = len(events)
            events.append(adjusted)
        query_matches = [
            item
            for item in events
            if item["original_event_index"] == event["event_index"]
        ]
        if not query_matches:
            raise ValueError(f"Selected RealDESED event fell outside sampled window for {sample['id']}")
        query_event = query_matches[0]

        sample["source_group"] = "real_desed"
        sample["waveform"] = waveform
        sample["length_sec"] = len(waveform) / float(self.sample_rate)
        sample["events"] = events
        sample["event_labels"] = [item["event_label"] for item in events]
        sample["query_event_index"] = int(query_event["event_index"])
        sample["query_event_label"] = display_label(query_event["event_label"])
        sample["prompt"] = prompt_for_event(query_event["event_label"])
        sample["relative_window"] = {
            **window,
            "start_sec": window["start_samples"] / float(self.sample_rate),
            "end_sec": window["end_samples"] / float(self.sample_rate),
            "length_sec": sample["length_sec"],
            "original_num_samples": original_num_samples,
            "original_length_sec": original_num_samples / float(self.sample_rate),
            "padded_samples": 0,
        }
        return sample

    def _selected_event(
        self,
        sample: dict[str, Any],
        query: dict[str, Any],
    ) -> dict[str, Any]:
        matches = [
            event
            for event in sample["events"]
            if event["event_index"] == query.get("event_index")
            or (
                event["event_label"] == query["event_label"]
                and abs(event["annotation_onset_sec"] - query["onset_sec"]) < 1e-6
                and abs(event["annotation_offset_sec"] - query["offset_sec"]) < 1e-6
            )
        ]
        if not matches:
            raise ValueError(f"Could not locate selected RealDESED event in {sample['id']}")
        return matches[0]

    def _sample_window(
        self,
        idx: int,
        original_num_samples: int,
        event: dict[str, Any],
    ) -> dict[str, int]:
        rng = np.random.default_rng(self.random_seed + idx)
        requested_min_samples = int(round(self.min_length_sec * self.sample_rate))
        requested_max_samples = int(round(self.max_length_sec * self.sample_rate))
        max_samples = min(requested_max_samples, original_num_samples)
        min_samples = min(requested_min_samples, max_samples)
        window_samples = int(rng.integers(min_samples, max_samples + 1))

        latest_start = max(0, event["onset_samples"])
        earliest_start = max(0, event["offset_samples"] - window_samples)
        max_recording_start = max(0, original_num_samples - window_samples)
        low = min(earliest_start, max_recording_start)
        high = min(latest_start, max_recording_start)
        if low > high:
            low = high
        start = int(rng.integers(low, high + 1)) if high > low else int(low)
        return {
            "start_samples": start,
            "end_samples": start + window_samples,
            "num_samples": window_samples,
        }

    def _event_in_window(
        self,
        event: dict[str, Any],
        window_start: int,
        window_samples: int,
    ) -> dict[str, Any] | None:
        window_end = window_start + window_samples
        clipped_onset = max(event["onset_samples"], window_start)
        clipped_offset = min(event["offset_samples"], window_end)
        if clipped_offset <= clipped_onset:
            return None
        onset_samples = clipped_onset - window_start
        offset_samples = clipped_offset - window_start
        adjusted = dict(event)
        adjusted["original_event_index"] = int(event["event_index"])
        adjusted["original_onset_samples"] = int(event["onset_samples"])
        adjusted["original_offset_samples"] = int(event["offset_samples"])
        adjusted["original_onset_sec"] = float(event["onset_sec"])
        adjusted["original_offset_sec"] = float(event["offset_sec"])
        adjusted["onset_samples"] = int(onset_samples)
        adjusted["offset_samples"] = int(offset_samples)
        adjusted["onset_sec"] = onset_samples / float(self.sample_rate)
        adjusted["offset_sec"] = offset_samples / float(self.sample_rate)
        adjusted["duration_sec"] = adjusted["offset_sec"] - adjusted["onset_sec"]
        return adjusted


def build_datasets(args: argparse.Namespace) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {}
    synthetic_split_sizes = {
        "train": args.train_examples,
        "validation": args.validation_examples,
        "test": args.test_examples,
    }
    if args.include_synthetic:
        from dataset.synthetic_sed import SyntheticSoundEventDetectionDataset

        for split, size in synthetic_split_sizes.items():
            if size > 0:
                datasets[f"synthetic_{split}"] = PromptedSyntheticOneEventDataset(
                    SyntheticSoundEventDetectionDataset(
                        foreground_dataset="esc50",
                        root=args.dataset_root,
                        split=split,
                        sample_rate=args.sample_rate,
                        min_length_sec=args.min_length_sec,
                        max_length_sec=args.max_length_sec,
                        size=size,
                        num_events=1,
                        unique_event_classes=True,
                        trim_silence=True,
                        random_seed=args.random_seed,
                        auto_download=not args.no_auto_download,
                    )
                )

    if args.real_desed_examples != 0:
        real_splits = (
            ("train", "validation", "test")
            if args.real_desed_split == "all"
            else (args.real_desed_split,)
        )
        datasets["real_desed"] = RelativeWindowRealDESEDDataset(
            root=args.real_desed_root,
            splits=real_splits,
            sample_rate=args.sample_rate,
            min_event_duration_sec=args.min_real_event_duration_sec,
            max_event_duration_sec=args.max_real_event_duration_sec,
            min_length_sec=args.min_length_sec,
            max_length_sec=args.max_length_sec,
            size=0 if args.real_desed_examples is None else args.real_desed_examples,
            random_seed=args.random_seed,
        )
    return datasets


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
        records["metadata"].append({
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
            "relative_window": sample.get("relative_window"),
        })


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
            "min_length_sec": args.min_length_sec,
            "max_length_sec": args.max_length_sec,
            "synthetic_split_sizes": {
                "train": args.train_examples,
                "validation": args.validation_examples,
                "test": args.test_examples,
            },
            "real_desed_split": args.real_desed_split,
            "real_desed_root_candidates": [str(path) for path in DEFAULT_REALDESED_ROOTS],
            "real_desed_root": None if real_desed_root is None else str(real_desed_root),
            "real_desed_examples_requested": args.real_desed_examples,
            "real_desed_min_event_duration_sec": args.min_real_event_duration_sec,
            "real_desed_max_event_duration_sec": args.max_real_event_duration_sec,
            "dataset_lengths": dataset_lengths,
            "include_synthetic": args.include_synthetic,
            "synthetic_num_events": 1,
            "synthetic_trim_silence": True,
            "real_desed_windowing": "random input window containing selected event, capped to recording length",
            "random_seed": args.random_seed,
            "activation_dtype": args.activation_dtype,
            "generation": None,
            "activation_format": "ragged list of one tensor per sample",
            "activation_sample_shape": "[layer_slot, selected_token, hidden]",
            "activation_layer_slot_0": "decoder input before first decoder layer",
            "activation_layer_slots_1_to_n": "outputs after decoder layers 0..n-1",
        }, sort_keys=True),
    }
    torch.save(payload, output_path)


def run_extraction(args: argparse.Namespace) -> Path:
    if args.min_length_sec > args.max_length_sec:
        raise ValueError("--min-length-sec must not exceed --max-length-sec")
    if args.real_desed_examples is not None and args.real_desed_examples < 0:
        raise ValueError("--real-desed-examples must be non-negative")

    model_id = args.model_id or default_model_id(args.model)
    datasets = build_datasets(args)
    dataset_lengths = {name: len(dataset) for name, dataset in datasets.items()}
    if not dataset_lengths:
        raise RuntimeError("No datasets selected. Use --include-synthetic or set --real-desed-examples above 0.")
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
                layer_slots, _batch, max_tokens, hidden_size = result["activations"].shape
                estimate_gb = (
                    total_examples
                    * layer_slots
                    * max_tokens
                    * hidden_size
                    * torch.empty((), dtype=storage_dtype(args.activation_dtype)).element_size()
                    / (1024 ** 3)
                )
                print(f"Estimated activation tensor size: {estimate_gb:.2f} GiB")
                if estimate_gb > args.max_output_gb and not args.allow_large_output:
                    raise RuntimeError(
                        "Estimated activation tensor exceeds "
                        f"--max-output-gb={args.max_output_gb:g}. Re-run with "
                        "--allow-large-output or reduce dataset sizes."
                    )
                checked_size = True
            extend_records(records, batch, result)

    if not records["activations"]:
        raise RuntimeError("No activations were captured")
    if layer_count is None:
        raise RuntimeError("Could not resolve decoder layer count")

    output_dir = Path(args.output_dir) / model_output_slug(model_id)
    if args.output_name is None:
        real_count_slug = "all" if args.real_desed_examples is None else str(args.real_desed_examples)
        source_slug = (
            "relative_time_synthetic_esc50_1event"
            if args.include_synthetic
            else "relative_time_realdesed"
        )
        synthetic_slug = (
            f"_train{args.train_examples}_val{args.validation_examples}_test{args.test_examples}"
            if args.include_synthetic
            else ""
        )
        output_name = (
            f"{args.model.replace('-', '_')}_{source_slug}_"
            f"len{args.min_length_sec:g}-{args.max_length_sec:g}s"
            f"{synthetic_slug}_realdesed{real_count_slug}_seed{args.random_seed}_"
            "all_decoder_text_activations.pt"
        )
    else:
        output_name = args.output_name
    output_path = output_dir / output_name
    save_records(output_path, records, args, model_id, layer_count, dataset_lengths, real_desed_root)
    print(f"Wrote activation bundle: {output_path}")
    print(f"Final file size: {output_path.stat().st_size / (1024 ** 3):.2f} GiB")
    return output_path


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
