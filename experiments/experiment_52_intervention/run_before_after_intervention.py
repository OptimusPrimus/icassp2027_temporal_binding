#!/usr/bin/env python3
"""Run per-event RealDESED relative temporal-position interventions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_41_DIR = REPO_ROOT / "experiments" / "experiment_41_temporal_IDs"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_PLOT_DIR = EXPERIMENT_DIR / "plots"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (  # noqa: E402
    DEFAULT_MODEL_IDS,
    OrderingIntervention,
    _batch_values,
    _validate_batch_lengths,
    build_model_interface,
    default_model_id,
    model_output_slug,
)
from experiments.experiment_41_temporal_IDs.plot_temporal_id_analysis import (  # noqa: E402
    center_by_train_class_mean,
    extract_raw_examples,
    load_bundle,
    temporal_ids_for_split,
    train_class_means,
)
CANDIDATE_TEXTS = [" before", " after"]
GROUP_ORDER = [
    "all",
    "predicted_before",
    "predicted_after",
    "actual_before",
    "actual_after",
]
GROUP_LABELS = {
    "all": "All",
    "predicted_before": "Predicted before",
    "predicted_after": "Predicted after",
    "actual_before": "Actually before",
    "actual_after": "Actually after",
}
QUERY_TOKEN_TYPES = ("part2", "part10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RealDESED before/after prediction with per-example relative "
            "temporal-ID directions that move query events to the "
            "beginning or end of the audio."
        )
    )
    parser.add_argument("--model", default="af-next", choices=tuple(DEFAULT_MODEL_IDS))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--intervention-layer", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--temporal-id-activation-path", type=Path, default=None)
    parser.add_argument(
        "--temporal-id-results-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "experiment_41_temporal_IDs" / "plots",
    )
    parser.add_argument("--time-field", choices=("onset", "center", "offset"), default="center")
    parser.add_argument("--token", default="query_event")
    parser.add_argument("--bin-width-sec", type=float, default=2.5)
    parser.add_argument("--first-bin-center-sec", type=float, default=3.75)
    parser.add_argument("--min-bin-count", type=int, default=None)
    parser.add_argument("--exclude-bin-centers-sec", type=float, nargs="*", default=[])
    parser.add_argument(
        "--use-global-direction",
        action="store_true",
        help=(
            "Use the normalized late-minus-early temporal-ID direction instead "
            "of per-event relative directions."
        ),
    )
    parser.add_argument("--early-range-sec", type=float, nargs=2, default=(2.5, 5.0))
    parser.add_argument("--late-range-sec", type=float, nargs=2, default=(25.0, 27.5))
    parser.add_argument("--synthetic-reference-duration-sec", type=float, default=30.0)
    parser.add_argument(
        "--beginning-target-sec",
        type=float,
        default=3.75,
        help="Synthetic temporal-ID time used as the default beginning target.",
    )
    parser.add_argument(
        "--end-target-sec",
        type=float,
        default=26.25,
        help="Synthetic temporal-ID time used as the default end target.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/paul/repos/domestic_sed_dataset/data/release",
    )
    parser.add_argument("--split", default="all", choices=("train", "validation", "test", "all"))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-event-duration-sec", type=float, default=2.5)
    parser.add_argument("--max-event-duration-sec", type=float, default=5.5)
    parser.add_argument("--min-gap-sec", type=float, default=1.0)
    parser.add_argument("--exclude-event-labels", nargs="*", default=["footsteps"])
    parser.add_argument("--max-audio-length-sec", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. Primary intervention results are written to "
            "intervention/{model_slug}; derived aligned and summary CSVs are "
            "written to aligned/{model_slug} and summary/{model_slug}."
        ),
    )
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--csv-prefix", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _safe_number(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def experiment_stem(args: argparse.Namespace) -> str:
    model_slug = args.model.replace("-", "_")
    prefix = args.csv_prefix or (
        f"{model_slug}_real_desed_prediction_{args.split}"
        f"_max{args.max_event_duration_sec:g}s_n{args.size}"
    )
    return (
        f"{prefix}_single_event_relative_temporal_move_layer{args.intervention_layer}"
        f"_alpha{_safe_number(args.alpha)}"
        f"{'_global_direction' if getattr(args, 'use_global_direction', False) else ''}"
    )


class RealDESEDOrderingDataset(Dataset):
    """RealDESED event-pair ordering dataset with experiment-52 filters."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "all",
        sample_rate: int = 16000,
        size: int | None = 1000,
        min_event_duration_sec: float = 2.5,
        max_event_duration_sec: float = 5.5,
        min_gap_sec: float = 1.0,
        excluded_event_labels: Sequence[str] | None = ("footsteps",),
        max_audio_length_sec: float = 30.0,
        random_seed: int = 0,
        cache_audio: bool = True,
    ):
        from dataset.real_desed import RealDESEDDataset

        self.root = root
        self.split = split
        self.sample_rate = int(sample_rate)
        self.size = size
        self.min_event_duration_sec = float(min_event_duration_sec)
        self.max_event_duration_sec = float(max_event_duration_sec)
        self.min_gap_sec = float(min_gap_sec)
        self.excluded_event_labels = set(excluded_event_labels or [])
        self.max_audio_length_sec = float(max_audio_length_sec)
        self.random_seed = int(random_seed)
        splits = ("train", "validation", "test") if split == "all" else (split,)
        self.base_datasets = [
            RealDESEDDataset(
                root=root,
                split=split_name,
                sample_rate=self.sample_rate,
                cache_audio=cache_audio,
            )
            for split_name in splits
        ]

        if self.min_event_duration_sec < 0:
            raise ValueError("min_event_duration_sec cannot be negative")
        if self.max_event_duration_sec <= 0:
            raise ValueError("max_event_duration_sec must be positive")
        if self.min_event_duration_sec > self.max_event_duration_sec:
            raise ValueError("min_event_duration_sec must not exceed max_event_duration_sec")
        if self.min_gap_sec < 0:
            raise ValueError("min_gap_sec cannot be negative")
        if self.max_audio_length_sec <= 0:
            raise ValueError("max_audio_length_sec must be positive")

        self.examples = self._build_examples()
        if not self.examples:
            raise ValueError(
                "No RealDESED ordering pairs matched the constraints: "
                "distinct separated events, duration range "
                f"{self.min_event_duration_sec:g}-{self.max_event_duration_sec:g}s, "
                f"minimum gap {self.min_gap_sec:g}s, and no duplicate selected labels."
            )

    @staticmethod
    def _display_label(label: str) -> str:
        return label.replace("_", " ")

    @staticmethod
    def _duration_sec(event: dict[str, Any]) -> float:
        return float(event["offset_sec"]) - float(event["onset_sec"])

    @staticmethod
    def _non_overlapping(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            float(left["offset_sec"]) <= float(right["onset_sec"])
            or float(right["offset_sec"]) <= float(left["onset_sec"])
        )

    def _gap_sec(self, first_event: dict[str, Any], second_event: dict[str, Any]) -> float:
        first_event, second_event = sorted(
            [first_event, second_event],
            key=lambda event: (float(event["onset_sec"]), float(event["offset_sec"])),
        )
        return float(second_event["onset_sec"]) - float(first_event["offset_sec"])

    def _eligible_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        label_counts = Counter(event["event_label"] for event in events)
        return [
            event
            for event in events
            if label_counts[event["event_label"]] == 1
            and event["event_label"] not in self.excluded_event_labels
            and self.min_event_duration_sec <= self._duration_sec(event) <= self.max_event_duration_sec
        ]

    def _build_examples(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.random_seed)
        examples = []
        for dataset_index, base_dataset in enumerate(self.base_datasets):
            for audio_index, recording in enumerate(base_dataset.examples):
                eligible_events = self._eligible_events(recording["events"])
                pair_index = 0
                for left_idx, left_event in enumerate(eligible_events):
                    for right_event in eligible_events[left_idx + 1:]:
                        if left_event["event_label"] == right_event["event_label"]:
                            continue
                        if not self._non_overlapping(left_event, right_event):
                            continue
                        if self._gap_sec(left_event, right_event) < self.min_gap_sec:
                            continue

                        first_event, second_event = sorted(
                            [left_event, right_event],
                            key=lambda event: (float(event["onset_sec"]), float(event["offset_sec"])),
                        )
                        window = self._audio_window(recording, first_event, second_event)
                        if window is None:
                            continue
                        query_is_first = bool(rng.integers(0, 2))
                        query_event = first_event if query_is_first else second_event
                        reference_event = second_event if query_is_first else first_event
                        examples.append({
                            "id": f"{base_dataset.split}_{recording['id']}__pair{pair_index:03d}",
                            "dataset_index": dataset_index,
                            "audio_index": audio_index,
                            "recording_split": base_dataset.split,
                            "recording_id": recording["id"],
                            "filename": recording["filename"],
                            "path": recording["path"],
                            "query_event": dict(query_event),
                            "reference_event": dict(reference_event),
                            "first_event": dict(first_event),
                            "second_event": dict(second_event),
                            "gap_sec": self._gap_sec(first_event, second_event),
                            "window_start_sec": window[0],
                            "window_end_sec": window[1],
                            "query_label": self._display_label(query_event["event_label"]),
                            "reference_label": self._display_label(reference_event["event_label"]),
                            "answer": "before" if query_is_first else "after",
                        })
                        pair_index += 1
        permutation = rng.permutation(len(examples)).tolist()
        examples = [examples[index] for index in permutation]
        if self.size is not None:
            if self.size < 0:
                raise ValueError("size must be non-negative or None")
            examples = examples[: self.size]
        return examples

    def _audio_window(
        self,
        recording: dict[str, Any],
        first_event: dict[str, Any],
        second_event: dict[str, Any],
    ) -> tuple[float, float] | None:
        first_onset = float(first_event["onset_sec"])
        second_offset = float(second_event["offset_sec"])
        pair_span = second_offset - first_onset
        if pair_span > self.max_audio_length_sec:
            return None

        recording_duration = float(recording.get("duration_sec") or recording["length_sec"])
        window_length = min(self.max_audio_length_sec, recording_duration)
        context = max(0.0, window_length - pair_span) / 2.0
        latest_start = max(0.0, recording_duration - window_length)
        window_start = min(max(0.0, first_onset - context), latest_start)
        window_end = window_start + window_length
        if first_onset < window_start or second_offset > window_end:
            return None
        return window_start, window_end

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _event_fields(
        prefix: str,
        event: dict[str, Any],
        window_start_sec: float,
    ) -> dict[str, Any]:
        relative_onset_sec = float(event["onset_sec"]) - window_start_sec
        relative_offset_sec = float(event["offset_sec"]) - window_start_sec
        return {
            f"{prefix}_event_label": event["event_label"],
            f"{prefix}_onset_sec": relative_onset_sec,
            f"{prefix}_offset_sec": relative_offset_sec,
            f"{prefix}_duration_sec": relative_offset_sec - relative_onset_sec,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        recording = self.base_datasets[example["dataset_index"]][example["audio_index"]]
        sample_rate = int(recording["sample_rate"])
        window_start_sec = float(example["window_start_sec"])
        window_end_sec = float(example["window_end_sec"])
        start_sample = int(round(window_start_sec * sample_rate))
        end_sample = int(round(window_end_sec * sample_rate))
        waveform = recording["waveform"][start_sample:end_sample]

        return {
            "id": example["id"],
            "dataset": "real_desed",
            "split": example["recording_split"],
            "recording_id": example["recording_id"],
            "filename": example["filename"],
            "path": example["path"],
            "waveform": waveform,
            "sample_rate": sample_rate,
            "length_sec": len(waveform) / float(sample_rate),
            "recording_length_sec": recording["length_sec"],
            "audio_window_start_sec": window_start_sec,
            "audio_window_end_sec": window_end_sec,
            "event_gap_sec": example["gap_sec"],
            "query_label": example["query_label"],
            "reference_label": example["reference_label"],
            "answer": example["answer"],
            **self._event_fields("query", example["query_event"], window_start_sec),
            **self._event_fields("reference", example["reference_event"], window_start_sec),
            **self._event_fields("first", example["first_event"], window_start_sec),
            **self._event_fields("second", example["second_event"], window_start_sec),
        }


class OrderingPredictionScorer(OrderingIntervention):
    """Next-token before/after scorer with optional temporal-ID addition."""

    def forward_prediction(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        teacher_forced_texts: Sequence[str] | str | None = None,
        sampling_rates: Sequence[int] | int | None = None,
        keywords: Sequence[str] = (" before", " after"),
        top_k: int = 10,
        temporal_direction: torch.Tensor | None = None,
        temporal_alpha: float = 1.0,
        intervention_layer: int | None = None,
        intervention_token_type: str | Sequence[str] = ("part2", "part10"),
        **forward_kwargs: Any,
    ) -> list[dict[str, Any]]:
        _validate_batch_lengths(audio_arrays, prompts)
        teacher_forced_texts = _batch_values(
            teacher_forced_texts,
            len(audio_arrays),
            default="",
        )
        self._validate_query_args(keywords, top_k)
        candidate_ids = self._candidate_token_ids(keywords)
        model_inputs, _tokens = self.interface.build_model_inputs(
            audio_arrays=audio_arrays,
            prompts=prompts,
            teacher_forced_texts=teacher_forced_texts,
            sampling_rates=sampling_rates,
        )
        if temporal_direction is None:
            with torch.no_grad():
                outputs = self.interface.forward(model_inputs, **forward_kwargs)
        else:
            if intervention_layer is None:
                raise ValueError("intervention_layer is required for temporal intervention")
            outputs = self._forward_with_temporal_direction_add(
                model_inputs,
                temporal_direction=temporal_direction,
                alpha=temporal_alpha,
                layer=intervention_layer,
                token_type=intervention_token_type,
                forward_kwargs=forward_kwargs,
            )
        return self._probability_results(
            outputs.logits,
            keywords,
            candidate_ids,
            top_k,
            attention_mask=model_inputs.get("attention_mask"),
        )

    def _forward_with_temporal_direction_add(
        self,
        model_inputs: Any,
        temporal_direction: torch.Tensor,
        alpha: float,
        layer: int,
        token_type: str | Sequence[str],
        forward_kwargs: dict[str, Any],
    ) -> Any:
        target_layer = self.decoder_layer(layer)
        token_mask = self.token_mask(model_inputs, token_type)
        if not bool(token_mask.any()):
            raise ValueError(f"No {token_type!r} query tokens found for temporal intervention")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden_states = self._hidden_states_from_layer_output(output)
            mask = token_mask.to(hidden_states.device)
            direction = temporal_direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if direction.numel() != hidden_states.shape[-1]:
                raise ValueError(
                    "Temporal-ID dimension does not match hidden size: "
                    f"{direction.numel()} vs {hidden_states.shape[-1]}"
                )
            edited_hidden_states = hidden_states.clone()
            for batch_index in range(hidden_states.shape[0]):
                sample_mask = mask[batch_index]
                if not bool(sample_mask.any()):
                    continue
                selected = hidden_states[batch_index, sample_mask]
                average_norm = selected.norm(dim=-1).mean()
                if torch.isfinite(average_norm) and average_norm > 0:
                    edit = direction * (float(alpha) * average_norm)
                    edited_hidden_states[batch_index, sample_mask] = selected + edit
            return self._replace_hidden_states_in_layer_output(output, edited_hidden_states)

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                return self.interface.forward(model_inputs, **forward_kwargs)
        finally:
            handle.remove()


def build_dataset(args: argparse.Namespace) -> RealDESEDOrderingDataset:
    return RealDESEDOrderingDataset(
        root=args.dataset_root,
        split=args.split,
        sample_rate=args.sample_rate,
        size=args.size,
        min_event_duration_sec=args.min_event_duration_sec,
        max_event_duration_sec=args.max_event_duration_sec,
        min_gap_sec=args.min_gap_sec,
        excluded_event_labels=args.exclude_event_labels,
        max_audio_length_sec=args.max_audio_length_sec,
        random_seed=args.random_seed,
    )


def probability_by_candidate(result: dict[str, Any]) -> dict[str, float]:
    return {
        item["text"].strip(): item["probability"]
        for item in result["keyword_probabilities"]
    }


def _range_label(prefix: str, start_sec: float, end_sec: float) -> str:
    return f"{prefix}_{start_sec:g}-{end_sec:g}s".replace(".", "p")


def _summary_sort_key(path: Path) -> tuple[int, float, str]:
    try:
        with path.open() as handle:
            summary = json.load(handle)
        metrics = summary.get("metrics") or []
        validation_r2 = [
            float(row["r2"])
            for row in metrics
            if row.get("model") == "linear" and row.get("split") == "validation"
        ]
        return (1, validation_r2[0] if validation_r2 else float("-inf"), str(path))
    except Exception:
        return (0, float("-inf"), str(path))


def discover_temporal_id_summary(
    results_dir: Path,
    model_id: str,
    layer: int,
    token: str,
    time_field: str,
) -> Path:
    model_slug = model_output_slug(model_id)
    analysis_dir = results_dir / model_slug / "temporal_id_analysis"
    candidates = sorted(
        analysis_dir.glob(f"*layer{layer}_token-{token}_{time_field}_summary.json")
    )
    candidates = [
        path
        for path in candidates
        if "_layers" not in path.name and path.name.endswith("_summary.json")
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not find an experiment-31 temporal-ID summary for "
            f"model={model_id!r}, layer={layer}, token={token!r}, time_field={time_field!r}. "
            f"Searched: {analysis_dir}"
        )
    return max(candidates, key=_summary_sort_key)


def discover_temporal_id_activation_bundle(
    model_id: str,
    activation_root: Path | None = None,
) -> Path:
    model_slug = model_output_slug(model_id)
    root = activation_root or (EXPERIMENT_41_DIR / "outputs" / "activations")
    activation_dir = root / model_slug
    candidates = sorted(activation_dir.glob("*all_decoder_text_activations.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find an experiment-31 activation bundle for "
            f"model={model_id!r}. Searched: {activation_dir}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def temporal_id_summary_layer(intervention_layer: int) -> int:
    return int(intervention_layer) + 1


def activation_path_config(
    activation_path: Path,
    args: argparse.Namespace,
    source: str,
    summary_errors: list[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    config = {
        "activation_path": activation_path,
        "summary_path": None,
        "summary_layer": None,
        "summary_layer_source": source,
        "summary_layer_matches_activation_slot": None,
        "bin_width_sec": args.bin_width_sec or 2.5,
        "first_bin_center_sec": args.first_bin_center_sec or 2.5,
        "min_bin_count": args.min_bin_count or 10,
        "exclude_bin_centers_sec": (
            args.exclude_bin_centers_sec
            if args.exclude_bin_centers_sec is not None
            else [1.25, 28.75]
        ),
        "max_time_sec": 30.0,
    }
    if summary_errors:
        config["summary_discovery_errors"] = [
            {"layer": layer, "error": error}
            for layer, error in summary_errors
        ]
    return config


def decoder_layer_activation_slot(
    bundle: dict[str, Any],
    intervention_layer: int,
) -> tuple[int, str | None]:
    layer_names = list(bundle.get("layer_names") or [])
    expected_name = f"decoder_layer_{int(intervention_layer)}_output"
    if layer_names:
        if expected_name not in layer_names:
            available = ", ".join(layer_names)
            raise ValueError(
                f"Could not find {expected_name!r} in temporal-ID activation bundle. "
                f"Available layer names: {available}"
            )
        slot = layer_names.index(expected_name)
        return slot, layer_names[slot]

    slot = temporal_id_summary_layer(intervention_layer)
    num_slots = int(bundle["activations"][0].shape[0])
    if slot >= num_slots:
        raise ValueError(
            "Temporal-ID activation slot for decoder layer "
            f"{intervention_layer} is outside available range [0, {num_slots - 1}]"
        )
    return slot, None


def resolve_temporal_id_config(args: argparse.Namespace, model_id: str) -> dict[str, Any]:
    if args.temporal_id_activation_path is not None:
        return activation_path_config(
            Path(args.temporal_id_activation_path),
            args,
            source="explicit_activation_path",
        )

    preferred_layer = temporal_id_summary_layer(args.intervention_layer)
    legacy_layer = int(args.intervention_layer)
    summary_path = None
    summary_layer = None
    errors = []
    for layer, source in (
        (preferred_layer, "decoder_layer_output_slot"),
        (legacy_layer, "legacy_intervention_layer"),
    ):
        if layer in {summary_layer for summary_layer, _source in errors}:
            continue
        try:
            summary_path = discover_temporal_id_summary(
                args.temporal_id_results_dir,
                model_id,
                layer,
                args.token,
                args.time_field,
            )
            summary_layer = layer
            summary_layer_source = source
            break
        except FileNotFoundError as error:
            errors.append((layer, str(error)))
    if summary_path is None or summary_layer is None:
        try:
            activation_path = discover_temporal_id_activation_bundle(model_id)
        except FileNotFoundError as activation_error:
            searched = "\n".join(error for _layer, error in errors)
            raise FileNotFoundError(f"{searched}\n{activation_error}") from activation_error
        return activation_path_config(
            activation_path,
            args,
            source="default_activation_bundle",
            summary_errors=errors,
        )

    with summary_path.open() as handle:
        summary = json.load(handle)
    return {
        "activation_path": Path(summary["activation_path"]),
        "summary_path": summary_path,
        "summary_layer": summary_layer,
        "summary_layer_source": summary_layer_source,
        "summary_layer_matches_activation_slot": summary_layer == preferred_layer,
        "bin_width_sec": args.bin_width_sec or float(summary["bin_width_sec"]),
        "first_bin_center_sec": args.first_bin_center_sec or float(summary["first_bin_center_sec"]),
        "min_bin_count": args.min_bin_count or int(summary["min_bin_count"]),
        "exclude_bin_centers_sec": (
            args.exclude_bin_centers_sec
            if args.exclude_bin_centers_sec is not None
            else list(summary.get("excluded_bin_centers_sec") or [])
        ),
        "max_time_sec": float(summary.get("max_time_sec") or 30.0),
    }


def temporal_id_inspection_table(
    train_table: Any,
    early_id: np.ndarray,
    late_id: np.ndarray,
    early_range_sec: Sequence[float],
    late_range_sec: Sequence[float],
    interval_width_sec: float = 2.5,
) -> dict[str, Any]:
    early_min, early_max = map(float, early_range_sec)
    late_min, late_max = map(float, late_range_sec)
    entries: list[dict[str, Any]] = [
        {
            "label": _range_label("early_range", early_min, early_max),
            "range_sec": [early_min, early_max],
            "count": int(
                train_table.counts[
                    (train_table.centers_sec >= early_min)
                    & (train_table.centers_sec <= early_max)
                ].sum()
            ),
            "source_bin_centers_sec": train_table.centers_sec[
                (train_table.centers_sec >= early_min)
                & (train_table.centers_sec <= early_max)
            ].tolist(),
            "vector": early_id,
        },
    ]

    start = early_max
    while start < late_min:
        end = min(start + interval_width_sec, late_min)
        mask = (train_table.centers_sec >= start) & (train_table.centers_sec < end)
        if bool(mask.any()):
            entries.append(
                {
                    "label": _range_label("interval", start, end),
                    "range_sec": [float(start), float(end)],
                    "count": int(train_table.counts[mask].sum()),
                    "source_bin_centers_sec": train_table.centers_sec[mask].tolist(),
                    "vector": train_table.vectors[mask].mean(axis=0),
                }
            )
        start = end

    late_mask = (train_table.centers_sec >= late_min) & (train_table.centers_sec <= late_max)
    entries.append(
        {
            "label": _range_label("late_range", late_min, late_max),
            "range_sec": [late_min, late_max],
            "count": int(train_table.counts[late_mask].sum()),
            "source_bin_centers_sec": train_table.centers_sec[late_mask].tolist(),
            "vector": late_id,
        }
    )

    return {
        "temporal_ids": torch.as_tensor(
            np.stack([entry["vector"] for entry in entries], axis=0),
            dtype=torch.float32,
        ),
        "labels": [entry["label"] for entry in entries],
        "ranges_sec": [entry["range_sec"] for entry in entries],
        "counts": [entry["count"] for entry in entries],
        "source_bin_centers_sec": [
            entry["source_bin_centers_sec"] for entry in entries
        ],
    }


def temporal_id_direction_from_experiment_41(
    args: argparse.Namespace,
    model_id: str,
    include_endpoint_tensors: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    config = resolve_temporal_id_config(args, model_id)
    activation_path = Path(config["activation_path"])
    if not activation_path.exists():
        raise FileNotFoundError(
            f"Experiment-51 activation bundle does not exist: {activation_path}. "
            "Pass --temporal-id-activation-path if the bundle is in a different location."
        )

    bundle = load_bundle(activation_path)
    activation_layer, activation_layer_name = decoder_layer_activation_slot(
        bundle,
        args.intervention_layer,
    )
    raw_examples = extract_raw_examples(
        bundle=bundle,
        layer=activation_layer,
        token_arg=args.token,
        time_field=args.time_field,
        first_bin_center=float(config["first_bin_center_sec"]),
        bin_width=float(config["bin_width_sec"]),
        max_time_sec=float(config["max_time_sec"]),
    )
    class_means = train_class_means(raw_examples)
    centered_examples = center_by_train_class_mean(raw_examples, class_means)
    train_table = temporal_ids_for_split(
        centered_examples,
        split="train",
        min_bin_count=int(config["min_bin_count"]),
        excluded_bin_centers={float(value) for value in config["exclude_bin_centers_sec"]},
    )

    early_min, early_max = map(float, args.early_range_sec)
    late_min, late_max = map(float, args.late_range_sec)
    early_mask = (train_table.centers_sec >= early_min) & (train_table.centers_sec <= early_max)
    late_mask = (train_table.centers_sec >= late_min) & (train_table.centers_sec <= late_max)
    if not bool(early_mask.any()) or not bool(late_mask.any()):
        raise ValueError(
            "Early and late ranges must each include at least one experiment-31 "
            f"train temporal-ID bin. Available centers: {train_table.centers_sec.tolist()}"
        )

    early_id = train_table.vectors[early_mask].mean(axis=0)
    late_id = train_table.vectors[late_mask].mean(axis=0)
    direction = late_id - early_id
    norm = np.linalg.norm(direction)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError(f"Temporal-ID direction has invalid norm {norm}")

    metadata = {
        **config,
        "activation_path": str(activation_path),
        "summary_path": None if config["summary_path"] is None else str(config["summary_path"]),
        "intervention_layer": int(args.intervention_layer),
        "temporal_id_activation_layer": int(activation_layer),
        "temporal_id_activation_layer_name": activation_layer_name,
        "train_bin_centers_sec": train_table.centers_sec.tolist(),
        "train_bin_counts": train_table.counts.tolist(),
        "early_range_sec": [early_min, early_max],
        "late_range_sec": [late_min, late_max],
        "direction_mode": "late_id_minus_early_id",
        "direction_norm": float(norm),
        "train_classes": len(class_means),
    }
    if include_endpoint_tensors:
        inspection_table = temporal_id_inspection_table(
            train_table=train_table,
            early_id=early_id,
            late_id=late_id,
            early_range_sec=(early_min, early_max),
            late_range_sec=(late_min, late_max),
        )
        metadata["_temporal_id_endpoint_tensors"] = {
            "early_id": torch.as_tensor(early_id, dtype=torch.float32),
            "late_id": torch.as_tensor(late_id, dtype=torch.float32),
            **inspection_table,
        }
    return torch.as_tensor(direction / norm, dtype=torch.float32), metadata


def _require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def load_and_align(baseline_csv: Path, intervention_csv: Path) -> pd.DataFrame:
    required = {
        "id",
        "before_probability",
        "after_probability",
        "predicted_class",
        "correct_class",
    }
    baseline = pd.read_csv(baseline_csv)
    intervention = pd.read_csv(intervention_csv)
    _require_columns(baseline, baseline_csv, required)
    _require_columns(intervention, intervention_csv, required)
    baseline = baseline.rename(
        columns={
            "before_probability": "before_probability_baseline",
            "after_probability": "after_probability_baseline",
            "predicted_class": "predicted_class_baseline",
            "correct_class": "correct_class_baseline",
        }
    )
    intervention = intervention.rename(
        columns={
            "before_probability": "before_probability_intervention",
            "after_probability": "after_probability_intervention",
            "predicted_class": "predicted_class_intervention",
            "correct_class": "correct_class_intervention",
        }
    )
    metadata_columns = [
        column
        for column in [
            "recording_id",
            "filename",
            "split",
            "event1",
            "event2",
            "query_event_label",
            "reference_event_label",
        ]
        if column in baseline.columns
    ]
    aligned = baseline[
        [
            "id",
            "before_probability_baseline",
            "after_probability_baseline",
            "predicted_class_baseline",
            "correct_class_baseline",
            *metadata_columns,
        ]
    ].merge(
        intervention[
            [
                "id",
                "before_probability_intervention",
                "after_probability_intervention",
                "predicted_class_intervention",
                "correct_class_intervention",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError("Baseline and intervention CSVs have no overlapping ids")

    aligned["before_probability_mass_change"] = (
        aligned["before_probability_intervention"] - aligned["before_probability_baseline"]
    )
    aligned["after_probability_mass_change"] = (
        aligned["after_probability_intervention"] - aligned["after_probability_baseline"]
    )
    aligned["candidate_probability_mass_change"] = (
        aligned["before_probability_mass_change"] + aligned["after_probability_mass_change"]
    )
    aligned["correct_probability_baseline"] = np.where(
        aligned["correct_class_baseline"] == "before",
        aligned["before_probability_baseline"],
        aligned["after_probability_baseline"],
    )
    aligned["correct_probability_intervention"] = np.where(
        aligned["correct_class_baseline"] == "before",
        aligned["before_probability_intervention"],
        aligned["after_probability_intervention"],
    )
    aligned["correct_probability_mass_change"] = (
        aligned["correct_probability_intervention"] - aligned["correct_probability_baseline"]
    )
    denominator = aligned["correct_probability_baseline"] - 0.5
    aligned["gt_probability_shift_towards_half"] = np.where(
        np.isclose(denominator, 0.0),
        np.nan,
        (
            aligned["correct_probability_baseline"]
            - aligned["correct_probability_intervention"]
        )
        / denominator,
    )
    aligned["prediction_changed"] = (
        aligned["predicted_class_baseline"] != aligned["predicted_class_intervention"]
    )
    return aligned


def group_mask(aligned: pd.DataFrame, group_name: str) -> pd.Series:
    if group_name == "all":
        return pd.Series(True, index=aligned.index)
    if group_name == "predicted_before":
        return aligned["predicted_class_baseline"] == "before"
    if group_name == "predicted_after":
        return aligned["predicted_class_baseline"] == "after"
    if group_name == "actual_before":
        return aligned["correct_class_baseline"] == "before"
    if group_name == "actual_after":
        return aligned["correct_class_baseline"] == "after"
    raise ValueError(f"Unknown group: {group_name}")


def summarize_groups(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in GROUP_ORDER:
        group_df = aligned.loc[group_mask(aligned, group_name)]
        after_change = pd.to_numeric(
            group_df["after_probability_mass_change"],
            errors="coerce",
        ).dropna()
        before_change = pd.to_numeric(
            group_df["before_probability_mass_change"],
            errors="coerce",
        ).dropna()
        correct_change = pd.to_numeric(
            group_df["correct_probability_mass_change"],
            errors="coerce",
        ).dropna()
        candidate_change = pd.to_numeric(
            group_df["candidate_probability_mass_change"],
            errors="coerce",
        ).dropna()
        gt_shift_towards_half = pd.to_numeric(
            group_df["gt_probability_shift_towards_half"],
            errors="coerce",
        ).dropna()
        rows.append({
            "group": group_name,
            "label": GROUP_LABELS[group_name],
            "count": int(len(after_change)),
            "mean_after_probability_mass_change": (
                float(after_change.mean()) if len(after_change) else np.nan
            ),
            "median_after_probability_mass_change": (
                float(after_change.median()) if len(after_change) else np.nan
            ),
            "std_after_probability_mass_change": (
                float(after_change.std(ddof=1)) if len(after_change) > 1 else 0.0
            ),
            "mean_before_probability_mass_change": (
                float(before_change.mean()) if len(before_change) else np.nan
            ),
            "mean_correct_probability_mass_change": (
                float(correct_change.mean()) if len(correct_change) else np.nan
            ),
            "mean_candidate_probability_mass_change": (
                float(candidate_change.mean()) if len(candidate_change) else np.nan
            ),
            "mean_gt_probability_shift_towards_half": (
                float(gt_shift_towards_half.mean()) if len(gt_shift_towards_half) else np.nan
            ),
            "median_gt_probability_shift_towards_half": (
                float(gt_shift_towards_half.median()) if len(gt_shift_towards_half) else np.nan
            ),
            "prediction_changed_count": int(group_df["prediction_changed"].sum()),
            "prediction_changed_rate": (
                float(group_df["prediction_changed"].mean()) if len(group_df) else np.nan
            ),
        })
    return pd.DataFrame(rows)


class RelativeTemporalIdInterpolator:
    def __init__(self, relative_time: np.ndarray, vectors: np.ndarray):
        order = np.argsort(relative_time)
        self.relative_time = np.asarray(relative_time, dtype=np.float64)[order]
        self.vectors = np.asarray(vectors, dtype=np.float32)[order]
        if len(self.relative_time) < 2:
            raise ValueError("At least two temporal-ID bins are required for interpolation")
        if np.any(np.diff(self.relative_time) <= 0.0):
            raise ValueError("Temporal-ID relative times must be strictly increasing")
        self.reference_direction = self.vectors[-1] - self.vectors[0]
        self.reference_direction_norm = float(np.linalg.norm(self.reference_direction))
        if self.reference_direction_norm <= 0.0 or not np.isfinite(
            self.reference_direction_norm
        ):
            raise ValueError(
                "Temporal-ID endpoint reference direction has invalid norm "
                f"{self.reference_direction_norm}"
            )

    def vector_at(self, relative_time: float) -> np.ndarray:
        clipped = float(np.clip(relative_time, self.relative_time[0], self.relative_time[-1]))
        return np.array(
            [
                np.interp(clipped, self.relative_time, self.vectors[:, dim])
                for dim in range(self.vectors.shape[1])
            ],
            dtype=np.float32,
        )


def build_relative_temporal_ids(
    args: argparse.Namespace,
    model_id: str,
) -> tuple[RelativeTemporalIdInterpolator, dict[str, Any]]:
    config = resolve_temporal_id_config(args, model_id)
    activation_path = Path(config["activation_path"])
    if not activation_path.exists():
        raise FileNotFoundError(
            f"Temporal-ID activation bundle does not exist: {activation_path}. "
            "Pass --temporal-id-activation-path if the bundle is in a different location."
        )

    bundle = load_bundle(activation_path)
    activation_layer, activation_layer_name = decoder_layer_activation_slot(
        bundle,
        args.intervention_layer,
    )
    raw_examples = extract_raw_examples(
        bundle=bundle,
        layer=activation_layer,
        token_arg=args.token,
        time_field=args.time_field,
        first_bin_center=float(config["first_bin_center_sec"]),
        bin_width=float(config["bin_width_sec"]),
        max_time_sec=float(config["max_time_sec"]),
    )
    class_means = train_class_means(raw_examples)
    centered_examples = center_by_train_class_mean(raw_examples, class_means)
    train_table = temporal_ids_for_split(
        centered_examples,
        split="train",
        min_bin_count=int(config["min_bin_count"]),
        excluded_bin_centers={float(value) for value in config["exclude_bin_centers_sec"]},
    )
    relative_time = train_table.centers_sec / float(args.synthetic_reference_duration_sec)
    interpolator = RelativeTemporalIdInterpolator(relative_time, train_table.vectors)
    metadata = {
        **config,
        "activation_path": str(activation_path),
        "summary_path": None if config["summary_path"] is None else str(config["summary_path"]),
        "intervention_layer": int(args.intervention_layer),
        "temporal_id_activation_layer": int(activation_layer),
        "temporal_id_activation_layer_name": activation_layer_name,
        "synthetic_reference_duration_sec": args.synthetic_reference_duration_sec,
        "train_bin_centers_sec": train_table.centers_sec.tolist(),
        "train_relative_time": relative_time.tolist(),
        "train_bin_counts": train_table.counts.tolist(),
        "reference_first_bin_center_sec": float(train_table.centers_sec[0]),
        "reference_last_bin_center_sec": float(train_table.centers_sec[-1]),
        "reference_first_relative_time": float(relative_time[0]),
        "reference_last_relative_time": float(relative_time[-1]),
        "reference_direction_source": "tau_last_minus_tau_first",
        "reference_direction_norm": float(interpolator.reference_direction_norm),
        "train_classes": len(class_means),
        "relative_time_range": [
            float(interpolator.relative_time[0]),
            float(interpolator.relative_time[-1]),
        ],
    }
    return interpolator, metadata


def event_time(sample: dict[str, Any], prefix: str, time_field: str) -> float:
    onset = float(sample[f"{prefix}_onset_sec"])
    offset = float(sample[f"{prefix}_offset_sec"])
    if time_field == "onset":
        return onset
    if time_field == "offset":
        return offset
    return 0.5 * (onset + offset)


def relative_event_time(sample: dict[str, Any], prefix: str, time_field: str) -> float:
    duration = float(sample["length_sec"])
    if duration <= 0.0:
        raise ValueError(f"Sample {sample['id']} has non-positive length_sec={duration}")
    return event_time(sample, prefix, time_field) / duration


INTERVENTION_SPECS = {
    "query_to_end": {
        "event_prefix": "query",
        "token_types": QUERY_TOKEN_TYPES,
        "target": "end",
        "measured_token": "after",
    },
    "query_to_beginning": {
        "event_prefix": "query",
        "token_types": QUERY_TOKEN_TYPES,
        "target": "beginning",
        "measured_token": "before",
    },
}


def batch_single_event_directions(
    batch: Sequence[dict[str, Any]],
    temporal_ids: RelativeTemporalIdInterpolator,
    time_field: str,
    event_prefix: str,
    target: str,
    beginning_target_relative_time: float,
    end_target_relative_time: float,
) -> tuple[torch.Tensor, list[dict[str, float | str]]]:
    directions = []
    metadata = []
    if target == "beginning":
        target_relative_time = float(beginning_target_relative_time)
    elif target == "end":
        target_relative_time = float(end_target_relative_time)
    else:
        raise ValueError(f"Unsupported target: {target}")
    target_tau = temporal_ids.vector_at(target_relative_time)
    for sample in batch:
        source_relative_time = relative_event_time(sample, event_prefix, time_field)
        source_tau = temporal_ids.vector_at(source_relative_time)
        direction = target_tau - source_tau
        norm = np.linalg.norm(direction)
        if norm > 0.0 and np.isfinite(norm):
            direction = direction / temporal_ids.reference_direction_norm
        directions.append(direction)
        metadata.append({
            "moved_event": event_prefix,
            "movement_target": target,
            "direction_source": "relative_tau_target_minus_tau_source_scaled_by_reference",
            "source_relative_time": float(source_relative_time),
            "target_relative_time": float(target_relative_time),
            "raw_direction_norm": float(norm),
            "reference_direction_norm": float(temporal_ids.reference_direction_norm),
            "direction_reference_scale": float(norm / temporal_ids.reference_direction_norm),
            "direction_norm": float(np.linalg.norm(direction)),
            "query_relative_time": float(relative_event_time(sample, "query", time_field)),
            "reference_relative_time": float(relative_event_time(sample, "reference", time_field)),
        })
    return torch.as_tensor(np.stack(directions), dtype=torch.float32), metadata


def batch_global_single_event_directions(
    batch: Sequence[dict[str, Any]],
    global_direction: torch.Tensor,
    time_field: str,
    event_prefix: str,
    target: str,
) -> tuple[torch.Tensor, list[dict[str, float | str]]]:
    if target == "beginning":
        signed_direction = -global_direction
        target_label = "global_early"
    elif target == "end":
        signed_direction = global_direction
        target_label = "global_late"
    else:
        raise ValueError(f"Unsupported target: {target}")

    direction = signed_direction.detach().cpu().float()
    directions = direction.unsqueeze(0).repeat(len(batch), 1)
    norm = float(direction.norm().item())
    metadata = []
    for sample in batch:
        metadata.append({
            "moved_event": event_prefix,
            "movement_target": target,
            "direction_source": "global_late_minus_early",
            "source_relative_time": float(relative_event_time(sample, event_prefix, time_field)),
            "target_relative_time": target_label,
            "raw_direction_norm": norm,
            "direction_norm": norm,
            "query_relative_time": float(relative_event_time(sample, "query", time_field)),
            "reference_relative_time": float(relative_event_time(sample, "reference", time_field)),
        })
    return directions, metadata


class SingleEventRelativeTemporalMoveScorer(OrderingPredictionScorer):
    def forward_prediction(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        teacher_forced_texts: Sequence[str] | str | None = None,
        sampling_rates: Sequence[int] | int | None = None,
        keywords: Sequence[str] = (" before", " after"),
        top_k: int = 10,
        directions: torch.Tensor | None = None,
        intervention_token_types: Sequence[str] | None = None,
        temporal_alpha: float = 1.0,
        intervention_layer: int | None = None,
        **forward_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if directions is None:
            return super().forward_prediction(
                audio_arrays=audio_arrays,
                prompts=prompts,
                teacher_forced_texts=teacher_forced_texts,
                sampling_rates=sampling_rates,
                keywords=keywords,
                top_k=top_k,
                **forward_kwargs,
            )
        if intervention_token_types is None:
            raise ValueError("intervention_token_types is required for single-event intervention")
        if intervention_layer is None:
            raise ValueError("intervention_layer is required for single-event intervention")

        from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (  # noqa: E402
            _batch_values,
            _validate_batch_lengths,
        )

        _validate_batch_lengths(audio_arrays, prompts)
        teacher_forced_texts = _batch_values(
            teacher_forced_texts,
            len(audio_arrays),
            default="",
        )
        self._validate_query_args(keywords, top_k)
        candidate_ids = self._candidate_token_ids(keywords)
        model_inputs, _tokens = self.interface.build_model_inputs(
            audio_arrays=audio_arrays,
            prompts=prompts,
            teacher_forced_texts=teacher_forced_texts,
            sampling_rates=sampling_rates,
        )
        outputs = self._forward_with_single_event_temporal_move(
            model_inputs,
            directions=directions,
            token_types=intervention_token_types,
            alpha=temporal_alpha,
            layer=intervention_layer,
            forward_kwargs=forward_kwargs,
        )
        return self._probability_results(
            outputs.logits,
            keywords,
            candidate_ids,
            top_k,
            attention_mask=model_inputs.get("attention_mask"),
        )

    def _forward_with_single_event_temporal_move(
        self,
        model_inputs: Any,
        directions: torch.Tensor,
        token_types: Sequence[str],
        alpha: float,
        layer: int,
        forward_kwargs: dict[str, Any],
    ) -> Any:
        target_layer = self.decoder_layer(layer)
        token_mask = self.token_mask(model_inputs, token_types)
        if not bool(token_mask.any()):
            raise ValueError(f"No {token_types!r} tokens found for single-event intervention")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden_states = self._hidden_states_from_layer_output(output)
            if directions.shape[-1] != hidden_states.shape[-1]:
                raise ValueError(
                    "Direction dimension does not match hidden size: "
                    f"{directions.shape[-1]} vs {hidden_states.shape[-1]}"
                )
            mask = token_mask.to(hidden_states.device)
            batch_directions = directions.to(device=hidden_states.device, dtype=hidden_states.dtype)
            edited = hidden_states.clone()
            for batch_index in range(hidden_states.shape[0]):
                if bool(mask[batch_index].any()):
                    selected = hidden_states[batch_index, mask[batch_index]]
                    average_norm = selected.norm(dim=-1).mean()
                    if not (torch.isfinite(average_norm) and average_norm > 0):
                        continue
                    edited[batch_index, mask[batch_index]] = (
                        selected + float(alpha) * average_norm * batch_directions[batch_index]
                    )
            return self._replace_hidden_states_in_layer_output(output, edited)

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                return self.interface.forward(model_inputs, **forward_kwargs)
        finally:
            handle.remove()


def write_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extras)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, args: argparse.Namespace, temporal_metadata: dict[str, Any]) -> None:
    payload = {
        "model": args.model,
        "model_id": args.model_id or default_model_id(args.model),
        "intervention_layer": args.intervention_layer,
        "alpha": args.alpha,
        "direction_mode": (
            "global_late_minus_early"
            if getattr(args, "use_global_direction", False)
            else "relative_tau_target_minus_tau_source"
        ),
        "interventions": {
            name: {
                "event_prefix": spec["event_prefix"],
                "token_types": list(spec["token_types"]),
                "target": spec["target"],
                "measured_token": spec["measured_token"],
            }
            for name, spec in INTERVENTION_SPECS.items()
        },
        "time_field": args.time_field,
        "beginning_target_sec": args.beginning_target_sec,
        "end_target_sec": args.end_target_sec,
        "beginning_target_relative_time": (
            args.beginning_target_sec / args.synthetic_reference_duration_sec
        ),
        "end_target_relative_time": (
            args.end_target_sec / args.synthetic_reference_duration_sec
        ),
        "real_desed_filters": {
            "split": args.split,
            "size": args.size,
            "min_event_duration_sec": args.min_event_duration_sec,
            "max_event_duration_sec": args.max_event_duration_sec,
            "min_gap_sec": args.min_gap_sec,
            "exclude_event_labels": args.exclude_event_labels,
            "max_audio_length_sec": args.max_audio_length_sec,
            "random_seed": args.random_seed,
        },
        "temporal_id_source": temporal_metadata,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def score_rows(
    args: argparse.Namespace,
    dataset: Any,
    model: SingleEventRelativeTemporalMoveScorer,
    temporal_ids: RelativeTemporalIdInterpolator | None = None,
    global_direction: torch.Tensor | None = None,
    intervention_name: str = "baseline",
    intervention_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list,
    )
    for batch in tqdm(dataloader, desc=f"Scoring RealDESED {intervention_name} examples"):
        prompts = [
            f"Does {sample['query_label']} occur before or after {sample['reference_label']}? "
            for sample in batch
        ]
        teacher_forced_inputs = [f"{sample['query_label']} occurs" for sample in batch]
        direction_metadata = [
            {
                "intervention_name": intervention_name,
                "moved_event": "",
                "movement_target": "",
                "direction_source": "",
                "source_relative_time": "",
                "target_relative_time": "",
                "raw_direction_norm": "",
                "reference_direction_norm": "",
                "direction_reference_scale": "",
                "direction_norm": "",
                "query_relative_time": "",
                "reference_relative_time": "",
            }
            for _ in batch
        ]
        directions = None
        intervention_token_types = None
        if temporal_ids is not None or global_direction is not None:
            if intervention_spec is None:
                raise ValueError("intervention_spec is required for interventions")
            if global_direction is not None:
                directions, direction_metadata = batch_global_single_event_directions(
                    batch,
                    global_direction,
                    args.time_field,
                    event_prefix=str(intervention_spec["event_prefix"]),
                    target=str(intervention_spec["target"]),
                )
            else:
                if temporal_ids is None:
                    raise ValueError("temporal_ids is required for relative directions")
                directions, direction_metadata = batch_single_event_directions(
                    batch,
                    temporal_ids,
                    args.time_field,
                    event_prefix=str(intervention_spec["event_prefix"]),
                    target=str(intervention_spec["target"]),
                    beginning_target_relative_time=(
                        args.beginning_target_sec / args.synthetic_reference_duration_sec
                    ),
                    end_target_relative_time=(
                        args.end_target_sec / args.synthetic_reference_duration_sec
                    ),
                )
            for item in direction_metadata:
                item["intervention_name"] = intervention_name
            intervention_token_types = intervention_spec["token_types"]

        results = model.forward_prediction(
            [sample["waveform"] for sample in batch],
            prompts,
            teacher_forced_texts=teacher_forced_inputs,
            sampling_rates=[sample["sample_rate"] for sample in batch],
            keywords=CANDIDATE_TEXTS,
            top_k=args.top_k,
            directions=directions,
            intervention_token_types=intervention_token_types,
            temporal_alpha=args.alpha,
            intervention_layer=args.intervention_layer,
        )
        for sample, prompt, teacher_forced_input, result, direction_info in zip(
            batch,
            prompts,
            teacher_forced_inputs,
            results,
            direction_metadata,
        ):
            probabilities = probability_by_candidate(result)
            before_probability = probabilities["before"]
            after_probability = probabilities["after"]
            correct_class = sample["answer"]
            rows.append({
                "id": sample["id"],
                "recording_id": sample["recording_id"],
                "filename": sample["filename"],
                "path": sample["path"],
                "split": sample["split"],
                "event1": sample["query_label"],
                "event2": sample["reference_label"],
                "prompt": prompt,
                "teacher_forced_input": teacher_forced_input,
                "correct_class": correct_class,
                "predicted_class": "before" if before_probability >= after_probability else "after",
                "before_probability": before_probability,
                "after_probability": after_probability,
                "correct_probability": probabilities[correct_class],
                "single_event_relative_temporal_move_intervention": (
                    temporal_ids is not None or global_direction is not None
                ),
                "intervention_name": intervention_name,
                "intervention_layer": args.intervention_layer,
                "alpha": args.alpha if temporal_ids is not None else "",
                "intervention_token_types": json.dumps(intervention_token_types or []),
                "query_label": sample["query_label"],
                "query_event_label": sample["query_event_label"],
                "query_onset_sec": sample["query_onset_sec"],
                "query_offset_sec": sample["query_offset_sec"],
                "query_duration_sec": sample["query_duration_sec"],
                "reference_label": sample["reference_label"],
                "reference_event_label": sample["reference_event_label"],
                "reference_onset_sec": sample["reference_onset_sec"],
                "reference_offset_sec": sample["reference_offset_sec"],
                "reference_duration_sec": sample["reference_duration_sec"],
                "first_event_label": sample["first_event_label"],
                "first_onset_sec": sample["first_onset_sec"],
                "first_offset_sec": sample["first_offset_sec"],
                "first_duration_sec": sample["first_duration_sec"],
                "second_event_label": sample["second_event_label"],
                "second_onset_sec": sample["second_onset_sec"],
                "second_offset_sec": sample["second_offset_sec"],
                "second_duration_sec": sample["second_duration_sec"],
                "length_sec": sample["length_sec"],
                "recording_length_sec": sample["recording_length_sec"],
                "audio_window_start_sec": sample["audio_window_start_sec"],
                "audio_window_end_sec": sample["audio_window_end_sec"],
                "event_gap_sec": sample["event_gap_sec"],
                "min_event_duration_sec": args.min_event_duration_sec,
                "max_event_duration_sec": args.max_event_duration_sec,
                "min_gap_sec": args.min_gap_sec,
                "excluded_event_labels": json.dumps(args.exclude_event_labels),
                "top10_tokens": json.dumps(result["top_tokens"]),
                **direction_info,
            })
    return rows


def add_flip_percent(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["prediction_flipped_percent"] = summary["prediction_changed_rate"] * 100.0
    return summary


def plot_probability_mass_changes(aligned: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True, sharey=True)
    specs = [
        ("before_probability_mass_change", 'Probability mass change for " before"', "#bfdbfe", "#1d4ed8"),
        ("after_probability_mass_change", 'Probability mass change for " after"', "#a7f3d0", "#047857"),
    ]
    rng = np.random.default_rng(0)
    for ax, (column, ylabel, face_color, edge_color) in zip(axes, specs):
        data = [
            aligned.loc[group_mask(aligned, group_name), column].dropna().to_numpy()
            for group_name in GROUP_ORDER
        ]
        labels = [
            f"{GROUP_LABELS[group_name]}\n(n={len(values)})"
            for group_name, values in zip(GROUP_ORDER, data)
        ]
        box = ax.boxplot(
            data,
            tick_labels=labels,
            showmeans=True,
            patch_artist=True,
            medianprops={"color": "#111111", "linewidth": 1.5},
            meanprops={
                "marker": "o",
                "markerfacecolor": "#ffffff",
                "markeredgecolor": "#111111",
                "markersize": 5,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor(face_color)
            patch.set_edgecolor(edge_color)
            patch.set_alpha(0.85)
        for index, values in enumerate(data, start=1):
            if len(values) == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                s=12,
                alpha=0.4,
                color="#334155",
                linewidths=0,
            )
        ax.axhline(0.0, color="#111111", linewidth=1.0, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Single-event relative temporal-position move intervention")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_type_dir(plot_root: Path, plot_type: str, model_slug: str) -> Path:
    directory = plot_root / plot_type / model_slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_type_dir(output_root: Path, output_type: str, model_slug: str) -> Path:
    directory = output_root / output_type / model_slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_plot_outputs(
    baseline_csv: Path,
    intervention_csv: Path,
    plot_root: Path,
    output_root: Path,
    model_slug: str,
    output_prefix: str,
    dpi: int,
    intervention_name: str,
    measured_token: str,
    make_plot: bool = True,
) -> dict[str, Path]:
    aligned = load_and_align(baseline_csv, intervention_csv)
    measured_column = f"{measured_token}_probability_mass_change"
    aligned["measured_token"] = measured_token
    aligned["measured_probability_mass_change"] = aligned[measured_column]
    summary = add_flip_percent(summarize_groups(aligned))
    summary["intervention_name"] = intervention_name
    summary["measured_token"] = measured_token
    summary["mean_measured_probability_mass_change"] = summary[
        f"mean_{measured_token}_probability_mass_change"
    ]
    aligned_path = (
        output_type_dir(output_root, "aligned", model_slug)
        / f"{output_prefix}_{intervention_name}_aligned.csv"
    )
    summary_path = (
        output_type_dir(output_root, "summary", model_slug)
        / f"{output_prefix}_{intervention_name}_summary.csv"
    )
    aligned.to_csv(aligned_path, index=False)
    summary.to_csv(summary_path, index=False)
    outputs = {"aligned": aligned_path, "summary": summary_path}
    if make_plot:
        plot_path = (
            plot_type_dir(plot_root, "probability_mass_change", model_slug)
            / f"{output_prefix}_{intervention_name}_probability_mass_change.png"
        )
        plot_probability_mass_changes(aligned, plot_path, dpi=dpi)
        outputs["plot"] = plot_path
    return outputs


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.synthetic_reference_duration_sec <= 0.0:
        raise ValueError("--synthetic-reference-duration-sec must be positive")
    if not 0.0 <= args.beginning_target_sec <= args.synthetic_reference_duration_sec:
        raise ValueError("--beginning-target-sec must be within the synthetic reference duration")
    if not 0.0 <= args.end_target_sec <= args.synthetic_reference_duration_sec:
        raise ValueError("--end-target-sec must be within the synthetic reference duration")
    if args.beginning_target_sec >= args.end_target_sec:
        raise ValueError("--beginning-target-sec must be smaller than --end-target-sec")

    model_id = args.model_id or default_model_id(args.model)
    model_slug = model_output_slug(model_id)
    output_dir = args.output_dir / "intervention" / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_ids = None
    global_direction = None
    if args.use_global_direction:
        global_direction, temporal_metadata = temporal_id_direction_from_experiment_41(
            args,
            model_id,
        )
        temporal_metadata = {
            **temporal_metadata,
            "direction_mode": "global_late_minus_early",
        }
    else:
        temporal_ids, temporal_metadata = build_relative_temporal_ids(args, model_id)
        temporal_metadata = {
            **temporal_metadata,
            "direction_mode": "relative_tau_target_minus_tau_source",
        }
    dataset = build_dataset(args)
    model = SingleEventRelativeTemporalMoveScorer(
        build_model_interface(args.model, model_id=model_id, device=args.device)
    )

    stem = experiment_stem(args)
    baseline_csv = output_dir / f"{stem}_baseline_probabilities.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    baseline_rows = score_rows(
        args,
        dataset,
        model,
        temporal_ids=None,
        intervention_name="baseline",
    )
    write_rows(baseline_csv, baseline_rows)
    write_metadata(metadata_path, args, temporal_metadata)

    outputs = {
        "baseline_csv": baseline_csv,
        "metadata": metadata_path,
    }
    for intervention_name, intervention_spec in INTERVENTION_SPECS.items():
        intervention_csv = output_dir / f"{stem}_{intervention_name}_probabilities.csv"
        intervention_rows = score_rows(
            args,
            dataset,
            model,
            temporal_ids=temporal_ids,
            global_direction=global_direction,
            intervention_name=intervention_name,
            intervention_spec=intervention_spec,
        )
        write_rows(intervention_csv, intervention_rows)
        outputs[f"{intervention_name}_csv"] = intervention_csv
        plot_outputs = write_plot_outputs(
            baseline_csv=baseline_csv,
            intervention_csv=intervention_csv,
            plot_root=args.plot_dir,
            output_root=args.output_dir,
            model_slug=model_slug,
            output_prefix=stem,
            dpi=args.dpi,
            intervention_name=intervention_name,
            measured_token=str(intervention_spec["measured_token"]),
            make_plot=not args.no_plot,
        )
        outputs.update(
            {
                f"{intervention_name}_{output_name}": path
                for output_name, path in plot_outputs.items()
            }
        )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")
    return outputs


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
