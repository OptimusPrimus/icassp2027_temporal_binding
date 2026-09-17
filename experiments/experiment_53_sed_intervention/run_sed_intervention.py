#!/usr/bin/env python3
"""Run RealDESED onset detection with event-position-dependent temporal-ID edits."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader
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
from experiments.experiment_52_intervention.run_before_after_intervention import (  # noqa: E402
    RelativeTemporalIdInterpolator,
    decoder_layer_activation_slot,
    resolve_temporal_id_config,
)
from experiments.fine_temporal_reasoning.real_desed_onset_dataset import (  # noqa: E402
    RealDESEDOnsetDataset,
)
from experiments.onset_detection.run_onset_detection import (  # noqa: E402
    parse_onset_seconds,
    query_object_token_mask,
)


PROMPT_TEMPLATE = (
    "Identify the onset time in seconds. "
    "Do not output anything else. The onset time of {target} is:"
)
GROUP_ORDER = ["all", "baseline_valid", "intervention_valid", "both_valid"]
GROUP_LABELS = {
    "all": "All",
    "baseline_valid": "Baseline valid",
    "intervention_valid": "Intervention valid",
    "both_valid": "Both valid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RealDESED onset prediction with per-example temporal-ID directions "
            "derived with the same relative temporal-ID interpolation used by "
            "experiment 52."
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
        default=EXPERIMENT_41_DIR / "plots",
    )
    parser.add_argument(
        "--time-field",
        choices=("onset", "center", "offset"),
        default="center",
        help="Event time used when extracting experiment-31 temporal IDs.",
    )
    parser.add_argument("--token", default="query_event")
    parser.add_argument(
        "--bin-width-sec",
        type=float,
        default=2.5,
        help="Temporal-ID bin width. Defaults to experiment-52 endpoint bins.",
    )
    parser.add_argument(
        "--first-bin-center-sec",
        type=float,
        default=3.75,
        help="First bin center. Defaults to experiment-52 endpoint bins.",
    )
    parser.add_argument("--min-bin-count", type=int, default=None)
    parser.add_argument(
        "--exclude-bin-centers-sec",
        type=float,
        nargs="*",
        default=[],
        help="Temporal-ID bin centers to exclude. Defaults to none.",
    )
    parser.add_argument("--synthetic-reference-duration-sec", type=float, default=30.0)
    parser.add_argument(
        "--target",
        choices=("beginning", "end"),
        default="end",
        help="Temporal-ID bin to move the queried event toward.",
    )
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--min-event-duration-sec", type=float, default=2.5)
    parser.add_argument("--max-event-duration-sec", type=float, default=5.5)
    parser.add_argument("--max-overlap-fraction", type=float, default=0.1)
    parser.add_argument("--exclude-event-labels", nargs="*", default=["footsteps"])
    parser.add_argument("--max-audio-length-sec", type=float, default=30.0)
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
        f"{model_slug}_real_desed_onset_{args.split}"
        f"_dur{args.min_event_duration_sec:g}-{args.max_event_duration_sec:g}s"
        f"_n{args.size}"
    )
    return (
        f"{prefix}_relative_temporal_id_{args.target}"
        f"_layer{args.intervention_layer}_alpha{_safe_number(args.alpha)}"
        f"_bin{_safe_number(args.bin_width_sec)}s"
    ).replace(".", "p")


class OnsetInterventionGenerator(OrderingIntervention):
    """Generation wrapper with optional temporal-ID addition."""

    def generate_onsets(
        self,
        audio_arrays: Sequence[Any],
        prompts: Sequence[str],
        event_labels: Sequence[str],
        sampling_rates: Sequence[int] | int | None = None,
        max_new_tokens: int = 40,
        temporal_direction: torch.Tensor | None = None,
        alpha: float = 1.0,
        layer: int | None = None,
    ) -> list[str]:
        model_inputs, _tokens = self.interface.build_model_inputs(
            audio_arrays=audio_arrays,
            prompts=prompts,
            teacher_forced_texts=[""] * len(audio_arrays),
            sampling_rates=sampling_rates,
        )
        if temporal_direction is None:
            return self.interface.generate(model_inputs, max_new_tokens=max_new_tokens)
        if layer is None:
            raise ValueError("layer is required for temporal intervention")
        return self._generate_with_temporal_direction_add(
            model_inputs,
            event_labels=event_labels,
            temporal_direction=temporal_direction,
            alpha=alpha,
            layer=layer,
            max_new_tokens=max_new_tokens,
        )

    def _generate_with_temporal_direction_add(
        self,
        model_inputs: Any,
        event_labels: Sequence[str],
        temporal_direction: torch.Tensor,
        alpha: float,
        layer: int,
        max_new_tokens: int,
    ) -> list[str]:
        target_layer = self.decoder_layer(layer)
        token_mask = query_object_token_mask(
            self.interface,
            model_inputs["input_ids"],
            event_labels,
            attention_mask=model_inputs.get("attention_mask"),
        )
        if not bool(token_mask.any()):
            raise ValueError("No queried event-label tokens found for temporal intervention")
        if not torch.all(token_mask.any(dim=1)):
            missing = [
                label
                for label, has_match in zip(event_labels, token_mask.any(dim=1).tolist())
                if not has_match
            ]
            raise ValueError(f"No queried event-label tokens found for: {missing}")

        def hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden_states = self._hidden_states_from_layer_output(output)
            mask = token_mask.to(hidden_states.device)
            if hidden_states.shape[1] < mask.shape[1]:
                return output
            if hidden_states.shape[1] > mask.shape[1]:
                mask = _align_prompt_mask_to_hidden_states(mask, hidden_states)
            direction = temporal_direction.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if direction.ndim == 1:
                if direction.numel() != hidden_states.shape[-1]:
                    raise ValueError(
                        "Temporal-ID dimension does not match hidden size: "
                        f"{direction.numel()} vs {hidden_states.shape[-1]}"
                    )
                batch_directions = direction.unsqueeze(0).expand(hidden_states.shape[0], -1)
            elif direction.ndim == 2:
                if direction.shape != (hidden_states.shape[0], hidden_states.shape[-1]):
                    raise ValueError(
                        "Temporal-ID batch direction shape must be [batch, hidden_size]: "
                        f"{tuple(direction.shape)} vs "
                        f"{(hidden_states.shape[0], hidden_states.shape[-1])}"
                    )
                batch_directions = direction
            else:
                raise ValueError(
                    "Temporal-ID direction must have shape [hidden_size] or "
                    f"[batch, hidden_size], got {tuple(direction.shape)}"
                )
            edited_hidden_states = hidden_states.clone()
            for batch_index in range(hidden_states.shape[0]):
                sample_mask = mask[batch_index]
                if not bool(sample_mask.any()):
                    continue
                selected = hidden_states[batch_index, sample_mask]
                average_norm = selected.norm(dim=-1).mean()
                if torch.isfinite(average_norm) and average_norm > 0:
                    edit = batch_directions[batch_index] * (float(alpha) * average_norm)
                    edited_hidden_states[batch_index, sample_mask] = selected + edit
            return self._replace_hidden_states_in_layer_output(output, edited_hidden_states)

        handle = target_layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                return self.interface.generate(model_inputs, max_new_tokens=max_new_tokens)
        finally:
            handle.remove()


def _align_prompt_mask_to_hidden_states(
    prompt_mask: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.shape[1] > prompt_mask.shape[1]:
        padding = torch.zeros(
            (prompt_mask.shape[0], hidden_states.shape[1] - prompt_mask.shape[1]),
            dtype=prompt_mask.dtype,
            device=prompt_mask.device,
        )
        return torch.cat([prompt_mask, padding], dim=1)
    return prompt_mask[:, : hidden_states.shape[1]]


def build_dataset(args: argparse.Namespace) -> RealDESEDOnsetDataset:
    return RealDESEDOnsetDataset(
        root=args.dataset_root,
        split=args.split,
        sample_rate=args.sample_rate,
        size=args.size,
        min_event_duration_sec=args.min_event_duration_sec,
        max_event_duration_sec=args.max_event_duration_sec,
        max_overlap_fraction=args.max_overlap_fraction,
        excluded_event_labels=args.exclude_event_labels,
        max_audio_length_sec=args.max_audio_length_sec,
        random_seed=args.random_seed,
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
    temporal_ids = RelativeTemporalIdInterpolator(relative_time, train_table.vectors)
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
        "reference_direction_norm": float(temporal_ids.reference_direction_norm),
        "train_classes": len(class_means),
        "relative_time_range": [
            float(temporal_ids.relative_time[0]),
            float(temporal_ids.relative_time[-1]),
        ],
        "lookup": "linear_relative_time_interpolation",
        "target": args.target,
        "beginning_target_sec": args.beginning_target_sec,
        "end_target_sec": args.end_target_sec,
    }
    return temporal_ids, metadata


def event_center_relative_time(sample: dict[str, Any]) -> float:
    duration = float(sample["length_sec"])
    if duration <= 0.0:
        raise ValueError(f"Sample {sample['id']} has non-positive length_sec={duration}")
    center_sec = (float(sample["event_onset_sec"]) + float(sample["event_offset_sec"])) / 2.0
    return center_sec / duration


def batch_event_relative_directions(
    batch: Sequence[dict[str, Any]],
    temporal_ids: RelativeTemporalIdInterpolator,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    target_sec = (
        args.beginning_target_sec
        if args.target == "beginning"
        else args.end_target_sec
    )
    target_relative_time = float(target_sec) / float(args.synthetic_reference_duration_sec)
    target_tau = temporal_ids.vector_at(target_relative_time)
    directions = []
    metadata = []
    for sample in batch:
        source_relative_time = event_center_relative_time(sample)
        source_tau = temporal_ids.vector_at(source_relative_time)
        direction = target_tau - source_tau
        norm = np.linalg.norm(direction)
        if norm > 0.0 and np.isfinite(norm):
            direction = direction / temporal_ids.reference_direction_norm
        directions.append(direction)
        metadata.append({
            "direction_source": "relative_tau_target_minus_tau_source_scaled_by_reference",
            "temporal_id_lookup": "linear_relative_time_interpolation",
            "source_relative_time": float(source_relative_time),
            "target": args.target,
            "target_relative_time": float(target_relative_time),
            "raw_direction_norm": float(norm),
            "reference_direction_norm": float(temporal_ids.reference_direction_norm),
            "direction_reference_scale": float(norm / temporal_ids.reference_direction_norm),
            "direction_norm": float(np.linalg.norm(direction)),
        })
    return torch.as_tensor(np.stack(directions), dtype=torch.float32), metadata


def score_rows(
    args: argparse.Namespace,
    dataset: Any,
    model: OnsetInterventionGenerator,
    temporal_direction: torch.Tensor | None = None,
    temporal_activation_file: Path | None = None,
) -> list[dict[str, Any]]:
    rows = []
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list,
    )
    for batch in tqdm(dataloader, desc="Generating RealDESED onset predictions"):
        prompts = [
            PROMPT_TEMPLATE.format(target=f"{sample['label']} sound")
            for sample in batch
        ]
        predictions = model.generate_onsets(
            [sample["waveform"] for sample in batch],
            prompts,
            [sample["label"] for sample in batch],
            sampling_rates=[sample["sample_rate"] for sample in batch],
            max_new_tokens=args.max_new_tokens,
            temporal_direction=temporal_direction,
            alpha=args.alpha,
            layer=args.intervention_layer,
        )
        for sample, prompt, raw_prediction in zip(batch, prompts, predictions):
            predicted_onset = parse_onset_seconds(raw_prediction)
            actual_onset = float(sample["event_onset_sec"])
            absolute_error = (
                None
                if predicted_onset is None
                else abs(predicted_onset - actual_onset)
            )
            rows.append({
                "id": sample["id"],
                "dataset": sample["dataset"],
                "recording_id": sample["recording_id"],
                "filename": sample["filename"],
                "path": sample["path"],
                "split": sample["split"],
                "label": sample["label"],
                "event_label": sample["event_label"],
                "prompt": prompt,
                "actual_onset_sec": actual_onset,
                "predicted_onset_sec": predicted_onset,
                "raw_prediction": raw_prediction,
                "absolute_error_sec": absolute_error,
                "event_offset_sec": sample["event_offset_sec"],
                "event_duration_sec": sample["event_duration_sec"],
                "sample_rate": sample["sample_rate"],
                "onset_samples": sample["onset_samples"],
                "offset_samples": sample["offset_samples"],
                "length_sec": sample["length_sec"],
                "recording_length_sec": sample["recording_length_sec"],
                "audio_window_start_sec": sample["audio_window_start_sec"],
                "audio_window_end_sec": sample["audio_window_end_sec"],
                "event_recording_onset_sec": sample["event_recording_onset_sec"],
                "event_recording_offset_sec": sample["event_recording_offset_sec"],
                "temporal_id_intervention": temporal_direction is not None,
                "intervention_layer": args.intervention_layer,
                "temporal_alpha": args.alpha if temporal_direction is not None else "",
                "temporal_id_activation_file": (
                    "" if temporal_activation_file is None else str(temporal_activation_file)
                ),
                "top10_tokens": "",
            })
    return rows


def score_relative_rows(
    args: argparse.Namespace,
    dataset: Any,
    model: OnsetInterventionGenerator,
    temporal_ids: RelativeTemporalIdInterpolator,
    temporal_activation_file: Path,
) -> list[dict[str, Any]]:
    rows = []
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list,
    )
    for batch in tqdm(dataloader, desc="Generating relative RealDESED onset predictions"):
        directions, direction_metadata = batch_event_relative_directions(batch, temporal_ids, args)
        batch_rows = score_rows(
            args,
            batch,
            model,
            temporal_direction=directions,
            temporal_activation_file=temporal_activation_file,
        )
        for row, metadata in zip(batch_rows, direction_metadata):
            row.update(metadata)
        rows.extend(batch_rows)
    return rows


def _require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def load_and_align(baseline_csv: Path, intervention_csv: Path) -> pd.DataFrame:
    required = {"id", "actual_onset_sec", "predicted_onset_sec", "absolute_error_sec"}
    baseline = pd.read_csv(baseline_csv)
    intervention = pd.read_csv(intervention_csv)
    _require_columns(baseline, baseline_csv, required)
    _require_columns(intervention, intervention_csv, required)
    metadata_columns = [
        column
        for column in [
            "dataset",
            "recording_id",
            "filename",
            "split",
            "label",
            "event_label",
            "event_duration_sec",
            "length_sec",
            "audio_window_start_sec",
            "audio_window_end_sec",
        ]
        if column in baseline.columns
    ]
    baseline = baseline.rename(
        columns={
            "predicted_onset_sec": "predicted_onset_sec_baseline",
            "raw_prediction": "raw_prediction_baseline",
            "absolute_error_sec": "absolute_error_sec_baseline",
        }
    )
    intervention = intervention.rename(
        columns={
            "predicted_onset_sec": "predicted_onset_sec_intervention",
            "raw_prediction": "raw_prediction_intervention",
            "absolute_error_sec": "absolute_error_sec_intervention",
        }
    )
    aligned = baseline[
        [
            "id",
            "actual_onset_sec",
            "predicted_onset_sec_baseline",
            "raw_prediction_baseline",
            "absolute_error_sec_baseline",
            *metadata_columns,
        ]
    ].merge(
        intervention[
            [
                "id",
                "predicted_onset_sec_intervention",
                "raw_prediction_intervention",
                "absolute_error_sec_intervention",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if aligned.empty:
        raise ValueError("Baseline and intervention CSVs have no overlapping ids")
    aligned["prediction_delta_sec"] = (
        aligned["predicted_onset_sec_intervention"]
        - aligned["predicted_onset_sec_baseline"]
    )
    aligned["absolute_error_delta_sec"] = (
        aligned["absolute_error_sec_intervention"]
        - aligned["absolute_error_sec_baseline"]
    )
    aligned["baseline_valid"] = aligned["predicted_onset_sec_baseline"].notna()
    aligned["intervention_valid"] = aligned["predicted_onset_sec_intervention"].notna()
    aligned["both_valid"] = aligned["baseline_valid"] & aligned["intervention_valid"]
    return aligned


def group_mask(aligned: pd.DataFrame, group_name: str) -> pd.Series:
    if group_name == "all":
        return pd.Series(True, index=aligned.index)
    if group_name == "baseline_valid":
        return aligned["baseline_valid"]
    if group_name == "intervention_valid":
        return aligned["intervention_valid"]
    if group_name == "both_valid":
        return aligned["both_valid"]
    raise ValueError(f"Unknown group: {group_name}")


def summarize_groups(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name in GROUP_ORDER:
        group_df = aligned.loc[group_mask(aligned, group_name)]
        prediction_delta = pd.to_numeric(
            group_df["prediction_delta_sec"],
            errors="coerce",
        ).dropna()
        error_delta = pd.to_numeric(
            group_df["absolute_error_delta_sec"],
            errors="coerce",
        ).dropna()
        baseline_error = pd.to_numeric(
            group_df["absolute_error_sec_baseline"],
            errors="coerce",
        ).dropna()
        intervention_error = pd.to_numeric(
            group_df["absolute_error_sec_intervention"],
            errors="coerce",
        ).dropna()
        rows.append({
            "group": group_name,
            "label": GROUP_LABELS[group_name],
            "count": int(len(group_df)),
            "baseline_valid_count": int(group_df["baseline_valid"].sum()),
            "intervention_valid_count": int(group_df["intervention_valid"].sum()),
            "both_valid_count": int(group_df["both_valid"].sum()),
            "mean_prediction_delta_sec": (
                float(prediction_delta.mean()) if len(prediction_delta) else np.nan
            ),
            "median_prediction_delta_sec": (
                float(prediction_delta.median()) if len(prediction_delta) else np.nan
            ),
            "std_prediction_delta_sec": (
                float(prediction_delta.std(ddof=1)) if len(prediction_delta) > 1 else 0.0
            ),
            "mean_absolute_error_delta_sec": (
                float(error_delta.mean()) if len(error_delta) else np.nan
            ),
            "median_absolute_error_delta_sec": (
                float(error_delta.median()) if len(error_delta) else np.nan
            ),
            "mean_absolute_error_baseline_sec": (
                float(baseline_error.mean()) if len(baseline_error) else np.nan
            ),
            "mean_absolute_error_intervention_sec": (
                float(intervention_error.mean()) if len(intervention_error) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def plot_type_dir(plot_root: Path, plot_type: str, model_slug: str) -> Path:
    directory = plot_root / plot_type / model_slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_type_dir(output_root: Path, output_type: str, model_slug: str) -> Path:
    directory = output_root / output_type / model_slug
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_outputs(
    baseline_csv: Path,
    intervention_csv: Path,
    plot_root: Path,
    output_root: Path,
    model_slug: str,
    output_prefix: str,
    dpi: int,
) -> dict[str, Path]:
    aligned = load_and_align(baseline_csv, intervention_csv)
    summary = summarize_groups(aligned)
    aligned_path = output_type_dir(output_root, "aligned", model_slug) / f"{output_prefix}_aligned.csv"
    summary_path = output_type_dir(output_root, "summary", model_slug) / f"{output_prefix}_summary.csv"
    prediction_delta_plot = (
        plot_type_dir(plot_root, "prediction_delta", model_slug)
        / f"{output_prefix}_prediction_delta.png"
    )
    scatter_plot = (
        plot_type_dir(plot_root, "predicted_onset_scatter", model_slug)
        / f"{output_prefix}_predicted_onset_scatter.png"
    )
    aligned.to_csv(aligned_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_delta_boxplot(
        aligned,
        prediction_delta_plot,
        value_column="prediction_delta_sec",
        ylabel="Intervention - baseline predicted onset (sec)",
        title="RealDESED onset prediction shift under temporal-ID intervention",
        dpi=dpi,
    )
    plot_predicted_onset_scatter(aligned, scatter_plot, dpi=dpi)
    return {
        "aligned": aligned_path,
        "summary": summary_path,
        "prediction_delta_plot": prediction_delta_plot,
        "predicted_onset_scatter": scatter_plot,
    }


def plot_delta_boxplot(
    aligned: pd.DataFrame,
    output_path: Path,
    value_column: str,
    ylabel: str,
    title: str,
    dpi: int,
) -> None:
    data = [
        pd.to_numeric(
            aligned.loc[group_mask(aligned, group_name), value_column],
            errors="coerce",
        ).dropna().to_numpy()
        for group_name in GROUP_ORDER
    ]
    labels = [
        f"{GROUP_LABELS[group_name]}\n(n={len(values)})"
        for group_name, values in zip(GROUP_ORDER, data)
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
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
        patch.set_facecolor("#bfdbfe")
        patch.set_edgecolor("#1d4ed8")
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for index, values in enumerate(data, start=1):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=14,
            alpha=0.45,
            color="#334155",
            linewidths=0,
        )
    ax.axhline(0.0, color="#111111", linewidth=1.0, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_onset_scatter(aligned: pd.DataFrame, output_path: Path, dpi: int) -> None:
    valid = aligned.loc[aligned["both_valid"]].copy()
    valid["actual_onset_sec"] = pd.to_numeric(valid["actual_onset_sec"], errors="coerce")
    valid["predicted_onset_sec_baseline"] = pd.to_numeric(
        valid["predicted_onset_sec_baseline"],
        errors="coerce",
    )
    valid["predicted_onset_sec_intervention"] = pd.to_numeric(
        valid["predicted_onset_sec_intervention"],
        errors="coerce",
    )
    valid = valid.dropna(
        subset=[
            "actual_onset_sec",
            "predicted_onset_sec_baseline",
            "predicted_onset_sec_intervention",
        ]
    )
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    if not valid.empty:
        ax.scatter(
            valid["actual_onset_sec"],
            valid["predicted_onset_sec_baseline"],
            s=18,
            alpha=0.55,
            label="Baseline",
            color="#475569",
            linewidths=0,
        )
        ax.scatter(
            valid["actual_onset_sec"],
            valid["predicted_onset_sec_intervention"],
            s=18,
            alpha=0.55,
            label="Intervention",
            color="#2563eb",
            linewidths=0,
        )
        low = float(
            np.nanmin(
                valid[
                    [
                        "actual_onset_sec",
                        "predicted_onset_sec_baseline",
                        "predicted_onset_sec_intervention",
                    ]
                ].to_numpy()
            )
        )
        high = float(
            np.nanmax(
                valid[
                    [
                        "actual_onset_sec",
                        "predicted_onset_sec_baseline",
                        "predicted_onset_sec_intervention",
                    ]
                ].to_numpy()
            )
        )
        ax.plot([low, high], [low, high], color="#111111", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Actual onset (sec)")
    ax.set_ylabel("Predicted onset (sec)")
    ax.set_title("RealDESED predicted vs actual onset")
    if not valid.empty:
        ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
        "direction_mode": "event_relative_target_minus_source_scaled_by_reference",
        "target": args.target,
        "temporal_id_time_field": args.time_field,
        "binning": {
            "bin_width_sec": args.bin_width_sec,
            "first_bin_center_sec": args.first_bin_center_sec,
            "first_segment_sec": [
                args.first_bin_center_sec - 0.5 * args.bin_width_sec,
                args.first_bin_center_sec + 0.5 * args.bin_width_sec,
            ],
            "last_segment_sec": [
                args.end_target_sec - 0.5 * args.bin_width_sec,
                args.end_target_sec + 0.5 * args.bin_width_sec,
            ],
            "assignment_time": "event_center",
            "lookup": "linear_relative_time_interpolation",
        },
        "real_desed_onset_filters": {
            "split": args.split,
            "size": args.size,
            "min_event_duration_sec": args.min_event_duration_sec,
            "max_event_duration_sec": args.max_event_duration_sec,
            "max_overlap_fraction": args.max_overlap_fraction,
            "exclude_event_labels": args.exclude_event_labels,
            "max_audio_length_sec": args.max_audio_length_sec,
            "random_seed": args.random_seed,
        },
        "temporal_id_source": temporal_metadata,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    model_id = args.model_id or default_model_id(args.model)
    model_slug = model_output_slug(model_id)
    output_dir = args.output_dir / "intervention" / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_ids, temporal_metadata = build_relative_temporal_ids(args, model_id)
    temporal_activation_file = Path(temporal_metadata["activation_path"])
    dataset = build_dataset(args)
    model = OnsetInterventionGenerator(
        build_model_interface(args.model, model_id=model_id, device=args.device)
    )

    stem = experiment_stem(args)
    baseline_csv = output_dir / f"{stem}_baseline_predictions.csv"
    intervention_csv = output_dir / f"{stem}_intervention_predictions.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    baseline_rows = score_rows(args, dataset, model)
    write_rows(baseline_csv, baseline_rows)
    intervention_rows = score_relative_rows(
        args,
        dataset,
        model,
        temporal_ids=temporal_ids,
        temporal_activation_file=temporal_activation_file,
    )
    write_rows(intervention_csv, intervention_rows)
    write_metadata(metadata_path, args, temporal_metadata)

    outputs = {
        "baseline_csv": baseline_csv,
        "intervention_csv": intervention_csv,
        "metadata": metadata_path,
    }
    if not args.no_plot:
        outputs.update(
            plot_outputs(
                baseline_csv,
                intervention_csv,
                args.plot_dir,
                args.output_dir,
                model_slug,
                stem,
                args.dpi,
            )
        )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")
    return outputs


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
