#!/usr/bin/env python3
"""Validate RealDESED temporal IDs against synthetic absolute/relative time curves."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
SYNTHETIC_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "experiment_41_temporal_IDs"
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs" / "activations"
DEFAULT_SYNTHETIC_INPUT_DIR = SYNTHETIC_EXPERIMENT_DIR / "outputs" / "activations"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_PLOT_DIR = EXPERIMENT_DIR / "plots"
PLOT_CATEGORY = "realdesed_relative_time_validation"
SYNTHETIC_REFERENCE_DURATION_SEC = 30.0
PREDICTION_HYPOTHESES = (
    "relative",
    "absolute",
    "absolute_pc1_relative_pc2",
    "relative_pc1_absolute_pc2",
)
PREDICTION_HYPOTHESIS_LABELS = {
    "relative": "Relative t/T",
    "absolute": "Absolute t/30",
    "absolute_pc1_relative_pc2": "Absolute PC1, relative PC2",
    "relative_pc1_absolute_pc2": "Relative PC1, absolute PC2",
}
DEFAULT_MODEL_IDS = {
    "af3": "nvidia/audio-flamingo-3-hf",
    "af-next": "nvidia/audio-flamingo-next-hf",
    "moss-audio": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    "qwen3-audio": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "qwen3-omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}
DEFAULT_LAYER_RANGES = {
    "af3": (14, 20),
    "af-next": (14, 20),
    "moss-audio": (16, 22),
    "qwen3-audio": (22, 28),
    "qwen3-omni": (22, 28),
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (  # noqa: E402
    model_output_slug,
)
from experiments.experiment_41_temporal_IDs.plot_temporal_id_analysis import (  # noqa: E402
    available_layer_count,
    bin_center_for,
    event_time_sec,
    layer_name_for,
    resolve_layer,
    resolve_token_index,
)


@dataclass(frozen=True)
class EventVector:
    dataset_kind: str
    split: str
    event_class: str
    recording_id: str
    time_sec: float
    duration_sec: float
    bin_center_sec: float
    vector: np.ndarray


@dataclass(frozen=True)
class SyntheticIds:
    centers_sec: np.ndarray
    relative_time: np.ndarray
    vectors: np.ndarray
    counts: np.ndarray
    class_counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a 2D synthetic temporal-ID trajectory from class-mean-centered "
            "synthetic embeddings, then test whether RealDESED embeddings are "
            "better predicted by relative time t/T or absolute time t/30."
        )
    )
    parser.add_argument(
        "activation_path",
        nargs="?",
        type=Path,
        help=(
            "RealDESED validation activation .pt bundle from run_save_relative_time_activations.py. "
            "If omitted, the newest bundle for --model under --input-dir is used."
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_MODEL_IDS),
        default="af3",
        help="Model key used to discover the newest activation bundle.",
    )
    parser.add_argument("--model-id", default=None, help="Override the model id used for discovery.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--synthetic-activation-path",
        type=Path,
        default=None,
        help=(
            "Fixed-30s synthetic activation .pt bundle from experiment_41_temporal_IDs. "
            "If omitted, the newest bundle for --model under --synthetic-input-dir is used."
        ),
    )
    parser.add_argument(
        "--synthetic-input-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_INPUT_DIR,
        help="Root containing experiment_41_temporal_IDs activation bundles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Root for PNG plots; files are written under <root>/<plot_type>/<model_slug>/.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help=(
            "Single layer slot to analyze. Overrides --min-layer/--max-layer. "
            "Use an integer layer slot, a layer name from the bundle, or 'last'."
        ),
    )
    parser.add_argument("--min-layer", type=int, default=None)
    parser.add_argument("--max-layer", type=int, default=None)
    parser.add_argument(
        "--token",
        default="query_event",
        help="Target token to analyze: query_event, first, last, index, or token string.",
    )
    parser.add_argument(
        "--time-field",
        choices=("onset", "center", "offset"),
        default="center",
        help="Event time used for synthetic bins and RealDESED validation.",
    )
    parser.add_argument(
        "--synthetic-split",
        choices=("train", "validation", "test", "all"),
        default="train",
        help="Synthetic split used to fit temporal IDs, PCA, and the quadratic trajectory.",
    )
    parser.add_argument("--bin-width-sec", type=float, default=2.5)
    parser.add_argument("--first-bin-center-sec", type=float, default=2.5)
    parser.add_argument("--min-bin-count", type=int, default=10)
    parser.add_argument("--exclude-bin-centers-sec", type=float, nargs="*", default=[0.0, 30.0])
    parser.add_argument("--max-time-sec", type=float, default=None)
    parser.add_argument("--synthetic-reference-duration-sec", type=float, default=SYNTHETIC_REFERENCE_DURATION_SEC)
    parser.add_argument(
        "--synthetic-length-tolerance-sec",
        type=float,
        default=0.25,
        help="Maximum allowed deviation from the 30s synthetic reference duration.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def newest_activation_path(input_dir: Path, model_id: str) -> Path:
    model_dir = input_dir / model_output_slug(model_id)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model output directory does not exist: {model_dir}")
    candidates = sorted(
        model_dir.glob("*all_decoder_text_activations.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No activation bundles found under {model_dir}")
    return candidates[0]


def plot_type_dir(plot_root: Path, plot_type: str, model_slug: str) -> Path:
    return plot_root / plot_type / model_slug


def discover_real_activation_path(args: argparse.Namespace) -> Path:
    if args.activation_path is not None:
        if not args.activation_path.exists():
            raise FileNotFoundError(f"RealDESED activation path does not exist: {args.activation_path}")
        return args.activation_path
    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    return newest_activation_path(args.input_dir, model_id)


def discover_synthetic_activation_path(args: argparse.Namespace) -> Path:
    if args.synthetic_activation_path is not None:
        if not args.synthetic_activation_path.exists():
            raise FileNotFoundError(
                f"Synthetic activation path does not exist: {args.synthetic_activation_path}"
            )
        return args.synthetic_activation_path
    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    return newest_activation_path(args.synthetic_input_dir, model_id)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected {path} to contain a dict payload")
    for key in ("activations", "metadata", "target_tokens", "target_lengths"):
        if key not in bundle:
            raise ValueError(f"{path} is missing required key {key!r}")
    return bundle


def resolve_layer_range(bundle: dict[str, Any], args: argparse.Namespace) -> list[int]:
    num_layers = available_layer_count(bundle)
    if args.layer is not None:
        return [resolve_layer(bundle, str(args.layer))]
    default_min, default_max = DEFAULT_LAYER_RANGES[args.model]
    min_layer = default_min if args.min_layer is None else int(args.min_layer)
    max_layer = default_max if args.max_layer is None else int(args.max_layer)
    if min_layer > max_layer:
        raise ValueError("--min-layer must be <= --max-layer")
    if min_layer < 0 or max_layer >= num_layers:
        raise ValueError(
            f"Layer range [{min_layer}, {max_layer}] is outside available range [0, {num_layers - 1}]"
        )
    return list(range(min_layer, max_layer + 1))


def validate_layers_available(
    real_bundle: dict[str, Any],
    synthetic_bundle: dict[str, Any],
    layers: list[int],
) -> None:
    real_count = available_layer_count(real_bundle)
    synthetic_count = available_layer_count(synthetic_bundle)
    missing = [layer for layer in layers if layer >= real_count or layer >= synthetic_count]
    if missing:
        raise ValueError(
            "Requested layers are not available in both bundles. "
            f"RealDESED bundle has {real_count} layer slots; "
            f"synthetic bundle has {synthetic_count}; missing: {missing}"
        )


def is_synthetic_metadata(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("source_group", "")).startswith("synthetic")
        or str(metadata.get("dataset", "")).startswith("synthetic_sed")
    )


def is_real_desed_metadata(metadata: dict[str, Any]) -> bool:
    return str(metadata.get("source_group", "")) == "real_desed" or str(metadata.get("dataset", "")) == "real_desed"


def normalized_token_slug(token: str) -> str:
    return (
        token.replace("/", "_")
        .replace(" ", "_")
        .replace("?", "question")
        .replace(":", "_")
    )


def extract_examples(
    bundle: dict[str, Any],
    layer: int,
    token_arg: str,
    time_field: str,
    first_bin_center: float,
    bin_width: float,
    max_time_sec: float,
) -> list[EventVector]:
    examples = []
    missing_token = 0
    for idx, metadata in enumerate(bundle["metadata"]):
        if not (is_synthetic_metadata(metadata) or is_real_desed_metadata(metadata)):
            continue
        time_sec = event_time_sec(metadata, time_field)
        if time_sec < 0.0:
            continue
        if is_synthetic_metadata(metadata) and time_sec >= max_time_sec:
            continue
        target_length = int(bundle["target_lengths"][idx])
        event_class = str(metadata["query_event_label"])
        token_index = resolve_token_index(
            bundle["target_tokens"][idx],
            target_length,
            token_arg,
            event_class,
        )
        if token_index is None:
            missing_token += 1
            continue
        activation = bundle["activations"][idx]
        duration = float(metadata.get("length_sec") or metadata.get("relative_window", {}).get("length_sec") or 0.0)
        if duration <= 0.0:
            continue
        examples.append(
            EventVector(
                dataset_kind="synthetic" if is_synthetic_metadata(metadata) else "real_desed",
                split=str(metadata.get("split", "")),
                event_class=event_class,
                recording_id=str(metadata.get("id", idx)),
                time_sec=float(time_sec),
                duration_sec=duration,
                bin_center_sec=bin_center_for(time_sec, first_bin_center, bin_width),
                vector=activation[layer, token_index].float().numpy(),
            )
        )
    if missing_token:
        print(f"Skipped {missing_token} examples without token {token_arg!r}")
    if not examples:
        raise ValueError("No examples matched the requested layer/token/time filters")
    return examples


def center_by_class_mean(examples: list[EventVector], dataset_kind: str) -> list[EventVector]:
    selected = [example for example in examples if example.dataset_kind == dataset_kind]
    grouped: dict[str, list[np.ndarray]] = {}
    for example in selected:
        grouped.setdefault(example.event_class, []).append(example.vector)
    if not grouped:
        raise ValueError(f"No {dataset_kind} examples found")
    class_means = {
        event_class: np.stack(vectors, axis=0).mean(axis=0)
        for event_class, vectors in grouped.items()
    }
    return [
        EventVector(
            dataset_kind=example.dataset_kind,
            split=example.split,
            event_class=example.event_class,
            recording_id=example.recording_id,
            time_sec=example.time_sec,
            duration_sec=example.duration_sec,
            bin_center_sec=example.bin_center_sec,
            vector=example.vector - class_means[example.event_class],
        )
        for example in selected
    ]


def select_synthetic_split(examples: list[EventVector], split: str) -> list[EventVector]:
    if split == "all":
        return examples
    selected = [example for example in examples if example.split == split]
    if not selected:
        raise ValueError(f"No synthetic examples found for --synthetic-split={split!r}")
    return selected


def validate_fixed_synthetic_duration(
    examples: list[EventVector],
    reference_duration_sec: float,
    tolerance_sec: float,
) -> None:
    bad = [
        example
        for example in examples
        if abs(example.duration_sec - reference_duration_sec) > tolerance_sec
    ]
    if bad:
        durations = np.array([example.duration_sec for example in bad], dtype=np.float64)
        raise ValueError(
            "Synthetic reference bundle contains non-30s examples after filtering: "
            f"{len(bad)}/{len(examples)} differ from {reference_duration_sec:g}s by more than "
            f"{tolerance_sec:g}s. Bad duration range: {durations.min():.3f}-{durations.max():.3f}s. "
            "Pass a fixed-30s experiment_41 bundle via --synthetic-activation-path."
        )


def synthetic_temporal_ids(
    centered_synthetic: list[EventVector],
    min_bin_count: int,
    excluded_bin_centers: set[float],
    reference_duration_sec: float,
) -> SyntheticIds:
    by_class_bin: dict[tuple[str, float], list[np.ndarray]] = {}
    for example in centered_synthetic:
        if example.bin_center_sec in excluded_bin_centers:
            continue
        by_class_bin.setdefault((example.event_class, example.bin_center_sec), []).append(example.vector)

    per_class_bin = {
        key: np.stack(vectors, axis=0).mean(axis=0)
        for key, vectors in by_class_bin.items()
    }
    by_bin: dict[float, list[np.ndarray]] = {}
    counts_by_bin: dict[float, int] = {}
    for (event_class, center), vector in per_class_bin.items():
        _ = event_class
        by_bin.setdefault(center, []).append(vector)
        counts_by_bin[center] = counts_by_bin.get(center, 0) + len(by_class_bin[(event_class, center)])
    by_bin = {
        center: vectors
        for center, vectors in by_bin.items()
        if counts_by_bin[center] >= min_bin_count
    }
    if not by_bin:
        raise ValueError("No synthetic temporal bins remain after exclusions and --min-bin-count")

    centers = np.array(sorted(by_bin), dtype=np.float64)
    vectors = np.stack([np.stack(by_bin[center], axis=0).mean(axis=0) for center in centers], axis=0)
    counts = np.array([counts_by_bin[center] for center in centers], dtype=np.int64)
    class_counts = np.array([len(by_bin[center]) for center in centers], dtype=np.int64)
    return SyntheticIds(
        centers_sec=centers,
        relative_time=centers / float(reference_duration_sec),
        vectors=vectors,
        counts=counts,
        class_counts=class_counts,
    )


def fit_pca_2d(vectors: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    if coords.shape[0] >= 2 and coords[-1, 0] < coords[0, 0]:
        pca.components_[0] *= -1.0
        coords[:, 0] *= -1.0
    return pca, coords


def fit_quadratic_curve(r_values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(r_values), r_values, r_values**2])
    coefficients, *_ = np.linalg.lstsq(design, coords, rcond=None)
    return coefficients


def predict_curve(r_values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    r_values = np.asarray(r_values, dtype=np.float64)
    return np.column_stack([np.ones_like(r_values), r_values, r_values**2]) @ coefficients


def multivariate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual_sq = np.sum((observed - predicted) ** 2, axis=1)
    denom = float(np.sum((observed - observed.mean(axis=0, keepdims=True)) ** 2))
    return {
        "mean_euclidean_distance": float(np.sqrt(residual_sq).mean()),
        "rmse_2d": float(np.sqrt(residual_sq.sum() / (2.0 * len(observed)))),
        "r2_multivariate": float(1.0 - residual_sq.sum() / denom) if denom > 0.0 else float("nan"),
    }


def per_pc_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = {}
    for pc_index in range(2):
        obs = observed[:, pc_index]
        pred = predicted[:, pc_index]
        residual = obs - pred
        denom = float(np.sum((obs - obs.mean()) ** 2))
        prefix = f"pc{pc_index + 1}"
        result[f"{prefix}_mae"] = float(np.mean(np.abs(residual)))
        result[f"{prefix}_rmse"] = float(np.sqrt(np.mean(residual**2)))
        result[f"{prefix}_r2"] = float(1.0 - np.sum(residual**2) / denom) if denom > 0.0 else float("nan")
    return result


def bootstrap_ci_by_recording(
    frame: pd.DataFrame,
    value_column: str,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    recordings = frame["recording_id"].drop_duplicates().to_numpy()
    if len(recordings) <= 1 or iterations <= 0:
        value = float(frame[value_column].mean())
        return value, value
    values = []
    grouped = {recording_id: group for recording_id, group in frame.groupby("recording_id")}
    for _ in range(iterations):
        sampled_ids = rng.choice(recordings, size=len(recordings), replace=True)
        sampled = pd.concat([grouped[recording_id] for recording_id in sampled_ids], ignore_index=True)
        values.append(float(sampled[value_column].mean()))
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def duration_bin_labels(durations: pd.Series) -> pd.Series:
    edges = [0.0, 20.0, 25.0, 30.0, math.inf]
    labels = ["<=20s", "20-25s", "25-30s", ">30s"]
    return pd.cut(durations, bins=edges, labels=labels, right=True, include_lowest=True)


def metrics_rows(
    frame: pd.DataFrame,
    observed_cols: list[str],
    rng: np.random.Generator,
    bootstrap_iterations: int,
    subset_name: str,
) -> list[dict[str, Any]]:
    observed = frame[observed_cols].to_numpy(dtype=np.float64)
    rows = []
    for hypothesis in PREDICTION_HYPOTHESES:
        predicted = frame[[f"pred_{hypothesis}_pc1", f"pred_{hypothesis}_pc2"]].to_numpy(dtype=np.float64)
        row = {
            "subset": subset_name,
            "hypothesis": hypothesis,
            "samples": int(len(frame)),
            "recordings": int(frame["recording_id"].nunique()),
        }
        row.update(multivariate_metrics(observed, predicted))
        row.update(per_pc_metrics(observed, predicted))
        rows.append(row)

    ci_low, ci_high = bootstrap_ci_by_recording(
        frame,
        "relative_over_absolute_improvement",
        rng,
        bootstrap_iterations,
    )
    rows.append({
        "subset": subset_name,
        "hypothesis": "relative_improvement",
        "samples": int(len(frame)),
        "recordings": int(frame["recording_id"].nunique()),
        "mean_euclidean_distance": float(frame["relative_over_absolute_improvement"].mean()),
        "rmse_2d": float("nan"),
        "r2_multivariate": float("nan"),
        "pc1_mae": float("nan"),
        "pc1_rmse": float("nan"),
        "pc1_r2": float("nan"),
        "pc2_mae": float("nan"),
        "pc2_rmse": float("nan"),
        "pc2_r2": float("nan"),
        "median_improvement": float(frame["relative_over_absolute_improvement"].median()),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "fraction_relative_lower_error": float((frame["error_relative"] < frame["error_absolute"]).mean()),
    })
    return rows


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def plot_error_comparison(frame: pd.DataFrame, output_base: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    values = [frame[f"error_{hypothesis}"].to_numpy() for hypothesis in PREDICTION_HYPOTHESES]
    positions = np.arange(1, len(PREDICTION_HYPOTHESES) + 1)
    parts = ax.violinplot(values, positions=positions, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("#94a3b8")
        body.set_edgecolor("#334155")
        body.set_alpha(0.45)
    ax.boxplot(values, positions=positions, widths=0.22, showfliers=False)
    ax.set_xticks(
        positions,
        [PREDICTION_HYPOTHESIS_LABELS[hypothesis] for hypothesis in PREDICTION_HYPOTHESES],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Euclidean error in 2D PCA space")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, output_base, dpi)


def plot_duration_bin_comparison(bin_frame: pd.DataFrame, output_base: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    hypotheses = list(PREDICTION_HYPOTHESES)
    bins = list(bin_frame["duration_bin"].drop_duplicates())
    x = np.arange(len(bins))
    width = min(0.18, 0.72 / len(hypotheses))
    offsets = (np.arange(len(hypotheses)) - (len(hypotheses) - 1) / 2.0) * width
    for offset, hypothesis in zip(offsets, hypotheses):
        selected = bin_frame[bin_frame["hypothesis"] == hypothesis].set_index("duration_bin").reindex(bins)
        y = selected["mean_error"].to_numpy()
        yerr = np.vstack([
            y - selected["ci_low"].to_numpy(),
            selected["ci_high"].to_numpy() - y,
        ])
        ax.bar(
            x + offset,
            y,
            width=width,
            label=PREDICTION_HYPOTHESIS_LABELS[hypothesis],
            yerr=yerr,
            capsize=3,
        )
    ax.set_xticks(x, bins)
    ax.set_xlabel("Recording duration bin")
    ax.set_ylabel("Mean Euclidean error")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig, output_base, dpi)


def duration_bin_metrics(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    rows = []
    for bin_label, group in frame.groupby("duration_bin", observed=True):
        for hypothesis in PREDICTION_HYPOTHESES:
            column = f"error_{hypothesis}"
            ci_low, ci_high = bootstrap_ci_by_recording(group, column, rng, bootstrap_iterations)
            rows.append({
                "duration_bin": str(bin_label),
                "hypothesis": hypothesis,
                "samples": int(len(group)),
                "recordings": int(group["recording_id"].nunique()),
                "mean_error": float(group[column].mean()),
                "median_error": float(group[column].median()),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
    return pd.DataFrame(rows)


def build_real_frame(
    centered_real: list[EventVector],
    pca: PCA,
    curve_coefficients: np.ndarray,
    synthetic_r_min: float,
    synthetic_r_max: float,
    reference_duration_sec: float,
) -> pd.DataFrame:
    real_vectors = np.stack([example.vector for example in centered_real], axis=0)
    coords = pca.transform(real_vectors)
    relative_r = np.array([example.time_sec / example.duration_sec for example in centered_real], dtype=np.float64)
    absolute_r = np.array([example.time_sec / reference_duration_sec for example in centered_real], dtype=np.float64)
    pred_relative = predict_curve(relative_r, curve_coefficients)
    pred_absolute = predict_curve(absolute_r, curve_coefficients)
    pred_absolute_pc1_relative_pc2 = pred_relative.copy()
    pred_absolute_pc1_relative_pc2[:, 0] = pred_absolute[:, 0]
    pred_relative_pc1_absolute_pc2 = pred_relative.copy()
    pred_relative_pc1_absolute_pc2[:, 1] = pred_absolute[:, 1]
    error_relative = np.linalg.norm(coords - pred_relative, axis=1)
    error_absolute = np.linalg.norm(coords - pred_absolute, axis=1)
    error_absolute_pc1_relative_pc2 = np.linalg.norm(coords - pred_absolute_pc1_relative_pc2, axis=1)
    error_relative_pc1_absolute_pc2 = np.linalg.norm(coords - pred_relative_pc1_absolute_pc2, axis=1)
    frame = pd.DataFrame({
        "dataset_kind": [example.dataset_kind for example in centered_real],
        "split": [example.split for example in centered_real],
        "event_class": [example.event_class for example in centered_real],
        "recording_id": [example.recording_id for example in centered_real],
        "time_sec": [example.time_sec for example in centered_real],
        "duration_sec": [example.duration_sec for example in centered_real],
        "relative_time": relative_r,
        "absolute_reference_time": absolute_r,
        "absolute_outside_synthetic_range": (absolute_r < synthetic_r_min) | (absolute_r > synthetic_r_max),
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "pred_relative_pc1": pred_relative[:, 0],
        "pred_relative_pc2": pred_relative[:, 1],
        "pred_absolute_pc1": pred_absolute[:, 0],
        "pred_absolute_pc2": pred_absolute[:, 1],
        "pred_absolute_pc1_relative_pc2_pc1": pred_absolute_pc1_relative_pc2[:, 0],
        "pred_absolute_pc1_relative_pc2_pc2": pred_absolute_pc1_relative_pc2[:, 1],
        "pred_relative_pc1_absolute_pc2_pc1": pred_relative_pc1_absolute_pc2[:, 0],
        "pred_relative_pc1_absolute_pc2_pc2": pred_relative_pc1_absolute_pc2[:, 1],
        "error_relative": error_relative,
        "error_absolute": error_absolute,
        "error_absolute_pc1_relative_pc2": error_absolute_pc1_relative_pc2,
        "error_relative_pc1_absolute_pc2": error_relative_pc1_absolute_pc2,
        "relative_over_absolute_improvement": error_absolute - error_relative,
    })
    frame["duration_bin"] = duration_bin_labels(frame["duration_sec"])
    return frame


def run_layer_analysis(
    args: argparse.Namespace,
    real_bundle: dict[str, Any],
    synthetic_bundle: dict[str, Any],
    real_activation_path: Path,
    synthetic_activation_path: Path,
    layer: int,
    output_dir: Path,
    plot_root: Path,
    model_slug: str,
) -> dict[str, Any]:
    print(f"Analyzing layer {layer} ({layer_name_for(real_bundle, layer)})")
    config = json.loads(synthetic_bundle.get("config_json", "{}"))
    max_time_sec = args.max_time_sec or float(config.get("length_sec") or args.synthetic_reference_duration_sec)
    synthetic_examples_all = extract_examples(
        bundle=synthetic_bundle,
        layer=layer,
        token_arg=str(args.token),
        time_field=args.time_field,
        first_bin_center=args.first_bin_center_sec,
        bin_width=args.bin_width_sec,
        max_time_sec=max_time_sec,
    )
    real_examples_all = extract_examples(
        bundle=real_bundle,
        layer=layer,
        token_arg=str(args.token),
        time_field=args.time_field,
        first_bin_center=args.first_bin_center_sec,
        bin_width=args.bin_width_sec,
        max_time_sec=max_time_sec,
    )
    synthetic_examples = select_synthetic_split(
        [example for example in synthetic_examples_all if example.dataset_kind == "synthetic"],
        args.synthetic_split,
    )
    validate_fixed_synthetic_duration(
        synthetic_examples,
        args.synthetic_reference_duration_sec,
        args.synthetic_length_tolerance_sec,
    )
    centered_synthetic = center_by_class_mean(synthetic_examples, "synthetic")
    centered_real = center_by_class_mean(real_examples_all, "real_desed")
    synthetic_ids = synthetic_temporal_ids(
        centered_synthetic,
        min_bin_count=args.min_bin_count,
        excluded_bin_centers={float(value) for value in args.exclude_bin_centers_sec},
        reference_duration_sec=args.synthetic_reference_duration_sec,
    )
    pca, synthetic_coords = fit_pca_2d(synthetic_ids.vectors)
    curve_coefficients = fit_quadratic_curve(synthetic_ids.relative_time, synthetic_coords)
    real_frame = build_real_frame(
        centered_real,
        pca,
        curve_coefficients,
        synthetic_r_min=float(synthetic_ids.relative_time.min()),
        synthetic_r_max=float(synthetic_ids.relative_time.max()),
        reference_duration_sec=args.synthetic_reference_duration_sec,
    )

    stem = (
        f"{real_activation_path.stem}_fixed30s-synthetic-ref_layer{layer}_"
        f"token-{normalized_token_slug(str(args.token))}_"
        f"{args.time_field}_synthetic-{args.synthetic_split}"
    )
    rng = np.random.default_rng(args.random_seed + layer)
    metrics = pd.DataFrame(metrics_rows(
        real_frame,
        ["pc1", "pc2"],
        rng,
        args.bootstrap_iterations,
        "all_extrapolated",
    ))
    in_range_frame = real_frame[~real_frame["absolute_outside_synthetic_range"]].copy()
    if not in_range_frame.empty and len(in_range_frame) < len(real_frame):
        metrics = pd.concat(
            [
                metrics,
                pd.DataFrame(metrics_rows(
                    in_range_frame,
                    ["pc1", "pc2"],
                    rng,
                    args.bootstrap_iterations,
                    "absolute_in_synthetic_range",
                )),
            ],
            ignore_index=True,
        )

    bin_metrics = duration_bin_metrics(real_frame, rng, args.bootstrap_iterations)
    synthetic_table = pd.DataFrame({
        "bin_center_sec": synthetic_ids.centers_sec,
        "relative_time": synthetic_ids.relative_time,
        "count": synthetic_ids.counts,
        "class_count": synthetic_ids.class_counts,
        "pc1": synthetic_coords[:, 0],
        "pc2": synthetic_coords[:, 1],
    })
    coefficients = pd.DataFrame(
        curve_coefficients,
        index=["intercept", "r", "r_squared"],
        columns=["pc1", "pc2"],
    )

    real_path = output_dir / f"{stem}_realdesed_predictions.csv"
    metrics_path = output_dir / f"{stem}_metrics.csv"
    bins_path = output_dir / f"{stem}_synthetic_temporal_ids.csv"
    coeff_path = output_dir / f"{stem}_quadratic_curve_coefficients.csv"
    duration_bins_path = output_dir / f"{stem}_duration_bin_metrics.csv"
    real_frame.to_csv(real_path, index=False)
    metrics.insert(0, "layer", layer)
    metrics.insert(1, "layer_name", layer_name_for(real_bundle, layer))
    metrics.to_csv(metrics_path, index=False)
    synthetic_table.to_csv(bins_path, index=False)
    coefficients.to_csv(coeff_path)
    bin_metrics.to_csv(duration_bins_path, index=False)

    plot_error_comparison(
        real_frame,
        plot_type_dir(plot_root, "error_comparison", model_slug) / f"{stem}_error_comparison",
        args.dpi,
    )
    plot_duration_bin_comparison(
        bin_metrics,
        plot_type_dir(plot_root, "duration_bin_comparison", model_slug) / f"{stem}_duration_bin_comparison",
        args.dpi,
    )

    outside_count = int(real_frame["absolute_outside_synthetic_range"].sum())
    summary = {
        "real_activation_path": str(real_activation_path),
        "synthetic_activation_path": str(synthetic_activation_path),
        "layer": layer,
        "layer_name": layer_name_for(real_bundle, layer),
        "synthetic_layer_name": layer_name_for(synthetic_bundle, layer),
        "token": str(args.token),
        "time_field": args.time_field,
        "synthetic_split": args.synthetic_split,
        "synthetic_reference_duration_sec": args.synthetic_reference_duration_sec,
        "synthetic_r_range": [float(synthetic_ids.relative_time.min()), float(synthetic_ids.relative_time.max())],
        "pca_explained_variance_ratio": {
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]),
            "sum": float(pca.explained_variance_ratio_[:2].sum()),
        },
        "samples": {
            "synthetic_events_centered": len(centered_synthetic),
            "synthetic_recordings": int(len({example.recording_id for example in centered_synthetic})),
            "realdesed_events_centered": len(centered_real),
            "realdesed_recordings": int(real_frame["recording_id"].nunique()),
            "absolute_outside_synthetic_range": outside_count,
            "absolute_inside_synthetic_range": int(len(real_frame) - outside_count),
        },
        "metrics": metrics.to_dict(orient="records"),
        "outputs": {
            "realdesed_predictions": str(real_path),
            "metrics": str(metrics_path),
            "synthetic_temporal_ids": str(bins_path),
            "quadratic_curve_coefficients": str(coeff_path),
            "duration_bin_metrics": str(duration_bins_path),
        },
    }
    summary_path = output_dir / f"{stem}_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(
        f"Wrote layer {layer}: {metrics_path} "
        f"({outside_count}/{len(real_frame)} absolute-time samples outside synthetic range)"
    )
    return summary


def plot_layer_overview(all_metrics: pd.DataFrame, output_path: Path, dpi: int) -> None:
    selected = all_metrics[
        (all_metrics["subset"] == "all_extrapolated")
        & (all_metrics["hypothesis"].isin(PREDICTION_HYPOTHESES))
    ].copy()
    if selected.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    colors = {
        "relative": "#2563eb",
        "absolute": "#dc2626",
        "absolute_pc1_relative_pc2": "#16a34a",
        "relative_pc1_absolute_pc2": "#9333ea",
    }
    for hypothesis in PREDICTION_HYPOTHESES:
        group = selected[selected["hypothesis"] == hypothesis].sort_values("layer")
        axes[0].plot(
            group["layer"],
            group["mean_euclidean_distance"],
            marker="o",
            color=colors[hypothesis],
            label=PREDICTION_HYPOTHESIS_LABELS[hypothesis],
        )
        axes[1].plot(
            group["layer"],
            group["r2_multivariate"],
            marker="o",
            color=colors[hypothesis],
            label=PREDICTION_HYPOTHESIS_LABELS[hypothesis],
        )
    axes[0].set_ylabel("Mean Euclidean error")
    axes[1].set_ylabel("Multivariate R2")
    for ax in axes:
        ax.set_xlabel("Layer slot")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    save_figure(fig, output_path.with_suffix(""), dpi)


def main() -> None:
    args = parse_args()
    if args.bin_width_sec <= 0:
        raise ValueError("--bin-width-sec must be positive")
    if args.min_bin_count < 1:
        raise ValueError("--min-bin-count must be at least 1")
    if args.synthetic_reference_duration_sec <= 0:
        raise ValueError("--synthetic-reference-duration-sec must be positive")

    real_activation_path = discover_real_activation_path(args)
    synthetic_activation_path = discover_synthetic_activation_path(args)
    real_bundle = load_bundle(real_activation_path)
    synthetic_bundle = load_bundle(synthetic_activation_path)
    layers = resolve_layer_range(real_bundle, args)
    validate_layers_available(real_bundle, synthetic_bundle, layers)
    model_slug = real_activation_path.parent.name
    output_dir = args.output_dir / PLOT_CATEGORY / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"RealDESED activation bundle: {real_activation_path}")
    print(f"Synthetic reference bundle: {synthetic_activation_path}")
    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Token: {args.token}")

    summaries = [
        run_layer_analysis(
            args,
            real_bundle,
            synthetic_bundle,
            real_activation_path,
            synthetic_activation_path,
            layer,
            output_dir,
            args.plot_dir,
            model_slug,
        )
        for layer in layers
    ]
    all_metrics = pd.concat(
        [pd.read_csv(summary["outputs"]["metrics"]) for summary in summaries],
        ignore_index=True,
    )
    range_stem = (
        f"{real_activation_path.stem}_fixed30s-synthetic-ref_layers{layers[0]}-{layers[-1]}_"
        f"token-{normalized_token_slug(str(args.token))}_{args.time_field}_"
        f"synthetic-{args.synthetic_split}"
    )
    all_metrics_path = output_dir / f"{range_stem}_all_metrics.csv"
    all_metrics.to_csv(all_metrics_path, index=False)
    overview_path = plot_type_dir(args.plot_dir, "layer_overview", model_slug) / f"{range_stem}_layer_overview.png"
    plot_layer_overview(all_metrics, overview_path, args.dpi)

    summary_path = output_dir / f"{range_stem}_summary.json"
    with summary_path.open("w") as handle:
        json.dump(
            {
                "real_activation_path": str(real_activation_path),
                "synthetic_activation_path": str(synthetic_activation_path),
                "layers": layers,
                "token": str(args.token),
                "time_field": args.time_field,
                "synthetic_split": args.synthetic_split,
                "layer_summaries": summaries,
                "outputs": {
                    "all_metrics": str(all_metrics_path),
                    "layer_overview": str(overview_path),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote all-layer metrics: {all_metrics_path}")
    print(
        all_metrics[
            (all_metrics["subset"] == "all_extrapolated")
            & (
                all_metrics["hypothesis"].isin([
                    *PREDICTION_HYPOTHESES,
                    "relative_improvement",
                ])
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
