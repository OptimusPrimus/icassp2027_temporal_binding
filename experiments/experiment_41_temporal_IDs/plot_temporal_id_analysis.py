#!/usr/bin/env python3
"""Plot and summarize temporal ID directions from captured decoder activations."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
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
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs" / "activations"
DEFAULT_PLOT_DIR = EXPERIMENT_DIR / "plots"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "outputs"
PLOT_CATEGORY = "temporal_id_analysis"
PLOT_TYPES = {
    "variance": "pca_rank_variance",
    "projection": "pca_2d_projection",
    "fits": "model_fits_pca_2d",
    "overview": "r2_overview",
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


@dataclass(frozen=True)
class ExampleVector:
    split: str
    event_class: str
    time_sec: float
    bin_center_sec: float
    vector: np.ndarray


@dataclass(frozen=True)
class TemporalIdTable:
    split: str
    centers_sec: np.ndarray
    vectors: np.ndarray
    counts: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract class-mean-centered temporal ID vectors from the synthetic "
            "training split of run_save_activations.py output, then run "
            "a layer range, plot "
            "PCA variance, plot a 2D PCA projection, and evaluate simple "
            "time-to-vector models on validation and test temporal IDs."
        )
    )
    parser.add_argument(
        "activation_path",
        nargs="?",
        type=Path,
        help=(
            "Activation .pt bundle from run_save_activations.py. If omitted, "
            "the newest bundle for --model under --input-dir is used."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Root directory for plot files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for CSV and JSON outputs.",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help=(
            "Single layer slot to analyze. Overrides --min-layer/--max-layer. "
            "Use an integer layer slot, a layer name from the bundle, or 'last'. "
            "Slot 0 is decoder input."
        ),
    )
    parser.add_argument(
        "--min-layer",
        type=int,
        default=None,
        help="First layer slot to analyze, inclusive. Defaults depend on --model.",
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="Last layer slot to analyze, inclusive. Defaults depend on --model.",
    )
    parser.add_argument(
        "--token",
        default="query_event",
        help=(
            "Target token to analyze. Defaults to 'query_event', which selects "
            "the last token corresponding to the queried event name. Also accepts "
            "'first', 'last', a 0-based selected-token index, or a token string "
            "such as '?' or 'yes'."
        ),
    )
    parser.add_argument(
        "--time-field",
        choices=("onset", "center", "offset"),
        default="center",
        help="Event time used for bin assignment.",
    )
    parser.add_argument("--bin-width-sec", type=float, default=2.5)
    parser.add_argument("--first-bin-center-sec", type=float, default=2.5)
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=10,
        help=(
            "Minimum number of examples required for a split/bin temporal ID. "
        ),
    )
    parser.add_argument(
        "--exclude-bin-centers-sec",
        type=float,
        nargs="*",
        default=[0.0, 30.0],
        help=(
            "Temporal ID bin centers to exclude after binning."
        ),
    )
    parser.add_argument("--max-time-sec", type=float, default=None)
    parser.add_argument("--piecewise-peak-sec", type=float, default=15.0)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def discover_activation_path(args: argparse.Namespace) -> Path:
    if args.activation_path is not None:
        if not args.activation_path.exists():
            raise FileNotFoundError(f"Activation path does not exist: {args.activation_path}")
        return args.activation_path

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]
    model_dir = args.input_dir / model_output_slug(model_id)
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


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected {path} to contain a dict payload")
    for key in ("activations", "metadata", "target_tokens", "target_lengths"):
        if key not in bundle:
            raise ValueError(f"{path} is missing required key {key!r}")
    return bundle


def resolve_layer(bundle: dict[str, Any], layer_arg: str) -> int:
    layer_names = list(bundle.get("layer_names") or [])
    if layer_arg == "last":
        return len(layer_names) - 1 if layer_names else int(bundle["activations"][0].shape[0]) - 1
    try:
        layer = int(layer_arg)
    except ValueError:
        if layer_arg not in layer_names:
            raise ValueError(
                f"Unknown layer {layer_arg!r}. Available layer names: {', '.join(layer_names)}"
            )
        layer = layer_names.index(layer_arg)

    num_layers = len(layer_names) if layer_names else int(bundle["activations"][0].shape[0])
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"Layer slot {layer} is outside available range [0, {num_layers - 1}]")
    return layer


def available_layer_count(bundle: dict[str, Any]) -> int:
    layer_names = list(bundle.get("layer_names") or [])
    if layer_names:
        return len(layer_names)
    return int(bundle["activations"][0].shape[0])


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
            f"Layer range [{min_layer}, {max_layer}] is outside available "
            f"range [0, {num_layers - 1}]"
        )
    return list(range(min_layer, max_layer + 1))


def normalized_token(token: str) -> str:
    return token.replace("Ġ", "").replace("▁", "").strip()


def normalized_text(text: str) -> str:
    return "".join(char.casefold() for char in text if char.isalnum())


def resolve_query_event_token_index(
    tokens: list[str],
    target_length: int,
    event_label: str,
) -> int | None:
    target = normalized_text(event_label)
    if not target:
        return None

    normalized_tokens = [
        normalized_text(normalized_token(token))
        for token in tokens[:target_length]
    ]
    best_match: tuple[int, int] | None = None
    for start in range(target_length):
        joined = ""
        last_nonempty = None
        for end in range(start, target_length):
            piece = normalized_tokens[end]
            if piece:
                joined += piece
                last_nonempty = end
            if joined == target and last_nonempty is not None:
                best_match = (start, last_nonempty)
                break
            if joined and not target.startswith(joined):
                break
        if best_match is not None:
            return best_match[1]

    label_parts = [part for part in event_label.split() if part]
    if label_parts:
        last_word = normalized_text(label_parts[-1])
        for idx in range(target_length - 1, -1, -1):
            if normalized_tokens[idx] and last_word.endswith(normalized_tokens[idx]):
                return idx
    return None


def resolve_token_index(
    tokens: list[str],
    target_length: int,
    token_arg: str,
    event_label: str,
) -> int | None:
    if target_length <= 0:
        return None
    if token_arg == "query_event":
        return resolve_query_event_token_index(tokens, target_length, event_label)
    if token_arg == "first":
        return 0
    if token_arg == "last":
        return target_length - 1
    try:
        token_index = int(token_arg)
    except ValueError:
        token_index = -1
    if token_index >= 0:
        return token_index if token_index < target_length else None

    for idx, token in enumerate(tokens[:target_length]):
        if token == token_arg or normalized_token(token) == token_arg:
            return idx
    return None


def event_time_sec(metadata: dict[str, Any], time_field: str) -> float:
    events = metadata.get("events") or []
    query_index = metadata.get("query_event_index")
    event = None
    for candidate in events:
        if candidate.get("event_index") == query_index:
            event = candidate
            break
    if event is None and len(events) == 1:
        event = events[0]
    if event is None:
        raise ValueError(f"Could not resolve queried event for sample {metadata.get('id')}")

    onset = float(event["onset_sec"])
    offset = float(event["offset_sec"])
    if time_field == "onset":
        return onset
    if time_field == "offset":
        return offset
    return 0.5 * (onset + offset)


def bin_center_for(time_sec: float, first_center: float, bin_width: float) -> float:
    bin_index = math.floor((time_sec - (first_center - 0.5 * bin_width)) / bin_width)
    return first_center + bin_index * bin_width


def is_synthetic_metadata(metadata: dict[str, Any]) -> bool:
    return (
        str(metadata.get("source_group", "")).startswith("synthetic")
        or str(metadata.get("dataset", "")).startswith("synthetic_sed")
    )


def extract_raw_examples(
    bundle: dict[str, Any],
    layer: int,
    token_arg: str,
    time_field: str,
    first_bin_center: float,
    bin_width: float,
    max_time_sec: float,
) -> list[ExampleVector]:
    examples = []
    missing_token = 0
    for idx, metadata in enumerate(bundle["metadata"]):
        if not is_synthetic_metadata(metadata):
            continue
        split = str(metadata.get("split"))
        if split not in {"train", "validation", "test"}:
            continue
        time_sec = event_time_sec(metadata, time_field)
        if time_sec < 0.0 or time_sec >= max_time_sec:
            continue
        target_length = int(bundle["target_lengths"][idx])
        token_index = resolve_token_index(
            bundle["target_tokens"][idx],
            target_length,
            token_arg,
            str(metadata["query_event_label"]),
        )
        if token_index is None:
            missing_token += 1
            continue
        activation = bundle["activations"][idx]
        vector = activation[layer, token_index].float().numpy()
        examples.append(
            ExampleVector(
                split=split,
                event_class=str(metadata["query_event_label"]),
                time_sec=time_sec,
                bin_center_sec=bin_center_for(time_sec, first_bin_center, bin_width),
                vector=vector,
            )
        )

    if not examples:
        raise ValueError("No synthetic examples matched the requested layer/token/time filters")
    if missing_token:
        print(f"Skipped {missing_token} synthetic examples without token {token_arg!r}")
    return examples


def train_class_means(examples: list[ExampleVector]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for example in examples:
        if example.split == "train":
            grouped.setdefault(example.event_class, []).append(example.vector)
    if not grouped:
        raise ValueError("No training examples found")
    return {
        event_class: np.stack(vectors, axis=0).mean(axis=0)
        for event_class, vectors in grouped.items()
    }


def center_by_train_class_mean(
    examples: list[ExampleVector],
    class_means: dict[str, np.ndarray],
) -> list[ExampleVector]:
    centered = []
    dropped = 0
    for example in examples:
        class_mean = class_means.get(example.event_class)
        if class_mean is None:
            dropped += 1
            continue
        centered.append(
            ExampleVector(
                split=example.split,
                event_class=example.event_class,
                time_sec=example.time_sec,
                bin_center_sec=example.bin_center_sec,
                vector=example.vector - class_mean,
            )
        )
    if dropped:
        print(f"Dropped {dropped} examples whose class was absent from train")
    return centered


def temporal_ids_for_split(
    examples: list[ExampleVector],
    split: str,
    min_bin_count: int,
    excluded_bin_centers: set[float],
) -> TemporalIdTable:
    grouped: dict[float, list[np.ndarray]] = {}
    for example in examples:
        if example.split == split:
            grouped.setdefault(example.bin_center_sec, []).append(example.vector)
    if not grouped:
        raise ValueError(f"No centered examples found for split {split!r}")

    excluded = {
        center: len(vectors)
        for center, vectors in grouped.items()
        if center in excluded_bin_centers
    }
    grouped = {
        center: vectors
        for center, vectors in grouped.items()
        if center not in excluded_bin_centers
    }
    if excluded:
        excluded_summary = ", ".join(
            f"{center:g}s({count})" for center, count in sorted(excluded.items())
        )
        print(f"Excluded {split} temporal ID bins: {excluded_summary}")

    dropped = {
        center: len(vectors)
        for center, vectors in grouped.items()
        if len(vectors) < min_bin_count
    }
    grouped = {
        center: vectors
        for center, vectors in grouped.items()
        if len(vectors) >= min_bin_count
    }
    if dropped:
        dropped_summary = ", ".join(
            f"{center:g}s({count})" for center, count in sorted(dropped.items())
        )
        print(
            f"Dropped sparse {split} temporal ID bins below "
            f"--min-bin-count={min_bin_count}: {dropped_summary}"
        )
    if not grouped:
        raise ValueError(
            f"No {split!r} bins remain after applying --min-bin-count={min_bin_count}"
        )

    centers = np.array(sorted(grouped), dtype=np.float64)
    vectors = np.stack([np.stack(grouped[center], axis=0).mean(axis=0) for center in centers], axis=0)
    counts = np.array([len(grouped[center]) for center in centers], dtype=np.int64)
    return TemporalIdTable(split=split, centers_sec=centers, vectors=vectors, counts=counts)


def cumulative_pca_variance(vectors: np.ndarray, max_rank: int) -> np.ndarray:
    if np.allclose(vectors, vectors[0]):
        return np.zeros(max_rank, dtype=np.float64)
    rank = min(max_rank, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=rank)
    pca.fit(vectors)
    values = np.cumsum(pca.explained_variance_ratio_)
    if rank < max_rank:
        values = np.pad(values, (0, max_rank - rank), constant_values=np.nan)
    return values


def fit_pca_2d(vectors: np.ndarray) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=2)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        coords = pca.fit_transform(vectors)
    if coords.shape[0] >= 2 and coords[-1, 0] < coords[0, 0]:
        pca.components_[0] *= -1.0
        coords[:, 0] *= -1.0
    return pca, coords


def design_matrix(times: np.ndarray, model_name: str, peak_sec: float) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if model_name == "linear":
        return np.column_stack([np.ones_like(times), times])
    if model_name == "quadratic":
        return np.column_stack([np.ones_like(times), times, times**2])
    if model_name == "piecewise_linear_peak":
        peak_height = np.maximum(0.0, 1.0 - np.abs(times - peak_sec) / peak_sec)
        return np.column_stack([np.ones_like(times), times, peak_height])
    raise ValueError(f"Unsupported model: {model_name}")


def fit_vector_model(times: np.ndarray, vectors: np.ndarray, model_name: str, peak_sec: float) -> np.ndarray:
    x_matrix = design_matrix(times, model_name, peak_sec)
    coefficients, *_ = np.linalg.lstsq(x_matrix, vectors, rcond=None)
    return coefficients


def predict_vector_model(
    times: np.ndarray,
    coefficients: np.ndarray,
    model_name: str,
    peak_sec: float,
) -> np.ndarray:
    return design_matrix(times, model_name, peak_sec) @ coefficients


def evaluate_models(
    train: TemporalIdTable,
    tables: list[TemporalIdTable],
    peak_sec: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for model_name in ("linear", "quadratic", "piecewise_linear_peak"):
        coefficients = fit_vector_model(train.centers_sec, train.vectors, model_name, peak_sec)
        for table in tables:
            predicted = predict_vector_model(
                table.centers_sec,
                coefficients,
                model_name,
                peak_sec,
            )
            predictions[f"{model_name}:{table.split}"] = predicted
            rows.append({
                "model": model_name,
                "split": table.split,
                "mae": mean_absolute_error(table.vectors.ravel(), predicted.ravel()),
                "r2": r2_score(table.vectors.ravel(), predicted.ravel()),
                "bins": int(len(table.centers_sec)),
                "examples": int(table.counts.sum()),
            })
    return pd.DataFrame(rows), predictions


def analysis_stem(activation_path: Path, layer: int, token: str, time_field: str) -> str:
    safe_token = token_slug(token)
    return f"{activation_path.stem}_layer{layer}_token-{safe_token}_{time_field}"


def token_slug(token: str) -> str:
    return (
        token.replace("/", "_")
        .replace(" ", "_")
        .replace("?", "question")
        .replace(":", "_")
    )


def write_temporal_id_csv(tables: list[TemporalIdTable], output_path: Path) -> None:
    rows = []
    for table in tables:
        for center, count in zip(table.centers_sec, table.counts):
            rows.append({
                "split": table.split,
                "bin_center_sec": center,
                "count": int(count),
            })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_variance(ranks: np.ndarray, variance: np.ndarray, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(ranks, variance * 100.0, marker="o", color="#2563eb", linewidth=2.0)
    ax.set_xticks(ranks)
    finite = variance[np.isfinite(variance)]
    ymax = min(100.0, float(finite.max() * 100.0) * 1.08) if len(finite) else 1.0
    ax.set_ylim(0.0, max(1.0, ymax))
    ax.set_xlabel("PCA rank")
    ax.set_ylabel("Cumulative variance captured (%)")
    ax.set_title("Temporal ID PCA variance")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_pca_projection(
    pca: PCA,
    train_table: TemporalIdTable,
    output_path: Path,
    dpi: int,
) -> None:
    id_coords = pca.transform(train_table.vectors)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    sc = ax.scatter(
        id_coords[:, 0],
        id_coords[:, 1],
        c=train_table.centers_sec,
        cmap="viridis",
        s=48,
        edgecolor="#111827",
        linewidth=0.4,
        label="Temporal IDs",
        zorder=3,
    )
    ax.plot(id_coords[:, 0], id_coords[:, 1], color="#111827", linewidth=1.0, alpha=0.8, zorder=2)
    for center, (x_coord, y_coord) in zip(train_table.centers_sec, id_coords):
        ax.annotate(f"{center:g}s", (x_coord, y_coord), xytext=(4, 4), textcoords="offset points", fontsize=8)
    colorbar = fig.colorbar(sc, ax=ax)
    colorbar.set_label("Bin center (s)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100.0:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100.0:.1f}%)")
    ax.set_title("Training temporal IDs in 2D PCA space")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_model_fits(
    pca: PCA,
    tables: list[TemporalIdTable],
    predictions: dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True, sharex=True, sharey=True)
    split_colors = {"train": "#111827", "validation": "#2563eb", "test": "#dc2626"}
    for ax, model_name in zip(axes, ("linear", "quadratic", "piecewise_linear_peak")):
        for table in tables:
            coords = pca.transform(table.vectors)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=28,
                color=split_colors[table.split],
                alpha=0.8,
                label=f"{table.split} IDs",
            )
            key = f"{model_name}:{table.split}"
            if key in predictions:
                predicted_coords = pca.transform(predictions[key])
                order = np.argsort(table.centers_sec)
                ax.plot(
                    predicted_coords[order, 0],
                    predicted_coords[order, 1],
                    color=split_colors[table.split],
                    linewidth=1.6,
                    alpha=0.8,
                )
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("PC1")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("PC2")
    axes[0].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_r2_overview(metrics: pd.DataFrame, output_path: Path, dpi: int) -> None:
    selected = metrics[metrics["split"].isin(["train", "validation"])].copy()
    if selected.empty:
        raise ValueError("No train/validation metrics available for R2 overview")

    model_names = ["linear", "quadratic", "piecewise_linear_peak"]
    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(15.0, 4.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    split_styles = {
        "train": {"color": "#111827", "marker": "o", "label": "train"},
        "validation": {"color": "#2563eb", "marker": "s", "label": "validation"},
    }
    for ax, model_name in zip(axes, model_names):
        model_metrics = selected[selected["model"] == model_name]
        for split, style in split_styles.items():
            split_metrics = model_metrics[model_metrics["split"] == split].sort_values("layer")
            if split_metrics.empty:
                continue
            ax.plot(
                split_metrics["layer"],
                split_metrics["r2"],
                linewidth=1.8,
                markersize=4.0,
                **style,
            )
        ax.axhline(0.0, color="#94a3b8", linewidth=0.8)
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("Layer slot")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("R2")
    axes[0].legend(loc="best")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def layer_name_for(bundle: dict[str, Any], layer: int) -> str | None:
    layer_names = list(bundle.get("layer_names") or [])
    return layer_names[layer] if layer < len(layer_names) else None


def run_layer_analysis(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    activation_path: Path,
    layer: int,
    max_time_sec: float,
    excluded_bin_centers: set[float],
    plot_dirs: dict[str, Path],
    results_dir: Path,
) -> dict[str, Any]:
    print(f"Analyzing layer {layer} ({layer_name_for(bundle, layer)})")
    raw_examples = extract_raw_examples(
        bundle=bundle,
        layer=layer,
        token_arg=str(args.token),
        time_field=args.time_field,
        first_bin_center=args.first_bin_center_sec,
        bin_width=args.bin_width_sec,
        max_time_sec=max_time_sec,
    )
    class_means = train_class_means(raw_examples)
    centered_examples = center_by_train_class_mean(raw_examples, class_means)

    train_table = temporal_ids_for_split(
        centered_examples,
        "train",
        args.min_bin_count,
        excluded_bin_centers,
    )
    validation_table = temporal_ids_for_split(
        centered_examples,
        "validation",
        args.min_bin_count,
        excluded_bin_centers,
    )
    test_table = temporal_ids_for_split(
        centered_examples,
        "test",
        args.min_bin_count,
        excluded_bin_centers,
    )
    tables = [train_table, validation_table, test_table]

    stem = analysis_stem(activation_path, layer, str(args.token), args.time_field)
    variance = cumulative_pca_variance(train_table.vectors, max_rank=5)
    variance_path = plot_dirs["variance"] / f"{stem}_pca_rank_variance.png"
    plot_variance(np.arange(1, 6), variance, variance_path, args.dpi)

    pca, _coords = fit_pca_2d(train_table.vectors)
    pca_path = plot_dirs["projection"] / f"{stem}_pca_2d_projection.png"
    plot_pca_projection(pca, train_table, pca_path, args.dpi)

    metrics, predictions = evaluate_models(
        train_table,
        tables,
        args.piecewise_peak_sec,
    )
    metrics.insert(0, "layer", layer)
    layer_name = layer_name_for(bundle, layer)
    metrics.insert(1, "layer_name", layer_name)
    metrics_path = results_dir / f"{stem}_model_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    bins_path = results_dir / f"{stem}_temporal_id_bins.csv"
    write_temporal_id_csv(tables, bins_path)

    fit_path = plot_dirs["fits"] / f"{stem}_model_fits_pca_2d.png"
    plot_model_fits(pca, tables, predictions, fit_path, args.dpi)

    summary_path = results_dir / f"{stem}_summary.json"
    summary = {
        "activation_path": str(activation_path),
        "layer": layer,
        "layer_name": layer_name,
        "token": str(args.token),
        "time_field": args.time_field,
        "bin_width_sec": args.bin_width_sec,
        "first_bin_center_sec": args.first_bin_center_sec,
        "min_bin_count": args.min_bin_count,
        "excluded_bin_centers_sec": sorted(excluded_bin_centers),
        "max_time_sec": max_time_sec,
        "piecewise_peak_sec": args.piecewise_peak_sec,
        "train_classes": len(class_means),
        "pca_cumulative_variance": {
            str(rank): None if np.isnan(value) else float(value)
            for rank, value in zip(range(1, 6), variance)
        },
        "metrics": metrics.to_dict(orient="records"),
        "outputs": {
            "pca_rank_variance": str(variance_path),
            "pca_2d_projection": str(pca_path),
            "model_fits_pca_2d": str(fit_path),
            "model_metrics": str(metrics_path),
            "temporal_id_bins": str(bins_path),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote layer {layer} metrics: {metrics_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.bin_width_sec <= 0:
        raise ValueError("--bin-width-sec must be positive")
    if args.min_bin_count < 1:
        raise ValueError("--min-bin-count must be at least 1")
    excluded_bin_centers = {float(value) for value in args.exclude_bin_centers_sec}

    activation_path = discover_activation_path(args)
    bundle = load_bundle(activation_path)
    config = json.loads(bundle.get("config_json", "{}"))
    max_time_sec = args.max_time_sec or float(config.get("length_sec") or 30.0)
    layers = resolve_layer_range(bundle, args)

    model_slug = activation_path.parent.name
    plot_dirs = {
        key: args.output_dir / plot_type / model_slug
        for key, plot_type in PLOT_TYPES.items()
    }
    for path in plot_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    results_dir = args.results_dir / PLOT_CATEGORY / model_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Activation bundle: {activation_path}")
    print(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Token: {args.token}")

    summaries = [
        run_layer_analysis(
            args=args,
            bundle=bundle,
            activation_path=activation_path,
            layer=layer,
            max_time_sec=max_time_sec,
            excluded_bin_centers=excluded_bin_centers,
            plot_dirs=plot_dirs,
            results_dir=results_dir,
        )
        for layer in layers
    ]

    metrics_frames = []
    for summary in summaries:
        metrics_path = Path(summary["outputs"]["model_metrics"])
        metrics_frames.append(pd.read_csv(metrics_path))
    all_metrics = pd.concat(metrics_frames, ignore_index=True)

    range_stem = (
        f"{activation_path.stem}_layers{layers[0]}-{layers[-1]}_"
        f"token-{token_slug(str(args.token))}_{args.time_field}"
    )
    all_metrics_path = results_dir / f"{range_stem}_all_model_metrics.csv"
    all_metrics.to_csv(all_metrics_path, index=False)
    overview_path = plot_dirs["overview"] / f"{range_stem}_r2_overview.png"
    plot_r2_overview(all_metrics, overview_path, args.dpi)

    summary_path = results_dir / f"{range_stem}_summary.json"
    summary = {
        "activation_path": str(activation_path),
        "layers": layers,
        "token": str(args.token),
        "time_field": args.time_field,
        "bin_width_sec": args.bin_width_sec,
        "first_bin_center_sec": args.first_bin_center_sec,
        "min_bin_count": args.min_bin_count,
        "excluded_bin_centers_sec": sorted(excluded_bin_centers),
        "max_time_sec": max_time_sec,
        "piecewise_peak_sec": args.piecewise_peak_sec,
        "layer_summaries": summaries,
        "outputs": {
            "all_model_metrics": str(all_metrics_path),
            "r2_overview": str(overview_path),
        },
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote all-layer metrics: {all_metrics_path}")
    print(f"Wrote R2 overview plot: {overview_path}")
    print(
        all_metrics[all_metrics["split"].isin(["train", "validation"])]
        .sort_values(["model", "split", "layer"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
