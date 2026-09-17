#!/usr/bin/env python3
"""Create a compact IEEE-style 2D PCA figure for temporal ID vectors."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_temporal_id_analysis import (
    center_by_train_class_mean,
    extract_raw_examples,
    fit_pca_2d,
    load_bundle,
    temporal_ids_for_split,
    token_slug,
    train_class_means,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs" / "activations"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots" / "paper"
PLOT_CATEGORY = "temporal_id_analysis"
DEFAULT_ANALYSIS_RESULTS_DIR = EXPERIMENT_DIR / "outputs" / PLOT_CATEGORY


@dataclass(frozen=True)
class ModelPanel:
    key: str
    label: str
    model_slug: str
    layer: int

    @property
    def decoder_layer(self) -> int:
        return self.layer - 1


MODEL_PANELS = (
    ModelPanel(
        key="af-next",
        label="AF-Next",
        model_slug="nvidia__audio-flamingo-next-hf",
        layer=17,
    ),
    ModelPanel(
        key="moss-audio",
        label="MOSS-Audio",
        model_slug="OpenMOSS-Team__MOSS-Audio-8B-Instruct",
        layer=19,
    ),
    ModelPanel(
        key="qwen-audio",
        label="Qwen3-Omni",
        model_slug="Qwen__Qwen3-Omni-30B-A3B-Instruct",
        layer=25,
    ),
)


@dataclass(frozen=True)
class PanelData:
    panel: ModelPanel
    centers_sec: np.ndarray
    coords: np.ndarray


def parse_keyed_paths(values: list[str] | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "Activation paths must be written as MODEL_KEY=PATH, "
                "for example af-next=/path/to/activations.pt"
            )
        key, path = value.split("=", 1)
        paths[key.strip()] = Path(path).expanduser()
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot side-by-side 2D PCA projections of temporal ID vectors for "
            "AF-next decoder layer 16, MOSS-audio decoder layer 18, and "
            "qwen3-omni decoder layer 24."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--analysis-results-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_RESULTS_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="temporal_id_pca_afnext16_moss18_qwen24_one_column",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=("pdf", "png", "svg"),
    )
    parser.add_argument(
        "--activation-path",
        action="append",
        default=None,
        help=(
            "Optional MODEL_KEY=PATH override. MODEL_KEY is one of: "
            + ", ".join(panel.key for panel in MODEL_PANELS)
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--token", default="query_event")
    parser.add_argument(
        "--time-field",
        default="center",
        choices=("onset", "center", "offset"),
    )
    parser.add_argument("--bin-width-sec", type=float, default=2.5)
    parser.add_argument("--first-bin-center-sec", type=float, default=2.5)
    parser.add_argument("--min-bin-count", type=int, default=10)
    parser.add_argument(
        "--exclude-bin-centers-sec",
        type=float,
        nargs="*",
        default=[0.0, 30.0],
    )
    parser.add_argument("--max-time-sec", type=float, default=None)
    parser.add_argument("--fig-width", type=float, default=3.5)
    parser.add_argument("--fig-height", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def newest_activation_path(input_dir: Path, model_slug: str) -> Path | None:
    model_dir = input_dir / model_slug
    if not model_dir.exists():
        return None
    candidates = sorted(
        model_dir.glob("*all_decoder_text_activations.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def summary_path(
    analysis_results_dir: Path,
    panel: ModelPanel,
    token: str,
    time_field: str,
) -> Path | None:
    analysis_dir = analysis_results_dir / panel.model_slug
    pattern = f"*layer{panel.layer}_token-{token_slug(token)}_{time_field}_summary.json"
    candidates = sorted(
        analysis_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def remap_activation_path(path: Path) -> Path | None:
    marker = Path("experiments") / "experiment_41_temporal_IDs" / "outputs"
    parts = path.parts
    marker_parts = marker.parts
    for idx in range(0, len(parts) - len(marker_parts) + 1):
        if parts[idx : idx + len(marker_parts)] == marker_parts:
            relative_path = Path(*parts[idx + len(marker_parts) :])
            candidates = [
                EXPERIMENT_DIR / "outputs" / "activations" / relative_path,
                EXPERIMENT_DIR / "outputs" / relative_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
    return None


def activation_path_for(
    panel: ModelPanel,
    args: argparse.Namespace,
    overrides: dict[str, Path],
) -> Path:
    if panel.key in overrides:
        path = overrides[panel.key]
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Activation override for {panel.key} does not exist: {path}"
        )

    local_path = newest_activation_path(args.input_dir, panel.model_slug)
    if local_path is not None:
        return local_path

    path = summary_path(args.analysis_results_dir, panel, args.token, args.time_field)
    if path is not None:
        with path.open() as handle:
            summary = json.load(handle)
        recorded = Path(summary["activation_path"])
        if recorded.exists():
            return recorded
        remapped = remap_activation_path(recorded)
        if remapped is not None:
            return remapped

    raise FileNotFoundError(
        f"No activation bundle found for {panel.label}. Looked under "
        f"{args.input_dir / panel.model_slug} and the layer {panel.layer} summary "
        f"under {args.analysis_results_dir / panel.model_slug}. Pass "
        f"--activation-path {panel.key}=/path/to/bundle.pt if it lives elsewhere."
    )


def panel_data(
    panel: ModelPanel,
    activation_path: Path,
    args: argparse.Namespace,
) -> PanelData:
    bundle = load_bundle(activation_path)
    max_time_sec = args.max_time_sec
    if max_time_sec is None:
        path = summary_path(args.analysis_results_dir, panel, args.token, args.time_field)
        if path is not None:
            with path.open() as handle:
                max_time_sec = float(json.load(handle).get("max_time_sec") or 30.0)
        else:
            max_time_sec = 30.0

    raw_examples = extract_raw_examples(
        bundle=bundle,
        layer=panel.layer,
        token_arg=args.token,
        time_field=args.time_field,
        first_bin_center=args.first_bin_center_sec,
        bin_width=args.bin_width_sec,
        max_time_sec=max_time_sec,
    )
    centered_examples = center_by_train_class_mean(
        raw_examples,
        train_class_means(raw_examples),
    )
    excluded = set(float(value) for value in args.exclude_bin_centers_sec)
    train_table = temporal_ids_for_split(
        centered_examples,
        "train",
        args.min_bin_count,
        excluded,
    )
    plot_table = train_table
    if args.split != "train":
        plot_table = temporal_ids_for_split(
            centered_examples,
            args.split,
            args.min_bin_count,
            excluded,
        )

    pca, _ = fit_pca_2d(train_table.vectors)
    return PanelData(
        panel=panel,
        centers_sec=plot_table.centers_sec,
        coords=pca.transform(plot_table.vectors),
    )


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 5.8,
            "axes.labelsize": 5.5,
            "axes.titlesize": 6.0,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.0,
            "legend.fontsize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def padded_limits(values: np.ndarray, fraction: float = 0.18) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = high - low
    if span <= 0.0:
        span = max(abs(high), 1.0)
    return low - fraction * span, high + fraction * span


def annotate_time_points(
    ax: plt.Axes,
    centers_sec: np.ndarray,
    coords: np.ndarray,
) -> list[plt.Annotation]:
    x_values = coords[:, 0]
    y_values = coords[:, 1]
    x_low, x_high = padded_limits(x_values)
    y_low, y_high = padded_limits(y_values)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    x_span = x_high - x_low
    y_span = y_high - y_low
    annotations: list[plt.Annotation] = []
    for idx, (center, (x_coord, y_coord)) in enumerate(zip(centers_sec, coords)):
        x_frac = (x_coord - x_low) / x_span
        y_frac = (y_coord - y_low) / y_span
        x_offset = -3.0 if x_frac > 0.72 else 3.0
        y_offset = -6.0 if y_frac > 0.72 else 3.0
        if 0.28 <= x_frac <= 0.72:
            x_offset *= 1.0 if idx % 2 == 0 else -1.0
        if 0.28 <= y_frac <= 0.72:
            y_offset *= 1.0 if idx % 3 != 1 else -1.0
        horizontal_alignment = "right" if x_frac > 0.72 else "left"
        vertical_alignment = "top" if y_frac > 0.72 else "bottom"
        annotation = ax.annotate(
            f"{center:g}s",
            (x_coord, y_coord),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=4.0,
            ha=horizontal_alignment,
            va=vertical_alignment,
            clip_on=True,
        )
        annotations.append(annotation)
    return annotations


def repel_text_labels(
    fig: plt.Figure,
    annotations_by_ax: list[tuple[plt.Axes, list[plt.Annotation]]],
    *,
    iterations: int = 100,
    padding_px: float = 1.5,
    max_offset_points: float = 18.0,
) -> None:
    """Relax offset-point annotations until their drawn bounding boxes separate."""
    if not any(annotations for _, annotations in annotations_by_ax):
        return

    px_to_points = 72.0 / fig.dpi
    for _ in range(iterations):
        moved = False
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, annotations in annotations_by_ax:
            axes_box = ax.get_window_extent(renderer).expanded(0.98, 0.98)
            boxes = [
                annotation.get_window_extent(renderer).expanded(
                    1.0 + padding_px / max(annotation.get_window_extent(renderer).width, 1.0),
                    1.0 + padding_px / max(annotation.get_window_extent(renderer).height, 1.0),
                )
                for annotation in annotations
            ]

            for idx, annotation in enumerate(annotations):
                box = boxes[idx]
                dx_px = 0.0
                dy_px = 0.0
                if box.x0 < axes_box.x0:
                    dx_px += axes_box.x0 - box.x0
                if box.x1 > axes_box.x1:
                    dx_px -= box.x1 - axes_box.x1
                if box.y0 < axes_box.y0:
                    dy_px += axes_box.y0 - box.y0
                if box.y1 > axes_box.y1:
                    dy_px -= box.y1 - axes_box.y1
                if dx_px or dy_px:
                    x_offset, y_offset = annotation.get_position()
                    annotation.set_position(
                        (
                            np.clip(
                                x_offset + dx_px * px_to_points,
                                -max_offset_points,
                                max_offset_points,
                            ),
                            np.clip(
                                y_offset + dy_px * px_to_points,
                                -max_offset_points,
                                max_offset_points,
                            ),
                        )
                    )
                    moved = True

            for first in range(len(annotations)):
                for second in range(first + 1, len(annotations)):
                    first_box = boxes[first]
                    second_box = boxes[second]
                    overlap_x = min(first_box.x1, second_box.x1) - max(
                        first_box.x0, second_box.x0
                    )
                    overlap_y = min(first_box.y1, second_box.y1) - max(
                        first_box.y0, second_box.y0
                    )
                    if overlap_x <= 0.0 or overlap_y <= 0.0:
                        continue

                    first_center = np.array(
                        [
                            0.5 * (first_box.x0 + first_box.x1),
                            0.5 * (first_box.y0 + first_box.y1),
                        ]
                    )
                    second_center = np.array(
                        [
                            0.5 * (second_box.x0 + second_box.x1),
                            0.5 * (second_box.y0 + second_box.y1),
                        ]
                    )
                    direction = first_center - second_center
                    if np.allclose(direction, 0.0):
                        direction = np.array([1.0, -1.0])
                    direction = direction / np.linalg.norm(direction)
                    shift_px = 0.5 * min(overlap_x, overlap_y) + padding_px
                    for annotation, sign in (
                        (annotations[first], 1.0),
                        (annotations[second], -1.0),
                    ):
                        x_offset, y_offset = annotation.get_position()
                        annotation.set_position(
                            (
                                np.clip(
                                    x_offset + sign * direction[0] * shift_px * px_to_points,
                                    -max_offset_points,
                                    max_offset_points,
                                ),
                                np.clip(
                                    y_offset + sign * direction[1] * shift_px * px_to_points,
                                    -max_offset_points,
                                    max_offset_points,
                                ),
                            )
                        )
                    moved = True
        if not moved:
            break


def plot_panels(
    data: list[PanelData],
    output_dir: Path,
    output_stem: str,
    formats: list[str],
    dpi: int,
    args: argparse.Namespace,
) -> None:
    set_paper_style()
    fig, axes = plt.subplots(
        1,
        len(data),
        figsize=(args.fig_width, args.fig_height),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    cmap = "viridis"
    annotations_by_ax: list[tuple[plt.Axes, list[plt.Annotation]]] = []

    for ax, item in zip(axes, data):
        coords = item.coords
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color="#1f2937",
            linewidth=0.65,
            alpha=0.75,
            zorder=1,
        )
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=item.centers_sec,
            cmap=cmap,
            s=14,
            edgecolor="#111827",
            linewidth=0.25,
            zorder=2,
        )
        annotations_by_ax.append((ax, annotate_time_points(ax, item.centers_sec, coords)))
        ax.set_title(f"{item.panel.label} L{item.panel.decoder_layer}", pad=2.0)
        ax.tick_params(length=2.0, pad=1.0)
        ax.grid(True, linewidth=0.25, alpha=0.25)
        ax.set_box_aspect(1.0)

    fig.supxlabel("PC1", y=0.02, fontsize=5.5)
    fig.supylabel("PC2", x=0.005, fontsize=5.5)
    repel_text_labels(fig, annotations_by_ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(output_path, dpi=dpi)
        print(f"Wrote {output_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    overrides = parse_keyed_paths(args.activation_path)
    unknown = sorted(set(overrides) - {panel.key for panel in MODEL_PANELS})
    if unknown:
        raise ValueError(f"Unknown activation-path model keys: {unknown}")

    data = [
        panel_data(panel, activation_path_for(panel, args, overrides), args)
        for panel in MODEL_PANELS
    ]
    plot_panels(data, args.output_dir, args.output_stem, args.formats, args.dpi, args)


if __name__ == "__main__":
    main()
