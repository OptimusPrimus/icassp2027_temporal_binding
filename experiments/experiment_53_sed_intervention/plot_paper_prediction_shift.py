#!/usr/bin/env python3
"""Create a compact paper boxplot figure for temporal-ID endpoint steering."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs" / "aligned"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots" / "paper"
DEFAULT_STATS_DIR = EXPERIMENT_DIR / "outputs" / "paper"
REQUIRED_COLUMNS = {
    "predicted_onset_sec_baseline",
    "predicted_onset_sec_intervention",
    "length_sec",
}
ENDPOINTS = ("beginning", "end")
ENDPOINT_TO_DIRECTION = {
    "beginning": "backward",
    "end": "forward",
}
BASELINE_HALVES = (
    ("E", "early", "start-half"),
    ("L", "late", "half-end"),
)


@dataclass(frozen=True)
class ModelPanel:
    key: str
    label: str
    model_slug: str
    filename_prefix: str
    layer: int
    alpha: float

    @property
    def alpha_label(self) -> str:
        return f"{self.alpha:g}".replace("-", "neg").replace(".", "p")


MODEL_PANELS = (
    ModelPanel(
        key="af-next",
        label="AF-Next",
        model_slug="nvidia__audio-flamingo-next-hf",
        filename_prefix="af_next",
        layer=16,
        alpha=1.0,
    ),
    ModelPanel(
        key="moss-audio",
        label="MOSS-Audio",
        model_slug="OpenMOSS-Team__MOSS-Audio-8B-Instruct",
        filename_prefix="moss_audio",
        layer=18,
        alpha=0.5,
    ),
    ModelPanel(
        key="qwen3-omni",
        label="Qwen3-Omni",
        model_slug="Qwen__Qwen3-Omni-30B-A3B-Instruct",
        filename_prefix="qwen3_omni",
        layer=24,
        alpha=1.0,
    ),
)


def parse_keyed_paths(values: list[str] | None) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "Input paths must be written as MODEL_KEY:ENDPOINT=PATH, "
                "for example af-next:beginning=/path/to/aligned.csv"
            )
        key, path = value.split("=", 1)
        if ":" not in key:
            raise ValueError(
                "Input path keys must include an endpoint: MODEL_KEY:ENDPOINT=PATH"
            )
        model_key, endpoint = key.split(":", 1)
        paths[(model_key.strip(), endpoint.strip())] = Path(path).expanduser()
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed onset prediction-shift boxplots for temporal-ID steering "
            "to the beginning versus to the end using relative temporal-ID "
            "interventions. Each model panel has forward and backward groups, "
            "split by baseline predictions in the early or late half of the audio."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="relative_temporal_id_endpoint_steering_onset_shift",
    )
    parser.add_argument(
        "--stats-name",
        default=None,
        help="Optional CSV filename for per-box statistics.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("pdf", "png", "svg"),
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help=(
            "Optional MODEL_KEY:ENDPOINT=PATH override. MODEL_KEY is one of "
            + ", ".join(panel.key for panel in MODEL_PANELS)
            + "; ENDPOINT is beginning or end."
        ),
    )
    parser.add_argument("--shift-threshold-sec", type=float, default=1.0)
    parser.add_argument("--fig-width", type=float, default=3.5)
    parser.add_argument("--fig-height", type=float, default=1.45)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Show outlier markers in the boxplots.",
    )
    return parser.parse_args()


def default_aligned_csv(input_dir: Path, panel: ModelPanel, endpoint: str) -> Path:
    filename = (
        f"{panel.filename_prefix}_real_desed_onset_all_dur2p5-5p5s_n1000"
        f"_relative_temporal_id_{endpoint}_layer{panel.layer}"
        f"_alpha{panel.alpha_label}_bin2p5s_aligned.csv"
    )
    return input_dir / panel.model_slug / filename


def aligned_csv_for(
    panel: ModelPanel,
    endpoint: str,
    input_dir: Path,
    overrides: dict[tuple[str, str], Path],
) -> Path:
    return overrides.get((panel.key, endpoint)) or default_aligned_csv(
        input_dir,
        panel,
        endpoint,
    )


def load_valid_shifts(path: Path, direction: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    valid = df.copy()
    valid["baseline_prediction_sec"] = pd.to_numeric(
        valid["predicted_onset_sec_baseline"],
        errors="coerce",
    )
    valid["intervention_prediction_sec"] = pd.to_numeric(
        valid["predicted_onset_sec_intervention"],
        errors="coerce",
    )
    valid["audio_length_sec"] = pd.to_numeric(valid["length_sec"], errors="coerce")
    valid = valid.dropna(
        subset=[
            "baseline_prediction_sec",
            "intervention_prediction_sec",
            "audio_length_sec",
        ]
    )
    valid["delta_t_sec"] = (
        valid["intervention_prediction_sec"] - valid["baseline_prediction_sec"]
    )
    if direction == "backward":
        valid["intended_direction_shift_sec"] = -valid["delta_t_sec"]
    elif direction == "forward":
        valid["intended_direction_shift_sec"] = valid["delta_t_sec"]
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return valid


def split_shift_values(shifts: pd.DataFrame) -> dict[str, np.ndarray]:
    half_point = 0.5 * shifts["audio_length_sec"]
    early = shifts["baseline_prediction_sec"] < half_point
    late = shifts["baseline_prediction_sec"] >= half_point
    return {
        "E": shifts.loc[early, "intended_direction_shift_sec"].to_numpy(dtype=np.float64),
        "L": shifts.loc[late, "intended_direction_shift_sec"].to_numpy(dtype=np.float64),
    }


def box_statistics(values: np.ndarray) -> dict[str, float | int]:
    count = int(len(values))
    if count == 0:
        return {
            "count": 0,
            "mean_target_shift_sec": math.nan,
            "median_target_shift_sec": math.nan,
            "q1_target_shift_sec": math.nan,
            "q3_target_shift_sec": math.nan,
            "min_target_shift_sec": math.nan,
            "max_target_shift_sec": math.nan,
            "fraction_abs_target_shift_gt_threshold": math.nan,
        }
    return {
        "count": count,
        "mean_target_shift_sec": float(np.mean(values)),
        "median_target_shift_sec": float(np.median(values)),
        "q1_target_shift_sec": float(np.percentile(values, 25)),
        "q3_target_shift_sec": float(np.percentile(values, 75)),
        "min_target_shift_sec": float(np.min(values)),
        "max_target_shift_sec": float(np.max(values)),
    }


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 7.0,
            "axes.labelsize": 5.5,
            "axes.titlesize": 6.0,
            "xtick.labelsize": 4.7,
            "ytick.labelsize": 5.0,
            "legend.fontsize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def shared_ylim(panel_values: list[list[np.ndarray]]) -> tuple[float, float]:
    return -15.0, 40.0


def plot_endpoint_steering(
    panels: tuple[ModelPanel, ...],
    panel_values: list[list[np.ndarray]],
    output_dir: Path,
    output_stem: str,
    formats: list[str],
    dpi: int,
    fig_width: float,
    fig_height: float,
    show_fliers: bool,
) -> None:
    set_paper_style()
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(fig_width, fig_height),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    y_min, y_max = shared_ylim(panel_values)

    labels = ["E", "L", "E", "L"]
    colors = ["#f59e0b", "#f59e0b", "#60a5fa", "#60a5fa"]
    edge_colors = ["#b45309", "#b45309", "#1d4ed8", "#1d4ed8"]

    for ax, panel, data in zip(axes, panels, panel_values):
        box = ax.boxplot(
            data,
            positions=[1, 2, 4, 5],
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            showfliers=show_fliers,
            tick_labels=labels,
            medianprops={"color": "#111111", "linewidth": 0.75},
            whiskerprops={"color": "#334155", "linewidth": 0.6},
            capprops={"color": "#334155", "linewidth": 0.6},
            meanprops={
                "marker": "o",
                "markerfacecolor": "#ffffff",
                "markeredgecolor": "#111111",
                "markeredgewidth": 0.45,
                "markersize": 2.4,
            },
            flierprops={
                "marker": ".",
                "markerfacecolor": "#64748b",
                "markeredgecolor": "#64748b",
                "markersize": 1.6,
                "alpha": 0.35,
            },
        )
        for patch, face_color, edge_color in zip(
            box["boxes"],
            colors,
            edge_colors,
        ):
            patch.set_facecolor(face_color)
            patch.set_edgecolor(edge_color)
            patch.set_alpha(0.78)
            patch.set_linewidth(0.65)

        ax.axhline(0.0, color="#111111", linewidth=0.6, linestyle="--", zorder=0)
        ax.axvline(3.0, color="#94a3b8", linewidth=0.45, alpha=0.65, zorder=0)
        ax.set_title(f"{panel.label} L{panel.layer}", pad=2.0)
        ax.set_xlim(0.3, 5.7)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linewidth=0.25, alpha=0.25)
        ax.tick_params(length=1.8, pad=1.0)
        ax.text(
            1.5,
            y_min - 0.17 * (y_max - y_min),
            "Forward",
            ha="center",
            va="top",
        )
        ax.text(
            4.5,
            y_min - 0.17 * (y_max - y_min),
            "Backward",
            ha="center",
            va="top",
        )

    axes[0].set_ylabel("Shift toward target (s)", labelpad=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(output_path, dpi=dpi)
        print(f"Wrote {output_path}")
    plt.close(fig)


def write_stats(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No statistics rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    overrides = parse_keyed_paths(args.input_csv)
    valid_keys = {
        (panel.key, endpoint)
        for panel in MODEL_PANELS
        for endpoint in ENDPOINTS
    }
    unknown = sorted(set(overrides) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown input-csv keys: {unknown}")

    panel_values: list[list[np.ndarray]] = []
    stats_rows: list[dict[str, object]] = []
    for panel in MODEL_PANELS:
        panel_data: dict[str, dict[str, np.ndarray]] = {}
        for endpoint in ENDPOINTS:
            path = aligned_csv_for(panel, endpoint, args.input_dir, overrides)
            if not path.exists():
                raise FileNotFoundError(
                    f"No aligned CSV found for {panel.label} {endpoint}: {path}. "
                    f"Pass --input-csv {panel.key}:{endpoint}=/path/to/aligned.csv to override."
                )
            direction = ENDPOINT_TO_DIRECTION[endpoint]
            shifts = load_valid_shifts(path, direction)
            values_by_range = split_shift_values(shifts)
            panel_data[direction] = values_by_range
            for half_label, half_name, half_description in BASELINE_HALVES:
                values = values_by_range[half_label]
                stats = box_statistics(values)
                fraction_large = (
                    float(np.mean(np.abs(values) > args.shift_threshold_sec))
                    if len(values)
                    else math.nan
                )
                stats_rows.append({
                    "model_key": panel.key,
                    "model_label": panel.label,
                    "model_slug": panel.model_slug,
                    "intervention_target": endpoint,
                    "intervention_direction": direction,
                    "baseline_half_label": half_label,
                    "baseline_half_name": half_name,
                    "baseline_half_description": half_description,
                    "input_csv": str(path),
                    **stats,
                    "fraction_abs_target_shift_gt_threshold": fraction_large,
                    "shift_threshold_sec": float(args.shift_threshold_sec),
                })
            mean_shift = shifts["intended_direction_shift_sec"].mean()
            frac_large = (
                shifts["intended_direction_shift_sec"].abs() > args.shift_threshold_sec
            ).mean()
            print(f"{panel.label} {endpoint}: {path}")
            print(f"  valid predictions: {len(shifts)}")
            print(f"  mean target shift: {mean_shift:.3f}s")
            print(
                f"  fraction |target shift|>{args.shift_threshold_sec:g}s: "
                f"{frac_large:.3f}"
            )
        panel_values.append([
            panel_data["forward"]["E"],
            panel_data["forward"]["L"],
            panel_data["backward"]["E"],
            panel_data["backward"]["L"],
        ])

    stats_name = args.stats_name or f"{args.output_stem}_per_box_stats.csv"
    stats_path = DEFAULT_STATS_DIR / stats_name
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_stats(stats_path, stats_rows)
    print(f"Wrote {stats_path}")

    plot_endpoint_steering(
        panels=MODEL_PANELS,
        panel_values=panel_values,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        formats=args.formats,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        show_fliers=args.show_fliers,
    )


if __name__ == "__main__":
    main()
