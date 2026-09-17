#!/usr/bin/env python3
"""Create a compact paper figure for relative temporal-ID intervention results."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
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
        alpha=0.5,
    ),
    ModelPanel(
        key="moss-audio",
        label="MOSS-Audio",
        model_slug="OpenMOSS-Team__MOSS-Audio-8B-Instruct",
        filename_prefix="moss_audio",
        layer=18,
        alpha=0.75,
    ),
    ModelPanel(
        key="qwen3-omni",
        label="Qwen3-Omni",
        model_slug="Qwen__Qwen3-Omni-30B-A3B-Instruct",
        filename_prefix="qwen3_omni",
        layer=24,
        alpha=1.25,
    ),
)

REQUIRED_COLUMNS = {
    "correct_class_baseline",
    "before_probability_mass_change",
    "after_probability_mass_change",
    "predicted_class_baseline",
    "predicted_class_intervention",
}


@dataclass(frozen=True)
class InterventionSpec:
    key: str
    direction: str
    intended_class: str
    value_column: str


INTERVENTIONS = (
    InterventionSpec(
        key="query_to_end",
        direction="forward",
        intended_class="after",
        value_column="after_probability_mass_change",
    ),
    InterventionSpec(
        key="query_to_beginning",
        direction="backward",
        intended_class="before",
        value_column="before_probability_mass_change",
    ),
)


def parse_keyed_paths(values: list[str] | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "Input paths must be written as MODEL_KEY=PATH, "
                "for example af-next=/path/to/aligned.csv"
            )
        key, path = value.split("=", 1)
        paths[key.strip()] = Path(path).expanduser()
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot side-by-side boxplots of pairwise-relative temporal-ID steering effects "
            "for AF-next, MOSS-audio, and Qwen3-omni."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="intervention_probability_shift_boxplots_three_models",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=("pdf", "png", "svg"),
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        default=None,
        help=(
            "Optional MODEL_KEY:INTERVENTION=PATH override, where INTERVENTION is "
            "query_to_end or query_to_beginning. "
            "For backwards compatibility, MODEL_KEY=PATH overrides query_to_end. "
            "MODEL_KEY is one of: "
            + ", ".join(panel.key for panel in MODEL_PANELS)
        ),
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional path for the flip-direction significance report CSV.",
    )
    parser.add_argument("--fig-width", type=float, default=3.5)
    parser.add_argument("--fig-height", type=float, default=1.45)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Show outlier markers in the boxplots.",
    )
    return parser.parse_args()


def default_aligned_csv(input_dir: Path, panel: ModelPanel, intervention_key: str) -> Path:
    filename = (
        f"{panel.filename_prefix}_real_desed_prediction_all_max5.5s_n1000"
        f"_single_event_relative_temporal_move_layer{panel.layer}_alpha{panel.alpha_label}"
        f"_{intervention_key}_aligned.csv"
    )
    return input_dir / panel.model_slug / filename


def aligned_csv_for(
    panel: ModelPanel,
    args: argparse.Namespace,
    overrides: dict[str, Path],
    intervention: InterventionSpec,
) -> Path:
    path = overrides.get(f"{panel.key}:{intervention.key}")
    if path is None and intervention.key == "query_to_end":
        path = overrides.get(panel.key)
    path = path or default_aligned_csv(args.input_dir, panel, intervention.key)
    if not path.exists():
        raise FileNotFoundError(
            f"No aligned CSV found for {panel.label}: {path}. "
            f"Pass --input-csv {panel.key}:{intervention.key}=/path/to/aligned.csv to override."
        )
    return path


def read_aligned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def probability_shift_by_baseline_class(df: pd.DataFrame, value_column: str) -> list[np.ndarray]:
    actual_before = df["correct_class_baseline"] == "before"
    actual_after = df["correct_class_baseline"] == "after"
    return [
        df.loc[actual_before, value_column].dropna().to_numpy(),
        df.loc[actual_after, value_column].dropna().to_numpy(),
    ]


def load_panel_data(paths: dict[str, Path]) -> list[np.ndarray]:
    values_by_direction = {
        "forward": [[], []],
        "backward": [[], []],
    }
    for intervention in INTERVENTIONS:
        df = read_aligned(paths[intervention.key])
        split_values = probability_shift_by_baseline_class(df, intervention.value_column)
        direction_values = values_by_direction[intervention.direction]
        for index, values in enumerate(split_values):
            direction_values[index].append(values)
    return [
        np.concatenate(values) if values else np.array([])
        for direction in ("forward", "backward")
        for values in values_by_direction[direction]
    ]


def exact_binomial_greater_pvalue(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    total = 0.0
    for value in range(k, n + 1):
        log_prob = math.lgamma(n + 1) - math.lgamma(value + 1) - math.lgamma(n - value + 1)
        log_prob += value * math.log(p) + (n - value) * math.log1p(-p)
        total += math.exp(log_prob)
    return min(1.0, total)


def flip_stats_for_frame(
    panel: ModelPanel,
    intervention: InterventionSpec,
    path: Path,
) -> dict[str, object]:
    df = read_aligned(path)
    baseline = df["predicted_class_baseline"].astype(str)
    intervened = df["predicted_class_intervention"].astype(str)
    intended = intervention.intended_class
    opposite = "before" if intended == "after" else "after"
    intended_baseline = baseline == intended
    opposite_baseline = baseline == opposite
    intended_stayed = intended_baseline & (intervened == intended)
    intended_changed_away = intended_baseline & (intervened == opposite)
    opposite_flipped_to_intended = opposite_baseline & (intervened == intended)
    opposite_stayed = opposite_baseline & (intervened == opposite)
    opposite_baseline_count = int(opposite_baseline.sum())
    opposite_flipped_count = int(opposite_flipped_to_intended.sum())
    p_value = exact_binomial_greater_pvalue(opposite_flipped_count, opposite_baseline_count)
    return {
        "model_key": panel.key,
        "model_label": panel.label,
        "model_slug": panel.model_slug,
        "layer": panel.layer,
        "alpha": panel.alpha,
        "direction": intervention.direction,
        "intervention": intervention.key,
        "intended_class": intended,
        "path": str(path),
        "n_examples": int(len(df)),
        "intended_baseline_count": int(intended_baseline.sum()),
        "intended_stayed_count": int(intended_stayed.sum()),
        "intended_changed_away_count": int(intended_changed_away.sum()),
        "opposite_baseline_count": opposite_baseline_count,
        "opposite_flipped_to_intended_count": opposite_flipped_count,
        "opposite_stayed_count": int(opposite_stayed.sum()),
        "opposite_flipped_to_intended_rate": (
            opposite_flipped_count / opposite_baseline_count
            if opposite_baseline_count
            else float("nan")
        ),
        "binomial_null_p": 0.5,
        "binomial_one_sided_greater_p": p_value,
    }


def aggregate_flip_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate_rows = []
    for model_key in sorted({str(row["model_key"]) for row in rows}):
        model_rows = [row for row in rows if row["model_key"] == model_key]
        for direction in ("forward", "backward", "all"):
            selected = (
                model_rows
                if direction == "all"
                else [row for row in model_rows if row["direction"] == direction]
            )
            if not selected:
                continue
            opposite_baseline_count = sum(int(row["opposite_baseline_count"]) for row in selected)
            opposite_flipped_count = sum(
                int(row["opposite_flipped_to_intended_count"]) for row in selected
            )
            aggregate_rows.append({
                "model_key": selected[0]["model_key"],
                "model_label": selected[0]["model_label"],
                "model_slug": selected[0]["model_slug"],
                "layer": selected[0]["layer"],
                "alpha": selected[0]["alpha"],
                "direction": direction,
                "intervention": "__aggregate__",
                "intended_class": "mixed" if direction == "all" else selected[0]["intended_class"],
                "path": "",
                "n_examples": sum(int(row["n_examples"]) for row in selected),
                "intended_baseline_count": sum(
                    int(row["intended_baseline_count"]) for row in selected
                ),
                "intended_stayed_count": sum(
                    int(row["intended_stayed_count"]) for row in selected
                ),
                "intended_changed_away_count": sum(
                    int(row["intended_changed_away_count"]) for row in selected
                ),
                "opposite_baseline_count": opposite_baseline_count,
                "opposite_flipped_to_intended_count": opposite_flipped_count,
                "opposite_stayed_count": sum(
                    int(row["opposite_stayed_count"]) for row in selected
                ),
                "opposite_flipped_to_intended_rate": (
                    opposite_flipped_count / opposite_baseline_count
                    if opposite_baseline_count
                    else float("nan")
                ),
                "binomial_null_p": 0.5,
                "binomial_one_sided_greater_p": exact_binomial_greater_pvalue(
                    opposite_flipped_count,
                    opposite_baseline_count,
                ),
            })
    return aggregate_rows


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
            "xtick.labelsize": 4.7,
            "ytick.labelsize": 5.0,
            "legend.fontsize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def probability_shift_ylim(panel_values: list[list[np.ndarray]]) -> tuple[float, float]:
    values = np.concatenate(
        [array for panel_data in panel_values for array in panel_data if len(array) > 0]
    )
    upper = float(np.nanmax(values))
    upper = max(0.05, upper * 1.08)
    return -0.1, upper


def plot_panels(
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
    y_min, y_max = probability_shift_ylim(panel_values)

    labels = ["B", "A", "B", "A"]
    colors = ["#60a5fa", "#60a5fa", "#f59e0b", "#f59e0b"]
    edge_colors = ["#1d4ed8", "#1d4ed8", "#b45309", "#b45309"]

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

    axes[0].set_ylabel("Shift toward target (p)", labelpad=1.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(output_path, dpi=dpi)
        print(f"Wrote {output_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    overrides = parse_keyed_paths(args.input_csv)
    valid_override_keys = {panel.key for panel in MODEL_PANELS} | {
        f"{panel.key}:{intervention.key}"
        for panel in MODEL_PANELS
        for intervention in INTERVENTIONS
    }
    unknown = sorted(set(overrides) - valid_override_keys)
    if unknown:
        raise ValueError(f"Unknown input-csv model keys: {unknown}")

    csv_paths = [
        {
            intervention.key: aligned_csv_for(panel, args, overrides, intervention)
            for intervention in INTERVENTIONS
        }
        for panel in MODEL_PANELS
    ]
    panel_values = [
        load_panel_data(panel_paths)
        for panel_paths in csv_paths
    ]
    stats_rows = []
    for panel, panel_paths in zip(MODEL_PANELS, csv_paths):
        for intervention in INTERVENTIONS:
            stats_rows.append(
                flip_stats_for_frame(
                    panel=panel,
                    intervention=intervention,
                    path=panel_paths[intervention.key],
                )
            )
    stats_rows.extend(aggregate_flip_stats(stats_rows))
    stats_path = args.stats_csv or DEFAULT_STATS_DIR / f"{args.output_stem}_flip_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    for panel, panel_paths, data in zip(
        MODEL_PANELS,
        csv_paths,
        panel_values,
    ):
        counts = ", ".join(str(len(values)) for values in data)
        print(f"{panel.label}:")
        for intervention in INTERVENTIONS:
            print(f"  {intervention.key}: {panel_paths[intervention.key]}")
        print(f"  examples per box: {counts}")
        aggregate_rows = [
            row
            for row in stats_rows
            if row["model_key"] == panel.key and row["intervention"] == "__aggregate__"
        ]
        for row in aggregate_rows:
            intended_baseline = int(row["intended_baseline_count"])
            intended_stayed = int(row["intended_stayed_count"])
            opposite_baseline = int(row["opposite_baseline_count"])
            opposite_flipped = int(row["opposite_flipped_to_intended_count"])
            rate = row["opposite_flipped_to_intended_rate"]
            p_value = row["binomial_one_sided_greater_p"]
            if row["direction"] == "forward":
                print(
                    "  forward: "
                    f"after->after {intended_stayed}/{intended_baseline}; "
                    f"before->after {opposite_flipped}/{opposite_baseline} "
                    f"({rate:.1%}; one-sided binomial p={p_value:.3g})"
                )
            elif row["direction"] == "backward":
                print(
                    "  backward: "
                    f"before->before {intended_stayed}/{intended_baseline}; "
                    f"after->before {opposite_flipped}/{opposite_baseline} "
                    f"({rate:.1%}; one-sided binomial p={p_value:.3g})"
                )
            else:
                print(
                    "  all query interventions: "
                    f"intended stays {intended_stayed}/{intended_baseline}; "
                    f"opposite->intended {opposite_flipped}/{opposite_baseline} "
                    f"({rate:.1%}; one-sided binomial p={p_value:.3g})"
                )
    print(f"Wrote {stats_path}")

    plot_panels(
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
