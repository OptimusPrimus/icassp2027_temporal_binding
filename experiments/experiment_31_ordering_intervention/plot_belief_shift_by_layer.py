import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedFormatter, FixedLocator
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots" / "belief_shift_by_layer"
DEFAULT_FILE_PATTERN = "ordering_intervention_syntheticsed_10s_*_layer*_*.csv"
BELIEF_SHIFT_YLIM = (-1.5, 1.5)
COMBINED_TOKEN_TYPES = ("audio", "text")
PROBABILITY_COLUMNS = (
    "before_probability",
    "after_probability",
    "before_after_probability_mass",
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_31_ordering_intervention.swap_tokens_intervention import (
    INTERVENTION_TOKEN_TYPES,
)


TOKEN_TYPE_COLORS = {
    "audio": "#1f77b4",
    "text": "#d62728",
    "events": "#2ca02c",
    "control": "#ff7f0e",
    "part1": "#9467bd",
    "part2": "#8c564b",
    "part3": "#e377c2",
    "part4": "#7f7f7f",
    "part5": "#bcbd22",
    "part6": "#17becf",
    "part7": "#aec7e8",
    "part8": "#ffbb78",
    "part9": "#98df8a",
    "part10": "#c5b0d5",
    "part11": "#c49c94",
    "part4+part6": "#ff7f0e",
}
FALLBACK_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
TOKEN_TYPE_LABELS = {
    "audio": "Audio",
    "text": "Text",
    "events": "Event-name spans",
    "control": "Control phrase: before or after",
    "part1": "Part 1: Does",
    "part2": "Part 2: query label",
    "part3": "Part 3: occur",
    "part4": "Part 4: before",
    "part5": "Part 5: or",
    "part6": "Part 6: after",
    "part7": "Part 7: reference label",
    "part8": "Part 8: ?",
    "part9": "Part 9: question-answer template",
    "part10": "Part 10: query label in answer",
    "part11": "Part 11: occurs in answer",
    "part4+part6": "before + after",
}
TOKEN_TYPE_YTICK_LABELS = {
    "audio": "Audio",
    "text": "Text",
    "events": "Events",
    "control": "Control",
    "part1": "P1: Does",
    "part2": "P2: query",
    "part3": "P3: occur",
    "part4": "P4: before",
    "part5": "P5: or",
    "part6": "P6: after",
    "part7": "P7: ref",
    "part8": "P8: ?",
    "part9": "P9: Q-A",
    "part10": "P10: query answer",
    "part11": "P11: occurs answer",
    "part4+part6": "before+after",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot experiment 31 ordering-intervention belief shifts by layer."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing model-specific experiment 31 CSV output folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base plot directory. Defaults to plots/detailed.",
    )
    parser.add_argument(
        "--file-pattern",
        default=DEFAULT_FILE_PATTERN,
        help="Glob pattern used to discover CSV files inside each model folder.",
    )
    parser.add_argument(
        "--model-slug",
        default=None,
        help="Optional model output folder name to plot, e.g. nvidia__audio-flamingo-3-hf.",
    )
    parser.add_argument(
        "--token-types",
        nargs="+",
        default=None,
        help="Optional intervention token types to include. Defaults to all found.",
    )
    parser.add_argument(
        "--summary-name",
        default="belief_shift_by_layer_summary.csv",
        help="Filename for the aggregated summary CSV.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def layer_from_filename(path: Path) -> int | None:
    match = re.search(r"_layer(\d+)_", path.name)
    if match is None:
        return None
    return int(match.group(1))


def read_intervention_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "belief_shift",
        "before_probability",
        "after_probability",
        "intervention_layer",
        "intervention_token_type",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if df["intervention_layer"].isna().all():
        filename_layer = layer_from_filename(path)
        if filename_layer is None:
            raise ValueError(f"Could not infer intervention layer for {path}")
        df["intervention_layer"] = filename_layer
    df["before_after_probability_mass"] = (
        pd.to_numeric(df["before_probability"], errors="coerce")
        + pd.to_numeric(df["after_probability"], errors="coerce")
    )

    selected_columns = [
        "belief_shift",
        "intervention_layer",
        "intervention_token_type",
        *[column for column in PROBABILITY_COLUMNS if column in df.columns],
    ]
    return df[selected_columns].copy()


def load_results(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No CSV files were provided")

    results = pd.concat(
        [read_intervention_csv(path) for path in paths],
        ignore_index=True,
    )
    results["belief_shift"] = pd.to_numeric(results["belief_shift"], errors="coerce")
    results["intervention_layer"] = pd.to_numeric(
        results["intervention_layer"],
        errors="coerce",
    )
    for column in PROBABILITY_COLUMNS:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    results = results.dropna(
        subset=["belief_shift", "intervention_layer", "intervention_token_type"]
    )
    results["intervention_layer"] = results["intervention_layer"].astype(int)
    return results


def discover_model_csv_groups(
    input_dir: Path,
    file_pattern: str,
    model_slug: str | None,
) -> list[tuple[str, list[Path]]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if model_slug is not None:
        model_dir = input_dir / model_slug
        paths = sorted(model_dir.glob(file_pattern))
        if not paths:
            raise FileNotFoundError(
                f"No CSV files matched {file_pattern!r} under {model_dir}"
            )
        return [(model_slug, paths)]

    groups = []
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        paths = sorted(model_dir.glob(file_pattern))
        if paths:
            groups.append((model_dir.name, paths))

    if groups:
        return groups

    direct_paths = sorted(input_dir.glob(file_pattern))
    if direct_paths:
        return [(input_dir.name, direct_paths)]

    raise FileNotFoundError(f"No CSV files matched {file_pattern!r} under {input_dir}")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = {"belief_shift": ["mean", "std", "count"]}
    for column in PROBABILITY_COLUMNS:
        if column in results.columns:
            aggregations[column] = ["mean"]

    summary = results.groupby(
        ["intervention_token_type", "intervention_layer"]
    ).agg(aggregations)
    summary.columns = [
        "_".join(column_parts).rstrip("_")
        for column_parts in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index().rename(
        columns={
            "belief_shift_mean": "belief_shift_mean",
            "belief_shift_std": "belief_shift_std",
            "belief_shift_count": "count",
        }
    )
    summary["belief_shift_sem"] = (
        summary["belief_shift_std"] / summary["count"].pow(0.5)
    )
    summary["belief_shift_ci95"] = 1.96 * summary["belief_shift_sem"]
    return summary.sort_values(["intervention_token_type", "intervention_layer"])


def summarize_across_layers(results: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "belief_shift": ["mean", "std", "count"],
    }
    for column in PROBABILITY_COLUMNS:
        if column in results.columns:
            aggregations[column] = ["mean"]

    summary = results.groupby("intervention_token_type").agg(aggregations)
    summary.columns = [
        "_".join(column_parts).rstrip("_")
        for column_parts in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index().rename(
        columns={
            "belief_shift_mean": "belief_shift_mean_across_layers",
            "belief_shift_std": "belief_shift_std_across_layers",
            "belief_shift_count": "count",
            "before_probability_mean": "before_probability_mean_across_layers",
            "after_probability_mean": "after_probability_mean_across_layers",
            "before_after_probability_mass_mean": (
                "before_after_probability_mass_mean_across_layers"
            ),
        }
    )
    summary["belief_shift_sem_across_layers"] = (
        summary["belief_shift_std_across_layers"] / summary["count"].pow(0.5)
    )
    summary["belief_shift_ci95_across_layers"] = (
        1.96 * summary["belief_shift_sem_across_layers"]
    )
    return summary.sort_values(
        "intervention_token_type",
        key=lambda column: column.map(token_type_sort_key),
    )


def token_type_sort_key(token_type: str) -> tuple:
    if token_type in INTERVENTION_TOKEN_TYPES:
        return (0, INTERVENTION_TOKEN_TYPES.index(token_type), token_type)
    part_order = {f"part{idx}": idx for idx in range(1, 12)}
    if "+" in token_type:
        parts = token_type.split("+")
        if all(part in part_order for part in parts):
            return (1, [part_order[part] for part in parts], token_type)
    return (2, token_type)


def token_types_to_plot(
    results: pd.DataFrame,
    requested_token_types: Sequence[str] | None,
) -> list[str]:
    if requested_token_types is not None:
        return list(requested_token_types)
    return sorted(
        results["intervention_token_type"].dropna().unique(),
        key=token_type_sort_key,
    )


def token_type_color(token_type: str) -> str:
    if token_type in TOKEN_TYPE_COLORS:
        return TOKEN_TYPE_COLORS[token_type]
    return FALLBACK_COLORS[abs(hash(token_type)) % len(FALLBACK_COLORS)]


def token_type_label(token_type: str) -> str:
    if token_type in TOKEN_TYPE_LABELS:
        return TOKEN_TYPE_LABELS[token_type]
    if "+" not in token_type:
        return token_type
    return " + ".join(TOKEN_TYPE_LABELS.get(part, part) for part in token_type.split("+"))


def token_type_ytick_label(token_type: str) -> str:
    return TOKEN_TYPE_YTICK_LABELS.get(token_type, token_type)


def output_name_token_type(token_type: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", token_type)


def boxplot_data(results: pd.DataFrame, token_type: str, layers: Sequence[int]) -> list:
    token_results = results[results["intervention_token_type"] == token_type]
    return [
        token_results.loc[
            token_results["intervention_layer"] == layer,
            "belief_shift",
        ].to_numpy()
        for layer in layers
    ]


def style_boxplot(boxplot: dict, color: str) -> None:
    for box in boxplot["boxes"]:
        box.set(facecolor=color, edgecolor="#303030", linewidth=0.9, alpha=0.72)
    for whisker in boxplot["whiskers"]:
        whisker.set(color="#303030", linewidth=0.9)
    for cap in boxplot["caps"]:
        cap.set(color="#303030", linewidth=0.9)
    for median in boxplot["medians"]:
        median.set(color="#111111", linewidth=1.2)
    for flier in boxplot["fliers"]:
        flier.set(
            marker="o",
            markerfacecolor=color,
            markeredgecolor="#303030",
            markersize=2.2,
            alpha=0.45,
        )


def set_layer_ticks(ax: plt.Axes, layers: Sequence[int]) -> None:
    ax.xaxis.set_major_locator(FixedLocator(layers))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(layer)) for layer in layers]))


def plot_token_type(
    results: pd.DataFrame,
    token_type: str,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    token_results = results[results["intervention_token_type"] == token_type]
    if token_results.empty:
        print(f"Skipping {token_type}: no rows found")
        return None

    layers = sorted(token_results["intervention_layer"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    boxplot = ax.boxplot(
        boxplot_data(results, token_type, layers),
        positions=layers,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
    )
    style_boxplot(boxplot, token_type_color(token_type))
    ax.axhline(0, color="#808080", linewidth=0.8, linestyle="--")
    ax.set_title(f"Belief shift by layer: {token_type_label(token_type)}")
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel("Belief shift")
    set_layer_ticks(ax, layers)
    ax.set_ylim(BELIEF_SHIFT_YLIM)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"belief_shift_by_layer_{output_name_token_type(token_type)}.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_combined_audio_text(
    results: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    token_types = [token for token in COMBINED_TOKEN_TYPES if token in set(results["intervention_token_type"])]
    if len(token_types) < 2:
        return None

    combined = results[results["intervention_token_type"].isin(token_types)]
    layers = sorted(combined["intervention_layer"].unique())
    offsets = {"audio": -0.18, "text": 0.18}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for token_type in token_types:
        token_results = combined[combined["intervention_token_type"] == token_type]
        token_layers = sorted(token_results["intervention_layer"].unique())
        positions = [layer + offsets[token_type] for layer in token_layers]
        boxplot = ax.boxplot(
            boxplot_data(combined, token_type, token_layers),
            positions=positions,
            widths=0.32,
            patch_artist=True,
            showfliers=True,
        )
        style_boxplot(boxplot, token_type_color(token_type))
        boxplot["boxes"][0].set_label(token_type_label(token_type))

    ax.axhline(0, color="#808080", linewidth=0.8, linestyle="--")
    ax.set_title("Belief shift by layer: audio and text token interventions")
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel("Belief shift")
    set_layer_ticks(ax, layers)
    ax.set_xlim(min(layers) - 0.6, max(layers) + 0.6)
    ax.set_ylim(BELIEF_SHIFT_YLIM)
    ax.legend(title="Token type")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "belief_shift_by_layer_audio_text.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_heatmap(
    summary: pd.DataFrame,
    token_types: Sequence[str],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    heatmap_summary = summary[
        summary["intervention_token_type"].isin(token_types)
    ].copy()
    if heatmap_summary.empty:
        print("Skipping heatmap: no rows found")
        return None

    ordered_token_types = sorted(
        heatmap_summary["intervention_token_type"].dropna().unique(),
        key=token_type_sort_key,
    )
    layers = sorted(heatmap_summary["intervention_layer"].dropna().unique())
    heatmap_data = (
        heatmap_summary.pivot(
            index="intervention_token_type",
            columns="intervention_layer",
            values="belief_shift_mean",
        )
        .reindex(index=ordered_token_types, columns=layers)
    )

    fig_height = max(5.0, 0.34 * len(ordered_token_types) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#eeeeee")
    image = ax.imshow(
        heatmap_data.to_numpy(),
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(
            vmin=BELIEF_SHIFT_YLIM[0],
            vcenter=0,
            vmax=BELIEF_SHIFT_YLIM[1],
        ),
        interpolation="nearest",
    )

    ax.set_title("Mean belief shift by layer and intervention token group")
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel("Intervention token group")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(int(layer)) for layer in layers], rotation=90)
    ax.set_yticks(range(len(ordered_token_types)))
    ax.set_yticklabels([token_type_ytick_label(token) for token in ordered_token_types])
    ax.set_xticks([idx - 0.5 for idx in range(1, len(layers))], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, len(ordered_token_types))], minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Mean belief shift")
    fig.tight_layout()

    output_path = output_dir / "belief_shift_by_layer_heatmap.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_model_group(
    model_slug: str,
    paths: Sequence[Path],
    output_root: Path,
    token_types_arg: Sequence[str] | None,
    summary_name: str,
    dpi: int,
) -> None:
    model_output_dir = output_root / model_slug
    layerwise_dir = model_output_dir / "layerwise"
    layerwise_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {model_slug}: {len(paths)} CSV files")
    results = load_results(paths)
    summary = summarize(results)
    across_layers_summary = summarize_across_layers(results)
    token_types = token_types_to_plot(results, token_types_arg)

    summary_path = model_output_dir / summary_name
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")

    across_layers_summary_path = (
        model_output_dir / f"{Path(summary_name).stem}_across_layers.csv"
    )
    across_layers_summary.to_csv(across_layers_summary_path, index=False)
    print(f"Wrote across-layer summary: {across_layers_summary_path}")

    for token_type in token_types:
        output_path = plot_token_type(results, token_type, layerwise_dir, dpi)
        if output_path is not None:
            print(f"Wrote layerwise plot: {output_path}")

    output_path = plot_combined_audio_text(results, layerwise_dir, dpi)
    if output_path is not None:
        print(f"Wrote layerwise plot: {output_path}")

    output_path = plot_heatmap(summary, token_types, model_output_dir, dpi)
    if output_path is not None:
        print(f"Wrote heatmap: {output_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    groups = discover_model_csv_groups(input_dir, args.file_pattern, args.model_slug)
    for model_slug, paths in groups:
        plot_model_group(
            model_slug=model_slug,
            paths=paths,
            output_root=output_dir,
            token_types_arg=args.token_types,
            summary_name=args.summary_name,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
