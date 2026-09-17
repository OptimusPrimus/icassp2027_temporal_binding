import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "plots" / "belief_shift_by_layer"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots" / "paper"
SUMMARY_NAME = "belief_shift_by_layer_summary.csv"
TOKEN_TYPES = ("audio", "text", "events", "control", "part11")
TOKEN_LABELS = {
    "audio": "Audio",
    "text": "Text",
    "events": "Event names",
    "control": "Control",
    "part11": "Last token",
}
TOKEN_COLORS = {
    "audio": "#0072B2",
    "text": "#D55E00",
    "events": "#009E73",
    "control": "#6b7280",
    "part11": "#CC79A7",
}
BELIEF_SHIFT_RANGE = (-0.1, 1.1)

MODEL_LABELS = {
    "nvidia__audio-flamingo-3-hf": "AF3",
    "nvidia__audio-flamingo-next-hf": "AF-Next",
    "OpenMOSS-Team__MOSS-Audio-8B-Instruct": "MOSS-Audio",
    "Qwen__Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni",
}
MODEL_ORDER = (
    "nvidia__audio-flamingo-next-hf",
    "OpenMOSS-Team__MOSS-Audio-8B-Instruct",
    "Qwen__Qwen3-Omni-30B-A3B-Instruct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact paper line plot summarizing experiment 31 belief "
            "shifts for the paper model set."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=(
            "Directory containing model plot folders with "
            f"{SUMMARY_NAME}. Defaults to plots/belief_shift_by_layer."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the paper figure outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default="belief_shift_all_models_audio_text_events_control_part11_lines_wide",
        help="Output filename stem. Extensions are controlled by --formats.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=("pdf", "png", "svg"),
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--model-slugs",
        nargs="+",
        default=None,
        help="Optional ordered list of model folder names to include.",
    )
    parser.add_argument(
        "--wide-fig-width",
        type=float,
        default=7.16,
        help="Figure width in inches for the compact paper figure.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--ymin",
        type=float,
        default=BELIEF_SHIFT_RANGE[0],
        help="Lower y-axis limit.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=BELIEF_SHIFT_RANGE[1],
        help="Upper y-axis limit.",
    )
    parser.add_argument(
        "--tick-step",
        type=int,
        default=8,
        help="Layer tick spacing.",
    )
    parser.add_argument(
        "--error-column",
        default="belief_shift_ci95",
        choices=("belief_shift_ci95", "belief_shift_sem", "belief_shift_std"),
        help="Summary column to use for transparent error bands.",
    )
    parser.add_argument(
        "--error-alpha",
        type=float,
        default=0.22,
        help="Transparency for error regions.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Omitted by default to save vertical space.",
    )
    return parser.parse_args()


def discover_summary_paths(
    input_dir: Path,
    model_slugs: Sequence[str] | None,
) -> list[tuple[str, Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if model_slugs is not None:
        paths = [(slug, input_dir / slug / SUMMARY_NAME) for slug in model_slugs]
    else:
        found = {
            path.parent.name: path
            for path in input_dir.glob(f"*/{SUMMARY_NAME}")
            if path.parent.is_dir()
        }
        ordered = [slug for slug in MODEL_ORDER if slug in found]
        paths = [(slug, found[slug]) for slug in ordered]

    missing = [str(path) for _, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing summary CSVs:\n" + "\n".join(missing))
    if not paths:
        raise FileNotFoundError(f"No {SUMMARY_NAME} files found under {input_dir}")
    return paths


def model_label(model_slug: str) -> str:
    return MODEL_LABELS.get(model_slug, model_slug.replace("__", "/"))


def read_summary(model_slug: str, path: Path, error_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "intervention_token_type",
        "intervention_layer",
        "belief_shift_mean",
        error_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame[frame["intervention_token_type"].isin(TOKEN_TYPES)].copy()
    if frame.empty:
        raise ValueError(f"{path} has no rows for token types: {TOKEN_TYPES}")
    frame["model_slug"] = model_slug
    for column in ("intervention_layer", "belief_shift_mean", error_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[error_column] = frame[error_column].fillna(0.0)
    frame = frame.dropna(
        subset=[
            "intervention_layer",
            "belief_shift_mean",
            error_column,
            "intervention_token_type",
        ]
    )
    frame["intervention_layer"] = frame["intervention_layer"].astype(int)
    return frame


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def layer_ticks(layers: Sequence[int], tick_step: int) -> list[int]:
    ticks = [layer for layer in layers if layer % tick_step == 0]
    if layers[-1] not in ticks:
        ticks.append(layers[-1])
    return ticks


def guide_layers(layers: Sequence[int], tick_step: int) -> list[int]:
    return [layer for layer in layers if layer % tick_step == 0]


def model_layers(summary: pd.DataFrame, model_slug: str) -> list[int]:
    model_frame = summary[summary["model_slug"] == model_slug]
    return list(
        range(
            int(model_frame["intervention_layer"].min()),
            int(model_frame["intervention_layer"].max()) + 1,
        )
    )


def plot_lines_wide(
    summary: pd.DataFrame,
    model_slugs: Sequence[str],
    output_dir: Path,
    output_stem: str,
    formats: Sequence[str],
    fig_width: float,
    dpi: int,
    ymin: float,
    ymax: float,
    tick_step: int,
    error_column: str,
    error_alpha: float,
    title: str | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    layer_groups = [model_layers(summary, model_slug) for model_slug in model_slugs]
    width_ratios = [max(1.0, len(layers) ** 0.7) for layers in layer_groups]
    title_height = 0.18 if title else 0.0
    fig_height = 1.32 + title_height
    fig, axes = plt.subplots(
        1,
        len(model_slugs),
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=False,
    )
    if len(model_slugs) == 1:
        axes = np.array([axes])

    for ax, model_slug, layers in zip(axes, model_slugs, layer_groups, strict=True):
        model_frame = summary[summary["model_slug"] == model_slug]
        x = np.asarray(layers, dtype=float)
        for token_type in TOKEN_TYPES:
            token_frame = (
                model_frame[model_frame["intervention_token_type"] == token_type]
                .sort_values("intervention_layer")
                .set_index("intervention_layer")
                .reindex(layers)
            )
            y = token_frame["belief_shift_mean"].to_numpy(dtype=float)
            error = token_frame[error_column].to_numpy(dtype=float)
            color = TOKEN_COLORS[token_type]
            ax.plot(x, y, color=color, linewidth=1.15, label=TOKEN_LABELS[token_type])
            ax.fill_between(
                x,
                y - error,
                y + error,
                color=color,
                alpha=error_alpha,
                linewidth=0,
            )

        for yline in (0.0, 1.0):
            ax.axhline(yline, color="#6b7280", linewidth=0.45, linestyle=":")
        for layer in guide_layers(layers, tick_step):
            ax.axvline(layer, color="#d0d0d0", linewidth=0.35, zorder=0)
        ax.set_title(model_label(model_slug), pad=2.0)
        ax.set_xlim(min(layers), max(layers))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([0.0, 1.0])
        ticks = layer_ticks(layers, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.35, alpha=0.75)
        ax.tick_params(axis="both", length=2.0, pad=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    fig.supxlabel("Layer", y=0.18, fontsize=9, x=0.2)
    fig.supylabel("Belief shift", x=0.047, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(TOKEN_TYPES),
        frameon=False,
        handlelength=1.5,
        columnspacing=0.75,
        bbox_to_anchor=(0.54, 0.0),
    )
    if title:
        fig.suptitle(title, y=0.985)
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.38,
        top=0.74 if title else 0.78,
        wspace=0.08,
    )

    output_paths = []
    for suffix in formats:
        output_path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def main() -> None:
    args = parse_args()
    summary_paths = discover_summary_paths(Path(args.input_dir), args.model_slugs)
    frames = [
        read_summary(model_slug, path, args.error_column)
        for model_slug, path in summary_paths
    ]
    summary = pd.concat(frames, ignore_index=True)
    wide_output_paths = plot_lines_wide(
        summary=summary,
        model_slugs=[model_slug for model_slug, _ in summary_paths],
        output_dir=Path(args.output_dir),
        output_stem=args.output_stem,
        formats=args.formats,
        fig_width=args.wide_fig_width,
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=args.ymax,
        tick_step=args.tick_step,
        error_column=args.error_column,
        error_alpha=args.error_alpha,
        title=args.title,
    )
    for output_path in wide_output_paths:
        print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
