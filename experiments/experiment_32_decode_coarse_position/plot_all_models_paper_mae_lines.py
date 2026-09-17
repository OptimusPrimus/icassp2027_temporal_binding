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
DEFAULT_CLASS_CV_INPUT_DIR = EXPERIMENT_DIR / "outputs" / "class_cv_split"
DEFAULT_REGULAR_INPUT_DIR = EXPERIMENT_DIR / "outputs" / "regular_split"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots" / "paper"
METRICS_NAME = "event_midpoint_metrics_by_layer.csv"
PREDICTIONS_NAME = "event_midpoint_predictions.csv"
CLASS_CV_METRICS_NAME = "event_midpoint_class_cv_metrics_by_layer.csv"
CLASS_CV_PREDICTIONS_NAME = "event_midpoint_class_cv_predictions.csv"
CONDITIONS = (
    "target_event_word",
    "alternative_event_control",
    "first_prompt_token_control",
)
CONDITION_LABELS = {
    "target_event_word": r"$\mathbf{h}_{L,q}(x) \rightarrow t_q$",
    "alternative_event_control": r"$\mathbf{h}_{L,q}(x) \rightarrow t_o$",
    "first_prompt_token_control": r"$\mathbf{h}_{L,1}(x) \rightarrow t_q$",
}
CONDITION_COLORS = {
    "target_event_word": "#0072B2",
    "alternative_event_control": "#D55E00",
    "first_prompt_token_control": "#009E73",
}
MODEL_LABELS = {
    "nvidia__audio-flamingo-next-hf": "AF-Next",
    "OpenMOSS-Team__MOSS-Audio-8B-Instruct": "MOSS-Audio",
    "Qwen__Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni",
}
MODEL_ORDER = (
    "nvidia__audio-flamingo-next-hf",
    "OpenMOSS-Team__MOSS-Audio-8B-Instruct",
    "Qwen__Qwen3-Omni-30B-A3B-Instruct",
)

DEFAULT_Y_TICKS = [3.0, 6.0]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact paper line plot summarizing experiment 32 event "
            "midpoint MAE for the three embedding-to-time estimators."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_CLASS_CV_INPUT_DIR),
        help=(
            "Directory containing model folders with class-CV metrics CSVs. Defaults to "
            "experiments/experiment_32_decode_coarse_position/outputs/class_cv_split."
        ),
    )
    parser.add_argument(
        "--metrics-name",
        default=CLASS_CV_METRICS_NAME,
        help="Metrics CSV filename to read from each model folder.",
    )
    parser.add_argument(
        "--predictions-name",
        default=CLASS_CV_PREDICTIONS_NAME,
        help="Predictions CSV filename used to compute error bands.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the paper figure outputs.",
    )
    parser.add_argument(
        "--wide-output-stem",
        default="event_midpoint_class_cv_mae_all_models_three_estimators_wide",
        help=(
            "Class-CV output filename stem. Extensions are controlled by --formats."
        ),
    )
    parser.add_argument(
        "--all-classes-input-dir",
        default=str(DEFAULT_REGULAR_INPUT_DIR),
        help=(
            "Directory containing model folders with regular-split metrics CSVs. Defaults to "
            "experiments/experiment_32_decode_coarse_position/outputs/regular_split."
        ),
    )
    parser.add_argument(
        "--all-classes-metrics-name",
        default=METRICS_NAME,
        help=(
            "Non-class-CV metrics CSV filename to read from each model folder. "
            "Set to an empty string to skip this extra plot."
        ),
    )
    parser.add_argument(
        "--all-classes-predictions-name",
        default=PREDICTIONS_NAME,
        help="Non-class-CV predictions CSV filename used to compute error bands.",
    )
    parser.add_argument(
        "--all-classes-output-stem",
        default="event_midpoint_all_classes_mae_all_models_three_estimators_wide",
        help=(
            "Output filename stem for the non-class-CV plot using all ESC-50 "
            "classes. Extensions are controlled by --formats."
        ),
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
        "--regressor",
        default="midpoint_nn",
        help="Regressor value to select from the metrics CSV.",
    )
    parser.add_argument(
        "--wide-fig-width",
        type=float,
        default=7.16,
        help="Figure width in inches.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ymin", type=float, default=2.0)
    parser.add_argument("--ymax", type=float, default=8.0)
    parser.add_argument(
        "--tick-step",
        type=int,
        default=8,
        help="Layer-slot tick and vertical-guide spacing.",
    )
    parser.add_argument(
        "--error-stat",
        default="ci95",
        choices=("ci95", "sem", "std", "none"),
        help="Error bars computed from per-example absolute errors.",
    )
    parser.add_argument(
        "--error-alpha",
        type=float,
        default=0.18,
        help="Transparency for MAE error regions.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Omitted by default to save vertical space.",
    )
    return parser.parse_args()


def discover_metrics_paths(
    input_dir: Path,
    model_slugs: Sequence[str] | None,
    metrics_name: str,
    predictions_name: str,
) -> list[tuple[str, Path, Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if model_slugs is not None:
        paths = [
            (slug, input_dir / slug / metrics_name, input_dir / slug / predictions_name)
            for slug in model_slugs
        ]
    else:
        found = {
            path.parent.name: path
            for path in input_dir.glob(f"*/{metrics_name}")
            if path.parent.is_dir()
        }
        paths = [
            (slug, found[slug], found[slug].parent / predictions_name)
            for slug in MODEL_ORDER
            if slug in found
        ]

    missing = [
        str(path)
        for _, metrics_path, predictions_path in paths
        for path in (metrics_path, predictions_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing metrics CSVs:\n" + "\n".join(missing))
    if not paths:
        raise FileNotFoundError(f"No {metrics_name} files found under {input_dir}")
    return paths


def model_label(model_slug: str) -> str:
    return MODEL_LABELS.get(model_slug, model_slug.replace("__", "/"))


def read_metrics(model_slug: str, path: Path, regressor: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"condition", "layer_slot", "regressor", "mae"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame[
        (frame["condition"].isin(CONDITIONS)) & (frame["regressor"] == regressor)
    ].copy()
    if frame.empty:
        raise ValueError(
            f"{path} has no rows for regressor={regressor!r} and conditions={CONDITIONS}"
        )
    frame["model_slug"] = model_slug
    frame["layer_slot"] = pd.to_numeric(frame["layer_slot"], errors="coerce")
    frame["mae"] = pd.to_numeric(frame["mae"], errors="coerce")
    frame = frame.dropna(subset=["condition", "layer_slot", "mae"])
    frame["layer_slot"] = frame["layer_slot"].astype(int)
    frame = frame[frame["layer_slot"] > 0].copy()
    frame["layer"] = frame["layer_slot"] - 1
    return frame


def read_error_bars(model_slug: str, path: Path, error_stat: str) -> pd.DataFrame:
    if error_stat == "none":
        return pd.DataFrame(
            columns=["model_slug", "condition", "layer_slot", "mae_error"]
        )

    frame = pd.read_csv(path)
    required = {"condition", "layer_slot", "absolute_error_sec"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    frame = frame[frame["condition"].isin(CONDITIONS)].copy()
    frame["layer_slot"] = pd.to_numeric(frame["layer_slot"], errors="coerce")
    frame["absolute_error_sec"] = pd.to_numeric(
        frame["absolute_error_sec"], errors="coerce"
    )
    frame = frame.dropna(subset=["condition", "layer_slot", "absolute_error_sec"])
    grouped = (
        frame.groupby(["condition", "layer_slot"])["absolute_error_sec"]
        .agg(["std", "count"])
        .reset_index()
    )
    if error_stat == "std":
        grouped["mae_error"] = grouped["std"]
    else:
        sem = grouped["std"] / grouped["count"].pow(0.5)
        grouped["mae_error"] = 1.96 * sem if error_stat == "ci95" else sem
    grouped["model_slug"] = model_slug
    grouped["layer_slot"] = grouped["layer_slot"].astype(int)
    grouped = grouped[grouped["layer_slot"] > 0].copy()
    grouped["layer"] = grouped["layer_slot"] - 1
    return grouped[["model_slug", "condition", "layer_slot", "layer", "mae_error"]]


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


def model_layer_slots(frame: pd.DataFrame, model_slug: str) -> list[int]:
    model_frame = frame[frame["model_slug"] == model_slug]
    return list(
        range(int(model_frame["layer"].min()), int(model_frame["layer"].max()) + 1)
    )


def guide_slots(slots: Sequence[int], tick_step: int) -> list[int]:
    return [slot for slot in slots if slot % tick_step == 0]


def tick_slots(slots: Sequence[int], tick_step: int) -> list[int]:
    ticks = guide_slots(slots, tick_step)
    if slots[-1] not in ticks:
        ticks.append(slots[-1])
    return ticks


def plot_lines_wide(
    metrics: pd.DataFrame,
    model_slugs: Sequence[str],
    output_dir: Path,
    output_stem: str,
    formats: Sequence[str],
    fig_width: float,
    dpi: int,
    ymin: float,
    ymax: float,
    tick_step: int,
    error_stat: str,
    error_alpha: float,
    title: str | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    slot_groups = [model_layer_slots(metrics, model_slug) for model_slug in model_slugs]
    width_ratios = [max(1.0, len(slots) ** 0.7) for slots in slot_groups]
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

    for ax, model_slug, slots in zip(axes, model_slugs, slot_groups, strict=True):
        model_frame = metrics[metrics["model_slug"] == model_slug]
        for condition in CONDITIONS:
            condition_frame = (
                model_frame[model_frame["condition"] == condition]
                .sort_values("layer")
                .set_index("layer")
                .reindex(slots)
            )
            y = condition_frame["mae"].to_numpy(dtype=float)
            color = CONDITION_COLORS[condition]
            ax.plot(slots, y, color=color, linewidth=1.15, label=CONDITION_LABELS[condition])
            if error_stat != "none":
                error = condition_frame["mae_error"].to_numpy(dtype=float)
                ax.fill_between(
                    slots,
                    y - error,
                    y + error,
                    color=color,
                    alpha=error_alpha,
                    linewidth=0,
                )

        for slot in guide_slots(slots, tick_step):
            ax.axvline(slot, color="#d0d0d0", linewidth=0.35, zorder=0)
        ax.set_title(model_label(model_slug), pad=2.0)
        ax.set_xlim(min(slots), max(slots))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(DEFAULT_Y_TICKS)
        ticks = tick_slots(slots, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(tick) for tick in ticks])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.35, alpha=0.75)
        ax.tick_params(axis="both", length=2.0, pad=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    fig.supxlabel("Layer", y=0.18, fontsize=9, x=0.2)
    fig.supylabel("MAE (s)", x=0.047, y=0.62, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
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


def load_plot_metrics(
    input_dir: Path,
    model_slugs: Sequence[str] | None,
    metrics_name: str,
    predictions_name: str,
    regressor: str,
    error_stat: str,
) -> tuple[pd.DataFrame, list[str]]:
    metrics_paths = discover_metrics_paths(
        input_dir,
        model_slugs,
        metrics_name,
        predictions_name,
    )
    frames = [
        read_metrics(model_slug, path, regressor)
        for model_slug, path, _ in metrics_paths
    ]
    metrics = pd.concat(frames, ignore_index=True)
    error_frames = [
        read_error_bars(model_slug, path, error_stat)
        for model_slug, _, path in metrics_paths
    ]
    errors = pd.concat(error_frames, ignore_index=True)
    if not errors.empty:
        metrics = metrics.merge(
            errors,
            on=["model_slug", "condition", "layer_slot", "layer"],
            how="left",
        )
    return metrics, [model_slug for model_slug, _, _ in metrics_paths]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    metrics, model_slugs = load_plot_metrics(
        input_dir,
        args.model_slugs,
        args.metrics_name,
        args.predictions_name,
        args.regressor,
        args.error_stat,
    )
    wide_output_paths = plot_lines_wide(
        metrics=metrics,
        model_slugs=model_slugs,
        output_dir=Path(args.output_dir),
        output_stem=args.wide_output_stem,
        formats=args.formats,
        fig_width=args.wide_fig_width,
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=args.ymax,
        tick_step=args.tick_step,
        error_stat=args.error_stat,
        error_alpha=args.error_alpha,
        title=args.title,
    )
    for output_path in wide_output_paths:
        print(f"Wrote class-CV plot: {output_path}")

    if args.all_classes_metrics_name and args.all_classes_output_stem:
        all_classes_metrics, all_classes_model_slugs = load_plot_metrics(
            Path(args.all_classes_input_dir),
            args.model_slugs,
            args.all_classes_metrics_name,
            args.all_classes_predictions_name,
            args.regressor,
            args.error_stat,
        )
        all_classes_output_paths = plot_lines_wide(
            metrics=all_classes_metrics,
            model_slugs=all_classes_model_slugs,
            output_dir=Path(args.output_dir),
            output_stem=args.all_classes_output_stem,
            formats=args.formats,
            fig_width=args.wide_fig_width,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=args.ymax,
            tick_step=args.tick_step,
            error_stat=args.error_stat,
            error_alpha=args.error_alpha,
            title=args.title,
        )
        for output_path in all_classes_output_paths:
            print(f"Wrote all-classes plot: {output_path}")


if __name__ == "__main__":
    main()
