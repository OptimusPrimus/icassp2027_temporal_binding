#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_32_decode_coarse_position.run_collect_activations import (
    DEFAULT_MODEL_IDS,
    default_model_id,
    model_output_slug,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXPERIMENT_DIR / "outputs"
REGULAR_SPLIT_OUTPUT_NAME = "regular_split"
CLASS_CV_SPLIT_OUTPUT_NAME = "class_cv_split"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots" / "event_time_decoding"
REGULAR_METRICS_NAME = "event_midpoint_metrics_by_layer.csv"
CLASS_CV_METRICS_NAME = "event_midpoint_class_cv_metrics_by_layer.csv"
DEFAULT_REGRESSOR = "midpoint_nn"

CONDITION_LABELS = {
    "target_event_word": "Target event word -> target event",
    "alternative_event_control": "Target event word -> alternative event",
    "first_prompt_token_control": "First prompt token -> target event",
}

SPLIT_LABELS = {
    REGULAR_SPLIT_OUTPUT_NAME: "Regular split",
    CLASS_CV_SPLIT_OUTPUT_NAME: "Class-CV split",
}

SPLIT_METRICS_NAMES = {
    REGULAR_SPLIT_OUTPUT_NAME: REGULAR_METRICS_NAME,
    CLASS_CV_SPLIT_OUTPUT_NAME: CLASS_CV_METRICS_NAME,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot event-midpoint decoding MAE from saved metrics CSVs."
    )
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_MODEL_IDS),
        default="af-next",
        help="Audio language model whose decoding metrics should be plotted.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Hugging Face model id used to resolve the output slug.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Explicit metrics CSV. Overrides --input-dir/--model.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Base directory containing regular_split and class_cv_split output "
            "directories. If --split selects one split, this may also point "
            "directly at that split directory."
        ),
    )
    parser.add_argument(
        "--split",
        choices=(REGULAR_SPLIT_OUTPUT_NAME, CLASS_CV_SPLIT_OUTPUT_NAME, "both"),
        default="both",
        help="Which decoding split plots to create.",
    )
    parser.add_argument(
        "--metrics-name",
        default=None,
        help=(
            "Override metrics CSV filename. By default the regular split uses "
            f"{REGULAR_METRICS_NAME!r} and class-CV uses {CLASS_CV_METRICS_NAME!r}."
        ),
    )
    parser.add_argument(
        "--regressor",
        default=DEFAULT_REGRESSOR,
        help="Regressor value to select when the metrics CSV has a regressor column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base plot directory. Split plots go below {split}/{model_slug}.",
    )
    parser.add_argument(
        "--output-name",
        default="event_midpoint_mae_by_layer.png",
        help="Plot filename.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def selected_splits(args: argparse.Namespace) -> list[str]:
    if args.metrics_file is not None:
        return ["metrics_file"]
    if args.split == "both":
        return [REGULAR_SPLIT_OUTPUT_NAME, CLASS_CV_SPLIT_OUTPUT_NAME]
    return [args.split]


def split_input_dir(base_input_dir: Path, split_name: str) -> Path:
    split_dir = base_input_dir / split_name
    if split_dir.exists():
        return split_dir
    return base_input_dir


def metrics_path(args: argparse.Namespace, split_name: str) -> tuple[Path, str]:
    model_id = args.model_id or default_model_id(args.model)
    model_slug = model_output_slug(model_id)
    if args.metrics_file is not None:
        return args.metrics_file, model_slug

    input_dir = split_input_dir(args.input_dir, split_name)
    metrics_name = args.metrics_name or SPLIT_METRICS_NAMES[split_name]
    return input_dir / model_slug / metrics_name, model_slug


def read_metrics(path: Path, regressor: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV does not exist: {path}")

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Metrics CSV is empty: {path}")

    required = {"condition", "layer_slot", "mae"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if regressor is not None and "regressor" in rows[0]:
        rows = [row for row in rows if row["regressor"] == regressor]
        if not rows:
            raise ValueError(f"{path} has no rows for regressor={regressor!r}")

    for row in rows:
        row["layer_slot"] = int(row["layer_slot"])
        row["mae"] = float(row["mae"])
    return rows


def save_mae_plot(
    metrics: list[dict[str, Any]],
    output_path: Path,
    model_slug: str,
    split_label: str,
    dpi: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111)

    for condition, label in CONDITION_LABELS.items():
        condition_rows = [
            row for row in metrics if row["condition"] == condition
        ]
        condition_rows.sort(key=lambda row: row["layer_slot"])
        ax.plot(
            [row["layer_slot"] for row in condition_rows],
            [row["mae"] for row in condition_rows],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            label=label,
        )

    ax.set_xlabel("Layer slot")
    ax.set_ylabel("MAE (seconds)")
    ax.set_title(
        f"Event midpoint prediction from saved text embeddings\n"
        f"{model_slug} - {split_label}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for split_name in selected_splits(args):
        path, model_slug = metrics_path(args, split_name)
        if split_name == "metrics_file":
            output_path = args.output_dir / model_slug / args.output_name
            split_label = "Metrics file"
        else:
            output_path = args.output_dir / split_name / model_slug / args.output_name
            split_label = SPLIT_LABELS[split_name]
        save_mae_plot(
            read_metrics(path, args.regressor),
            output_path,
            model_slug,
            split_label,
            args.dpi,
        )
        print(f"Wrote {split_label} plot: {output_path}")


if __name__ == "__main__":
    main()
