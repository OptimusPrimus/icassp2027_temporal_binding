#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_32_decode_coarse_position.run_collect_activations import (
    DEFAULT_MODEL_IDS,
    DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_DIR,
    default_model_id,
    model_output_slug,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
REGULAR_SPLIT_OUTPUT_NAME = "regular_split"
CLASS_CV_SPLIT_OUTPUT_NAME = "class_cv_split"
DEFAULT_FILE_PATTERN = "*all_decoder_text_activations.pt"
ESC50_CLASS_COUNT = 50
REGRESSOR = "midpoint_nn"
RIDGE_ALPHA = 10.0


CONDITION_LABELS = {
    "target_event_word": "Target event word -> target event",
    "alternative_event_control": "Target event word -> alternative event",
    "first_prompt_token_control": "First prompt token -> target event",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train layer-wise event-midpoint predictors from the text-token "
            "embeddings saved by run_collect_activations.py."
        )
    )
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_MODEL_IDS),
        default="af3",
        help="Audio language model whose saved activation file should be used.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Hugging Face model id used to resolve the output slug.",
    )
    parser.add_argument(
        "--activation-file",
        type=Path,
        default=None,
        help="Explicit saved activation .pt file. Overrides --input-dir/--model.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing model-slug subdirectories with activation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base result directory. Regular split outputs go below "
            f"{REGULAR_SPLIT_OUTPUT_NAME}/{{model_slug}} and class-CV outputs go below "
            f"{CLASS_CV_SPLIT_OUTPUT_NAME}/{{model_slug}}."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument(
        "--class-fold-count",
        type=int,
        default=5,
        help=(
            "Number of ESC-50 class folds for the held-out-class evaluation. "
            "Default 5 gives 40 training classes and 10 held-out test classes per fold."
        ),
    )
    parser.add_argument(
        "--class-split-seed",
        type=int,
        default=0,
        help="Random seed for assigning ESC-50 classes to held-out folds.",
    )
    parser.add_argument(
        "--skip-class-cv",
        action="store_true",
        help="Only experiments without the class CV split",
    )
    return parser.parse_args()


def load_activation_file(path: Path) -> dict[str, Any]:
    result = torch.load(path, map_location="cpu")
    if not isinstance(result, dict):
        raise ValueError(f"Expected {path} to contain a dict, got {type(result)}")
    for key in ("activations", "metadata", "target_tokens", "layer_names"):
        if key not in result:
            raise ValueError(f"{path} is missing required key: {key}")
    return result


def discover_activation_file(args: argparse.Namespace) -> tuple[Path, str]:
    model_id = args.model_id or default_model_id(args.model)
    model_slug = model_output_slug(model_id)
    if args.activation_file is not None:
        return args.activation_file, model_slug

    model_dir = args.input_dir / model_slug
    candidates = sorted(
        model_dir.glob(DEFAULT_FILE_PATTERN),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No activation files matched {DEFAULT_FILE_PATTERN!r} in {model_dir}"
        )
    if len(candidates) > 1:
        print(f"Found {len(candidates)} activation files; using newest: {candidates[0]}")
    return candidates[0], model_slug


def selected_tokens(result: dict[str, Any], index: int) -> list[str]:
    tokens = result["target_tokens"][index]
    target_lengths = result.get("target_lengths")
    if torch.is_tensor(target_lengths):
        length = int(target_lengths[index].item())
        return list(tokens[:length])
    return list(tokens)


def token_piece(token: str) -> str:
    text = str(token)
    text = text.replace("Ġ", " ")
    text = text.replace("▁", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("</w>", "")
    return text


def token_text(token: str) -> str:
    return token_piece(token).strip()


def is_content_token(token: str) -> bool:
    text = token_text(token)
    if not text:
        return False
    if text.startswith("<|") and text.endswith("|>"):
        return False
    if text in {"<s>", "</s>", "<pad>", "user", "assistant", "system"}:
        return False
    return True


def first_prompt_token_index(tokens: list[str]) -> int:
    for index, token in enumerate(tokens):
        if is_content_token(token):
            return index
    raise ValueError("Could not find a first prompt token")


def target_event_last_token_index(tokens: list[str]) -> int:
    for index, token in enumerate(tokens):
        text = token_piece(token)
        if "?" not in text:
            continue
        before_question = text.split("?", 1)[0].strip()
        if before_question:
            return index
        for previous in range(index - 1, -1, -1):
            if is_content_token(tokens[previous]):
                return previous
    raise ValueError("Could not find the target event token before '?'")


def event_midpoint(event: dict[str, Any]) -> float:
    return (float(event["onset_sec"]) + float(event["offset_sec"])) / 2.0


def target_event(metadata: dict[str, Any]) -> dict[str, Any]:
    query_index = int(metadata["query_event_index"])
    return metadata["events"][query_index]


def alternative_event(metadata: dict[str, Any]) -> dict[str, Any]:
    query_index = int(metadata["query_event_index"])
    for index, event in enumerate(metadata["events"]):
        if index != query_index:
            return event
    raise ValueError("Could not find an alternative event")


def event_label(event: dict[str, Any]) -> str:
    label = event.get("event_label")
    if label is None:
        raise ValueError("Event metadata is missing event_label")
    return str(label)


def selected_event_label(metadata: dict[str, Any], condition: str) -> str:
    _select_token, select_event = condition_spec(condition)
    return event_label(select_event(metadata))


def split_event_labels(result: dict[str, Any], split: str) -> list[str]:
    labels = []
    for index, metadata in enumerate(result["metadata"]):
        if metadata.get("split") != split:
            continue
        for event in metadata.get("events", []):
            if isinstance(event, dict) and event.get("event_label"):
                labels.append(str(event["event_label"]))
    if not labels:
        raise ValueError(f"No event labels found for split: {split}")
    return labels


def is_esc50_result(result: dict[str, Any]) -> bool:
    return any(
        "esc50" in str(metadata.get("dataset", "")).lower()
        for metadata in result["metadata"]
    )


def esc50_class_folds(
    result: dict[str, Any],
    train_split: str,
    eval_split: str,
    fold_count: int,
    split_seed: int,
) -> list[dict[str, Any]]:
    if not is_esc50_result(result):
        raise ValueError("Class-wise cross-validation currently expects ESC-50 data")
    if fold_count <= 1:
        raise ValueError("--class-fold-count must be greater than 1")

    classes = sorted(set(split_event_labels(result, train_split)))
    if len(classes) != ESC50_CLASS_COUNT:
        raise ValueError(
            f"Expected {ESC50_CLASS_COUNT} ESC-50 classes in {train_split!r}, "
            f"found {len(classes)}"
        )
    eval_classes_available = set(split_event_labels(result, eval_split))
    missing_eval_classes = sorted(set(classes) - eval_classes_available)
    if missing_eval_classes:
        raise ValueError(
            f"{eval_split!r} is missing held-out class examples for: "
            f"{', '.join(missing_eval_classes)}"
        )
    if len(classes) % fold_count != 0:
        raise ValueError(
            f"{len(classes)} classes cannot be split evenly into {fold_count} folds"
        )

    shuffled_classes = list(classes)
    random.Random(split_seed).shuffle(shuffled_classes)
    fold_size = len(classes) // fold_count
    folds = []
    for fold_index in range(fold_count):
        start = fold_index * fold_size
        held_out_classes = sorted(shuffled_classes[start:start + fold_size])
        train_classes = sorted(set(classes) - set(held_out_classes))
        folds.append(
            {
                "fold_index": fold_index,
                "class_split_seed": split_seed,
                "train_event_classes": train_classes,
                "test_event_classes": held_out_classes,
            }
        )
    return folds


def split_indices(result: dict[str, Any], split: str) -> list[int]:
    indices = [
        index
        for index, metadata in enumerate(result["metadata"])
        if metadata.get("split") == split
    ]
    if not indices:
        raise ValueError(f"No examples found for split: {split}")
    return indices


def split_indices_for_event_classes(
    result: dict[str, Any],
    split: str,
    condition: str,
    allowed_event_classes: set[str],
) -> list[int]:
    indices = [
        index
        for index, metadata in enumerate(result["metadata"])
        if metadata.get("split") == split
        and selected_event_label(metadata, condition) in allowed_event_classes
    ]
    if not indices:
        raise ValueError(
            f"No {split!r} examples found for condition {condition!r} and "
            f"{len(allowed_event_classes)} selected event class(es)"
        )
    return indices


def condition_spec(
    condition: str,
) -> tuple[Callable[[list[str]], int], Callable[[dict[str, Any]], dict[str, Any]]]:
    if condition == "target_event_word":
        return target_event_last_token_index, target_event
    if condition == "alternative_event_control":
        return target_event_last_token_index, alternative_event
    if condition == "first_prompt_token_control":
        return first_prompt_token_index, target_event
    raise ValueError(f"Unsupported condition: {condition}")


def build_xy(
    result: dict[str, Any],
    example_indices: list[int],
    layer_slot: int,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    select_token, select_event = condition_spec(condition)
    xs = []
    ys = []
    rows = []

    for example_index in example_indices:
        activations = result["activations"][example_index]
        if not torch.is_tensor(activations) or activations.ndim != 3:
            raise ValueError(
                f"Expected activations[{example_index}] to have shape "
                "[layer, selected_token, hidden]"
            )
        tokens = selected_tokens(result, example_index)
        token_index = select_token(tokens)
        if token_index >= activations.shape[1]:
            raise ValueError(
                f"Token index {token_index} exceeds activation token dimension "
                f"{activations.shape[1]} for example {example_index}"
            )

        metadata = result["metadata"][example_index]
        event = select_event(metadata)
        xs.append(activations[layer_slot, token_index].float())
        ys.append(event_midpoint(event))
        rows.append(
            {
                "example_index": example_index,
                "sample_id": metadata.get("id", ""),
                "split": metadata.get("split", ""),
                "prompt": metadata.get("prompt", ""),
                "token_index": token_index,
                "token": tokens[token_index],
                "query_event_label": metadata.get("query_event_label", ""),
                "target_event_label": target_event(metadata).get("event_label", ""),
                "alternative_event_label": alternative_event(metadata).get("event_label", ""),
                "target_midpoint_sec": event_midpoint(target_event(metadata)),
                "alternative_midpoint_sec": event_midpoint(alternative_event(metadata)),
                "label_midpoint_sec": event_midpoint(event),
            }
        )

    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32), rows


class MidpointRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 30.0 * torch.sigmoid(self.net(x).squeeze(-1))


def train_midpoint_regressor(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    seed: int,
    epochs: int = 12,
    batch_size: int = 1024,
) -> MidpointRegressor:
    torch.manual_seed(1000 + seed)

    model = MidpointRegressor(x_train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=3e-3,
    )

    for _epoch in range(epochs):
        permutation = torch.randperm(len(x_train))
        model.train()
        for start in range(0, len(x_train), batch_size):
            ids = permutation[start:start + batch_size]
            pred = model(x_train[ids])
            loss = F.smooth_l1_loss(
                pred,
                y_train[ids],
                beta=1.0,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def fit_predict_midpoint_nn(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0).clamp_min(1e-5)
    x_train = (x_train - feature_mean) / feature_std
    x_eval = (x_eval - feature_mean) / feature_std

    model = train_midpoint_regressor(
        x_train,
        y_train,
        seed=seed,
    )
    model.eval()
    with torch.no_grad():
        return model(x_eval)


def fit_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    return fit_predict_midpoint_nn(
        x_train,
        y_train,
        x_eval,
        seed=seed,
    )


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, train_mean: float) -> dict[str, float]:
    errors = y_pred - y_true
    abs_errors = errors.abs()
    centered = y_true - y_true.mean()
    ss_res = float((errors ** 2).sum().item())
    ss_tot = float((centered ** 2).sum().item())
    baseline_abs = (torch.full_like(y_true, train_mean) - y_true).abs()
    return {
        "mae": float(abs_errors.mean().item()),
        "median_absolute_error": float(abs_errors.median().item()),
        "rmse": math.sqrt(float((errors ** 2).mean().item())),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "baseline_train_mean_mae": float(baseline_abs.mean().item()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_class_cv_metrics(
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for condition in CONDITION_LABELS:
        condition_folds = [row for row in fold_rows if row["condition"] == condition]
        layer_slots = sorted({row["layer_slot"] for row in condition_folds})
        for layer_slot in layer_slots:
            layer_folds = [
                row for row in condition_folds if row["layer_slot"] == layer_slot
            ]
            layer_predictions = [
                row
                for row in prediction_rows
                if row["condition"] == condition and row["layer_slot"] == layer_slot
            ]
            abs_errors = torch.tensor(
                [row["absolute_error_sec"] for row in layer_predictions],
                dtype=torch.float32,
            )
            rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "layer_slot": layer_slot,
                    "layer_name": layer_folds[0]["layer_name"],
                    "regressor": layer_folds[0]["regressor"],
                    "ridge_alpha": layer_folds[0]["ridge_alpha"],
                    "fold_count": len(layer_folds),
                    "n_train_mean": float(
                        torch.tensor([row["n_train"] for row in layer_folds], dtype=torch.float32).mean().item()
                    ),
                    "n_eval_total": sum(row["n_eval"] for row in layer_folds),
                    "mae": float(abs_errors.mean().item()),
                    "mae_fold_mean": float(
                        torch.tensor([row["mae"] for row in layer_folds], dtype=torch.float32).mean().item()
                    ),
                    "mae_fold_std": (
                        float(
                            torch.tensor([row["mae"] for row in layer_folds], dtype=torch.float32)
                            .std(unbiased=True)
                            .item()
                        )
                        if len(layer_folds) > 1
                        else 0.0
                    ),
                    "baseline_train_mean_mae_fold_mean": float(
                        torch.tensor(
                            [row["baseline_train_mean_mae"] for row in layer_folds],
                            dtype=torch.float32,
                        ).mean().item()
                    ),
                }
            )
    return rows


def run_class_cv(
    args: argparse.Namespace,
    result: dict[str, Any],
    layer_names: list[str],
    outdir: Path,
) -> None:
    class_folds = esc50_class_folds(
        result,
        train_split=args.train_split,
        eval_split=args.eval_split,
        fold_count=args.class_fold_count,
        split_seed=args.class_split_seed,
    )
    print(
        f"Class CV: {len(class_folds)} folds, "
        f"{len(class_folds[0]['train_event_classes'])} train classes/fold, "
        f"{len(class_folds[0]['test_event_classes'])} held-out classes/fold"
    )

    fold_metrics_rows = []
    prediction_rows = []

    for condition in CONDITION_LABELS:
        for layer_slot, layer_name in enumerate(layer_names):
            pooled_actual = []
            pooled_predicted = []
            for class_fold in class_folds:
                train_classes = set(class_fold["train_event_classes"])
                test_classes = set(class_fold["test_event_classes"])
                train_indices = split_indices_for_event_classes(
                    result,
                    args.train_split,
                    condition,
                    train_classes,
                )
                eval_indices = split_indices_for_event_classes(
                    result,
                    args.eval_split,
                    condition,
                    test_classes,
                )
                x_train, y_train, _ = build_xy(
                    result,
                    train_indices,
                    layer_slot,
                    condition,
                )
                x_eval, y_eval, eval_rows = build_xy(
                    result,
                    eval_indices,
                    layer_slot,
                    condition,
                )

                y_pred = fit_predict(
                    x_train,
                    y_train,
                    x_eval,
                    seed=class_fold["fold_index"],
                )

                row_metrics = regression_metrics(
                    y_eval,
                    y_pred,
                    train_mean=float(y_train.mean().item()),
                )
                fold_metrics_rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "layer_slot": layer_slot,
                        "layer_name": layer_name,
                        "fold_index": class_fold["fold_index"],
                        "regressor": REGRESSOR,
                        "ridge_alpha": RIDGE_ALPHA,
                        "n_train": int(x_train.shape[0]),
                        "n_eval": int(x_eval.shape[0]),
                        "train_event_classes": json.dumps(class_fold["train_event_classes"]),
                        "test_event_classes": json.dumps(class_fold["test_event_classes"]),
                        **row_metrics,
                    }
                )
                for eval_row, actual, predicted in zip(
                    eval_rows,
                    y_eval.tolist(),
                    y_pred.tolist(),
                ):
                    prediction_rows.append(
                        {
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "layer_slot": layer_slot,
                            "layer_name": layer_name,
                            "fold_index": class_fold["fold_index"],
                            "fold_train_classes": json.dumps(class_fold["train_event_classes"]),
                            "fold_test_classes": json.dumps(class_fold["test_event_classes"]),
                            **eval_row,
                            "selected_event_class": selected_event_label(
                                result["metadata"][eval_row["example_index"]],
                                condition,
                            ),
                            "actual_midpoint_sec": float(actual),
                            "predicted_midpoint_sec": float(predicted),
                            "absolute_error_sec": abs(float(predicted) - float(actual)),
                        }
                    )
                pooled_actual.append(y_eval)
                pooled_predicted.append(y_pred)

            actual = torch.cat(pooled_actual)
            predicted = torch.cat(pooled_predicted)
            pooled_metrics = regression_metrics(
                actual,
                predicted,
                train_mean=float("nan"),
            )
            print(
                f"class-cv {condition} layer={layer_slot:02d} "
                f"pooled_mae={pooled_metrics['mae']:.3f}s"
            )

    class_cv_metrics_rows = aggregate_class_cv_metrics(
        fold_metrics_rows,
        prediction_rows,
    )
    write_csv(outdir / "event_midpoint_class_cv_metrics_by_layer.csv", class_cv_metrics_rows)
    write_csv(outdir / "event_midpoint_class_cv_fold_metrics_by_layer.csv", fold_metrics_rows)
    write_csv(outdir / "event_midpoint_class_cv_predictions.csv", prediction_rows)

    summary = {
        "enabled": True,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "class_fold_count": args.class_fold_count,
        "class_split_seed": args.class_split_seed,
        "classes_per_test_fold": len(class_folds[0]["test_event_classes"]),
        "train_classes_per_fold": len(class_folds[0]["train_event_classes"]),
        "folds": class_folds,
        "best_by_condition": {},
    }
    for condition in CONDITION_LABELS:
        rows = [row for row in class_cv_metrics_rows if row["condition"] == condition]
        best = min(rows, key=lambda row: row["mae"])
        summary["best_by_condition"][condition] = {
            key: best[key]
            for key in (
                "condition_label",
                "layer_slot",
                "layer_name",
                "mae",
                "mae_fold_mean",
                "mae_fold_std",
                "baseline_train_mean_mae_fold_mean",
            )
        }
    with (outdir / "event_midpoint_class_cv_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote class-CV metrics: {outdir / 'event_midpoint_class_cv_metrics_by_layer.csv'}")


def run(args: argparse.Namespace) -> Path:
    activation_file, model_slug = discover_activation_file(args)
    result = load_activation_file(activation_file)
    layer_names = list(result["layer_names"])
    train_indices = split_indices(result, args.train_split)
    eval_indices = split_indices(result, args.eval_split)
    regular_outdir = args.output_dir / REGULAR_SPLIT_OUTPUT_NAME / model_slug
    class_cv_outdir = args.output_dir / CLASS_CV_SPLIT_OUTPUT_NAME / model_slug
    regular_outdir.mkdir(parents=True, exist_ok=True)

    print(f"Activation file: {activation_file}")
    print(f"Train split: {args.train_split} ({len(train_indices)} examples)")
    print(f"Eval split: {args.eval_split} ({len(eval_indices)} examples)")
    print(f"Regular-split output directory: {regular_outdir}")
    if not args.skip_class_cv:
        print(f"Class-CV output directory: {class_cv_outdir}")

    metrics_rows = []
    prediction_rows = []

    for condition in CONDITION_LABELS:
        for layer_slot, layer_name in enumerate(layer_names):
            x_train, y_train, _ = build_xy(
                result,
                train_indices,
                layer_slot,
                condition,
            )
            x_eval, y_eval, eval_rows = build_xy(
                result,
                eval_indices,
                layer_slot,
                condition,
            )

            y_pred = fit_predict(
                x_train,
                y_train,
                x_eval,
                seed=layer_slot,
            )

            row_metrics = regression_metrics(
                y_eval,
                y_pred,
                train_mean=float(y_train.mean().item()),
            )
            metrics_row = {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "layer_slot": layer_slot,
                "layer_name": layer_name,
                "regressor": REGRESSOR,
                "ridge_alpha": RIDGE_ALPHA,
                "n_train": int(x_train.shape[0]),
                "n_eval": int(x_eval.shape[0]),
                **row_metrics,
            }
            metrics_rows.append(metrics_row)
            print(
                f"{condition} layer={layer_slot:02d} "
                f"mae={row_metrics['mae']:.3f}s "
                f"rmse={row_metrics['rmse']:.3f}s "
                f"baseline_mae={row_metrics['baseline_train_mean_mae']:.3f}s"
            )

            for eval_row, actual, predicted in zip(eval_rows, y_eval.tolist(), y_pred.tolist()):
                prediction_rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "layer_slot": layer_slot,
                        "layer_name": layer_name,
                        **eval_row,
                        "actual_midpoint_sec": float(actual),
                        "predicted_midpoint_sec": float(predicted),
                        "absolute_error_sec": abs(float(predicted) - float(actual)),
                    }
                )

    write_csv(regular_outdir / "event_midpoint_metrics_by_layer.csv", metrics_rows)
    write_csv(regular_outdir / "event_midpoint_predictions.csv", prediction_rows)

    summary = {
        "activation_file": str(activation_file),
        "model_slug": model_slug,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "best_by_condition": {},
    }
    for condition in CONDITION_LABELS:
        rows = [row for row in metrics_rows if row["condition"] == condition]
        best = min(rows, key=lambda row: row["mae"])
        summary["best_by_condition"][condition] = {
            key: best[key]
            for key in (
                "condition_label",
                "layer_slot",
                "layer_name",
                "mae",
                "rmse",
                "r2",
                "baseline_train_mean_mae",
            )
        }
    with (regular_outdir / "event_midpoint_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if not args.skip_class_cv:
        run_class_cv(args, result, layer_names, class_cv_outdir)

    print(f"Wrote regular-split metrics and predictions: {regular_outdir}")
    if not args.skip_class_cv:
        print(f"Wrote class-CV metrics and predictions: {class_cv_outdir}")
    return regular_outdir


def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    run(args)


if __name__ == "__main__":
    main()
