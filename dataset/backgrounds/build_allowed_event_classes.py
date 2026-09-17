#!/usr/bin/env python3
"""Build per-background allowed event classes for synthetic SED mixing.

This script writes:

- ``allowed_event_classes.json``: foreground labels that may be mixed into each
  background recording.
- ``background_event_detections.json``: the detection scores used to make that
  decision.

The default ``ast-audioset`` backend uses a supervised AudioSet-tagging model
over sliding windows, maps its labels onto ESC-50 classes, and aggregates max
confidence across the recording. ``hf-zero-shot`` remains available for CLAP-
style prompt classification, and ``metadata`` is a fast local sanity check
based only on Freesound metadata and filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from dataset.synthetic_sed import (
    AUDIO_EXTENSIONS,
    DEFAULT_BACKGROUND_ROOT,
    DEFAULT_DATASET_ROOT,
    _load_audio_mono,
)


DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "backgrounds"
DEFAULT_AUDIOSET_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_ZERO_SHOT_MODEL = "laion/clap-htsat-unfused"
NSYNTH_SPLITS = ("train", "valid", "test")
UNRESTRICTED_ALLOWLIST_DATASETS = {"nsynth", "gsc"}
ESC50_AUDIOSET_ALIASES = {
    "airplane": ["aircraft", "airplane", "fixed-wing aircraft"],
    "breathing": ["breathing", "respiration"],
    "brushing_teeth": ["toothbrush", "brushing teeth"],
    "can_opening": ["can opening", "opening can"],
    "car_horn": ["car horn", "vehicle horn", "horn"],
    "cat": ["cat", "meow", "purr"],
    "chainsaw": ["chainsaw"],
    "chirping_birds": ["bird", "bird vocalization", "chirp", "tweet"],
    "church_bells": ["church bell", "bell"],
    "clapping": ["clapping", "applause"],
    "clock_alarm": ["alarm clock", "alarm"],
    "clock_tick": ["tick-tock", "clock", "tick"],
    "coughing": ["cough"],
    "cow": ["cow", "moo"],
    "crackling_fire": ["fire", "crackle", "crackling fire"],
    "crickets": ["cricket"],
    "crow": ["crow", "caw"],
    "crying_baby": ["baby cry", "crying baby", "infant cry"],
    "dog": ["dog", "bark", "bow-wow"],
    "door_wood_creaks": ["door", "creak"],
    "door_wood_knock": ["door", "knock"],
    "drinking_sipping": ["drink", "sip", "gulp"],
    "engine": ["engine", "motor"],
    "fireworks": ["fireworks", "explosion", "firecracker"],
    "footsteps": ["footstep", "walk", "steps"],
    "frog": ["frog", "croak"],
    "glass_breaking": ["glass", "shatter", "breaking"],
    "hand_saw": ["saw", "handsaw"],
    "helicopter": ["helicopter", "rotorcraft"],
    "hen": ["chicken", "hen", "cluck"],
    "insects": ["insect", "buzz"],
    "keyboard_typing": ["typing", "keyboard", "typewriter"],
    "laughing": ["laughter", "laugh"],
    "mouse_click": ["mouse", "click", "computer mouse"],
    "pig": ["pig", "oink"],
    "pouring_water": ["pour", "water"],
    "rain": ["rain", "rainfall"],
    "rooster": ["rooster", "cock-a-doodle-doo"],
    "sea_waves": ["ocean", "wave", "surf", "sea"],
    "sheep": ["sheep", "bleat"],
    "siren": ["siren"],
    "sneezing": ["sneeze"],
    "snoring": ["snore", "snoring"],
    "thunderstorm": ["thunder", "thunderstorm"],
    "toilet_flush": ["toilet flush", "flush"],
    "train": ["train", "railroad"],
    "vacuum_cleaner": ["vacuum cleaner", "vacuum"],
    "washing_machine": ["washing machine"],
    "water_drops": ["water drop", "drip"],
    "wind": ["wind"],
}


def normalize_label(label: str) -> str:
    return label.strip().replace("_", " ")


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def term_pattern(term: str) -> re.Pattern[str]:
    tokens = [re.escape(token) for token in re.split(r"[\s_-]+", term.casefold()) if token]
    if not tokens:
        tokens = [re.escape(term.casefold())]
    body = r"[\s_-]+".join(tokens)
    return re.compile(rf"(?<!\w){body}s?(?!\w)")


def discover_backgrounds(background_root: Path) -> list[dict[str, Any]]:
    metadata = load_freesound_metadata(background_root)
    metadata_by_path = {
        str((background_root / row["local_file"]).resolve()): row
        for row in metadata
        if row.get("local_file")
    }

    backgrounds = []
    for path in sorted(background_root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in AUDIO_EXTENSIONS:
            continue
        row = metadata_by_path.get(str(path.resolve()), {})
        background_id = str(path.relative_to(background_root))
        backgrounds.append({
            "id": background_id,
            "path": str(path),
            "category": row.get("category", path.parent.name),
            "metadata": row,
            "text": background_text(path, row),
        })
    return backgrounds


def load_freesound_metadata(background_root: Path) -> list[dict[str, Any]]:
    json_path = background_root / "metadata.json"
    csv_path = background_root / "metadata.csv"
    if json_path.exists():
        with json_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return []


def background_text(path: Path, row: dict[str, Any]) -> str:
    pieces = [
        path.stem,
        path.parent.name,
        str(row.get("category", "")),
        str(row.get("category_label", "")),
        str(row.get("name", "")),
        str(row.get("description", "")),
        str(row.get("tags", "")),
    ]
    return " ".join(pieces).casefold()


def load_event_classes(root: Path, datasets: list[str]) -> dict[str, list[str]]:
    classes = {}
    if "esc50" in datasets:
        classes["esc50"] = load_esc50_classes(root)
    if "gsc" in datasets:
        classes["gsc"] = load_gsc_classes(root)
    if "nsynth" in datasets:
        classes["nsynth"] = load_nsynth_classes(root)
    return classes


def load_esc50_classes(root: Path) -> list[str]:
    path = root / "ESC50" / "ESC-50-master" / "meta" / "esc50.csv"
    with path.open() as handle:
        return sorted({row["category"] for row in csv.DictReader(handle)})


def load_gsc_classes(root: Path) -> list[str]:
    gsc_root = root / "GSC"
    labels = [
        path.name
        for path in gsc_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ]
    return sorted(labels)


def load_nsynth_classes(root: Path) -> list[str]:
    labels = set()
    for split in NSYNTH_SPLITS:
        metadata_path = root / "NSynth" / f"nsynth-{split}" / "examples.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        for meta in metadata.values():
            family = meta.get("instrument_family_str", "music note")
            source = meta.get("instrument_source_str", "")
            label = f"{source} {family}".strip() or "music note"
            labels.add(label)
            labels.add(family)
    return sorted(labels or {"music note"})


def candidate_labels_for_esc50(label: str) -> list[str]:
    aliases = ESC50_AUDIOSET_ALIASES.get(label, [normalize_label(label)])
    labels = [normalize_text(alias) for alias in aliases]
    labels.append(normalize_text(normalize_label(label)))
    return sorted({alias for alias in labels if alias})


def audioset_label_matches(audio_label: str, candidate_label: str) -> bool:
    audio_label = normalize_text(audio_label)
    candidate_label = normalize_text(candidate_label)
    if not audio_label or not candidate_label:
        return False
    return candidate_label == audio_label or candidate_label in audio_label


def detect_with_metadata(
    backgrounds: list[dict[str, Any]],
    classes: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, float]]]:
    detections = {}
    for background in backgrounds:
        per_dataset = {}
        for dataset_name, labels in classes.items():
            per_dataset[dataset_name] = {}
            for label in labels:
                candidates = {label, normalize_label(label)}
                if dataset_name == "esc50":
                    candidates.update(candidate_labels_for_esc50(label))
                candidates.update(
                    token
                    for token in re.split(r"[\s_-]+", label)
                    if len(token) >= 4
                )
                if any(term_pattern(term).search(background["text"]) for term in candidates):
                    per_dataset[dataset_name][label] = 1.0
        detections[background["id"]] = per_dataset
    return detections


def detect_with_ast_audioset(
    backgrounds: list[dict[str, Any]],
    classes: dict[str, list[str]],
    model_name: str,
    threshold: float,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    batch_size: int,
    device: str,
    local_files_only: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    torch_device = torch.device(normalize_torch_device(device))
    model.to(torch_device)
    model.eval()

    id_to_label = {
        int(index): label
        for index, label in model.config.id2label.items()
    }
    esc50_labels = classes.get("esc50", [])
    label_aliases = {
        label: candidate_labels_for_esc50(label)
        for label in esc50_labels
    }

    detections = {}
    for background_index, background in enumerate(backgrounds, start=1):
        print(f"[{background_index}/{len(backgrounds)}] AST tagging {background['id']}", flush=True)
        audio = _load_audio_mono(background["path"], sample_rate)
        windows = audio_windows(audio, sample_rate, window_sec, hop_sec)
        scores = {label: 0.0 for label in esc50_labels}
        raw_audioset_scores: dict[str, float] = {}

        for batch in batched(windows, batch_size):
            inputs = feature_extractor(
                batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(torch_device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.sigmoid(logits).detach().cpu().numpy()

            for window_scores in probabilities:
                for class_index, score in enumerate(window_scores):
                    audio_label = id_to_label.get(class_index, "")
                    if not audio_label:
                        continue
                    score = float(score)
                    if score < threshold:
                        continue
                    raw_audioset_scores[audio_label] = max(
                        raw_audioset_scores.get(audio_label, 0.0),
                        score,
                    )
                    for esc50_label, aliases in label_aliases.items():
                        if any(audioset_label_matches(audio_label, alias) for alias in aliases):
                            scores[esc50_label] = max(scores[esc50_label], score)

        detections[background["id"]] = {
            "audioset": dict(sorted(
                raw_audioset_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )),
            "esc50": {
                label: score
                for label, score in scores.items()
                if score >= threshold
            }
        }
    return detections


def normalize_torch_device(device: str) -> str:
    if device == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device in {"-1", "cpu"}:
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def normalize_zero_shot_device(device: str) -> int:
    if device == "auto":
        import torch

        return 0 if torch.cuda.is_available() else -1
    if device == "cpu":
        return -1
    return int(device)


def detect_with_hf_zero_shot(
    backgrounds: list[dict[str, Any]],
    classes: dict[str, list[str]],
    model_name: str,
    threshold: float,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    label_batch_size: int,
    device: str,
) -> dict[str, dict[str, dict[str, float]]]:
    from transformers import pipeline

    classifier = pipeline(
        task="zero-shot-audio-classification",
        model=model_name,
        device=normalize_zero_shot_device(device),
    )
    detections = {}
    for background_index, background in enumerate(backgrounds, start=1):
        print(f"[{background_index}/{len(backgrounds)}] zero-shot tagging {background['id']}", flush=True)
        audio = _load_audio_mono(background["path"], sample_rate)
        windows = audio_windows(audio, sample_rate, window_sec, hop_sec)
        per_dataset = {}
        for dataset_name, labels in classes.items():
            scores = {label: 0.0 for label in labels}
            for window in windows:
                for batch in batched(labels, label_batch_size):
                    normalized_batch = [normalize_label(label) for label in batch]
                    predictions = classifier(
                        window,
                        candidate_labels=normalized_batch,
                    )
                    for prediction in predictions:
                        raw_label = prediction["label"]
                        try:
                            index = normalized_batch.index(raw_label)
                        except ValueError:
                            continue
                        label = batch[index]
                        scores[label] = max(scores[label], float(prediction["score"]))
            per_dataset[dataset_name] = {
                label: score
                for label, score in scores.items()
                if score >= threshold
            }
        detections[background["id"]] = per_dataset
    return detections


def audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
) -> list[np.ndarray]:
    window_samples = int(round(window_sec * sample_rate))
    hop_samples = int(round(hop_sec * sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_sec and hop_sec must be positive")
    if len(audio) <= window_samples:
        return [audio]

    windows = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        windows.append(audio[start:start + window_samples])
    if not windows or len(audio) - (len(windows) - 1) * hop_samples > window_samples:
        windows.append(audio[-window_samples:])
    return windows


def batched(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("label_batch_size must be positive")
    return [values[start:start + batch_size] for start in range(0, len(values), batch_size)]


def build_allowed_classes(
    backgrounds: list[dict[str, Any]],
    classes: dict[str, list[str]],
    detections: dict[str, dict[str, dict[str, float]]],
    backend: str | None = None,
    model_name: str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "description": (
            "NSynth and GSC labels are always allowed. Other labels are allowed "
            "when the detector did not find that class in the background."
        ),
        "unrestricted_allowlist_datasets": sorted(UNRESTRICTED_ALLOWLIST_DATASETS),
        "detector": {
            "backend": backend,
            "model": model_name,
            "threshold": threshold,
        },
        "classes": classes,
        "backgrounds": {},
    }
    for background in backgrounds:
        background_detections = detections.get(background["id"], {})
        allowed = {}
        for dataset_name, labels in classes.items():
            if dataset_name in UNRESTRICTED_ALLOWLIST_DATASETS:
                allowed[dataset_name] = labels
                continue
            detected_labels = set(background_detections.get(dataset_name, {}))
            allowed[dataset_name] = [
                label
                for label in labels
                if label not in detected_labels
            ]
        result["backgrounds"][background["id"]] = {
            "path": background["path"],
            "category": background["category"],
            "allowed_event_classes": allowed,
            "detected_event_classes": background_detections,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-root", type=Path, default=DEFAULT_BACKGROUND_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--category", action="append", help="only process this background category; repeatable")
    parser.add_argument("--max-backgrounds", type=int, help="cap the number of backgrounds to process")
    parser.add_argument("--datasets", nargs="+", choices=["esc50", "gsc", "nsynth"], default=["esc50", "gsc", "nsynth"])
    parser.add_argument("--backend", choices=["ast-audioset", "hf-zero-shot", "metadata"], default="ast-audioset")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HuggingFace model ID. Defaults to MIT AST for ast-audioset and "
            "LAION CLAP for hf-zero-shot."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--hop-sec", type=float, default=5.0)
    parser.add_argument("--label-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8, help="audio windows per AST forward pass")
    parser.add_argument("--device", default="auto", help="Torch device for ast-audioset, e.g. auto, cpu, cuda:0, or 0")
    parser.add_argument("--zero-shot-device", default="auto", help="auto, -1/cpu for CPU, otherwise CUDA device index")
    parser.add_argument("--local-files-only", action="store_true", help="load AST model files only from the local HuggingFace cache")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backgrounds = discover_backgrounds(args.background_root)
    if args.category:
        requested_categories = set(args.category)
        backgrounds = [
            background
            for background in backgrounds
            if background["category"] in requested_categories
        ]
    if args.max_backgrounds is not None:
        if args.max_backgrounds <= 0:
            raise ValueError("--max-backgrounds must be positive")
        backgrounds = backgrounds[:args.max_backgrounds]
    if not backgrounds:
        raise FileNotFoundError(f"No background audio files found under {args.background_root}")

    classes = load_event_classes(args.dataset_root, args.datasets)

    if args.backend == "metadata":
        model_name = "metadata"
        detections = detect_with_metadata(backgrounds, classes)
    elif args.backend == "hf-zero-shot":
        model_name = args.model or DEFAULT_ZERO_SHOT_MODEL
        detections = detect_with_hf_zero_shot(
            backgrounds=backgrounds,
            classes=classes,
            model_name=model_name,
            threshold=args.threshold,
            sample_rate=args.sample_rate,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            label_batch_size=args.label_batch_size,
            device=args.zero_shot_device,
        )
    else:
        model_name = args.model or DEFAULT_AUDIOSET_MODEL
        detections = detect_with_ast_audioset(
            backgrounds=backgrounds,
            classes=classes,
            model_name=model_name,
            threshold=args.threshold,
            sample_rate=args.sample_rate,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=args.local_files_only,
        )

    allowlist = build_allowed_classes(
        backgrounds,
        classes,
        detections,
        backend=args.backend,
        model_name=model_name,
        threshold=args.threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowlist_path = args.output_dir / "allowed_event_classes.json"
    detections_path = args.output_dir / "background_event_detections.json"
    allowlist_path.write_text(json.dumps(allowlist, indent=2) + "\n", encoding="utf-8")
    detections_path.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {allowlist_path}")
    print(f"Wrote {detections_path}")
    print(f"Backgrounds: {len(backgrounds)}")
    for dataset_name, labels in classes.items():
        print(f"{dataset_name} classes: {len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
