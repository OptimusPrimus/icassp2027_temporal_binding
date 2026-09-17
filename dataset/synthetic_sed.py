#!/usr/bin/env python3
"""Synthetic sound event detection dataset with background ambience.

The dataset returns variable-length mixtures, their unmixed background and
foreground tracks, and DCASE-style event annotations: each foreground event has
an event label, onset time, and offset time. It can also materialize those
examples to wav files plus metadata/events TSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from dataset.audio_utils import load_audio_mono as _load_audio_mono
from dataset.esc import download_esc50
from dataset.gsc import download_gsc
from dataset.nsynth import download_nsynth


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_BACKGROUND_ROOT = DEFAULT_DATASET_ROOT / "backgrounds" / "freesound_ambience"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
DATASET_SUBDIRS = {
    "esc50": "ESC50",
    "nsynth": "NSynth",
    "gsc": "GSC",
}
BACKGROUND_MODES = ("freesound", "silence")
UNRESTRICTED_ALLOWLIST_DATASETS = {"nsynth", "gsc"}
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
    "testing": "test",
}
NSYNTH_SPLIT_ALIASES = {
    "validation": "valid",
    "valid": "valid",
    "val": "valid",
    "testing": "test",
    "test": "test",
    "train": "train",
}
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def _as_range(value: float | int | tuple[float, float] | tuple[int, int], name: str) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} tuple must have length 2")
        low, high = float(value[0]), float(value[1])
    else:
        low = high = float(value)
    if low > high:
        raise ValueError(f"{name} minimum must not exceed maximum")
    return low, high


def _display_label(label: str) -> str:
    return label.replace("_", " ")


def _term_pattern(term: str) -> re.Pattern[str]:
    tokens = [re.escape(token) for token in re.split(r"[\s_-]+", term.casefold()) if token]
    if not tokens:
        tokens = [re.escape(term.casefold())]
    body = r"[\s_-]+".join(tokens)
    return re.compile(rf"(?<!\w){body}s?(?!\w)")


def _stable_sort_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def trim_leading_trailing_silence(
    audio: np.ndarray,
    top_db: float = 40.0,
    min_remaining_samples: int = 1,
) -> np.ndarray:
    """Trim low-energy leading/trailing samples using a peak-relative threshold."""

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio[:min_remaining_samples]

    threshold = peak * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(np.abs(audio) > threshold)
    if len(active) == 0:
        return audio[:min_remaining_samples]

    start = int(active[0])
    end = int(active[-1]) + 1
    if end - start < min_remaining_samples:
        center = (start + end) // 2
        half = min_remaining_samples // 2
        start = max(0, center - half)
        end = min(len(audio), start + min_remaining_samples)
    return audio[start:end].astype(np.float32, copy=False)


def mix_sound(
    background_waveform: np.ndarray,
    foreground_audio: np.ndarray | list[np.ndarray],
    position_samples: int | list[int] | None = None,
    normalize_peak: bool = True
) -> np.ndarray:
    """Mix trimmed foreground audio into a background at the given sample position."""

    background = np.asarray(background_waveform, dtype=np.float32)
    waveform = background.copy()
    if position_samples is None:
        raise ValueError("position_samples is required")

    foregrounds = foreground_audio if isinstance(foreground_audio, list) else [foreground_audio]
    positions = position_samples if isinstance(position_samples, list) else [position_samples]
    if len(foregrounds) != len(positions):
        raise ValueError("foreground_audio and position_samples must have the same length")

    for audio, position in zip(foregrounds, positions):
        event_audio = np.asarray(audio, dtype=np.float32)
        position = int(position)
        if position < 0:
            raise ValueError("position_samples cannot be negative")
        offset = min(len(waveform), position + len(event_audio))
        if offset > position:
            waveform[position:offset] += event_audio[:offset - position]
    if normalize_peak:
        peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
        if peak > 1.0:
            waveform = waveform / peak
    return waveform.astype(np.float32, copy=False)



class SyntheticSoundEventDetectionDataset(Dataset):
    """
    Variable-length synthetic SED dataset.

    Parameters
    ----------
    size:
        Number of mixtures.
    num_events:
        Either an integer or ``(min_events, max_events)``.
    snr_db:
        Either a fixed per-event foreground/background SNR or
        ``(min_db, max_db)``. A separate SNR is sampled for every foreground
        event and applied against the background segment under that event.
    split:
        ``train``, ``validation``, and ``test`` are disjoint. ESC-50 uses
        folds 1-3 / 4 / 5, GSC uses deterministic speaker-grouped 60/20/20
        partitions, NSynth uses its native train/valid/test partitions, and
        FreeSound backgrounds use deterministic 60/20/20 partitions keyed by
        the FreeSound ID at the start of each filename.
    """

    def __init__(
        self,
        foreground_dataset: str = "esc50",
        root: str | Path | None = None,
        background_mode: str = "freesound",
        background_root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = 16000,
        min_length_sec: float = 30.0,
        max_length_sec: float = 30.0,
        size: int = 1000,
        num_events: int | tuple[int, int] = 1,
        unique_event_classes: bool = False,
        snr_db: float | tuple[float, float] = (-5.0, 0.0),
        max_event_length_sec: float | None = None,
        trim_silence: bool = True,
        trim_top_db: float = 40.0,
        background_allowlist_path: str | Path | None = None,
        require_background_allowlist: bool = True,
        random_seed: int = 0,
        auto_download: bool = True,
    ):
        self.foreground_dataset = foreground_dataset
        self.root = Path(root or DEFAULT_DATASET_ROOT)
        self.background_mode = background_mode
        self.background_root = self._resolve_background_root(background_root)
        self.split = SPLIT_ALIASES.get(split, split)
        self.sample_rate = int(sample_rate)
        self.min_length_sec = float(min_length_sec)
        self.max_length_sec = float(max_length_sec)
        self.size = int(size)
        self.trim_silence = bool(trim_silence)
        self.trim_top_db = float(trim_top_db)
        self.background_allowlist_path = (
            None
            if background_allowlist_path is None
            else Path(background_allowlist_path)
        )
        self.require_background_allowlist = bool(require_background_allowlist)
        self.max_event_length_sec = (
            None if max_event_length_sec is None else float(max_event_length_sec)
        )
        self.unique_event_classes = bool(unique_event_classes)
        self.max_event_samples = (
            None
            if self.max_event_length_sec is None
            else int(round(self.max_event_length_sec * sample_rate))
        )
        self.random_seed = int(random_seed)
        self.auto_download = bool(auto_download)

        self.num_event_range = _as_range(num_events, "num_events")
        self.snr_range = _as_range(snr_db, "snr_db")

        # Audio is decoded/resampled at most once per path. Foreground processing
        # (silence trimming and optional truncation) is cached separately.
        self._audio_cache: dict[str, np.ndarray] = {}
        self._foreground_audio_cache: dict[str, np.ndarray] = {}

        self._validate_config()
        if int(self.num_event_range[1]) > 0:
            self.sources = self._load_foreground_sources(auto_download=auto_download)
            if not self.sources:
                raise ValueError(
                    f"No foreground samples found for {foreground_dataset!r} split {split!r}"
                )
        else:
            self.sources = []

        self.source_labels = sorted({source["label"] for source in self.sources})
        self.forbidden_background_terms = self._background_forbidden_terms(
            self.source_labels,
        )
        if self.background_mode == "silence":
            self.require_background_allowlist = False
        self.background_allowlist = (
            None
            if self.background_mode == "silence"
            else self._load_background_allowlist()
        )
        self.filter_background_metadata = (
            self.background_mode != "silence" and self.background_allowlist is None
        )
        self.backgrounds = self._load_background_sources()
        if not self.backgrounds:
            raise ValueError(
                f"No background files found for split {self.split!r} under {self.background_root!r} "
                "after leakage filtering"
            )

        self.examples = self._build_examples()
        print(
            f"Loaded {len(self.examples)} synthetic SED samples "
            f"from {len(self.sources)} foreground and {len(self.backgrounds)} background sources."
        )

    def _validate_config(self) -> None:
        if self.foreground_dataset not in DATASET_SUBDIRS:
            raise ValueError("foreground_dataset must be one of: esc50, nsynth, gsc")
        if self.background_mode not in BACKGROUND_MODES:
            raise ValueError("background_mode must be one of: freesound, silence")
        if self.split not in {"train", "validation", "test", "all"}:
            raise ValueError("split must be one of: train, validation, test, all")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.min_length_sec <= 0 or self.max_length_sec <= 0:
            raise ValueError("mixture lengths must be positive")
        if self.min_length_sec > self.max_length_sec:
            raise ValueError("min_length_sec must not exceed max_length_sec")
        if self.min_length_sec < 10.0 or self.max_length_sec > 60.0:
            raise ValueError("mixture lengths must stay within 10-60 seconds")
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.num_event_range[0] < 0:
            raise ValueError("num_events cannot be negative")
        if int(self.num_event_range[0]) != self.num_event_range[0]:
            raise ValueError("num_events values must be integers")
        if int(self.num_event_range[1]) != self.num_event_range[1]:
            raise ValueError("num_events values must be integers")
        if self.max_event_samples is not None and self.max_event_samples <= 0:
            raise ValueError("max_event_length_sec must be positive when provided")

    def _resolve_background_root(self, background_root: str | Path | None) -> Path:
        if background_root is not None:
            return Path(background_root)
        return DEFAULT_BACKGROUND_ROOT

    def _dataset_root(self, dataset: str) -> Path:
        subdir = DATASET_SUBDIRS[dataset]
        if self.root.name == subdir:
            return self.root
        return self.root / subdir

    def _load_foreground_sources(
        self,
        auto_download: bool,
    ) -> list[dict[str, Any]]:
        if self.foreground_dataset == "esc50":
            return self._load_esc50_sources(auto_download)
        if self.foreground_dataset == "nsynth":
            return self._load_nsynth_sources(auto_download)
        return self._load_gsc_sources(auto_download)

    def _load_esc50_sources(self, auto_download: bool) -> list[dict[str, Any]]:
        root = self._dataset_root("esc50")
        esc_root = Path(download_esc50(str(root))) if auto_download else root / "ESC-50-master"
        folds = None
        if self.split == "train":
            folds = {1, 2, 3}
        elif self.split == "validation":
            folds = {4}
        elif self.split in {"test", "testing"}:
            folds = {5}

        sources = []
        with (esc_root / "meta" / "esc50.csv").open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row["category"]
                if folds is not None and int(row["fold"]) not in folds:
                    continue
                path = esc_root / "audio" / row["filename"]
                if path.exists():
                    sources.append({
                        "id": row["filename"],
                        "path": str(path),
                        "label": label,
                        "split_key": f"esc50-fold-{row['fold']}",
                        "target": int(row["target"]),
                        "fold": int(row["fold"]),
                    })
        return sources

    def _load_nsynth_sources(self, auto_download: bool) -> list[dict[str, Any]]:
        split = NSYNTH_SPLIT_ALIASES.get(self.split, self.split)
        splits = ("train", "valid", "test") if split == "all" else (split,)
        root = self._dataset_root("nsynth")

        sources = []
        for native_split in splits:
            nsynth_root = (
                Path(download_nsynth(native_split, str(root)))
                if auto_download
                else root / f"nsynth-{native_split}"
            )
            with (nsynth_root / "examples.json").open() as handle:
                metadata = json.load(handle)
            audio_dir = nsynth_root / "audio"
            for note_id, meta in metadata.items():
                family = meta.get("instrument_family_str", "music note")
                source = meta.get("instrument_source_str", "")
                label = f"{source} {family}".strip() or "music note"
                path = audio_dir / f"{note_id}.wav"
                if path.exists():
                    pitch = meta["pitch"]
                    frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
                    sources.append({
                        "id": note_id,
                        "path": str(path),
                        "label": label,
                        "split": "validation" if native_split == "valid" else native_split,
                        "split_key": f"nsynth-{native_split}",
                        "pitch": pitch,
                        "frequency": frequency,
                        "family": family,
                        "source": source,
                        "instrument": meta.get("instrument"),
                    })
        return sorted(sources, key=lambda source: source["id"])

    def _partition_keys(self, keys: list[str] | set[str]) -> dict[str, str]:
        """Assign stable keys to deterministic 60/20/20 train/validation/test splits."""
        ordered = sorted(set(keys), key=_stable_sort_key)
        n = len(ordered)
        if n == 0:
            return {}

        n_train = int(round(SPLIT_RATIOS["train"] * n))
        n_validation = int(round(SPLIT_RATIOS["validation"] * n))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_validation = min(max(1, n_validation), n - n_train - 1)
        else:
            n_train = min(n, max(1, n_train))
            n_validation = min(n - n_train, max(0, n_validation))
        n_test = n - n_train - n_validation

        assignments: dict[str, str] = {}
        for key in ordered[:n_train]:
            assignments[key] = "train"
        for key in ordered[n_train:n_train + n_validation]:
            assignments[key] = "validation"
        for key in ordered[n_train + n_validation:n_train + n_validation + n_test]:
            assignments[key] = "test"
        return assignments

    @staticmethod
    def _gsc_speaker_id(path: Path) -> str:
        stem = path.stem
        return stem.split("_nohash_", 1)[0] if "_nohash_" in stem else stem.split("_", 1)[0]

    def _load_gsc_sources(self, auto_download: bool) -> list[dict[str, Any]]:
        root = self._dataset_root("gsc")
        gsc_root = Path(download_gsc(str(root))) if auto_download else root

        candidates: list[tuple[str, Path, str, str]] = []
        speaker_ids: set[str] = set()
        for label_dir in sorted(gsc_root.iterdir()):
            label = label_dir.name
            if not label_dir.is_dir() or label.startswith("_"):
                continue
            for path in sorted(label_dir.glob("*.wav")):
                rel_path = f"{label}/{path.name}"
                speaker_id = self._gsc_speaker_id(path)
                speaker_ids.add(speaker_id)
                candidates.append((rel_path, path, label, speaker_id))

        split_by_speaker = self._partition_keys(speaker_ids)
        sources = []
        for rel_path, path, label, speaker_id in candidates:
            split = split_by_speaker[speaker_id]
            if self.split != "all" and split != self.split:
                continue
            sources.append({
                "id": rel_path,
                "path": str(path),
                "label": label,
                "split": split,
                "split_key": f"gsc-speaker-{speaker_id}",
                "speaker_id": speaker_id,
            })
        return sources

    def _background_forbidden_terms(
        self,
        labels: list[str],
    ) -> list[str]:
        terms = set()
        for label in labels:
            terms.add(label)
            terms.add(_display_label(label))
            for token in re.split(r"[\s_-]+", label):
                if len(token) >= 4:
                    terms.add(token)
        if self.foreground_dataset == "nsynth":
            terms.update({"music", "note", "instrument", "synth", "piano", "guitar"})
        return sorted({term.casefold() for term in terms if term})

    def _load_background_sources(self) -> list[dict[str, Any]]:
        if self.background_mode == "silence":
            return [{
                "id": "silence",
                "path": "",
                "category": "silence",
                "type": "silence",
                "metadata_leakage_terms": [],
                "allowed_event_classes": None,
            }]

        metadata = self._load_background_metadata()
        metadata_by_path = {
            str((self.background_root / row["local_file"]).resolve()): row
            for row in metadata
            if row.get("local_file")
        }

        paths = [
            path
            for path in sorted(self.background_root.rglob("*"))
            if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
        ]
        if not paths:
            return []

        split_paths = self._split_background_paths(paths)
        backgrounds = []
        for path in split_paths:
            row = metadata_by_path.get(str(path.resolve()), {})
            blob = self._background_text(path, row)
            hits = self._metadata_leakage_terms(blob, self.forbidden_background_terms)
            if hits and self.filter_background_metadata:
                continue
            background = {
                "id": str(path.relative_to(self.background_root)),
                "path": str(path),
                "category": row.get("category", path.parent.name),
                "metadata_leakage_terms": hits,
            }
            background["allowed_event_classes"] = self._allowed_event_classes_for_background(
                background,
            )
            if self.require_background_allowlist and background["allowed_event_classes"] is None:
                continue
            if int(self.num_event_range[1]) > 0 and not self._eligible_sources_for_background(background):
                continue
            backgrounds.append(background)
        return backgrounds

    def _load_background_metadata(self) -> list[dict[str, Any]]:
        json_path = self.background_root / "metadata.json"
        csv_path = self.background_root / "metadata.csv"
        if json_path.exists():
            with json_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        if csv_path.exists():
            with csv_path.open(encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        return []

    def _load_background_allowlist(self) -> dict[str, Any] | None:
        candidates = []
        if self.background_allowlist_path is not None:
            candidates.append(self.background_allowlist_path)
        candidates.extend([
            self.background_root / "allowed_event_classes.json",
            self.background_root.parent / "allowed_event_classes.json",
        ])
        for path in candidates:
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    data = json.load(handle)
                data["_path"] = str(path)
                return data
        if self.require_background_allowlist:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"Could not find background allowlist. Searched: {searched}")
        return None

    def _allowed_event_classes_for_background(self, background: dict[str, Any]) -> Any:
        if self.background_allowlist is None:
            return None

        entries = self.background_allowlist.get("backgrounds", {})
        keys = [
            background["id"],
            background["path"],
            str(Path(background["path"]).resolve()),
        ]
        for key in keys:
            if key in entries:
                return entries[key].get("allowed_event_classes")

        categories = self.background_allowlist.get("categories", {})
        category = background.get("category")
        if category in categories:
            return categories[category].get("allowed_event_classes")
        return None

    def _allowed_labels_for_background(self, background: dict[str, Any]) -> set[str] | None:
        if self.foreground_dataset in UNRESTRICTED_ALLOWLIST_DATASETS:
            return None
        allowed_event_classes = background.get("allowed_event_classes")
        if allowed_event_classes is None:
            return None
        allowed = allowed_event_classes.get(self.foreground_dataset)
        if allowed is None:
            return set()
        if isinstance(allowed, dict):
            return {label for label, is_allowed in allowed.items() if is_allowed}
        return set(allowed)

    def _source_matches_allowed_labels(
        self,
        source: dict[str, Any],
        allowed_labels: set[str],
    ) -> bool:
        labels = {
            source["label"],
            _display_label(source["label"]),
        }
        if self.foreground_dataset == "nsynth":
            labels.update({
                source.get("family", ""),
                source.get("source", ""),
                "music note",
            })
        return bool(labels & allowed_labels)

    def _eligible_sources_for_background(self, background: dict[str, Any]) -> list[dict[str, Any]]:
        allowed_labels = self._allowed_labels_for_background(background)
        if allowed_labels is None:
            return self.sources
        return [
            source
            for source in self.sources
            if self._source_matches_allowed_labels(source, allowed_labels)
        ]

    def _unique_source_labels(self, sources: list[dict[str, Any]]) -> set[str]:
        return {source["label"] for source in sources}

    def _backgrounds_with_enough_unique_event_classes(
        self,
        num_events: int,
    ) -> list[dict[str, Any]]:
        return [
            background
            for background in self.backgrounds
            if len(
                self._unique_source_labels(
                    self._eligible_sources_for_background(background),
                ),
            ) >= num_events
        ]

    def _sample_unique_class_sources(
        self,
        eligible_sources: list[dict[str, Any]],
        num_events: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        sources_by_label: dict[str, list[dict[str, Any]]] = {}
        for source in eligible_sources:
            sources_by_label.setdefault(source["label"], []).append(source)

        labels = list(sources_by_label)
        if len(labels) < num_events:
            raise ValueError(
                f"Cannot sample {num_events} unique event classes from "
                f"{len(labels)} eligible {self.foreground_dataset} class(es)"
            )

        label_indices = rng.permutation(len(labels))[:num_events]
        sampled_sources = []
        for label_index in label_indices:
            label_sources = sources_by_label[labels[int(label_index)]]
            source_index = int(rng.integers(0, len(label_sources)))
            sampled_sources.append(label_sources[source_index])
        return sampled_sources

    @staticmethod
    def _freesound_id(path: Path) -> str:
        """Return the FreeSound ID encoded before the first underscore."""
        return path.stem.split("_", 1)[0]

    def _split_background_paths(self, paths: list[Path]) -> list[Path]:
        if self.split == "all":
            return paths

        # Split by FreeSound ID, not by absolute or relative pathname. This keeps
        # split membership stable if the dataset directory is moved and keeps all
        # files derived from the same FreeSound item in the same partition.
        freesound_ids = {self._freesound_id(path) for path in paths}
        split_by_id = self._partition_keys(freesound_ids)
        return [
            path
            for path in paths
            if split_by_id[self._freesound_id(path)] == self.split
        ]

    def _background_text(self, path: Path, row: dict[str, Any]) -> str:
        pieces = [
            path.stem,
            path.parent.name,
            str(row.get("category", "")),
            str(row.get("category_label", "")),
            str(row.get("name", "")),
            str(row.get("description", "")),
            str(row.get("tags", "")),
            str(row.get("leakage_terms", "")),
        ]
        return " ".join(pieces).casefold()

    def _metadata_leakage_terms(self, text: str, terms: list[str]) -> list[str]:
        return [term for term in terms if _term_pattern(term).search(text)]

    def _build_examples(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.random_seed)
        examples = []
        for base_index in range(self.size):
            examples.append(self._build_base_example(base_index, rng))
        return examples

    def _build_base_example(self, base_index: int, rng: np.random.Generator) -> dict[str, Any]:
        length_sec = float(rng.uniform(self.min_length_sec, self.max_length_sec))
        num_samples = int(round(length_sec * self.sample_rate))
        num_events = int(rng.integers(
            int(self.num_event_range[0]),
            int(self.num_event_range[1]) + 1,
        ))
        unique_event_classes = getattr(self, "unique_event_classes", False)
        background_candidates = self.backgrounds
        if unique_event_classes and num_events > 0:
            background_candidates = self._backgrounds_with_enough_unique_event_classes(
                num_events,
            )
            if not background_candidates:
                raise ValueError(
                    f"Cannot sample {num_events} unique event classes from any background "
                    f"for {self.foreground_dataset!r} split {self.split!r}"
                )
        background = background_candidates[int(rng.integers(0, len(background_candidates)))]
        background_offset_fraction = float(rng.random())
        eligible_sources = self._eligible_sources_for_background(background)
        if num_events > 0 and not eligible_sources:
            raise ValueError(
                f"Background {background['id']!r} has no allowed {self.foreground_dataset} "
                "foreground classes"
            )

        background_is_silence = background.get("type") == "silence" or not background.get("path")
        events = []
        if unique_event_classes:
            sampled_sources = self._sample_unique_class_sources(eligible_sources, num_events, rng)
        else:
            sampled_sources = []
            source_indices = list(rng.permutation(len(eligible_sources)))
            for _event_index in range(num_events):
                if source_indices:
                    source = eligible_sources[int(source_indices.pop())]
                else:
                    source = eligible_sources[int(rng.integers(0, len(eligible_sources)))]
                sampled_sources.append(source)

        for event_index in range(num_events):
            source = sampled_sources[event_index]
            event_samples = min(self._event_sample_count(source["path"]), num_samples)
            onset_samples = self._sample_event_onset(rng, num_samples, event_samples)
            offset_samples = min(num_samples, onset_samples + event_samples)
            snr_db = (
                math.inf
                if background_is_silence
                else float(rng.uniform(self.snr_range[0], self.snr_range[1]))
            )
            events.append({
                "event_index": event_index,
                "source": source,
                "onset_samples": onset_samples,
                "offset_samples": offset_samples,
                "duration_samples": offset_samples - onset_samples,
                "snr_db": snr_db,
            })

        return {
            "id": f"sed_{self.foreground_dataset}_{self.split}_{base_index:06d}",
            "base_index": base_index,
            "length_sec": num_samples / self.sample_rate,
            "num_samples": num_samples,
            "background": background,
            "background_offset_fraction": background_offset_fraction,
            "events": events,
        }

    def _get_audio(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if key not in self._audio_cache:
            self._audio_cache[key] = _load_audio_mono(path, self.sample_rate)
        return self._audio_cache[key]

    def _get_foreground_audio(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if key not in self._foreground_audio_cache:
            audio = self._get_audio(path)
            if self.trim_silence:
                audio = trim_leading_trailing_silence(audio, self.trim_top_db)
            if self.max_event_samples is not None:
                audio = audio[:self.max_event_samples]
            self._foreground_audio_cache[key] = audio.astype(np.float32, copy=False)
        return self._foreground_audio_cache[key]

    def _event_sample_count(self, path: str) -> int:
        return len(self._get_foreground_audio(path))

    @staticmethod
    def _sample_event_onset(
        rng: np.random.Generator,
        num_samples: int,
        event_samples: int,
    ) -> int:
        max_onset = max(0, num_samples - event_samples)
        return int(rng.integers(0, max_onset + 1)) if max_onset > 0 else 0

    def __len__(self) -> int:
        return len(self.examples)

    def _load_event_audio(self, event: dict[str, Any]) -> np.ndarray:
        audio = self._get_foreground_audio(event["source"]["path"])
        return audio[:event["duration_samples"]].astype(np.float32, copy=False)

    def _load_background_audio(self, example: dict[str, Any]) -> np.ndarray:
        num_samples = example["num_samples"]
        path = example["background"]["path"]
        if example["background"].get("type") == "silence" or not path:
            return np.zeros(num_samples, dtype=np.float32)
        audio = self._get_audio(path)
        if len(audio) == 0:
            raise ValueError(f"Background audio is empty: {path}")
        if len(audio) < num_samples:
            repeats = int(math.ceil(num_samples / len(audio)))
            audio = np.tile(audio, repeats)
        max_offset = len(audio) - num_samples
        offset = int(round(example["background_offset_fraction"] * max_offset)) if max_offset > 0 else 0
        return audio[offset:offset + num_samples].astype(np.float32, copy=False)

    def _scale_event_to_snr(
        self,
        event_audio: np.ndarray,
        background_segment: np.ndarray,
        snr_db: float,
    ) -> np.ndarray:
        # +inf is the explicit SNR convention for foreground over digital silence.
        if math.isinf(snr_db) or not np.any(background_segment):
            return event_audio
        event_rms = _rms(event_audio)
        background_rms = _rms(background_segment)
        if event_rms <= 0.0 or background_rms <= 0.0:
            return event_audio
        target_event_rms = background_rms * (10.0 ** (snr_db / 20.0))
        return event_audio * (target_event_rms / event_rms)

    def _load_scaled_event_audios(
        self,
        background: np.ndarray,
        events: list[dict[str, Any]],
    ) -> list[np.ndarray]:
        foregrounds = []
        for event in events:
            event_audio = self._load_event_audio(event)
            onset = event["onset_samples"]
            offset = min(len(background), onset + len(event_audio))
            if offset > onset:
                event_audio = event_audio[:offset - onset]
                event_audio = self._scale_event_to_snr(
                    event_audio,
                    background,
                    event["snr_db"],
                )
            else:
                event_audio = event_audio[:0]
            foregrounds.append(event_audio.astype(np.float32, copy=False))
        return foregrounds

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        background = self._load_background_audio(example)
        foregrounds = self._load_scaled_event_audios(
            background,
            example["events"],
        )
        foreground_onset_samples = [
            event["onset_samples"]
            for event in example["events"]
        ]
        waveform = mix_sound(background, foregrounds, foreground_onset_samples)

        events = []
        for event in example["events"]:
            source = event["source"]
            events.append({
                "event_index": event["event_index"],
                "event_label": source["label"],
                "onset_samples": event["onset_samples"],
                "offset_samples": event["offset_samples"],
                "onset_sec": event["onset_samples"] / self.sample_rate,
                "offset_sec": event["offset_samples"] / self.sample_rate,
                "duration_sec": (event["offset_samples"] - event["onset_samples"]) / self.sample_rate,
                "snr_db": event["snr_db"],
                "source_id": source["id"],
                "source_path": source["path"],
            })

        return {
            "id": example["id"],
            "dataset": f"synthetic_sed_{self.foreground_dataset}",
            "split": self.split,
            "waveform": waveform.astype(np.float32, copy=False),
            "background_waveform": background.astype(np.float32, copy=False),
            "foreground_waveforms": foregrounds,
            "foreground_onset_samples": foreground_onset_samples,
            "sample_rate": self.sample_rate,
            "length_sec": example["num_samples"] / self.sample_rate,
            "event_snr_db": [event["snr_db"] for event in example["events"]],
            "background_id": example["background"]["id"],
            "background_path": example["background"]["path"],
            "background_allowed_event_classes": example["background"].get("allowed_event_classes"),
            "events": events,
        }



def _parse_range(values: list[str], cast: Any) -> Any:
    parsed = [cast(value) for value in values]
    if len(parsed) == 1:
        return parsed[0]
    if len(parsed) == 2:
        return (parsed[0], parsed[1])
    raise argparse.ArgumentTypeError("expected one or two values")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default="SynthSED")
    parser.add_argument("--foreground-dataset", choices=sorted(DATASET_SUBDIRS), default="esc50")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--background-mode",
        choices=BACKGROUND_MODES,
        default="freesound",
        help="background source mode: freesound ambience files or silence",
    )
    parser.add_argument("--background-root", type=Path, default=DEFAULT_BACKGROUND_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test", "testing", "all"], default="train")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-length-sec", type=float, default=30.0)
    parser.add_argument("--max-length-sec", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--num-events", nargs="+", default=["1"], help="one fixed value or min max")
    parser.add_argument(
        "--unique-event-classes",
        action="store_true",
        help="sample at most one foreground event from each event class per mixture",
    )
    parser.add_argument("--snr-db", nargs="+", default=["-5.0", "0.0"], help="one fixed per-event value or min max")
    parser.add_argument("--max-event-length-sec", type=float)
    parser.add_argument("--no-trim-silence", action="store_true")
    parser.add_argument("--trim-top-db", type=float, default=40.0)
    parser.add_argument("--background-allowlist-path", type=Path)
    parser.add_argument("--require-background-allowlist", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--no-auto-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = SyntheticSoundEventDetectionDataset(
        foreground_dataset=args.foreground_dataset,
        root=args.root,
        background_mode=args.background_mode,
        background_root=args.background_root,
        split=args.split,
        sample_rate=args.sample_rate,
        min_length_sec=args.min_length_sec,
        max_length_sec=args.max_length_sec,
        size=args.size,
        num_events=_parse_range(args.num_events, int),
        unique_event_classes=args.unique_event_classes,
        snr_db=_parse_range(args.snr_db, float),
        max_event_length_sec=args.max_event_length_sec,
        trim_silence=not args.no_trim_silence,
        trim_top_db=args.trim_top_db,
        background_allowlist_path=args.background_allowlist_path,
        require_background_allowlist=args.require_background_allowlist,
        random_seed=args.random_seed,
        auto_download=not args.no_auto_download,
    )
    print(f"Wrote {len(dataset)} {dataset.split} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
