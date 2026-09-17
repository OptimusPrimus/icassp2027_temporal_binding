#!/usr/bin/env python3
"""RealDESED sound event detection dataset.

The dataset expects the release layout:

    root/{train,validation,test}/audio/*.wav
    root/{train,validation,test}/annotations.csv

``annotations.csv`` must contain aggregated event rows with
``filename,class,onset,offset`` columns.
"""

from __future__ import annotations

import csv
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from dataset.audio_utils import load_audio_mono as _load_audio_mono


DEFAULT_REALDESED_ROOT = Path(
    os.environ.get(
        "REALDESED_ROOT",
        "/home/paul/repos/domestic_sed_dataset/data/release",
    )
)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "dev": "validation",
    "testing": "test",
}


class RealDESEDDataset(Dataset):
    """Real domestic sound event detection recordings with event annotations."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "train",
        sample_rate: int = 16000,
        include_metadata: bool = True,
        cache_audio: bool = True,
        **kwargs: Any,
    ):
        del kwargs

        self.root = Path(root) if root is not None else DEFAULT_REALDESED_ROOT
        self.split = SPLIT_ALIASES.get(split, split)
        self.sample_rate = int(sample_rate)
        self.include_metadata = bool(include_metadata)
        self.cache_audio = bool(cache_audio)
        self._audio_cache: dict[str, np.ndarray] = {}

        self._validate_config()
        self.split_root = self.root / self.split
        self.audio_dir = self.split_root / "audio"
        self.annotations_path = self.split_root / "annotations.csv"
        self.metadata_path = self.split_root / "metadata.csv"

        self.annotations_by_filename = self._load_annotations()
        self.metadata_by_filename = self._load_metadata()
        self.examples = self._load_examples()
        if not self.examples:
            raise ValueError(
                f"No RealDESED audio files found for split {self.split!r} under {self.audio_dir!r}"
            )

        self.classes = sorted({
            event["event_label"]
            for events in self.annotations_by_filename.values()
            for event in events
        })

    def _validate_config(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    def _load_annotations(self) -> dict[str, list[dict[str, Any]]]:
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Could not find annotations: {self.annotations_path}")

        annotations: dict[str, list[dict[str, Any]]] = {}
        with self.annotations_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"filename", "class", "onset", "offset"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{self.annotations_path} is missing required columns: {sorted(missing)}"
                )

            for row in reader:
                filename = row["filename"]
                onset_sec = float(row["onset"])
                offset_sec = float(row["offset"])
                if offset_sec < onset_sec:
                    raise ValueError(
                        f"Invalid RealDESED event in {filename!r}: "
                        f"offset {offset_sec} is before onset {onset_sec}"
                    )
                annotations.setdefault(filename, []).append({
                    "event_label": row["class"],
                    "onset_sec": onset_sec,
                    "offset_sec": offset_sec,
                })

        for events in annotations.values():
            events.sort(key=lambda event: (event["onset_sec"], event["offset_sec"], event["event_label"]))
        return annotations

    def _load_metadata(self) -> dict[str, dict[str, str]]:
        if not self.include_metadata or not self.metadata_path.exists():
            return {}

        with self.metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "filename" not in (reader.fieldnames or []):
                return {}
            return {row["filename"]: dict(row) for row in reader}

    def _load_examples(self) -> list[dict[str, Any]]:
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Could not find RealDESED audio directory: {self.audio_dir}")

        examples = []
        for path in sorted(self.audio_dir.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in AUDIO_EXTENSIONS:
                continue
            filename = path.name
            examples.append({
                "id": path.stem,
                "filename": filename,
                "path": str(path),
                "duration_sec": self._audio_duration_sec(path),
                "events": self.annotations_by_filename.get(filename, []),
                "metadata": self.metadata_by_filename.get(filename, {}),
            })
        return examples

    @staticmethod
    def _audio_duration_sec(path: Path) -> float | None:
        if path.suffix.casefold() != ".wav":
            return None
        try:
            with wave.open(str(path), "rb") as handle:
                return handle.getnframes() / float(handle.getframerate())
        except (wave.Error, EOFError):
            return None

    def __len__(self) -> int:
        return len(self.examples)

    def _get_audio(self, path: str | Path) -> np.ndarray:
        key = str(path)
        if self.cache_audio and key in self._audio_cache:
            return self._audio_cache[key]

        audio = _load_audio_mono(path, self.sample_rate).astype(np.float32, copy=False)
        if self.cache_audio:
            self._audio_cache[key] = audio
        return audio

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        waveform = self._get_audio(example["path"])
        length_sec = len(waveform) / float(self.sample_rate)
        num_samples = len(waveform)

        events = []
        for event_index, event in enumerate(example["events"]):
            onset_samples = int(round(event["onset_sec"] * self.sample_rate))
            offset_samples = int(round(event["offset_sec"] * self.sample_rate))
            onset_samples = min(max(0, onset_samples), num_samples)
            offset_samples = min(max(onset_samples, offset_samples), num_samples)
            onset_sec = onset_samples / float(self.sample_rate)
            offset_sec = offset_samples / float(self.sample_rate)
            events.append({
                "event_index": event_index,
                "event_label": event["event_label"],
                "onset_samples": onset_samples,
                "offset_samples": offset_samples,
                "onset_sec": onset_sec,
                "offset_sec": offset_sec,
                "duration_sec": offset_sec - onset_sec,
                "annotation_onset_sec": event["onset_sec"],
                "annotation_offset_sec": event["offset_sec"],
                "source_id": example["filename"],
                "source_path": example["path"],
            })

        sample = {
            "id": example["id"],
            "dataset": "real_desed",
            "split": self.split,
            "waveform": waveform.astype(np.float32, copy=False),
            "sample_rate": self.sample_rate,
            "length_sec": length_sec,
            "events": events,
            "event_labels": [event["event_label"] for event in events],
            "path": example["path"],
            "filename": example["filename"],
        }
        if self.include_metadata:
            sample["metadata"] = example["metadata"]
        return sample
