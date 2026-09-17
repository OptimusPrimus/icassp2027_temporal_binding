import argparse
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "synthetic_examples"
FOREGROUND_DATASET = "esc50"
CLIP_LENGTH_SEC = 10.0
FIRST_ONSET_SEC = 0.0
SECOND_ONSET_SEC = 5.0
NUM_EVENTS = 2


class FixedPositionSyntheticSedOrderingDataset(Dataset):
    """Synthetic SED ordering examples with fixed 0s and 5s foreground onsets."""

    def __init__(
        self,
        root: str | Path,
        sample_rate: int = 16000,
        size: int = 1000,
        random_seed: int = 0,
        auto_download: bool = True,
    ):
        from dataset.synthetic_sed import SyntheticSoundEventDetectionDataset, mix_sound

        self.sample_rate = int(sample_rate)
        self._mix_sound = mix_sound
        self.first_onset_samples = int(round(FIRST_ONSET_SEC * self.sample_rate))
        self.second_onset_samples = int(round(SECOND_ONSET_SEC * self.sample_rate))
        self.clip_samples = int(round(CLIP_LENGTH_SEC * self.sample_rate))
        self.event_samples = self.second_onset_samples - self.first_onset_samples

        self.sed_dataset = SyntheticSoundEventDetectionDataset(
            foreground_dataset=FOREGROUND_DATASET,
            root=root,
            split="train",
            sample_rate=self.sample_rate,
            min_length_sec=CLIP_LENGTH_SEC,
            max_length_sec=CLIP_LENGTH_SEC,
            size=size,
            num_events=NUM_EVENTS,
            unique_event_classes=True,
            max_event_length_sec=SECOND_ONSET_SEC,
            trim_silence=False,
            random_seed=random_seed,
            auto_download=auto_download,
        )

    def __len__(self):
        return len(self.sed_dataset)

    @staticmethod
    def _display_label(label):
        return str(label).replace("_", " ")

    def _fixed_event_audio(self, event):
        audio = self.sed_dataset._get_foreground_audio(event["source"]["path"])
        audio = audio[: self.event_samples].astype(np.float32, copy=False)
        if len(audio) >= self.event_samples:
            return audio

        padded = np.zeros(self.event_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded

    def _scaled_event_audio(self, event, background):
        event_audio = self._fixed_event_audio(event)
        return self.sed_dataset._scale_event_to_snr(
            event_audio,
            background,
            event["snr_db"],
        ).astype(np.float32, copy=False)

    def _mix(self, background, first_audio, second_audio):
        return self._mix_sound(
            background,
            [first_audio, second_audio],
            [self.first_onset_samples, self.second_onset_samples],
        )

    def __getitem__(self, idx):
        example = self.sed_dataset.examples[idx]
        events = example["events"]
        if len(events) != NUM_EVENTS:
            raise ValueError(f"Expected {NUM_EVENTS} events, got {len(events)}")

        background = self.sed_dataset._load_background_audio(example)
        if len(background) != self.clip_samples:
            background = background[: self.clip_samples]
            if len(background) < self.clip_samples:
                padded = np.zeros(self.clip_samples, dtype=np.float32)
                padded[: len(background)] = background
                background = padded

        first_event, second_event = events
        first_audio = self._scaled_event_audio(first_event, background)
        second_audio = self._scaled_event_audio(second_event, background)

        waveform = self._mix(background, first_audio, second_audio)
        swapped_waveform = self._mix(background, second_audio, first_audio)

        query_is_first = (idx % 2) == 0
        query_event = first_event if query_is_first else second_event
        reference_event = second_event if query_is_first else first_event

        return {
            "id": example["id"],
            "waveform": waveform,
            "swapped_waveform": swapped_waveform,
            "sample_rate": self.sample_rate,
            "query_label": self._display_label(query_event["source"]["label"]),
            "reference_label": self._display_label(reference_event["source"]["label"]),
            "answer": "before" if query_is_first else "after",
            "first_label": self._display_label(first_event["source"]["label"]),
            "second_label": self._display_label(second_event["source"]["label"]),
            "first_source_id": first_event["source"]["id"],
            "second_source_id": second_event["source"]["id"],
            "background_id": example["background"]["id"],
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize fixed-position SyntheticSED ordering examples as WAV "
            "files plus metadata."
        )
    )
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=5,
        help="Number of synthetic examples to save.",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--random-seed", type=int, default=15)
    parser.add_argument("--no-auto-download", action="store_true")
    return parser.parse_args()


def sample_metadata(sample, waveform_path, swapped_waveform_path):
    return {
        "id": sample["id"],
        "waveform_path": str(waveform_path),
        "swapped_waveform_path": str(swapped_waveform_path),
        "sample_rate": sample["sample_rate"],
        "query_label": sample["query_label"],
        "reference_label": sample["reference_label"],
        "answer": sample["answer"],
        "first_label": sample["first_label"],
        "second_label": sample["second_label"],
        "first_source_id": sample["first_source_id"],
        "second_source_id": sample["second_source_id"],
        "background_id": sample["background_id"],
        "clip_length_sec": CLIP_LENGTH_SEC,
        "first_onset_sec": FIRST_ONSET_SEC,
        "second_onset_sec": SECOND_ONSET_SEC,
    }


def save_examples(args):
    if args.num_examples < 1:
        raise ValueError("--num-examples must be positive")

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import soundfile as sf

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FixedPositionSyntheticSedOrderingDataset(
        root=args.dataset_root,
        sample_rate=args.sample_rate,
        size=args.num_examples,
        random_seed=args.random_seed,
        auto_download=not args.no_auto_download,
    )

    metadata_rows = []
    for idx in range(args.num_examples):
        sample = dataset[idx]
        stem = f"synthetic_ordering_{idx:04d}_{sample['id']}"
        waveform_path = output_dir / f"{stem}.wav"
        swapped_waveform_path = output_dir / f"{stem}_swapped.wav"
        metadata_path = output_dir / f"{stem}.json"

        sf.write(waveform_path, sample["waveform"], sample["sample_rate"])
        sf.write(swapped_waveform_path, sample["swapped_waveform"], sample["sample_rate"])

        metadata = sample_metadata(sample, waveform_path, swapped_waveform_path)
        with metadata_path.open("w") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        metadata_rows.append({**metadata, "metadata_path": str(metadata_path)})

    metadata_jsonl_path = output_dir / "metadata.jsonl"
    with metadata_jsonl_path.open("w") as handle:
        for row in metadata_rows:
            json.dump(row, handle)
            handle.write("\n")

    print(f"Wrote {args.num_examples} examples to {output_dir}")
    print(f"Wrote metadata index: {metadata_jsonl_path}")
    return output_dir


def main():
    save_examples(parse_args())


if __name__ == "__main__":
    main()
