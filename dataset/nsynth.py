import os
import json
import tarfile
import urllib.request
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


# ============================================================
# NSYNTH DOWNLOAD
# ============================================================

NSYNTH_URLS = {
    "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
    "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
}

ROOT = "dataset/NSynth"


def download_nsynth(split="test", root=ROOT):
    """
    Download NSynth split automatically.

    Parameters
    ----------
    split : str
        One of: train / valid / test
    root : str
        Download directory
    """

    os.makedirs(root, exist_ok=True)

    if split not in NSYNTH_URLS:
        raise ValueError(f"Unknown split: {split}")

    url = NSYNTH_URLS[split]

    archive_path = os.path.join(root, f"nsynth-{split}.tar.gz")
    extract_path = os.path.join(root, f"nsynth-{split}")

    if os.path.exists(extract_path):
        print(f"NSynth {split} already downloaded.")
        return extract_path

    print(f"Downloading NSynth {split} split...")

    urllib.request.urlretrieve(url, archive_path)

    print("Extracting...")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)

    os.remove(archive_path)

    print("Done.")

    return extract_path


# ============================================================
# DATASET
# ============================================================

class NSynthDataset(Dataset):
    """
    Generic NSynth dataset.

    Supports filtering by:
    - instrument family
    - source
    - velocity
    - pitch range
    - instrument id
    """

    def __init__(
        self,
        split="test",
        root=ROOT,
        instrument_family=None,
        instrument_source=None,
        velocity=None,
        pitch_range=(21, 108),
        instrument_id=None,
        sample_rate=16000,
        length_sec=5.0,
        auto_download=True,
        **kwargs
    ):
        del kwargs

        self.sample_rate = sample_rate
        self.length_sec = length_sec
        self.num_samples = int(sample_rate * length_sec)
        if auto_download:
            self.nsynth_root = download_nsynth(split=split, root=root)
        else:
            self.nsynth_root = os.path.join(root, f"nsynth-{split}")

        metadata_path = os.path.join(self.nsynth_root, "examples.json")
        audio_dir = os.path.join(self.nsynth_root, "audio")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.samples = []

        low_pitch, high_pitch = pitch_range

        for note_id, meta in metadata.items():

            pitch = meta["pitch"]

            if not (low_pitch <= pitch <= high_pitch):
                continue

            if velocity is not None and meta["velocity"] != velocity:
                continue

            if instrument_family is not None:
                if meta["instrument_family_str"] != instrument_family:
                    continue

            if instrument_source is not None:
                if meta["instrument_source_str"] != instrument_source:
                    continue

            if instrument_id is not None:
                if meta["instrument"] != instrument_id:
                    continue

            audio_path = os.path.join(audio_dir, f"{note_id}.wav")

            if not os.path.exists(audio_path):
                continue

            freq = 440.0 * (2.0 ** ((pitch - 69) / 12.0))

            self.samples.append({
                "id": note_id,
                "path": audio_path,
                "pitch": pitch,
                "frequency": freq,
                "velocity": meta["velocity"],
                "family": meta["instrument_family_str"],
                "source": meta["instrument_source_str"],
                "instrument": meta["instrument"]
            })

        self.samples = sorted(self.samples, key=lambda x: x["pitch"])

        print(f"Loaded {len(self.samples)} NSynth samples.")

    def __len__(self):
        return len(self.samples)

    def load_audio(self, path):

        audio, _ = librosa.load(path, sr=self.sample_rate)

        if len(audio) >= self.num_samples:
            raise ValueError(f"Audio file {path} is longer than {self.length_sec} seconds.")
        elif len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))

        return audio

    def __getitem__(self, idx):

        meta = self.samples[idx]

        waveform = self.load_audio(meta["path"])

        return {
            "id": meta["id"],
            "waveform": waveform,
            "sample_rate": self.sample_rate,
            "frequency": meta["frequency"],
            "midi": meta["pitch"],
            "velocity": meta["velocity"],
            "family": meta["family"],
            "source": meta["source"],
            "instrument": meta["instrument"]
        }

if __name__ == "__main__":
    download_nsynth(split="test", root=ROOT)
    download_nsynth(split="train", root=ROOT)
    download_nsynth(split="val", root=ROOT)
