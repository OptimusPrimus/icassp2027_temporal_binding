import os
import tarfile
import urllib.request

import librosa
import numpy as np
from torch.utils.data import Dataset


GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ROOT = "dataset/GSC"


def download_gsc(root=ROOT):
    """
    Download the Google Speech Commands v0.02 dataset automatically.
    """

    os.makedirs(root, exist_ok=True)

    marker_path = os.path.join(root, ".gsc_v0.02_complete")
    if os.path.exists(marker_path):
        print("Google Speech Commands already downloaded.")
        return root

    has_wavs = False
    for entry in os.listdir(root):
        entry_path = os.path.join(root, entry)
        if os.path.isdir(entry_path) and entry != "_background_noise_":
            if any(name.endswith(".wav") for name in os.listdir(entry_path)):
                has_wavs = True
                break

    if has_wavs:
        with open(marker_path, "w") as f:
            f.write("present\n")
        print("Google Speech Commands already present.")
        return root

    archive_path = os.path.join(root, "speech_commands_v0.02.tar.gz")

    print("Downloading Google Speech Commands v0.02...")
    urllib.request.urlretrieve(GSC_URL, archive_path)

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=root)

    os.remove(archive_path)

    with open(marker_path, "w") as f:
        f.write("downloaded\n")

    print("Done.")
    return root


class GSCDataset(Dataset):
    """
    Generic Google Speech Commands v0.02 dataset.

    Supports filtering by command label and official train/validation/testing
    split. Each item returns a fixed-length waveform, matching the ESC-50 and
    NSynth dataset classes in this package.
    """

    def __init__(
        self,
        root=ROOT,
        command=None,
        split="all",
        sample_rate=16000,
        length_sec=1.0,
        auto_download=True,
        **kwargs
    ):
        del kwargs

        self.sample_rate = sample_rate
        self.length_sec = length_sec
        self.num_samples = int(sample_rate * length_sec)

        if self.num_samples <= 0:
            raise ValueError("length_sec must be positive")

        if auto_download:
            self.gsc_root = download_gsc(root)
        else:
            self.gsc_root = root

        self.samples = self._load_samples(command=command, split=split)
        self.classes = sorted(list(set([sample["label"] for sample in self.samples])))
        self.class_to_idx = {
            label: idx
            for idx, label in enumerate(self.classes)
        }

        for sample in self.samples:
            sample["target"] = self.class_to_idx[sample["label"]]

        print(f"Loaded {len(self.samples)} Google Speech Commands samples.")

    def _split_paths(self, filename):
        split_path = os.path.join(self.gsc_root, filename)
        if not os.path.exists(split_path):
            return set()

        paths = set()
        with open(split_path) as f:
            for line in f:
                rel_path = line.strip()
                if rel_path:
                    paths.add(rel_path)
        return paths

    def _normalize_split(self, split):
        aliases = {
            "valid": "validation",
            "val": "validation",
            "test": "testing",
        }
        split = aliases.get(split, split)
        valid_splits = {"all", "train", "validation", "testing"}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {sorted(valid_splits)}")
        return split

    def _load_samples(self, command, split):
        split = self._normalize_split(split)
        validation_paths = self._split_paths("validation_list.txt")
        testing_paths = self._split_paths("testing_list.txt")

        samples = []
        for label in sorted(os.listdir(self.gsc_root)):
            label_dir = os.path.join(self.gsc_root, label)
            if not os.path.isdir(label_dir):
                continue
            if label.startswith("_"):
                continue
            if command is not None and label != command:
                continue

            for filename in sorted(os.listdir(label_dir)):
                if not filename.endswith(".wav"):
                    continue

                rel_path = os.path.join(label, filename)
                sample_split = "train"
                if rel_path in validation_paths:
                    sample_split = "validation"
                elif rel_path in testing_paths:
                    sample_split = "testing"

                if split != "all" and sample_split != split:
                    continue

                samples.append({
                    "id": rel_path,
                    "path": os.path.join(self.gsc_root, rel_path),
                    "label": label,
                    "split": sample_split,
                })

        return samples

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
            "label": meta["label"],
            "target": meta["target"],
            "split": meta["split"],
        }
