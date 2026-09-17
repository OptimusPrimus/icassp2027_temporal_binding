import os
import csv
import zipfile
import requests
import numpy as np
import librosa
from torch.utils.data import Dataset


ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def download_esc50(root="dataset/ESC50"):
    """
    Download ESC-50 automatically.
    """

    os.makedirs(root, exist_ok=True)

    extract_path = os.path.join(root, "ESC-50-master")

    if os.path.exists(extract_path):
        print("ESC-50 already downloaded.")
        return extract_path

    zip_path = os.path.join(root, "esc50.zip")

    print("Downloading ESC-50...")

    r = requests.get(ESC50_URL, stream=True)

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Extracting...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)

    os.remove(zip_path)

    print("Done.")

    return extract_path


class ESC50Dataset(Dataset):

    def __init__(
        self,
        root="dataset/ESC50",
        sample_rate=16000,
        length_sec=5.0,
        fold=None,
        category=None,
        auto_download=True,
        **kwargs
    ):
        del kwargs

        self.sample_rate = sample_rate
        self.length_sec = length_sec
        self.num_samples = int(sample_rate * length_sec)

        if auto_download:
            self.esc_root = download_esc50(root)
        else:
            self.esc_root = os.path.join(root, "ESC-50-master")

        self.audio_dir = os.path.join(self.esc_root, "audio")
        self.meta_file = os.path.join(self.esc_root, "meta", "esc50.csv")

        self.samples = []

        with open(self.meta_file) as f:
            reader = csv.DictReader(f)

            for row in reader:

                if fold is not None and int(row["fold"]) != fold:
                    continue

                if category is not None and row["category"] != category:
                    continue

                audio_path = os.path.join(self.audio_dir, row["filename"])

                if not os.path.exists(audio_path):
                    continue

                self.samples.append({
                    "id": row["filename"],
                    "path": audio_path,
                    "label": row["category"],
                    "fold": int(row["fold"]),
                    "target": int(row["target"])
                })

        print(f"Loaded {len(self.samples)} ESC-50 samples.")
        self.classes = sorted(list(set([r["label"] for r in self.samples])))

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
            "fold": meta["fold"]
        }
