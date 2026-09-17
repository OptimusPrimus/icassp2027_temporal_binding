from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np


def _load_audio_with_ffmpeg(path: str | Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {message}")
    if not result.stdout:
        raise RuntimeError(f"ffmpeg decoded no audio from {path}")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def load_audio_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load audio as mono float32, resampled to ``sample_rate`` when needed."""
    try:
        import soundfile as sf

        audio, source_sample_rate = sf.read(path, always_2d=False, dtype="float32")
    except Exception as soundfile_error:
        try:
            return _load_audio_with_ffmpeg(path, sample_rate)
        except Exception as ffmpeg_error:
            try:
                # Keep librosa as a fallback for environments with audioread
                # support, but avoid numba JIT/cache setup because this path
                # only decodes audio.
                os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
                import librosa

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="PySoundFile failed.*")
                    warnings.filterwarnings(
                        "ignore",
                        message="librosa.core.audio.__audioread_load.*",
                        category=FutureWarning,
                    )
                    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
                return audio.astype(np.float32, copy=False)
            except Exception as librosa_error:
                raise RuntimeError(
                    f"Could not decode audio file {path!s}. "
                    f"soundfile: {soundfile_error}; "
                    f"ffmpeg: {ffmpeg_error}; "
                    f"librosa: {librosa_error}"
                ) from librosa_error

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if source_sample_rate != sample_rate:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(int(source_sample_rate), int(sample_rate))
        audio = resample_poly(
            audio,
            sample_rate // divisor,
            source_sample_rate // divisor,
        ).astype(np.float32)
    return audio.astype(np.float32, copy=False)
