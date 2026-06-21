"""Audio loading helpers (uses the system libsndfile via ``soundfile``).

ffmpeg (7.1) on the system handles container/codec decoding for formats that
``soundfile`` cannot read directly; ``librosa.load`` falls back to it.
"""

from typing import Any

import numpy as np

__all__ = ["load_audio", "audio_num_samples", "speed_perturb"]


def speed_perturb(waveform: np.ndarray, factor: float, sample_rate: int = 16000) -> np.ndarray:
    """Kaldi-style speed perturbation: change tempo *and* pitch by ``factor``.

    ``factor > 1`` makes the utterance faster/shorter. Implemented by resampling the
    signal (so duration becomes ``len/factor``) and keeping the original sample rate.
    ``factor == 1`` is a no-op.
    """
    if factor == 1.0 or factor <= 0:
        return waveform
    import librosa

    target = int(round(sample_rate / factor))
    return librosa.resample(waveform, orig_sr=sample_rate, target_sr=target).astype(np.float32)


def load_audio(ref: Any, target_sr: int = 16000) -> np.ndarray:
    """Load ``ref`` into a mono float32 waveform at ``target_sr``.

    ``ref`` may be a file path, a ``{"array": ..., "sampling_rate": ...}`` dict
    (🤗 datasets audio feature), or an already-decoded numpy array / list.
    """
    if isinstance(ref, np.ndarray):
        return ref.astype(np.float32)
    if isinstance(ref, list):
        return np.asarray(ref, dtype=np.float32)
    if isinstance(ref, dict) and "array" in ref:
        arr = np.asarray(ref["array"], dtype=np.float32)
        sr = ref.get("sampling_rate", target_sr)
        if sr != target_sr:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        return arr
    # Treat as a path.
    import librosa

    arr, _ = librosa.load(ref, sr=target_sr, mono=True)
    return arr.astype(np.float32)


def audio_num_samples(ref: Any, target_sr: int = 16000) -> int:
    """Return the number of samples for ``ref`` without fully decoding when possible.

    Used by the length-grouped sampler. For file paths this reads only the header
    via ``soundfile.info``; otherwise it falls back to decoding.
    """
    if isinstance(ref, (str,)):
        try:
            import soundfile as sf

            info = sf.info(ref)
            n = int(info.frames)
            if info.samplerate and info.samplerate != target_sr:
                n = int(n * target_sr / info.samplerate)
            return n
        except Exception:
            pass
    return int(len(load_audio(ref, target_sr)))
