"""Regression test for #20: length-grouped sampler manifest lengths + disk cache."""

import json

from glide.config.schema import GlideConfig, Modality


def test_compute_audio_lengths_from_manifest(tmp_path):
    from glide.data.build import compute_audio_lengths
    from glide.data.jsonl import JsonlDataset

    path = tmp_path / "manifest.jsonl"
    with open(path, "w") as f:
        for r in [{"audio": "/nonexistent/a.wav", "duration": 1.0},
                  {"audio": "/nonexistent/b.wav", "duration": 2.5}]:
            f.write(json.dumps(r) + "\n")

    ds = JsonlDataset(str(path), lazy=True)
    cfg = GlideConfig()
    cfg.modality = Modality.SPEECH
    cfg.data.train_jsonl_path = str(path)
    cfg.data.duration_field = "duration"
    cfg.speech.sample_rate = 16000
    # No file access (the audio paths don't exist) -> proves it read the manifest.
    assert compute_audio_lengths(ds, cfg) == [16000, 40000]


def test_lengths_cache_round_trip(tmp_path):
    from glide.data.build import _read_lengths_cache, _write_lengths_cache

    cache = str(tmp_path / "m.jsonl.glide-lengths.json")
    _write_lengths_cache(cache, "key123", [1, 2, 3])
    assert _read_lengths_cache(cache, "key123", 3) == [1, 2, 3]
    # Wrong key or wrong length -> cache miss (never reuse a stale scan).
    assert _read_lengths_cache(cache, "other", 3) is None
    assert _read_lengths_cache(cache, "key123", 2) is None
