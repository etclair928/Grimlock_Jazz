# =================================================================
# MODULE: separation_engine/demucs_engine.py
# Separation Engine (GRIMLOCK_6.0_DESIGN_DECISIONS.md §4 / §7): 6-stem
# Demucs (htdemucs_6s) to pull guitar and piano out of the "other"
# junk-drawer at the AUDIO layer, before Pitch Engine ever runs - clean
# audio in, clean notes out, rather than sorting notes downstream.
#
# Model lifetime is owned by Model Registry, not this module - it asks
# for the model by name/device and never keeps its own private
# reference. Audio comes from AudioEngine.stereo() (the one place
# resampling happens); this module never calls librosa/soundfile.
# =================================================================

from __future__ import annotations

import gc
import time
from typing import Dict, Optional

import numpy as np

from audio_engine import AudioEngine, AudioTrack
from core import StemType, Separation
from model_registry import get_demucs_model

_STEM_NAME_TO_TYPE: Dict[str, StemType] = {
    "drums": StemType.DRUMS,
    "bass": StemType.BASS,
    "other": StemType.OTHER,
    "vocals": StemType.VOCALS,
    "guitar": StemType.GUITAR,
    "piano": StemType.PIANO,
}


def separate(
        engine: AudioEngine,
        track: AudioTrack,
        model_name: str = "htdemucs_6s",
        device: str = "cpu",
        seed: Optional[int] = 0,
) -> Separation:
    """Runs Demucs on `track` and returns a Separation whose `stems` dict
    holds one AudioTrack per stem the model actually produces (6 for
    htdemucs_6s: drums/bass/other/vocals/guitar/piano; 4 for plain
    htdemucs). Each stem AudioTrack goes through the normal decode-shape
    invariants (non-writeable samples, content hash) - it's canonical
    audio like any other, just produced by a model instead of a file.

    `seed` makes separation reproducible. `apply_model(shifts>0)` draws a
    RANDOM time-shift offset for test-time augmentation; left unseeded it
    makes separation - and therefore EVERY downstream stage - differ
    run-to-run on identical input. That non-determinism silently confounds
    every A/B measurement and is what made the note_support filter
    untestable (GRIMLOCK_6.0_OPEN_PROBLEMS.md §III; DESIGN_DECISIONS §10.3
    lists this as the first fix to land). Seeding keeps the shifts=1 quality
    benefit while making the result deterministic. Pass seed=None to opt back
    into non-deterministic behaviour."""
    import random
    import torch
    from demucs.apply import apply_model

    start_time = time.time()
    model = get_demucs_model(model_name, device)

    audio = engine.stereo(track, model.samplerate)
    if audio.shape[0] != model.audio_channels:
        if audio.shape[0] == 1 and model.audio_channels == 2:
            audio = np.repeat(audio, 2, axis=0)
        else:
            raise ValueError(
                f"Track has {audio.shape[0]} channels, model expects "
                f"{model.audio_channels} - no defined conversion for this case"
            )

    # .copy() (not just ascontiguousarray) because `audio` is the engine's
    # read-only cached view - torch.from_numpy requires a writable buffer.
    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio).copy()).float().unsqueeze(0).to(torch.device(device))

    # Seed every RNG demucs versions have used for the shifts offset (Python
    # random, numpy, torch) right before the call, so the random time-shift is
    # fixed and separation is reproducible. Cheap; unblocks all measurement.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    with torch.no_grad():
        sources = apply_model(
            model, audio_tensor, device=device, shifts=1, split=True, overlap=0.25, progress=False,
        )

    stems: Dict[StemType, AudioTrack] = {}
    for idx, stem_name in enumerate(model.sources):
        stem_type = _STEM_NAME_TO_TYPE.get(stem_name)
        if stem_type is None:
            continue
        stem_audio = sources[0, idx].cpu().numpy().astype(np.float32)
        stem_audio.setflags(write=False)
        duration_seconds = stem_audio.shape[-1] / model.samplerate
        stems[stem_type] = AudioTrack(
            samples=stem_audio,
            sample_rate=model.samplerate,
            num_channels=stem_audio.shape[0],
            duration_seconds=duration_seconds,
            source_path=f"{track.source_path}::separated::{stem_name}",
        )

    del audio_tensor, sources
    gc.collect()  # a full-track apply_model() call holds large intermediate
                  # tensors - on a long track this is the single biggest
                  # memory consumer of the whole pipeline, and nothing else
                  # references it once the stem AudioTracks are built.
    separation_time = time.time() - start_time

    return Separation(
        stems=stems,
        model_used=model_name,
        confidence=1.0,
        separation_time_seconds=separation_time,
    )


__all__ = ["separate"]
