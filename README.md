# Grimlock (Jazz) 6.0

A ground-up rebuild of the Grimlock audio-to-MIDI transcriber. Where 5.x
(Symphony) grew by accretion, 6.0 is built on two disciplines:

- **One type per concept** — a single canonical `core` type layer, no
  colliding duplicate `NoteEvent`/`StageResult`/`Confidence` definitions.
- **Amplify, don't fight the models** — the detection models (Basic Pitch,
  CREPE) produce the notes, and their pitches are *frozen* once detected.
  Every later stage (rhythm, instrument attribution, harmonic legitimacy,
  quantization) attaches **Annotations** to those notes; nothing downstream
  rewrites a model's note into a different-pitched one. `output.engrave()`
  is the only place that decides whether to use a raw value or an
  annotated hypothesis.

## Pipeline

`orchestration/conductor.py :: transcribe_file(audio_path, output_midi_path,
...)` runs a linear sequence:

1. **Audio Engine** — one canonical decode + cached transforms (STFT/CQT/mel/
   onset envelope), all resampling behind one boundary.
2. **Separation Engine** — Demucs `htdemucs_6s` (drums/bass/other/vocals/
   guitar/piano), seeded (`separation_seed=0`) for reproducibility, with a
   timbre-based harmonic-stem merge.
3. **Pitch Engine** — Basic Pitch per stem (the transcription) + a CREPE
   bass read for the low end.
4. **Instrument Attribution** — per-note timbre fingerprint → voice-line
   streaming → one resolved instrument family per line (written as an
   Annotation).
5. **Rhythm Engine** — tempo / meter / groove + drum onset detection and
   classification.
6. **Epistemic referee** — reconciles contradictory scalar witnesses
   (e.g. tempo across 4 witnesses via anchor-and-align).
7. **Acoustic witnesses** — AnechoicMa (silence/resonance oracle) +
   SchoenbergMirror (harmonic-legitimacy auditor), evidence-only.
8. **Key intelligence**, **quantization** (Annotation-only lattice/duration
   hypotheses), then **Scribe Engraver** → MIDI (raw timing by default;
   `use_quantized_timing=True` opts into the grid).

Output MIDI is written to `transcriptions/` (gitignored).

## Environment (shared with Symphony — read before touching the venv)

Jazz has **no requirements.txt of its own**: it runs on the **same two-layer
Python environment** as the Symphony repo, and imports the same native stack.
See `../Symphony/requirements.txt` (top-of-file block) and
`../Symphony/README.md` → *Environment* for the authoritative writeup. In
short:

- **Global Python 3.11** provides the modern DSP + PyTorch world: `torch`,
  `torchaudio`, `demucs`, `librosa`, `numpy` (1.26.x), `scipy` (≥1.13),
  `numba`, `madmom`, `soundfile`, `crepe`.
- **The `.venv`** (created with `include-system-site-packages = true`) holds
  **only** the TensorFlow-pinned stack — `tensorflow` 2.15, `tf-keras` 2.15,
  `basic-pitch` — walled off so TF's `numpy<2` / old-`ml-dtypes` pins can't
  clash with the global torch/librosa stack. The layering (venv over global),
  not duplication, is what lets Basic Pitch (TF) and Demucs (torch) run in one
  process.
- A `scipy.signal.gaussian` runtime shim (scipy ≥1.13 removed the re-export
  Basic Pitch still calls) lives in `pitch_engine/basic_pitch_engine.py` and
  in the `tools/` scripts.

> **Do not rebuild the `.venv`.** Verify the environment by *running* the
> pipeline on a short clip, not by reinstalling. A destructive rebuild
> re-creates the exact dependency clash the split exists to avoid.

Runs are launched with the Symphony `.venv` interpreter, e.g.
`../Symphony/.venv/Scripts/python.exe`, with the Jazz repo on `sys.path`.

## Tools

- **`tools/naked_basic_pitch.py`** — audio → MIDI through **Basic Pitch
  only**, zero Grimlock (no separation, council, rhythm, or quantization). An
  A/B reference for what the model alone hears. Outputs to `transcriptions/`
  tagged `_naked_basic_pitch` so they're never confused with full-pipeline
  MIDI (`*_jazz*`, `*_6.0*`).
  ```bash
  python tools/naked_basic_pitch.py "<file-or-folder>" [more...]
  ```
- **`tools/build_stem_cache.py`** — one-off builder for the local test
  library `Input/` (below). Pre-separates each song into its 6 htdemucs_6s
  stems (seed 0, matching the pipeline) so stem-level analysis doesn't
  re-run Demucs by hand.

## `Input/` — local test library (gitignored)

`Input/<Song>/` holds a source audio file plus a `stems/` folder of its 6
separated stems. Purely a convenience cache for testing and stem-level
analysis — it is **not** read by the pipeline (the Conductor always separates
fresh) and is kept out of git (large, reproducible audio). Rebuild or extend
it with `tools/build_stem_cache.py`.

## Tests

```bash
python -m pytest
```

`tests/test_audit_fixes.py` pins the arithmetic underneath the pipeline's
decisions — key-signature mode, notation durations, the quantizer's grid
phase, meter propagation, the MIDI beat-tick alignment, and MusicXML
validity. Every one is a pure function over plain data: no audio, no models,
no separation, so the whole suite runs in about a minute and most of it in
milliseconds. It exists because this project measures its *decisions*
carefully and had nothing pinning the conversions those measurements sit on
— a correct measurement can rest on a wrong conversion indefinitely.

`university/test_detectors.py` is a separate `__main__` script (synthetic
+/- tests per pattern detector); run it directly.

## Design docs

`GRIMLOCK_6.0_DESIGN_DECISIONS.md` (the architecture bible) and
`GRIMLOCK_6.0_OPEN_PROBLEMS.md` (open questions / the notation frontier) hold
the full rationale, cluster boundaries, and the governing laws.

## Contributing

Only Claude and DeepSeek write code for this project directly — other AI tools
are for research/discussion, not implementation. Nothing lands half-wired: a
new module's change includes wiring it into the pipeline, or it doesn't land
yet.
