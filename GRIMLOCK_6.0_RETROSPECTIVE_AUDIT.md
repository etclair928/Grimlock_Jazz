# Retrospective Audit — lessons from the last stretch, applied backwards

Written 2026-08-22, after a long working session in which **eight separate
findings turned out to be a bad measurement rather than a bad implementation.**
That ratio is the most useful thing the session produced, and it shapes this
document: every claim below names how it was measured, and the ones that are
leads rather than results say so.

The question this answers: *what do we now know that the original code could
not have known, and where does that knowledge pay off retrospectively?*

---

## The ten lessons, and where each one still applies

### 1. Put a fallible stage where its errors are cheap

The deepest one. A stage that **generates** can invent; a stage that
**describes** can only mislabel. Today separation runs first and decides *what
notes exist*, so every leak becomes notes — multiplied by the stem count.

Measured on Burden of Sentiment, scored against Klangio:

| candidate | notes | precision | recall | F1 |
|---|---|---|---|---|
| raw Basic Pitch, no separation | 1602 | 0.202 | 0.236 | **0.218** |
| our `other` stem alone | 1977 | 0.168 | 0.243 | 0.199 |
| our full six-stem pipeline | 3425 | 0.147 | **0.369** | 0.211 |

The whole pipeline adds 1823 notes and does not beat feeding the raw mix to
Basic Pitch. Note the split honestly: separation genuinely **improves recall**
(0.369 vs 0.236) and badly hurts precision. It finds real quiet notes and
buries them.

**Retrospective fix: detect once on the mix; use stems only to attribute.**
For a note at f0 spanning t0–t1, compare energy at f0 across stems and label it
by whichever holds most. A leak then costs a wrong *name*, never a phantom
*note*. The machinery exists — `instrument_attribution/fingerprint.py` computes
per-note MFCC/centroid/bandwidth/ZCR from any audio and never needed stems at
all.

Caveat that must not be lost: this **trades recall for precision**. It is the
right trade only because a page with twice the notes the music has is less
usable than one missing some.

### 2. Audit every output for a reader

`tuplet_divisor` was computed, documented, and dropped one line before use —
which silently disabled the entire deliberate tuplet path. `RANGE_ANNOTATION_KIND`
was written on every run since 2026-08-06 and read by nothing, and turned out
to be the sharpest stem-quality witness we have.

Both were found by a mechanical scan, not by reading code. **Still open:**

| output | status |
|---|---|
| `DURATION_ANNOTATION_KIND` | written every run, read by nothing |
| `KEY_FIT_ANNOTATION_KIND` | written on 100% of notes, read by nothing |
| `PATTERN_STUDY_ANNOTATION_KIND` | written, read by nothing |
| `result.tempo_confidence` | never read |
| `result.tempo_contention` / `meter_contention` | never read — the referee's disagreement is discarded |
| `result.onset_refined_count` / `wobble_group_count` | never read |
| `DOUBLE_TIME_RATIO_LO` / `_HI` | defined, referenced nowhere |
| `classify_drift` verdict | computed and logged, acted on by nothing |

The contention fields are the interesting ones: when the tempo witnesses
disagree, we record it and throw it away.

### 3. A verdict is only useful if it discriminates

The counterweight to lesson 2 — not every unread verdict deserves wiring.
Tested by asking whether a verdict fires *differently* on stems and songs we
handled well versus badly:

| witness | Hopeful vocals (good) | Burden vocals (bad) | Chopin piano (our best) | usable? |
|---|---|---|---|---|
| range implausible | **0%** | **8%** | **0%** | **yes** |
| Schoenberg "uncertain" | 16% | 93% | **63%** | no |
| key-fit out of key | 9% | 8% | **15%** | no |

Range separates cleanly. The other two fire hardest on our best output —
Chopin is chromatic and modulating, so out-of-key notes there are real music.
**Range is now wired; the other two must stay explanatory.**

### 4. A guarantee that can be violated is not a guarantee

`notatable_at_most` returned `min(allowed)` when nothing fit under its ceiling
— a value *longer* than the bound it existed to enforce. Fifteen notes were
asked for ≤1/12 of a beat and given 1/8, overrunning by exactly 1/24 each and
displacing every onset after them. That single line produced the whole k/24
family of defects.

**Generalise:** any function whose name promises a bound (`at_most`, `clamp`,
`limit`, `max_`, `nearest`) must be checked for the branch where the bound
cannot be met. Returning 0 or None and letting the caller decide is correct;
returning something out of bounds is not.

### 5. Never lengthen to fit

User directive, and the same bug wearing different clothes: *"do not force
rhythmic values into spaces they won't or can't possibly fit. If it doesn't fit
then it must be something else that can actually fit."*

I reintroduced this error **after** fixing lesson 4 — `_notatable_chain`
originally folded a sub-notatable residue into the previous note to keep a tied
chain summing exactly. Growing a note to absorb a leftover pushes the next
onset. Shortening is always safe; lengthening never is.

### 6. Measure on the path that ships

Every "off-grid onsets = 0" reported during the k/24 work was measured on a
**re-export**, which built its score on a flat clock because the intermediate
did not store the beat grid. The pipeline's own file had 2310 onsets; the
re-export of the same pkl had 2602. Different engravings of the same notes.

Fixed by storing `beat_times_ms` / `bar_origin_ms` in the intermediate. The
general rule: **a diagnostic that reconstructs its input is measuring the
reconstruction.**

### 7. Do not re-derive the rule you are auditing

The first `check_tuplets` reimplemented the anchoring rule and reported 44
off-beat groups on a page that has 9 — it demanded a whole-beat anchor the
published edition disproves. `tools/tuplet_audit.py` already held the
calibrated version. Now split into `measure_tuplets()` (returns) and `audit()`
(prints), with a test asserting the printer holds no rules of its own.

### 8. A rule that cannot fire is dead code wearing a caveat

Removed under this heading: a repetition gate whose condition was
unsatisfiable, and an in-memory barline clamp that reported zero elements
touched while the written file still had thirteen crossings.

**Still open — `quantization/pitch_wobble_collapse.py` fires on 0.0% of notes
on every song in the corpus**, including ones with real prominent vocals. It
has four AND-ed gates; gate 4 requires continuous f0 spread ≤ 0.6 semitones,
and ordinary vocal vibrato spans 1–2. *This is a lead, not a measured result* —
the threshold has not been swept. Worth doing, because vocals are the worst
stem we have by every other measure.

### 9. Check docstrings against code

`_content_hash` promised "sampling a slice… keeps decode-time overhead flat
regardless of track length" and then converted the entire array to float32,
serialised all of it, and sliced afterwards — ~290MB of transient allocation to
obtain 64KB, and a MemoryError on a long file.

Eleven other functions make bounded/cheap claims in their docstrings and have
not been checked: `_drain_webhook`, `stream_into_lines`, `_events_to_voices`,
`_smooth_slots`, `_build_beat_tick_scales`, `sample_beat_accents`,
`load_cached_separation`, `detect_neighbor_tone`.

### 10. Scope — pass one instrument's notes

`output/playability.py` warns in its own header that feeding it an ensemble
asks whether one pianist can play the whole band. A diagnostic written to
evaluate the register split then ran `assign_hands` over **every** part
including bass and vocal lines, and reported 14 unplayable chords on Hopeful
and HRV. Correctly scoped: 6 and 3.

**Still open:** `output/piano_reduction.py` reads raw `note.end_ms` six times
and consults **no annotations at all** — so the treble/bass split is computed
on fragmented, un-consolidated notes, while sustain-recovery and consolidation
verdicts sit unread. Same shape as the notation-quantizer bug.

---

## Ranked by expected value

1. **Detect once, attribute after** (lesson 1). The only change here that could
   move F1 by a large step rather than a fraction. Prototype-able without
   touching the pipeline: detect on the mix, label using stems already on disk,
   score against the current output.
2. **Fragmentation** — 26% of Burden's notes are the same pitch re-struck
   within 120ms of the previous one ending, 685 of them at a gap of exactly
   zero. Klangio writes 213 whole notes where we write 37. Note that our tie
   behaviour is *already correct* (1.22 noteheads per sounding note vs
   Klangio's 1.20) — this is too many notes, not one note split badly.
3. **`piano_reduction` reading raw notes** (lesson 10). Cheap, and the staff
   split is currently decided on data we already know is wrong.
4. **Sweep the wobble gates** (lesson 8). A pass that never fires on the stem
   that needs it most.
5. **Consume tempo/meter contention** (lesson 2). We measure witness
   disagreement and discard it; that is the signal for "ask the user" in
   guided mode.

## Not worth doing

- Wiring `KEY_FIT` or Schoenberg `uncertain` as filters — measured, they do not
  discriminate (lesson 3).
- A better separation model. Every separator leaks; separate-then-detect
  multiplies the leak by the stem count regardless of model quality. The
  architecture is the problem, not the constant.
- Chasing Chopin's F1. Its local tempo swings 1.53× (44→68 bpm) and only 33% of
  the piece sits within 15% of *any* single tempo. Keep it as a bug-finder —
  in that role it has been the most productive file in the project.
