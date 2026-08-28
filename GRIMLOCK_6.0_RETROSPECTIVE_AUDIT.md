# Retrospective Audit — lessons from the last stretch, applied backwards

Written 2026-08-22, after a long working session in which **eight separate
findings turned out to be a bad measurement rather than a bad implementation.**
That ratio is the most useful thing the session produced, and it shapes this
document: every claim below names how it was measured, and the ones that are
leads rather than results say so.

The question this answers: *what do we now know that the original code could
not have known, and where does that knowledge pay off retrospectively?*

---

## The thirteen lessons, and where each one still applies

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

**PROTOTYPED 2026-08-23** — `separation_engine/stem_attribution.py`. Validated
by asking whether f0 energy across stems can recover a label the pipeline
already assigned, on Hopeful:

| detected as | recovered | leaks to |
|---|---|---|
| vocals | **97%** | — |
| bass | 64% | 14% vocals, 11% drums |
| harmonic (guitar+piano+other) | 59% | **31% vocals** |

72% overall; 78% among uncontested notes. A first scoring said 54% — that was
my own confound, counting an `other` note found in `guitar.wav` as an error
when the pipeline merges those three stems precisely because htdemucs_6s
reassigns content between them.

**Not yet good enough to replace per-stem detection**: a quarter of notes
would be mislabelled, and the weak spot is the harmonic family, which is
exactly where Demucs itself is weakest. What it does establish is that the
mechanism works where separation is clean (97% on vocals), and that a shared
note can be reported as CONTESTED rather than silently duplicated into two
stems. The end-to-end path — detect on mix, attribute, notate, score — has
**not** been run.

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

**RESOLVED, against the suspicion.** `pitch_wobble_collapse` fires on 0.0% of
notes, and the lead recorded here was that gate 4 (continuous f0 spread ≤ 0.6
semitones) was too tight for vocal vibrato. Swept on Hopeful's real vocals
audio:

| ceilings | candidates | pass gate 3 | pass gate 4 | median f0 spread |
|---|---|---|---|---|
| 86/30 (shipped) | 5 | 0 | 0 | — |
| 150/30 | 34 | 5 | **0** | **1.66 st** |
| 300/80 | 71 | 15 | **0** | **1.61 st** |

The surviving candidates measure ~1.6 semitones of continuous pitch spread.
That is not a flat pitch flickering across a quantization boundary; it is
real melodic motion. **Gate 4 is doing its job and the 0% is correct** —
loosening it would delete melody. Do not touch this pass.

One genuine sub-finding survives: `_tempo_adaptive_ceilings` returns
`min(constant, 0.85 × sixteenth)`, so at 148bpm the member ceiling is 86ms
whatever `PITCH_WOBBLE_MAX_MEMBER_DURATION_MS` says — the flat constant is
inert, and only 20% of vocal notes (median duration 128ms) can even seed a
group. Worth knowing before anyone tunes that constant expecting an effect.

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

**FALSE POSITIVE, corrected.** The scan flagged `output/piano_reduction.py`
for reading raw `note.end_ms` six times while consulting no annotations. It is
not a defect: `piano_reduction` operates on `NotationNote`, whose `end_ms` is
documented as "the end the page uses (sustain-extended when available)" and has
`_consolidated_end_ms` applied at both construction sites before it is built.
The module reads raw fields because its input is already the resolved value.

Recorded rather than deleted, because the scan itself is worth keeping and its
false-positive mode is worth knowing: **"reads raw fields" only indicts a
module whose input is a raw Note.**

### 11. A reference is written in its own units — check them before comparing

Added 2026-08-25. The bass was reported broken on this evidence: *"human bass
41–60, ours 29–75 — we bottom a clean twelve semitones below the real floor."*
It was wrong, and wrong in a way no amount of reading our own code could have
caught, because the defect was in the **reference**, not the subject.

Bass guitar is a transposing instrument. The published transcription's part
carries `<transpose><octave-change>-1</octave-change></transpose>`: it is
written an octave above where it sounds. Comparing its *written* pitches to our
*sounding* pitches manufactured a twelve-semitone error out of nothing. Scored
correctly:

| | human (sounding) | ours |
|---|---|---|
| floor | 29 | **29** — exact match |
| notes below floor | — | **0 (0%)** |
| notes above ceiling | — | 52 (6%) |
| median | 39 | 42 |

Our bass floor was never wrong. The real defect is a three-semitone upward skew
and a 6% tail above the ceiling, which looks like guitar and piano bleeding
into the bass stem — a *separation* problem, not a *pitch* problem, and one
that lesson 1 already predicts.

**Generalise: a score is a document with units, not a list of pitches.**
Before any part of a reference is used as ground truth, read its `<transpose>`,
its `<divisions>`, and its key signature. Guitar sounds an octave down;
clarinet, trumpet and horn are not in C; a percussion part has no pitch at all.
Every one of those silently produces a confident, precise, wrong number.

This one cost a module. `pitch_engine/bass_octave.py` was designed, built,
tested and wired to fix an error that does not exist — kept, off by default,
because the mechanism is sound and honest about what it has not yet shown.

**Seventeen findings this stretch turned out to be a bad measurement rather
than a bad implementation.** The headline ratio at the top of this document was
eight; it keeps growing faster than the defect count, which is itself the
finding.

### 12. Score the property the pass actually changes

Added 2026-08-28, from the bass. `bass_octave.py` changes exactly one thing:
which octave a note sits in. It was judged first by notes-in-range (+0.8
points), then by counts of physically impossible notes (zero movement), then by
pitch-class distance — which was **identical before and after, necessarily**,
because an octave correction preserves pitch class by construction. That number
was reported as evidence. It could not have been anything else.

The test that answered it aligned our bass to the human's in TIME and asked,
per note, whether it sat in the same octave as the human note sounding at that
moment:

| | before | after |
|---|---|---|
| octave correct | 72.4% | **91.1%** |
| fixed / broken | — | 118 / 21 |

**Generalise: a metric that is mathematically insensitive to the change under
test will report "no effect" no matter how large the effect is.** Before
measuring, ask what property the pass alters and whether the metric can see it.
Aggregate distributions are the usual offenders — they are cheap, they look
rigorous, and they routinely cannot discriminate.

The corollary bit twice here: two full sessions concluded the bass was fine (or
broken in the opposite direction) on aggregate evidence alone.

### 13. A verdict that only damps confidence cannot fix a label

Also 2026-08-28. `analyze_key_stability` grew a scale-coverage guard that
correctly found Clocks' global reading of Bbm to be wrong — "Bbm spells notes
this recording does not use; it covers 92% of what was played against Ab's
97%." It wrote that verdict, with its full reasoning, into the MusicBox trail
on every run. The Conductor read `stability.confidence` and discarded
`stability.key`, so **the page said Bbm through every run** while the trail
explained why it should say Ab. The published transcription is in Ab.

Reported last session as "Clocks moves Bbm to Ab." That was measured on the
function in isolation and never on the pipeline — lesson 6 again, and this time
self-inflicted.

Now wired, and safe because it is selective rather than trusted: across seven
songs it changes two and leaves Chopin (B), Burden (F), HRV (Bm), Grey (Gm) and
YSGS (Cm) untouched. When the guards do not fire, `stability.key` IS the global
reading and the assignment is a no-op.

**This is the fourth "computed and read by nothing" in the register**, after
`tuplet_divisor`, `RANGE_ANNOTATION_KIND`, and a repair pass whose write
condition named its stat keys. The pattern is not an accident of any one
author: a verdict is easy to compute, easy to log, and invisible when unread.

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
3. **Consume tempo/meter contention** (lesson 2). We measure witness
   disagreement and discard it; that is the signal for "ask the user" in
   guided mode. **Done 2026-08-23** — persisted in the intermediate and
   surfaced in the health panel.
4. ~~`piano_reduction` reading raw notes~~ — false positive, see lesson 10.
5. ~~Sweep the wobble gates~~ — swept, the pass is correct, see lesson 8.

## Not worth doing

- Wiring `KEY_FIT` or Schoenberg `uncertain` as filters — measured, they do not
  discriminate (lesson 3).
- A better separation model. Every separator leaks; separate-then-detect
  multiplies the leak by the stem count regardless of model quality. The
  architecture is the problem, not the constant.
- Chasing Chopin's F1. Its local tempo swings 1.53× (44→68 bpm) and only 33% of
  the piece sits within 15% of *any* single tempo. Keep it as a bug-finder —
  in that role it has been the most productive file in the project.
