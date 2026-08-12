# Grimlock University — a study layer for musical pattern knowledge

*Workshop of a pasted three-part proposal (jazz patterns → classical patterns →
guitar technique patterns). Same treatment as every other external proposal in
this project: keep / cut / already-have, grounded against what Jazz actually is,
with the failure modes named before a line is built.*

**Status:** design review. Nothing built. Nothing wired.
**Hard constraint (user, 2026-08-01):** *add this on top; do not change how Jazz
works yet; **this information should be logged.*** That constraint is not a
limitation to work around — it is the correct architecture, and §4 makes it the
governing law.

---

## 0. Verdict in one sentence

> The proposal's reframe — **discover musical patterns before assigning voices**
> — is right and fills a gap Jazz's own docs already named (§XIII.3, "notation
> needs a level of representation above the note"). But the code as written
> would corrupt the pipeline (it recreates `Note` objects, destroying the `id`
> the entire `AnnotationStore` is keyed by), several detectors are provably
> dead (the Alberti-bass test can never return true), it re-implements four
> things Jazz already does better, and it proposes ~60 hand-tuned detectors in a
> project whose defining problem is *having no way to score a hand-tuned
> threshold*. **Grimlock University is the right container for the good idea:
> a layer that studies and reports, never decides.**

---

## 1. The one true idea (KEEP)

```
Current: Notes → Voices → Score
Should:  Notes → Patterns → Musical Objects → Voices → Score
```

A copyist doesn't see `C4 E4 G4 C5 E5 G5`; they see *an arpeggio*. Jazz's own
open-problems doc says the same thing from the other side (§XIII.3): nothing
groups N detections into one notated object. Three specific corollaries worth
keeping verbatim:

- **Pattern knowledge is instrument-specific.** "Guitar is not piano with six
  strings" is the sharpest line in the whole proposal. Piano patterns are
  *harmonic*; guitar patterns are *technique*; wind/voice patterns are
  *breath*. One universal detector is the wrong shape.
- **Pattern knowledge is dialect-specific.** Bach's sequences, Mozart's Alberti,
  Debussy's planing, flamenco's compás, and metal's gallop are different
  vocabularies, not one taxonomy with more entries.
- **Three separate optimization problems** (acoustic accuracy / musical analysis
  / engraving compression) — naming them separately is genuinely clarifying, and
  it explains why solving Basic Pitch harder never fixed the page.

---

## 2. What is wrong with the proposal as written (CUT or FIX)

Not stylistic quibbles — these are correctness and law violations.

### 2.1 It would corrupt the pipeline (the fatal one)

`MusicalStructureOrchestrator.process()` and `ClassicalMusicPipeline.process()`
both **return new `Note` objects** built from dicts:

```python
note = Note(pitch=..., start_ms=..., ...)   # brand-new object → brand-new id
note.voice = note_dict.get('voice', 1)      # attribute injection
```

Two failures at once:

1. **Note identity is destroyed.** Every annotation in Jazz is keyed by
   `note.id` (`AnnotationStore`). Recreating notes orphans *every* annotation
   already written — quantization, sustain, ties, key-fit, harmonic legitimacy,
   consolidation, section labels. The pipeline would silently lose its entire
   interpretation layer.
2. **It mutates the detection floor.** The frozen-`Note` law (bible §2.2) is the
   single reason 6.0's playback is faithful; 5.x's un-freezing took fidelity
   **88% → 22%**. A pass that returns rewritten notes is that experiment again.

Also: `from core.order_types import Note` is Symphony's *archived* module. In
Jazz it is `core.note_types`, imported as `from core import Note`. The code has
never run against this repo.

### 2.2 Several detectors are provably dead code

- **`_detect_alberti_bass`** — takes a 4-note window, computes
  `sorted([p - root for p in pattern])` (**4 elements**) and compares to
  `[0, 4, 7]` (**3 elements**). A 4-element list can never equal a 3-element
  list. **Always returns `None`.** The single most-cited pattern in the proposal
  never fires.
- **`_detect_scale`** — requires
  `max(normalized) - min(normalized) == len(normalized) - 1` where `normalized`
  is `(p - root) % 12`. For a real ascending scale run the mod-12 wrap makes
  this essentially never true. Dead in practice.
- **`_detect_gallop`** — compares raw millisecond intervals with `==` for
  equality (`pattern[0] == pattern[1]`). Live performance timing never produces
  exactly-equal floats. Dead.
- **`_detect_tremolo_picking`** — `if all(intervals == 2 or intervals == 1)` on a
  numpy array is a truthiness error, not a comparison; and `_detect_lip_trill`
  has the same bug.
- **`detect_pattern()` returns the first match only** (`Optional[...]`), so a
  passage can only ever be *one* pattern — directly contradicting the proposal's
  own hierarchical-pattern goal.

The lesson isn't "fix these five bugs." It's that **~60 hand-authored templates
with no test corpus will be mostly dead or mostly noise, and nobody will know
which.** §7 is the answer to that.

### 2.3 Patterns cannot be mapped back to notes

`MusicalPattern.notes` is `List[int]` — **pitches, not note identities**. So
`StructureToVoiceMapper._get_notes_from_patterns()` reconstructs fake keys:

```python
note_ids.add(f"{pitch}_{pattern.start_ms:.0f}")   # every note gets the PATTERN's start
```

…and matches them against `f"{note['pitch']}_{note['start_ms']:.0f}"`. These
agree only for the pattern's first note. **The voice assignment is a near-total
no-op**, silently. This is the representational flaw underneath the whole design,
and it is the first thing §5 fixes.

### 2.4 It re-implements what Jazz already has, worse

| Proposal builds | Jazz already has | Verdict |
|---|---|---|
| `_group_by_phrases` (gap-threshold heuristic) | `check/structure.py` — real form detection (`detect_form` → `Section`) + `find_motifs` → `Motif`, already annotated per note | **Use Jazz's.** |
| `context.get('key_root', pitches[0] % 12)` — key guessed from the first note | `key_intelligence.analyze_key` → `KeyResult(key, confidence)` + per-note `key_fit` annotations | **Use Jazz's.** |
| `beat_duration = 500  # 120 BPM` hard-coded in `_detect_syncopation` | `TempoMeter` with resolved bpm, `beat_times_ms`, `downbeat_times_ms`, measured `swing_ratio` | **Use Jazz's.** |
| `_detect_era` from chromatic/chord ratios | `instrument_attribution.fingerprint.TimbreFingerprint` (centroid/rolloff/MFCC) + `resolve` families | **Use Jazz's** (and see §6.2). |
| `ClassicalHarmonyAnalyzer._detect_chords` | `acoustic_witness.schoenberg_mirror` harmonic legitimacy per note | **Complementary** — gate on it. |

### 2.5 The measured reality of this repo contradicts two premises

1. **You cannot dispatch pattern dialects by instrument on band material.**
   `separation_engine/stem_merge.py` merges guitar+piano+other into one stem
   because htdemucs_6s reassigns the same content between them over time —
   **measured 53% duplication** on Hopeful. The page's families
   (`bright_lead`/`mid_body`/`warm_sustained`) are *brightness buckets, not
   instruments*. So "run the flamenco detectors on the guitar" has no reliable
   guitar to run on. (Isolated-instrument input is a different, valid case —
   §6.2.)
2. **Patterns found on raw Basic Pitch output are patterns found in artifacts.**
   Measured this session on the produced page: **52% of adjacent notes in a
   voice are same-pitch rearticulations** (machine-gun), **39% are ≤16th
   fragments**. A "tremolo" or "ostinato" detector run on that will fire
   constantly on *detector noise*. **Pattern study must run downstream of
   `note_consolidation`.**

### 2.6 The scope is unfalsifiable as proposed

~60 pattern types × 5 eras × 6 instrument families, each with hand-authored
thresholds, in the one project whose central documented problem (§VI) is *"we
are optimizing without an accessible loss function."* A system of 60 guesses
that never crashes is precisely the inherited reflex (bible §10) that 6.0
exists to break. §7 is the mitigation, and it is non-negotiable.

---

## 3. The plug-in map (what the user asked for)

Grimlock University consumes existing intelligence; it does not re-derive it.
Every arrow is **read-only**.

```
                    ┌─────────────────────────── GRIMLOCK UNIVERSITY ──────────────┐
key_intelligence ──▶│ key, mode, per-note key_fit   → scale/chord/cadence dialects │
  (key_detector)    │                                                              │
                    │                                                              │
rhythm_engine ─────▶│ TempoMeter: bpm, beat_times_ms, downbeat_times_ms,           │
 (tempo/meter/      │ swing_ratio  → BEAT-RELATIVE pattern grammar (gallop,        │
  groove/downbeat)  │                compás, shuffle, Alberti) — never raw ms      │
                    │                                                              │
instrument_         │ TimbreFingerprint (centroid/rolloff/MFCC) + resolved family  │
 attribution ──────▶│  → WHICH DEPARTMENT may enroll this stem (the dialect gate)  │
 (fingerprint,      │ VoiceLine / line_id → candidate horizontal streams           │
  resolve,          │                                                              │
  voice_continuity) │                                                              │
                    │                                                              │
check ─────────────▶│ Section / Motif / repeat_group / repeat_drift                │
 (structure, check) │  → phrase windows FOR FREE + recurrence evidence             │
                    │                                                              │
acoustic_witness ──▶│ harmonic_legitimacy, note_support, anechoic activity         │
 (schoenberg_mirror,│  → do NOT study patterns built on unsupported notes          │
  note_support)     │                                                              │
                    │                                                              │
quantization ──────▶│ consolidation (primary/absorbed), notation timing,           │
 (note_consolidation│ is_tuplet, tie_candidate → study MUSIC, not BP confetti      │
  rhythm_inference) │                                                              │
                    └──────────────────┬───────────────────────────────────────────┘
                                       │  (evidence only)
                                       ▼
                       Annotations (kind="pattern_study")  +  MusicBox log
                                       │
                                       ▼
                        transcriptions/<song>_university.json   (corpus)
```

**Nothing flows back.** No arrow returns into the pipeline in v1. That is the
user's constraint and §4's law.

---

## 4. The governing law: a university studies; it does not decide

The name is the design. A university **observes, hypothesizes, publishes, and is
peer-reviewed** — it does not run the factory floor.

1. **Read-only.** `study()` takes `(notes, annotations, findings)` and returns a
   `StudyReport`. It **never returns notes**, never constructs a `Note`, never
   calls `replace()` on one. The frozen floor is untouched by construction.
2. **Everything is logged, nothing is applied.** Findings land in three places
   and no others: a `pattern_study` **Annotation** (keyed by real `note.id`), a
   **MusicBox** decision entry (`stage_name="grimlock_university"`,
   `reversible=True`), and a **per-song JSON** corpus file.
3. **No consumer in v1.** Not the engraver, not `piano_reduction`, not
   `voice_continuity`. The measured `register-split + ≤4 rhythmic-independence`
   path stays exactly as shipped. A pattern layer that quietly changed the page
   would violate both the user's constraint and §6.1's discipline.
4. **Every claim carries evidence + confidence + provenance**, like every other
   witness in 6.0. A detector that cannot say *why* it fired does not ship.
5. **Falsifiability is a shipping requirement**, not a nice-to-have (§7).

Add one enum value: `Provenance.GRIMLOCK_UNIVERSITY = "grimlock_university"`.
That, plus a new package, is the *entire* footprint on existing code.

---

## 5. Architecture (the corrected data model)

### 5.1 The note-ID fix (§2.3)

```python
@dataclass(frozen=True)
class PatternObservation:
    pattern: str                 # "alberti_bass", "gallop", "compas_solea", ...
    department: str              # "universal" | "keyboard" | "guitar" | "wind" | "jazz" | ...
    note_ids: Tuple[str, ...]    # REAL Note.id values — the whole fix
    start_ms: float
    end_ms: float
    confidence: float
    evidence: Dict[str, Any]     # the numbers that made it fire
    falsifier: str               # what would prove this wrong (§7)
```

`note_ids` (not pitches) is what makes an observation *joinable* back to the
frozen notes, to annotations, to sections, and to a future consumer. Everything
in §2.3 dissolves.

### 5.2 Beat-relative grammar, not milliseconds

The proposal hard-codes `beat_duration = 500`. Every rhythmic pattern in
Grimlock University is expressed as a **position/duration in beats**, derived
from `TempoMeter.beat_times_ms` + `downbeat_times_ms`. A gallop is
`[0.5, 0.25, 0.25]` of a beat — true at any tempo. Compás is a 12-beat accent
cycle — meaningless in milliseconds. This is also why the layer must run *after*
tempo/meter resolution, which the plug-in map already guarantees.

### 5.3 Witnesses, not a classifier

`detect_pattern()` returning the first match is wrong. Every applicable detector
runs; **all** firings are recorded with confidence; overlaps are kept, not
silently discarded. If a passage is simultaneously "arpeggio" and "sweep
picking," that *is* the finding — and it is exactly what the `epistemic.referee`
arbitration pattern exists to reconcile later, if a consumer ever needs one
answer. In v1 we publish the disagreement.

### 5.4 Windows come from `check`, not from a gap heuristic

Candidate windows = `Section` spans, `Motif` occurrences, and `VoiceLine`
segments already computed. This kills the proposal's O(n²) all-window sweep
(every size 2–12 at every offset, on 2000+ notes) and grounds every window in
something musically motivated.

---

## 6. The curriculum

### 6.1 Departments, in dependency order

| # | Department | Detectors (small, high-precision) | Why first/last |
|---|---|---|---|
| 1 | **Universal** | scale run, arpeggio/broken chord, sequence (transposed repeat), pedal/drone, ostinato, cadence approach | Applies to every instrument and dialect; needs only key + beat grid, both already resolved |
| 2 | **Keyboard/harmonic** | Alberti bass, block-chord homophony, walking bass, comping stab, chord-melody | The merged "other" stem is the input we actually have (§2.5) |
| 3 | **Rhythmic dialect** | shuffle/swing, gallop, syncopated anticipation, compás cycle | `swing_ratio` + downbeats already measured; testable on End Transmission (67% offbeat) |
| 4 | **Guitar technique** | power chord, palm-mute, tremolo picking, sweep, Travis alternating bass, campanella | **Gated on isolated guitar input** (§6.2) |
| 5 | **Wind/voice** | breath phrase, melisma, register break | Needs monophonic stems — vocals qualifies |
| 6 | **Era/style** | Baroque sequence, Impressionist planing, whole-tone, cluster | Last: highest ambiguity, lowest payoff, most tunable-by-wishful-thinking |

Ship department 1 alone first. If universal detectors can't pass §7, no dialect
department will.

### 6.2 The dialect gate (the scope-control mechanism)

**Do not run flamenco detectors on a gospel track.** Enrollment is decided by
evidence already available:

- **Isolated instrument input** (a `guitar.wav`, a solo piano recording) →
  full instrument department unlocked. This is the honest home for the guitar
  and classical-piano vocabularies, and it is real: `stems/Hopeful/` contains
  per-instrument files, and users transcribe solo material.
- **Band mix / merged harmonic stem** → **universal + keyboard/harmonic +
  rhythmic only.** Instrument-specific dialects are *not* offered, because
  §2.5's 53% measurement says the instrument label isn't trustworthy.
- **Timbre fingerprint + key + tempo + swing** narrow it further (a 62bpm 6/4
  worship track does not enroll in metal-gallop).

The gate is what keeps 60 detectors from becoming 60 noise sources.

---

## 7. Falsification: the null model (the missing piece, and the best part)

The proposal has no way to know whether any detector detects anything. Jazz's
§VI says there's no loss function and the ear is the only oracle. **For pattern
detection specifically, that is escapable — cheaply.**

> **A pattern detector that fires as often on shuffled notes as on real music is
> detecting nothing.**

The test, per detector, needs **zero ground truth and zero labeling**:

1. Run detector *D* over the real note stream → firing rate `r_real`, mean
   confidence.
2. Build null streams that destroy the structure *D* claims to find while
   preserving everything else:
   - **pitch-shuffle** (permute pitches, keep rhythm) — kills melodic patterns
   - **time-shuffle** (permute IOIs, keep pitches) — kills rhythmic patterns
   - **circular-rotate** one voice against the others — kills alignment patterns
3. Run *D* over N null streams → `r_null`.
4. **Report the lift** `r_real / r_null` and a p-value from the null
   distribution.

Then the taxonomy of outcomes is decidable:

| Result | Meaning | Action |
|---|---|---|
| `r_real ≈ 0` | dead detector (the Alberti bug, §2.2) | fix or delete |
| `r_real ≈ r_null` | fires on noise | delete — this is the honest kill |
| `r_real ≫ r_null` | detecting real structure | keep, and it earned it |
| `r_real ≈ 1.0` (fires everywhere) | threshold too loose | tighten |

This is the same statistical move as the injection–recovery harness (§IX.2) but
**an order of magnitude cheaper** — no synthesis, no soundfont, no rendering. It
also composes with the standing law: the **naked-BP baseline** ([the ground
truth for what the audio contains]) bounds *content*, while the null model
bounds *structure*. Together they cover both axes without a single human label.

Additionally, every detector ships with **a synthetic positive and a synthetic
negative unit test** (a hand-written Alberti figure must fire; a random cluster
must not). That alone would have caught four of the five dead detectors in §2.2
before they were ever run on audio.

---

## 8. First slice (small, logged, falsifiable)

**Slice U0 — "The university opens with one department, and grades itself."**

1. New package `university/` + `Provenance.GRIMLOCK_UNIVERSITY`. No edits to any
   existing stage.
2. `PatternObservation` (§5.1) and a `study(notes, annotations, findings)`
   entry point that is **read-only by signature**.
3. **Department 1 only**, four detectors: *scale run*, *arpeggio*, *sequence*,
   *ostinato/pedal* — each with a synthetic +/- unit test, each expressed
   beat-relative, each consuming key + tempo + sections from the plug-in map.
4. **The null-model harness (§7)** — built in the same slice, not after.
5. Output: `pattern_study` annotations + MusicBox entries + one JSON per song.
   **Pipeline output byte-identical** (assert it: re-export a page before/after
   and diff).
6. Run on the four songs already transcribed with saved `.pkl` intermediates
   (Hopeful, prospering, End Transmission, A Stranger's Smile) — **seconds each,
   no re-transcription**, since the intermediates exist.
7. **Publish the lift table.** Keep only detectors that beat their null.

If the four universal detectors can't beat a shuffle, we've learned that cheaply
and stopped — which is the entire point of building the grader in slice one.

---

## 9. Forks

1. **Read-only study layer, logged, zero consumers in v1** — my strong lean:
   **yes**, and it is what you asked for. (The alternative — wiring patterns into
   voice assignment now — would overwrite the measured register-split/≤4-voice
   result and break §2.1's law.)
2. **Null-model falsification built in slice 1, not later?** Lean: **yes,
   non-negotiable** — it's the only thing standing between 60 detectors and 60
   guesses.
3. **Department order: universal → keyboard/harmonic → rhythmic → instrument
   dialects?** Lean: **yes**, and gate instrument dialects on isolated-instrument
   input (§6.2), since the merged stem can't support them.
4. **Guitar/classical dialects: build only when an isolated stem is the input?**
   Lean: **yes.** That's the honest scope — and it's where the earlier TAB /
   fretboard-DP idea (`PERFORMANCE_ENGRAVING` §6) naturally rejoins this work.
5. **Corpus format** — per-song JSON next to the transcription. Lean: **yes**;
   it accumulates into the dataset that eventually makes the *next* question
   (which patterns predict good engraving?) answerable.

---

## 9b. BUILT + MEASURED (2026-08-06)

Shipped: `university/` (`observation_types`, `detectors`, `study`, `apply`,
`null_model`, `test_detectors`), `Provenance.GRIMLOCK_UNIVERSITY`,
`build_routed_score(honor_university=...)`, conductor flags
`university_mode` (`off`/`study`/`apply`) + `university_corpus_path`, and
`tools/university_report.py`. **16/16 detector unit tests pass.**

**The safety guarantee holds, verified:** OFF and STUDY produce a
byte-identical page (parts/notes/sounding-time equal on every song tested).

### The null model earned its place immediately — it caught three things

1. **A bug in the harness itself.** Grading a *melodic* detector against a
   *time*-shuffle is meaningless — time-shuffle leaves the pitch sequence
   intact, so a scale run survives it and always scores lift 1.0 ("NOISE").
   `scale_run` was really 0.034 vs **0.004** pitch-shuffle = **8.65× REAL**,
   masked by the invalid null. Fixed: each detector declares which nulls
   actually destroy its claimed structure (`_VALID_NULLS`).
2. **A genuinely bad detector.** `arpeggio` fired **more** on shuffled pitches
   (0.153) than on real music (0.111) — three leaping notes have a good chance
   of forming *some* triad under *some* rotation. Tightened (≥4 notes + the
   span must actually *outline* the chord); it now grades **REAL** on all three
   songs (lift 1.81 / 9.33 / 3.44).
3. **A detector that does not know anything.** `pedal_point` grades **NOISE**
   on all three songs (lift 0.34 / 0.72 / 1.3). Same-pitch adjacency arises by
   chance in a small pitch vocabulary. **Demoted** — the §7 rule applied to its
   own authors.

### Final grades (Hopeful / End Transmission / prospering)

| detector | verdict | lift | note |
|---|---|---|---|
| `scale_run` | **REAL** | 8.65 / – / 5.81 | strongest signal in the curriculum |
| `arpeggio` | **REAL** | 1.81 / 9.33 / 3.44 | only after being tightened |
| `sequence` | **REAL** | 1.80 / – / ∞ | rare but clean |
| `ostinato` | **REAL** | – / – / ∞ | fires rarely |
| `alberti_bass` | unproven | – | never fires: there is no Alberti bass in worship/rock. Correct, not broken. |
| `pedal_point` | **NOISE → demoted** | 0.34 / 0.72 / 1.3 | |
| `rearticulation` | **diagnostic only** | – | see below |

### The most useful negative result

`rearticulation` finds **0** runs with consolidation on and **63** with it off
(Hopeful). `note_consolidation` has already absorbed exactly the machine-gun
fragments it targets. So it is not broken and not redundant-by-accident — it is
**a sharp probe for whether consolidation is doing its job**, and in the
pipeline path it is correctly silent.

**Consequence: APPLY currently changes nothing on the page.** Its one page-
altering lever (suppressing low-confidence re-strikes) has nothing left to
suppress, and its other output (voice cohesion, 102 notes tagged on Hopeful) is
written but **not yet consumed by the voicer**. This is stated plainly rather
than dressed up: today APPLY is *safe and inert*. Making cohesion actually
bind notes into one voice inside `piano_reduction.assign_voices` is the next
slice, and it must be measured against the existing register-split/≤4-voice
result before it ships.

### Coverage is small on purpose

2.9–3.8% of notes carry a pattern. That is a *precision-first* curriculum on
dense band material where instrument identity is not recoverable (§2.5). Wider
coverage should come from detectors that beat their null, not from loosening
the ones that already passed.

---

## 9c. Streaming fixed, then detectors added (2026-08-06)

Done in that order deliberately: streaming raises the *ceiling*, detectors fill
it. Adding detectors first would have measured them against an artificially
starved input.

### Step 1 — streaming (`university/streams.py`)

The curriculum was reading lines from `voice_continuity.stream_into_lines`,
which is a **consolidator for instrument identity**, not a melodic-line finder:
*any* sustain overlap starts a new line, and its 300 ms gap is absolute (at
148 bpm a beat is 405 ms). Result: lines averaged **1.8–2.2 notes**, so a
detector needing 4 consecutive notes almost never fired.

The replacement groups by **onset succession**, tolerates sustain overlap, and
uses a **beat-relative** gap — with one guard that must never regress: notes
struck within 45 ms are a **chord, not a continuation** (otherwise every block
chord reads as an arpeggio). `voice_continuity` is untouched; the pipeline's
instrument attribution still uses it exactly as before.

| | notes reachable (in a line ≥4) | mean line length |
|---|---|---|
| Hopeful | 43.2% → **81.0%** | 2.19 → 4.89 |
| prospering | 29.3% → **77.4%** | 1.84 → 4.56 |
| End Transmission | 30.9% → **68.0%** | 1.76 → 3.43 |

Critically, **every detector still graded REAL afterwards** — longer lines gave
genuine patterns room to appear rather than breeding false positives.

### Step 2 — three new detectors, then graded

`neighbor_tone` (step away and return / turn), `gap_fill` (a leap answered by
stepwise motion the other way), and `chord` (the first **vertical** detector —
simultaneity cannot exist inside a monophonic line, and dense harmonic material
is where most notes are).

**The grader caught its own blind spot:** `chord` fired 29/52/12 times but
received **no grade at all** — the null model only ran detectors on melodic
lines, so a vertical detector silently scored zero and would have entered the
curriculum ungraded. Fixed by dispatching vertical detectors over the whole
stem in the null model too.

### Final tiers (3 songs, null-graded)

| tier | detectors | evidence |
|---|---|---|
| **GRADUATED** (may alter the page) | `scale_run` 3/3, `arpeggio` 3/3, `neighbor_tone` 3/3, `ostinato` 2/2 | lift 1.6–12.8 |
| **PROVISIONAL** (studied + logged, may **not** alter the page) | `chord` 2/3, `gap_fill` 2/3, `sequence` 1/2 | REAL on some material, NOISE on other |
| **FAILED** | `pedal_point` 0/3 | NOISE everywhere — demoted |

The tier split is the §7 rule applied where it actually costs something: a
detector that is only *sometimes* right still gets to publish, but it does not
get to move a notehead.

### Coverage, and what APPLY now does

| | coverage before | after | gestures kept intact (voicer off → on) |
|---|---|---|---|
| Hopeful | 2.9% | **22.7%** | 34.4% → **53.4%** (189 gestures) |
| prospering | 3.8% | **23.7%** | 45.5% → **55.6%** (99) |
| End Transmission | 2.9% | **15.3%** | 33.9% → **48.2%** (56) |

Coverage is up ~6–8×, and cohesion now touches 183–587 notes per song instead
of ~100. OFF remains byte-identical to Grimlock Jazz.

---

## 10. The owned thesis

> The proposal is right that Grimlock has been missing the layer between notes
> and voices, and right that it must be instrument- and dialect-aware. It is
> wrong that the way to get there is sixty hand-tuned detectors wired into voice
> assignment — that path recreates notes (destroying annotation identity),
> ships dead code nobody can detect is dead, and multiplies the one problem this
> project has never solved: an unscoreable threshold. **Grimlock University
> keeps the insight and inverts the posture.** It reads everything the existing
> intelligences already know, studies the notes without touching them, publishes
> what it finds with the evidence that produced it, and — crucially — **grades
> its own findings against a null model so a detector that knows nothing gets
> caught and killed.** It changes nothing about how Jazz works. It only starts
> writing down what Jazz has been hearing all along. When a detector has
> survived its null, *then* it has earned the right to be consulted by the page.
