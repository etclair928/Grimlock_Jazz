# Grimlock Perception Model — the ladder of hearing

*How this doc came to be: a listen-through of one song (the Gospel of John mashup)
where the user described, by ear, everything they heard — pulse, feel, meter,
groove, the ensemble's roles — and we measured each claim against the actual
audio and onset data to see whether it was really there. It was. This document
generalizes that exercise into the model Grimlock should reason with: **what is
there to look for in any piece of music, and where Grimlock currently stands on
each.***

> One song is a lens, not a template. The **framework** here is general; the
> **numbers** (67 BPM, 62% swing, etc.) are that song's fingerprint. Every
> observable below must be emitted *with a confidence*, because for some songs a
> layer is loud and obvious and for others it is faint or absent.

---

## The core idea

From two raw signals — **when each event happens** and **how loud it is** —
plus detected pitch, you can climb from the waveform to musical meaning one
inference at a time. The user's own framing splits every observation into:

- **`phys://`** — directly measurable in the wave (onset times, autocorrelation
  pulse, swing %, spectral centroid, pitch).
- **`hyp://`** — inferred / *felt*, reconstructed by the listener's brain (the
  razor-sharp virtual tactus, the downbeat felt through harmony not loudness,
  the "missing" middle triplet, a late note read as *intent* not error).

A transcriber that only reads `phys://` stops at "note at time T." Grimlock's
job is to also build the `hyp://` layer — with earned confidence.

---

## The seven layers, and where Grimlock stands

Status legend: **BUILT** (exists, works) · **PARTIAL** (exists, but stops short
of the musical payload) · **BLIND** (not looked at yet).

| # | Layer | Observable (`phys://`) | The concepts (incl. `hyp://`) | Grimlock module | Status |
|---|-------|------------------------|-------------------------------|-----------------|--------|
| 1 | onsets + energy | when + how loud each event is | the substrate | `rhythm_engine/onsets.py` (dual-witness) | **BUILT** |
| 2 | pulse / tactus | onset autocorrelation, IOI clustering | tempo, tempo-**octave**, *virtual tactus*, pulse *clarity*, drift | `octave_correction.py`, `note_onset_tempo_witness.py`, `lattice_witness.py`, `referee.resolve_tempo` | **BUILT** (octave just improved; note-onset witness reads ~19% high, #271) |
| 3 | subdivision / feel | onset phase *inside* the beat | straight vs triplet vs shuffle, the **swing ratio as a number** | `meter.py` (ratio_family), `groove_field.py` | **PARTIAL** — detects swung/compound; does not emit the numeric ratio or named pocket |
| 4 | meter / the bar | accent-by-position **+ harmonic-change timing** | time signature, downbeat, backbeat, **anacrusis**, *distributed downbeat* | `meter.py` (`TimeSignatureDetector`, downbeat phase) | **PARTIAL** — samples only the onset envelope; **blind to the harmonic downbeat** and to multi-source fusion |
| 5 | groove / microtiming | deviation off-grid, cross-instrument phase | the **pocket**, laid-back vs pushed, Dilla spread, somatic bounce | `groove_field.py`, `pulse_field.py` | **PARTIAL** — measures phase delta; the pocket is not surfaced as a first-class, named finding |
| 6 | phrase / voice | note grouping, gaps, contour | phrases, **voice-lines**, register, **per-voice rhythmic floor** | `instrument_attribution/voice_continuity.py` | **PARTIAL** — fragments into hundreds of micro-lines instead of instrument-level voices |
| 7 | harmony + role | bass-root spans, pitch-class, timbre features | harmonic rhythm, chords, key, *where harmony lives*, instrument **role** | `key_intelligence/`, `instrument_attribution/resolve.py` | **PARTIAL** — key yes; harmonic *rhythm* no; timbre buckets by brightness only, and trusts stem names |

---

## Two keystones that gate the whole ladder

Neither is a layer; both decide whether *any* layer reads cleanly.

1. **Consolidate before you analyze.** Fragments lie. In the worked example the
   bass came out of Basic Pitch as 1,306 notes; it took collapsing to 434 roots
   and then to chord-spans before a sane harmonic rhythm (~1 chord / 2.2 beats)
   appeared. The same fragmentation makes VoiceContinuity emit ~889 micro-lines.
   A consolidation pass (merge time-contiguous same-pitch fragments into held
   notes/roots) is the floor the ladder stands on. **Status: BLIND** — no
   dedicated consolidation step exists before attribution/harmony.

2. **Fuse sources — the truth is distributed.** No single stream states the
   meter, the harmony, or the pulse. In the example: the **bass** owned beat 1
   and the "4&" pickup; the **drums** owned the beat-3 backbeat; the **melodic
   bed** carried the tactus; the **trumpets** held the chords. Grimlock already
   fuses witnesses for *tempo* (`referee.resolve_tempo`). The leap is doing the
   same for **every** layer — a downbeat is a vote, a pocket is a vote, an
   instrument identity is a vote. **Status: BUILT for tempo, BLIND elsewhere.**

---

## What the worked example measured (the evidence behind the statuses)

All from the Gospel mashup; numbers are that song's, the *methods* are general.

- **Virtual tactus.** The 67 BPM pulse is as strong in the drumless melodic bed
  as in the drums (autocorrelation 0.54 = 0.54) — but low-clarity (1.15× vs
  3.16×). The period is `phys://`; the sharpness the listener feels is `hyp://`.
  The pulse rides the **melodic stabs**, not the (too-smooth) bass (bass ACF only
  0.33). Stable at 67 ± 1.8 BPM across the track at just 2.4 onsets/sec.
- **Feel.** Off-beat onsets land at **62% of the beat** — a Bernard Purdie
  half-time pocket (not straight 50, not hard 66).
- **Meter.** 4/4 with a strong **"4&" anacrusis** confirmed in *two* independent
  stems (drums and bass). The "1 & 3" macro pulse the ear feels is **not** in the
  onset energy (flat 1.0×) — it is **harmonic** (chords turning over), i.e.
  `hyp://`. The downbeat is **distributed** across the ensemble.
- **Harmony.** The bass maps the harmony at the **chord grain** (~1 change / 2.2
  beats), not per-note (it walks). The chords are actually held **up high in the
  trumpet section**, not the bass.
- **Over-detection, by a musical rule.** The user's "the bass never subdivides
  smaller than an 8th-note triplet" is a hard law: **68% of detected bass
  intervals are faster than that floor** → provable over-detection. The piano
  legitimately reaches 16ths ~2× more than the bass → the rhythmic floor is
  **per-instrument.**

### Second worked example — Hopeful (6/4 shuffle), and a self-correction

Applying the same x-ray to a *different* song — the pipeline's hardest fixture,
which took a dozen tasks (#240–252) to land at 145/6-4 — both confirmed the
method and exposed a flaw in a fix shipped this session.

- **The merged coherent instrument reveals the meter the drums obscure.** The
  merged piano bed's autocorrelation shows a clean **3:1 nested hierarchy** —
  140 BPM (quarter) → 44 BPM (dotted-half = the "1 & 4" macro pulse) → 26 BPM
  (the 6-beat bar). That nesting *is* the 6/4. The drum-heavy full mix — what the
  pipeline weights — read 152 with a messier hierarchy. **Read the metrical
  hierarchy off the instrument that spells the meter, not the loudest stem.**
- **Bar accent confirmed the shuffle shape:** macro on **1 & 4**, upbeats on
  **3 & 6** (beat 3 the loudest — the anacrusis pull into the second macro beat).
- **Self-correction: the compound-tactus fix shipped this session is too narrow.**
  It measures triplet salience at `beat ÷ 3` (a subdivision *below* the tactus).
  On Hopeful that reads ~0.007 — blind — because Hopeful's compound feel is a
  **3:1 grouping ABOVE the tactus** (three quarters per dotted-half). Gospel's
  triplet sat below the tactus (caught); Hopeful's sits above it (missed). **The
  general detector is "a strong 3:1 or 2:1 nesting ANYWHERE in the ACF
  ratio-hierarchy," not a fixed `beat/3` contrast.**
- **Whole-song rhythmic grid.** Per the ear, *everything* in Hopeful is 8th-note
  triplets — nothing faster anywhere. So the grid floor is a single lattice (the
  8th-triplet); 43% of detected piano / 38% of bass intervals fall below it → all
  over-detection. The floor is sometimes **per-song**, not only per-instrument.
- **Tempo is a curve, not a number (measured).** A smart triplet-8th metronome
  test: onsets lock only **40%** to a *rigid* grid at ANY tempo 145–152, but
  **60%** to an *elastic* grid re-locked every 8 s. The local quarter wanders
  **138–152 BPM (median ~147)** — which is why a tap-tempo site (145), Moises
  (145) and hand-counting (148) all disagree: each samples a different slice of a
  breathing tempo. There is no scalar answer; "145 or 148?" is literally
  unanswerable. Grimlock must carry a **tempo curve** and quantize to the *local*
  grid (the elastic TimeMap warp) — concrete evidence that the layer-2/4 elastic
  grid is mandatory, not optional.
- **How to SEE the subdivision (the method that finally worked).** The `beat/3`
  autocorrelation-contrast that read 0.007 was the WRONG instrument — dense
  running triplets smear it. The right method: **consolidate the notes → put them
  on the elastic beat → read each onset's WITHIN-BEAT position.** Do that and the
  triplets are unmistakable: onsets peak at **1/3 and 2/3** (the triplet partials)
  while the straight "and" at **1/2 is a valley** (grid-fit triplet 44% > 16th 42%
  > straight 24%; the empty 1/2 is the decisive tell). All three steps are
  required — the same *consolidate → track the breathing grid → read the phase*
  discipline that defines the whole model.

---

## Build / tweak sketch (prioritized, honest)

### Tweak — exists, stops short of the payload
- **Emit the swing ratio + named pocket (layer 3/5).** `groove_field.py` already
  measures the timing; add the numeric ratio (e.g. 62%) and a label
  (straight / light / Purdie / hard) as a first-class finding with confidence.
  Small, high visible value.
- **Fix the note-onset tempo witness ~19% high bias (#271).** Known, isolated.
- **Surface the pocket as a finding, not an internal number (layer 5).**
- **Generalize compound / subdivision detection (layer 2/3).** Two fixes, both
  proven this session. **(a)** The compound-tactus fix only checks `beat/3`
  salience — it catches a triplet *subdivision* (Gospel) but is blind to a
  compound *grouping above* the tactus (Hopeful 6/4, salience ~0.007); score any
  strong 3:1 / 2:1 nesting in the ACF ratio-hierarchy. **(b)** To classify the
  *subdivision type* (triplet vs straight vs 16th), do NOT use `beat/3`
  autocorrelation contrast (blind on dense material) — fold **consolidated** note
  onsets onto the **elastic** beat and read the **within-beat phase histogram**
  (triplet ⇒ peaks at 1/3 & 2/3, empty 1/2). Small changes; they catch what the
  ear hears.
- **Tempo curve + elastic quantization (layer 2/4).** Represent tempo as a
  per-window curve (not a scalar) and quantize to the local grid. Measured as
  mandatory on Hopeful (rigid 40% vs elastic 60% lock; tempo wanders 138–152).

### Build — genuinely missing, in leverage order
1. **Consolidation pass (keystone).** Before attribution/harmony, merge
   over-detected fragments into stable notes/roots. Unblocks layers 6 and 7 and
   the harmonic-rhythm witness at once. *Highest leverage — everything else
   depends on it.*
2. **Rhythmic-floor filter — per-instrument AND per-song (layer 6, feeds #268).**
   A musical over-detection filter keyed to a known grid floor: Gospel's bass
   can't subdivide past the 8th-triplet while its piano reaches 16ths
   (per-instrument); Hopeful is *entirely* 8th-note triplets, so the whole song
   has one lattice floor (per-song). Anything below the floor is artifact — far
   stronger than a statistical "drop short notes" rule. Needs the instrument /
   the detected grid (see #4, #7).
3. **Multi-source downbeat fusion (layer 4, #255).** Combine bass-root accents,
   drum accents, and harmonic-change timing into one downbeat vote — because no
   single stream states the bar. Reuses the `referee` fusion pattern.
4. **Harmonic-rhythm witness (layer 4/7).** Chord-change timing from consolidated
   bass roots *and* the upper-structure (trumpet) content — the `hyp://` downbeat
   evidence the onset-only `TimeSignatureDetector` is blind to.
5. **Real timbre discrimination (layer 7).** Move `resolve.py` past centroid-only
   brightness buckets to actual family ID (brass vs reed vs EP vs piano) from the
   already-computed bandwidth/ZCR/MFCC — the discriminating features are present
   in the data (trumpet ~3 kHz vs EP ~900 Hz cleanly separate).
6. **Stop trusting stem names (layer 6/7).** Demucs scatters/mislabels one
   instrument across stems (guitar-stem = EP; vocal-stem = trumpet). Merge all
   pitched stems and re-attribute from the audio via VoiceContinuity +
   TimbreIntelligence — which is what they were designed for. (A/B this session:
   merging the 4 melodic stems cut ~54% duplicate notes.)
7. **Stem-aware metrical hierarchy (layer 2/4).** Read the ACF ratio-hierarchy
   from the merged *coherent-instrument* bed (the piano — the thing spelling the
   meter), not the drum-weighted full mix. On Hopeful this exposed the 6/4
   (140→44→26, a clean 3:1 nesting) that the full mix obscured (152, messy).
8. **Similarity-gated stem merge (keystone / layer 6).** Cluster Demucs stems by
   spectral fingerprint (low/mid/high shape + centroid) and merge only true
   duplicates while keeping genuinely-different instruments apart. Hopeful:
   guitar/piano/other are one piano (identical shape) → merge (−40% notes, voice
   lines 2.6→4.3 notes/line). Gospel: those same three were EP/trumpet/piano
   (distinct) → don't collapse. Smarter than the current *always*-merge-
   guitar+piano+other rule; also recovers scattered frequency **body** (a stem
   missing its lows may have leaked them into the bass stem — the "underwater"
   vocal).

### Already landed this session
- **Compound-tactus octave correction** (layer 2) — picks the metrical level with
  a clean triplet *subdivision*, fixing the 136→67 double-tempo error (Gospel)
  without regressing straight songs. **Caveat found later:** too narrow — blind to
  compound *groupings above* the tactus (Hopeful 6/4). See the tweak above:
  generalize to the ACF ratio-hierarchy.
- **Brass GM patch map** (layer 7, a stopgap, not real discrimination).

---

*The end state — "Grimlock sees it this clearly" — is every rung emitting its own
measured evidence **and** its own confidence, with the layers cross-checking each
other: harmony confirming the downbeat, feel confirming the subdivision, roles
confirming the voices. The listener hears a whole; the machine reconstructs that
whole from the parts instead of stopping at the note.*
