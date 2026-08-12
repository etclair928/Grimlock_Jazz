# Grimlock 6.0 — Performance-Aware Engraving (Hands, Grand Staff, Dueling Pianos, Guitar TAB)

*A review of a pasted "dueling pianos → intelligent voice/hand/staff engraving"
essay, ground-truthed against what Jazz already is. Companion to
[GRIMLOCK_6.0_NOTATION_GRAPH.md](GRIMLOCK_6.0_NOTATION_GRAPH.md) — it does not
restate that doc, it extends it at one seam the notation-graph doc explicitly
left open: **voice ownership and the level above the note** (§XIII.3).*

**Status:** Pre-build review. No code proposed until a fork below is chosen.
**Policy note:** the essay is an external-AI proposal — research/opinion to
pressure-test, per project convention. Implementers remain Claude/DeepSeek.

> **2026-07-30 — read §11 first; it changes the target.** A measured pass
> against the real artifacts refutes the *hand-recovery* framing of §3 (there is
> no recoverable "piano" identity in a band mix — see §11.1), but the user's
> clarified intent rescues the idea in a better-posed form: **treat the grand
> staff as a harmonic *reduction/presentation* of the junk-drawer stem — chord
> aggregation + register-based staff split — not a reconstruction of a
> pianist's two hands.** That version needs no instrument identity and runs on
> the messy stem we actually have. §11 is the real design; §3–§6 are kept as the
> record of how we got there.

---

## 0. One sentence

> Applied to Jazz, roughly **80% of this essay is already in your open-problems
> and notation-graph docs**, **three of its load-bearing ideas are things Jazz
> already ran as experiments and rejected with measurements**, and the genuinely
> new, valuable ~15% is one thing the essay itself buries: the **piano
> performance→engraving cascade — hand/staff assignment and the grand staff —
> which Jazz has never designed** (today it emits one staff per voice/family
> with a naive mean-pitch clef). That, plus a clean framing of dueling pianos
> and guitar TAB, is the whole opportunity.

---

## 1. The uncomfortable finding: you've mostly been here already

Read against `GRIMLOCK_6.0_OPEN_PROBLEMS.md` and `_NOTATION_GRAPH.md`, the essay
is largely a re-derivation of Jazz's own prior analysis:

| Essay proposal | Jazz's existing position | Verdict |
|---|---|---|
| "Voice separation is 5 coupled inference problems, a cascade" | §XIII.3 (level above the note), §XIII.5 (data association), the whole `NotationScore` thesis | **Already your thesis.** Keep. |
| "Model note↔note as typed relationships / affinity graph → Musical Relationship Graph" | §XIV: ChatGPT's identical "universal Musical Relationship Graph" was reviewed and **rejected as model-everything-jointly scope-creep**; edge-entropy diagnostic kept | **Already rejected.** Don't rebuild. |
| "Maintain uncertainty / multiple hypotheses per note (SLAM-style), un-freeze detection" | §XIII.7 + bible §8: this exact experiment took fidelity **88% → 22%**; the note is frozen *on purpose* | **Already rejected, empirically.** Do not reopen. |
| "Global constraint/optimization solver over all hypotheses (JPDA/Kalman everywhere)" | §7 (notation graph) + open-problems §X.3: the **7.0 horizon**, gated behind a ground-truth harness that does not exist | **Deferred by design.** Gate stands. |
| Stage 1 cleanup: overtone de-ghost, consolidation, micro-timing preservation | `note_consolidation`, `micro_note_purge`, `onset_refinement`, three-timeline law | **Built.** |
| "Detect pedal / sustain provenance (why is a note still sounding)" | `quantization/sustain_recovery.py` — extends note ends from BP's **posteriorgram** + narrowband f0 energy, opt-in annotation | **Built** (the useful half — see §4-C). |
| Voice separation by pitch+time cost | `instrument_attribution/voice_continuity.stream_into_lines` | **Built** (and measured to fail on dense polyphony — §2). |
| "Rest cleanup / register-continuity voicing / don't over-voice" | exporter `_events_to_voices` (register continuity, measured 8.2→2.5 semitone jump); register+sparse-merge tried and **reverted** (rest ratio 1.10→1.24) | **Built and tuned.** |
| Ties / tuplets / spelling / measures as owned symbolic facts | `tie_reconstruction`, `rhythm_inference` per-beat `is_tuplet`, `key_detector`, `NotationScore` | **Built / partially built.** |
| MusicXML as a view over a symbolic object | `output/notation_score.py` + `output/musicxml_exporter.py` | **Built.** |
| Injection–recovery / ground-truth evaluation | §IX.2, §VI, §XIII.4 — named, ranked, **unbuilt** | **Your open gap** (see §7). |

The essay's contribution to Jazz is therefore *not* its architecture — you have
a better-defended version of that architecture already. It is that it points a
bright light at the **one region your docs mention only in passing**: piano
hands and the grand staff (§XIII.3's "left hand / right hand … undefined").

---

## 2. The three traps — do not let this essay reopen them

Each of these is seductive, and each is a settled question in your own logs.
Naming them so a future read of this essay doesn't relitigate:

1. **Un-freezing the note / per-note multi-hypotheses / "spatial uncertainty
   like SLAM."** The essay's "this note belongs 40% Gesture A / 35% B / 25% C
   until later" is precisely the 5.x design that produced **88% → 22%**
   fidelity. Playback is faithful *because* the note is frozen. Uncertainty
   lives in interpretation layers (voice, duration, hand), never at detection.

2. **The universal relationship / knowledge graph.** §XIV already dismantled the
   identical proposal. A small, notation-driven set of edges on `NotationScore`
   earns its place; "a graph of all musical facts" does not.

3. **A global solver you tune without ground truth.** The essay's JPDA/Kalman/
   annealing-over-everything is the 7.0 horizon. The gate (§7 of the notation
   graph doc) is non-negotiable: **no global solver before the harness.**
   Four conclusions this session were overturned by one measurement; an
   unscoreable optimizer is a confident-guess generator.

The essay's *good* technical instinct — proper **data association** for voice
tracking (JPDA/MHT, deferred commitment, a line holding multiple candidate
continuations) — is already your §XIII.5 / §XIV.5 **#2 lever**, correctly scoped
as "research-grade, its own effort." Keep it there. It is the upstream fix for
the confetti, but it is not what this essay uniquely adds.

---

## 3. The genuinely new, valuable delta: the piano hand/grand-staff cascade

Here is what Jazz has genuinely *not* designed, and what the essay is actually
good for. Today `build_notation_score` makes **one part per (family, voice)** and
`musicxml_exporter._clef_for(mean_pitch)` picks a clef from a static pitch
threshold. For a **piano**, that is wrong in a specific, fixable way:

- A piano is **one instrument on two staves** (grand staff), split by *hand /
  register role*, not one-staff-per-voice and not one static clef.
- The current pipeline has **no concept of hand, no grand-staff pairing, no
  dynamic clef, no cross-staff** — because Jazz grew up on multi-instrument
  jazz stems where "one staff per instrument family" was the right default.

The essay's cascade, stripped of its rejected parts, is the right shape for
*this* gap:

```
frozen Notes (piano stem)
  → [have] consolidation / onset-refine / sustain-recovery / rhythm-inference
  → [NEW] gesture grouping           (§XIII.3 level-above-note: run/arpeggio/trill/tremolo/Alberti)
  → [NEW] hand/staff assignment      (span + movement-cost DP; OUTPUT = staff, hand as evidence)
  → [have] voice within a hand       (voice_continuity, once data-association lands)
  → [NEW] grand-staff engraving      (staff pairing, dynamic clef w/ hysteresis, cross-staff, hidden rests)
  → [have] NotationScore → MusicXML
```

Everything marked `[NEW]` becomes **another edge on `NotationScore`**, built
from evidence, frozen `Note` untouched — exactly the discipline the notation
graph doc already established. Specifically:

- a `hand` / `staff` field on `NotationNote` (like `is_tuplet`, `tie_start`),
- a grand-staff *pairing* concept in `NotationScore` (two `NotationPart`s that
  are one instrument), which the exporter renders as a braced piano part with
  per-staff dynamic clefs instead of two unrelated staves.

This is additive, measurable, and does not touch the frozen detection floor.

---

## 4. Things the essay (and the current pipeline) is still not seeing

### A. Staff assignment ≠ hand assignment. Make **staff** the target; hand is evidence.
The essay treats "which physical hand" as a prerequisite everything hangs on.
But readers need a clean **staff split**; the true hand is (a) partly ill-posed
(expert pianists finger differently) and (b) not what MusicXML encodes. Solve
for **staff** (a two-way, well-defined choice), and let a hand/span/movement-cost
model *inform* it rather than gate it. This de-risks the single most failure-
prone stage and keeps it inside the frozen-note law (staff is an annotation, not
a fact about the note).

### B. Over-splitting is the failure mode — you already have the receipts.
The confetti (776 voices / 2,183 notes) and the reverted sparse-merge
(1.10→1.24 rest ratio) are the *engraving-domain mirror* of the over-detection
arc. Any hand/staff/gesture layer must default to **fewer** objects and make
splitting earn its place. The exporter's register-continuity `_events_to_voices`
is the model to imitate: cheap, measured, rest-neutral.

### C. `sustain_recovery` is the *acoustically correct* half of "sustain
provenance" — and finger-vs-pedal is both under-determined and unnecessary.
The essay wants a state machine (finger-held → pedal-held → resonance → decay).
Jazz already has the part that matters: `sustain_recovery` extends a note to its
true acoustic end using the **posteriorgram + narrowband f0 energy** (so, unlike
a raw Basic-Pitch note list, you *do* keep acoustic evidence). What it doesn't
do — separate "finger held it" from "pedal held it" — is (a) genuinely
under-determined even with the posteriorgram, and (b) **not needed for
engraving**: an over-long, acoustically-supported duration becomes a **tie/held
note**, which `tie_reconstruction` already draws. Keep the one useful spin-off:
**pedal-necessity as a weak prior for hand/staff** ("no single hand could have
held this while the other was busy" → don't charge a hand for holding it). Drop
the state machine.

### D. Enharmonic spelling is a required, forgotten sub-problem.
The exporter passes `key` to music21 but nothing tracks *harmonic function*, so
accidentals (G♯ vs A♭) are music21's guess. This is §XIII.3's "spelling" edge;
it's cheap to name now so the grand-staff work doesn't ship pretty-but-
mis-spelled pages.

### E. Dynamic clef needs **hysteresis**, not a threshold.
`_clef_for(mean_pitch)` per staff is the static rule the essay rightly warns
against. A good engraver switches clef only when the register move is
*sustained*, and penalizes churn. This is a tiny per-staff DP with a switch cost
— a direct, measurable upgrade to the existing function.

### F. The evaluation gap is *smaller* for this work than for detection (§7).
Your §VI lament ("no loss function; the ear is the only oracle") is true for
*musicality* and *detection statistics*. It is **much less true for
staff/hand/voice**, because those have real human ground truth (engraved scores,
fingering datasets). This is the sharpest thing to see: the new cascade is,
ironically, **more supervisable than the work you've been fighting.**

---

## 5. Dueling pianos, placed honestly in Jazz's architecture

In Jazz, two pianos is not a new subsystem — it decomposes onto layers you have:

- **If the two pianos are panned** (most live/studio dueling-piano recordings):
  it's a `separation_engine` problem. The spatial cue (M/S, ITD/ILD azimuth
  masking) becomes **one more witness** feeding source assignment — the
  "helper, not fighter" pattern. Cheap, and the right home is upstream of the
  pitch engine.
- **If they share timbre and are mono/center** (the essay's hard case): this is
  the *same* multi-target data-association problem as intra-stem voice tracking
  (§XIII.5), just with two objects instead of N voices. The **register-role +
  gesture-continuity** machinery from §3 is what disambiguates it — not a
  bespoke net.
- **Cut the neural moonshot.** Fine-tuning a Demucs variant on synthetic
  2-piano MAESTRO mixes is a multi-month research project that yields nothing
  for the actual prize (one complex piano → grand staff). Keep only the
  physical-constraint downstream idea: polyphony/span caps can *reassign*
  ambiguous notes between two already-separated streams.

Net: dueling pianos = **a spatial-cue witness in `separation_engine`** (when
panning exists) **+ the same data-association lever you already ranked #2.** No
new architecture.

---

## 6. Guitar is a different problem, and you already named it the trouble spot

Your docs repeatedly cite "guitar is a mess of polyphony" as *the* page problem.
Two distinct fixes, and the essay conflates them:

- **Guitar in standard notation** = a single treble-8vb staff, rarely >2 voices.
  Here the win is the same voice/data-association lever plus the notation-
  specific **de-merge** you already floated in §XIV.4 (keep guitar as its own
  thinner staff rather than folding it into a brightness-rebucketed "merged
  family"). This is arguably a *bigger, cheaper* guitar win than anything in the
  essay.
- **Guitar TAB** = the genuinely new sibling. Assign each note to a
  `(string, fret)` — a **fretboard-position DP** with its own constraints (6
  strings, 4 fretting fingers, fret-span ≤ ~4–5, open strings, position inertia,
  chord-shape playability). It reuses the *shape* of the hand-assignment DP from
  §3 but swaps the entire constraint model. Treat it as a separate exporter over
  a `(string, fret)` annotation — after the piano cascade proves the DP pattern.

---

## 7. The gate you already wrote — and the one thing I'd add

Your gate stands: **no global solver until the injection–recovery harness
(§IX.2) exists.** But §4-F changes what that harness can be *for this work*:

- For **detection statistics**, keep §IX.2 as written: synthesize audio from a
  known MIDI, mix into real stems, score precision/recall vs. injected truth.
- For the **engraving cascade specifically** (staff, hand, voice, clef), you can
  do better than synthesis, because human answers exist:
  - **ASAP** (Aligned Scores And Performances) — MusicXML scores aligned to
    MAESTRO performances. Real performances *with* the engraved answer.
  - **PIG** (Piano fingering dataset, ~150 pieces) — ground-truth **hand +
    finger** labels. The oracle for the performance layer.
  - **MV2H** (McLeod & Steedman) — the standard multi-layer transcription metric
    (multi-pitch, **voice**, meter, note value, harmony), so results are
    comparable to the literature.
- Scoreable, permutation-invariant metrics for the new layer: **staff accuracy**
  (headline), **voice agreement** (pairwise-F1 over same-voice relations, not raw
  labels), **clef-change count** (penalize churn), **hand accuracy** vs PIG.

This is the piece that makes the piano cascade *not* a confident-guess generator:
unlike the musicality axis, it has a real loss function. Build a thin version of
this harness first, on 5–10 engraved piano pieces, and let it gate every slice.

---

## 8. Smallest measurable first slice

Consistent with your slice discipline, and aimed at the named gap — not "build
the graph":

**Slice 1 — "A piano stem becomes a real grand staff," measured.**
1. Stand up the thin engraving-eval harness (§7) on a handful of engraved piano
   MusicXML pieces; record the *current* Jazz output's staff/voice numbers as
   baseline.
2. Add a `staff` (and optional `hand`) field to `NotationNote`, filled by a
   **span + movement-cost DP over gesture/register** — output is the two-way
   staff choice, defaulting conservatively (few clef changes, few voices).
3. Teach `NotationScore` a **grand-staff pairing** (two parts = one piano) and
   the exporter to render it braced with **hysteretic per-staff clefs** (§4-E)
   instead of `_clef_for(mean_pitch)`.
4. Re-measure staff accuracy + clef-change count against the harness **and**
   confirm the perceptual axis is untouched (frozen notes → playback identical,
   per §XIII.4). Ship only on a measured win.

No dueling pianos, no TAB, no gesture layer yet — each is a later slice justified
by a harness gain. Gesture grouping (§XIII.3) is the natural Slice 2 because it
*also* helps beaming and the tuplet gate, not just hands.

---

## 9. Forks (yes/no)

1. **Is the piano grand-staff / hand-as-evidence cascade (§3) the target?** My
   strong lean: **yes** — it's the one region your docs left open and the essay's
   only genuine contribution to Jazz.
2. **Staff as the primary output, hand as evidence (§4-A)?** Lean: **yes.** Or do
   you specifically want fingering output (needs PIG, harder, more ill-posed)?
3. **Build the engraving-eval harness first (§7), using ASAP/PIG/MV2H?** Lean:
   **yes, non-negotiable** — it's the gate you already wrote, now with real
   labels for this layer.
4. **Dueling pianos: panned-case spatial witness only, cut the neural net?**
   Lean: **yes.**
5. **Guitar: prioritize the notation-specific de-merge (standard staff) over
   TAB?** Lean: **de-merge first** (cheap, big page win you already identified);
   **TAB later** as a sibling of the piano DP.

---

## 10. The owned thesis (extending §8 of the notation-graph doc)

> Jazz already froze the note, already owns the middle (`NotationScore`), and
> already ranked voice-via-data-association as the #2 page lever. What it never
> designed is the **piano's own body** on the page: two hands, one grand staff,
> a clef that moves only when the music does. That is the essay's one real gift —
> and unlike almost everything else Grimlock fights, it has **ground truth**
> (engraved scores, fingering data), so it can be *measured* into place one
> slice at a time instead of guessed. Everything else the essay proposes, you
> have already built, already rejected with data, or already gated behind the
> harness. Build the harness, build the grand staff, and leave the cathedral on
> the horizon.

---

## 11. Deeper: measured against the real artifacts (2026-07-30) — and the retargeted goal

Before forking, I measured my own §3 claim against Jazz's actual produced pages
and stems. It broke in a useful way, and the user's clarified intent reshaped it
into something better-posed.

### 11.1 — The measurements that killed "hand recovery"

- **The page has no piano.** Every produced `transcriptions/*.musicxml` for
  Hopeful shows parts `vocals / bright_lead / mid_body / warm_sustained / bass /
  drums`. Those middle three are **brightness buckets, not instruments.** Piano
  identity does not reach notation.
- **Why: the harmonic stems are not separable.** `separation_engine/stem_merge.py`
  documents it with numbers — htdemucs_6s's guitar/piano/other heads **swap the
  same content between stems over time.** Measured on Hopeful: guitar 1549 +
  piano 1564 + other 1532 = 4645 notes, but the three summed and transcribed
  once = **2183**. **~53% of "piano/guitar/other" notes are the same music
  counted 2–3×.** The merge exists precisely because "piano" is not a stable
  identity.
- **The isolated piano stem is the weakest thing on the record.** Hopeful stem
  energy: vocals rms .084 / drums .070 / bass .052 / **merged-harmonic .039** /
  guitar .026 / other .022 / **piano .012 (active only 12%).** The `piano.wav`
  is mostly bleed, not a performance.
- **The page is a data dump.** Rest/note ratio is **1.1–1.37** on Hopeful (more
  rests than notes) — the confetti, quantified. This is the user's "everything
  is just dumped on the staff."

**Conclusion:** recovering *which hand played what* is a category error on band
material — there is no clean piano, and true hands are unknowable. Drop it.

### 11.2 — The retargeted goal (what the user actually asked for)

> "When we look at the Piano or the Junk Drawer stem — can we get *sensible
> notation* out of it? Cluster notes into chords. Shape it toward piano
> notation. Rhythms in different registers. Right now everything is dumped on
> the staff."

This is not hand recovery. It is a **harmonic reduction**: take the junk-drawer
(merged harmonic / `OTHER`) stem's notes and *present* them as a readable
grand-staff piano reduction. Crucially, **a reduction needs no instrument
identity and no true-hand knowledge** — the grand staff becomes a *presentation
convention for dense harmonic content*, split by **register**, not by hand. That
is well-posed on exactly the messy stem we have, and it is measurable.

### 11.3 — The four moves, grounded in what exists

1. **De-clutter first (mostly built, wire it on).** The rest>note ratio is part
   real polyphony, part over-fragmentation. `note_consolidation` (absorb
   same-pitch fragments), `micro_note_purge`, and `sustain_recovery` (true
   acoustic ends → held notes/ties instead of gaps) already exist and are
   opt-in. A reduction should run them *on* by default — fewer, longer noteheads
   before anything is placed.

2. **Harmonic-aware chord aggregation (upgrade `_chord_events`).** Today the
   exporter groups notes whose onsets fall within a flat **30 ms** window into a
   chord. On a jittery merged stem that both over-merges (unrelated simultaneous
   content) and under-merges (a rolled/arpeggiated chord spread past 30 ms). The
   sharper question the user named — *"what cluster of notes is a chord?"* — is:
   group by **onset proximity + shared offset + harmonic coherence** (do these
   pitches fit a chord template / the local key from `key_detector`?), with the
   window scaled to the beat, not a constant. This is where a chord becomes one
   notehead-stack instead of a smear. **But chord-stacking is only one branch of
   a single decision (see §13): notes stack into a chord only when they share a
   rhythm; when they diverge rhythmically they become separate voices, not a block
   chord.** Block-chording everything is the opposite failure from confetti and
   equally wrong. *Honest limit:* "one chord vs. two instruments at once" is the
   same data-association ambiguity — but for a *reduction* the readable answer
   (group the simultaneity) is usually correct regardless of true source.

3. **Register-based staff split — the achievable "grand staff."** Assign each
   note/chord to **treble or bass** by register, with a boundary that moves only
   when the music sustains a move (**hysteresis / a switch cost**). This is a
   small, bounded **Viterbi/DP over {treble, bass}** — *not* the rejected global
   solver (§2.3): it is local, per-slice, inspectable, and needs no ground-truth
   weights beyond "don't churn the split." It replaces the naive
   `_clef_for(mean_pitch)` and is what makes dense content *look* like piano
   instead of a dump. It is the essay's hand-DP, honestly relabeled as a
   register-presentation DP.

4. **Per-register rhythm falls out.** Once split, each staff voices
   independently (existing `_events_to_voices` register-continuity). The bass
   staff's slower pulse and the treble's faster motion — the "rhythms in
   different registers" — separate naturally because they're now on different
   staves. No new machinery.

### 11.4 — Where to run it, and the one real tension

Run the reduction on the **merged junk-drawer stem** (`OTHER +merged_harmonic`),
*not* the isolated `piano.wav` — because §11.1 proves the isolated stem is
unreliable bleed and the merge is the de-duplicated truth. The reduction is a
**notation view of the junk drawer**, presented as a grand staff. This also
means it composes with, and does not fight, the merge decision Jazz already
made and measured.

The one tension to name honestly: a reduction of *all* harmonic content onto one
grand staff will still be denser than a real piano part, because it contains
guitar + keys + pads at once. That is fine for a **readable reduction** (the
goal), and it is strictly better than today's brightness-bucket dump. If the user
later wants a *true* per-instrument piano part, that requires separation Jazz
does not have (§11.1) — a different, research-grade project.

### 11.5 — Smallest measurable slice (supersedes §8)

**Slice 1 — "The junk drawer becomes a two-staff reduction," measured.**
1. Build a `NotationScore` path that takes the merged-harmonic notes as **one
   instrument**, runs consolidation + sustain-recovery **on**, and emits **one
   braced grand staff** via a register-split DP (§11.3.3) instead of N
   brightness parts.
2. Upgrade chord aggregation to beat-scaled + harmonic-coherence grouping
   (§11.3.2).
3. **Metrics (proxies for "sensible," ear is final oracle):** rest/note ratio
   (target: well under 1.0 vs today's 1.24), noteheads-per-beat per staff (target:
   musically plausible, not 5+), chord-stack rate (are simultaneities becoming
   chords?), and clef/register-split churn (should be low). Compare against the
   current `Hopeful_*.musicxml` baselines already on disk.
4. Confirm the perceptual axis is untouched (frozen notes → playback identical).

This is one instrument, one grand staff, on the stem we actually have — the
cheapest honest test of "can we make the junk drawer read like piano."

### 11.6 — Revised forks (supersede §9)

1. **Target = harmonic *reduction* of the junk-drawer stem onto a grand staff
   (§11.2), not hand/instrument recovery?** Lean: **yes** — it matches your
   stated intent and is the only version well-posed on band material.
2. **Run the reduction on the merged stem, with consolidation/sustain ON by
   default (§11.4)?** Lean: **yes.**
3. **Register-split DP + harmonic chord aggregation as Slice 1 (§11.5)?** Lean:
   **yes** — both are bounded, inspectable, no ground-truth weights needed.
4. **Build the thin proxy-metric harness (§11.5.3) before tuning?** Lean:
   **yes**, but lightweight — these proxies are computable from the MusicXML
   we already emit; no external dataset needed for this slice.
5. **Guitar:** the same reduction machinery gives a cleaner guitar *staff* for
   free once content isn't brightness-bucketed; **TAB stays a later sibling**
   and, per §11.1, only becomes real on genuinely isolated guitar input.

---

## 12. Playability as the objective function (2026-07-30, user steer)

The user's sharpest steer: *once we assume/identify the content is piano, apply
intelligence about whether it is even realistic to **play on a piano**, and only
then engrave it so it makes real-world sense.* This is not a fourth feature — it
is the **objective function** §11's reduction was missing, and it may be the
single most important idea in this whole document.

### 12.1 — Why this is the keystone: an unplayable page is *objectively* wrong

§VI of the open-problems doc laments that engraving has **no loss function** —
"the ear is the only oracle." Playability is a partial escape from that, on the
one axis that matters here: **a page a human physically cannot play is wrong,
with no ear and no dataset required.** Fifteen notes spanning four octaves struck
at one instant is not a hard-to-read chord — it is *impossible*, and its
impossibility is computable. That gives the engraving layer a real, gradient-ish
correctness signal that detection never had. Build the metric and you can
optimize against it.

### 12.2 — Playability does three jobs at once

1. **Reality-check / de-ghost (input sanity).** An "impossible" cluster is
   evidence: it is over-detection (harmonics/ghosts), or it is genuinely
   *multiple instruments* in the junk drawer (which §11.1 measured is common).
   Either way the reduction must prune, split, or fold it — the impossibility is
   the trigger.
2. **Drive the staff-split (§11.3.3 objective).** The register-split DP's
   objective is no longer just "low churn" — it is **"produce two staves each of
   which is physically playable."** Feasibility becomes the cost function, not an
   afterthought.
3. **Force an honest piano *reduction*.** When content genuinely exceeds two
   hands (routine for a summed guitar+keys+pads stem), do what a human arranger
   does reducing an orchestral score: **keep the most salient content, fold or
   drop the rest, present what remains playably.** This is precisely "engrave it
   where it makes real-world sense."

### 12.3 — The constraints, concretely (graded, not binary)

Per instant, and per hand after the split:

- **Span:** one hand covers ≤ ~an octave comfortably; ≤ a 10th (~14–16 st) for
  large hands or a roll; beyond that is unplayable-as-a-block.
- **Finger count:** ≤5 struck notes per hand; ≤ ~8 struck total across two hands.
- **Temporal reachability:** consecutive same-hand events must be reachable at
  tempo — a hand cannot leap three octaves in 50 ms while holding notes
  (movement cost, the essay's inertia term, used for *feasibility* not identity).
- **Pedal as a feasibility *relaxer* (where the pedal idea finally earns its
  keep):** a sustained bass note under moving upper voices need not occupy a
  finger for its whole life if the sustain pedal can hold it. So "held" ≠
  "finger-occupied," and `sustain_recovery` / pedal-necessity re-enters here with
  a concrete job — it *relaxes* the finger-count constraint rather than
  describing provenance for its own sake. (This is the one genuinely useful
  survivor of the essay's sustain state machine, §4-C.)

### 12.4 — The honest hard part: *what* to drop is a salience judgment

Deciding which notes survive a reduction (keep the melody, keep the bass, keep
harmonically essential tones; drop doublings and inner filler) is a
musical-salience call, and here the **ear stays the oracle.** But a crude
salience — note confidence × register-extremity (outer voices matter most) ×
onset-salience/harmonic-fit — gives a defensible first cut, and playability gates
the rest: you drop lowest-salience notes *until the instant becomes playable.*

### 12.5 — Fits the frozen-note law exactly

Every playability judgment is an **annotation**, never a mutation. A note flagged
"unplayable in this cluster / dropped from the reduction" is untouched in the
performance layer — **playback stays faithful** (the three-timeline law), and the
reduction is one *view* among several. This is the same discipline as ties,
tuplets, and voices: witnesses testify, the reduction view decides, the decision
is logged and reversible.

### 12.6 — Revised smallest slice: build the *checker* before the *reducer*

Measure before build (the project's through-line). The cheapest, zero-risk first
step is diagnostic, not generative:

**Slice 0 — the playability checker.** A function that scores any
`NotationScore` (or existing `*.musicxml`) for physical playability per beat:
per-staff span, struck-note count, and same-staff reachability, with the pedal
relaxer applied. Run it on the current `Hopeful_*.musicxml` baselines and **get
the number**: what fraction of the existing page is literally unplayable on one
piano? That single measurement tells us how large the reduction problem is,
gives Slice 1 its baseline and its loss function, and changes no output. Only
after that do we build the register-split-with-playability-objective reducer
(§11.5), now optimizing against a metric that already exists.

### 12.7 — Slice 0 result (2026-07-30): playability is NOT the bottleneck

Built (`output/playability.py`, `tools/playability_report.py`) and run on the
produced pages. The measurement **falsified the premise that the page is an
unplayable dense cluster-dump.** Union of the harmonic parts (bright_lead +
mid_body + warm_sustained), two-hand feasibility per instant, two brackets
(octave / tenth hand):

| Page (harmonic scope) | playable (time) | mean notes-at-once | max | >10 fingers |
|---|---|---|---|---|
| Hopeful_notation | **95.6–98.9%** | 3.3 | 9 | 0.0% |
| Hopeful_familyonly (all crammed to 4 staves) | **97.9–99.4%** | 3.1 | 9 | 0.0% |
| Hopeful_voicefix | **96.6–99.0%** | 3.2 | 10 | 0.0% |
| You_Say (busier arrangement) | **86.4–92.0%** | 4.4 | 14 | 1.2% |

Findings, measured:

1. **True impossibility (>10 fingers) is 0–1.2% of the time.** The "cluster of
   fifteen notes dumped on the staff" is *not what is happening* — instantaneous
   density is mostly ≤5 notes (68–97% of the time), mean ~3–4. A note-*dropping*
   reduction (§12.2/§12.3) has almost nothing to drop. **This deflates the §12
   reduction emphasis** — the checker did its job and overturned it.
2. **The small unplayable sliver is dominated by SPAN, not count** — and span
   failures are mostly the **grand-staff signal, not impossibility**: folding the
   bass in (`harmonic_bass`) spikes span-failures to ~12% with max spans of
   ~66 st, because a bass note sounding under a treble cluster wants **two
   staves** (LH bass clef / RH treble clef / empty middle), which is exactly what
   a register split provides. The pedal relaxer (unbuilt) would reclassify more
   of this sliver as playable, so the true playable rate is *higher* still.
3. **The real mush is horizontal, not vertical.** High note counts (2000–2700)
   + low simultaneity (~3.3) + rest/note ≥ 1.1 means the pages are *thin,
   fragmented, fast-moving lines scattered across voice-slots and off the grid* —
   a rhythm/voicing problem, not a density problem.
4. **Density is song-dependent.** Sparse worship (Hopeful) is ~all playable;
   busier arrangements (You_Say) have a real ~8–14% dense/wide tail. So the
   reduction layer is a *tail-handler for busy songs*, not a front-line stage.

**Consequence for the plan (re-ranked):**

- **Register-split (grand staff) moves to the front.** It is what resolves the
  span sliver *and* what shrinks per-staff polyphony for the voicer. Doubly
  justified by the data.
- **The rhythmic-independence voicer + rhythm cleanup (consolidation ON)** is
  the other real lever — because the mush is horizontal.
- **Playability becomes a guardrail, not a stage.** Keep the checker as a
  regression metric and as the objective term that *keeps each staff of the
  register split playable*; keep note-dropping only as a small tail-reducer for
  the densest instants (mostly on busy songs). It is no longer Slice 1's
  front-half.

The through-line held: a cheap measurement changed the plan before a line of
generative code was written.

---

## 13. Voices, not block chords: rhythmic independence as the primitive (2026-07-30, user steer)

The user's correction to §11.3.2: the target is **not** block chords, and it is
**not** the confetti we get now. It is **independent voices with independent
rhythms** — Voice 1 a syncopated melody, Voice 2 a held inner note or moving
counter-line — which is what real piano writing *is*. Both failure modes we can
already produce (everything-a-block-chord; everything-its-own-line) destroy the
music in opposite directions.

### 13.1 — One primitive generates both chords *and* voices

Chords and voices are not competing strategies; they are the two outcomes of a
single test the copyist runs (the pasted essay had this right, and it is the one
part of it worth keeping verbatim):

> **Do these simultaneous notes share a rhythm?**
> **Yes** (same onset *and* same offset/duration) → one **chord** in one voice.
> **No** (they diverge — one sustains while the other moves) → **separate
> voices**, each with its own rhythm.

So "chord aggregation" (§11.3.2) and "voice separation" are the same decision
seen from two sides. Get the rhythmic-independence test right and both fall out.

### 13.2 — Why syncopation *forces* this

Syncopation is the case that makes it non-optional. A melody note landing off the
beat, over a chord held from the downbeat, has a **different rhythm** from that
chord. Write them as one block chord and you have either lied about the rhythm or
spawned a mess of tied rests to fake it. The *only* clean rendering is Voice 1 =
the syncopated line (its own stems/beams), Voice 2 = the held chord. This is
exactly the user's "syncopated rhythms written more cleanly," and it is
unreachable without real voices. Block-chording is what produces "the cacophony
of shit."

### 13.3 — The measured hard truth (don't hand-wave it)

Proper voice separation on dense polyphony is your **#2 page lever**
(§XIV.5), it is **research-grade** (data association: JPDA/MHT), and your greedy
`stream_into_lines` was measured to produce **776 voices for 2,183 notes** — it
shatters overlapping polyphony into single-note confetti (§6.1). "Just make voice
continuity smarter" *is* the hard problem, not a tweak. It cannot be waved at.

### 13.4 — The unlock: ordering + the 4-voice cap as a *regularizer*

Here is where the last three turns compound into something tractable. The
confetti happened because greedy streaming tried to separate voices across **all**
the harmonic content at once — a summed guitar+keys+pads stem, genuinely 5+ deep,
where "no legal voice merge exists" (§XIV.4). That is the wrong problem to solve.

Reorder so voice separation never sees that mess:

```
1. REDUCE   (§12 playability)  → drop/fold to what one piano can play
2. SPLIT    (§11.3.3 register) → assign to treble / bass staff
3. VOICE    (§13, per staff)   → separate into ≤4 rhythmically-coherent voices
```

Two consequences make step 3 solvable where §6.1 failed:

- **Divide and conquer.** Voicing *one register of a playable reduction* is a far
  smaller, better-posed problem than voicing the full junk drawer. Most of the
  polyphony that caused confetti is gone (reduced) or on the other staff (split).
- **The 4-voice cap is not just an export limit — it is the model.** Finale and
  MuseScore allow **≤4 voices per staff** because that is what human piano
  notation *uses*. So the task is not free clustering (which floats to 776); it is
  **assign these notes to at most 4 voices such that each voice is a coherent
  rhythmic stream.** The constraint tames the search and matches the medium. A
  separator that produces 776 voices simply isn't modeling the target notation;
  bake the cap in and confetti is structurally impossible.

Concretely, step 3 becomes a **bounded assignment** per staff: ≤4 voice slots,
cost = rhythmic incoherence within a slot (onset-grid conflicts, wild
register/rhythm jumps) + gap/rest penalty, with the chord/voice test (§13.1)
deciding whether a simultaneity joins a slot as a chord tone or opens a new slot.
Small K (≤4) keeps it a real search, not a global anneal — it is *not* the
rejected 7.0 solver (§2.3).

### 13.5 — "Intelligently pick out piano," honestly

We cannot isolate a piano *signal* — §11.1 measured that identity is not
recoverable from the merged stem (53% cross-stem duplication). So "pick out
piano" cannot mean audio separation. What it *can* mean, and what actually serves
the goal: **select the pianistically-coherent, playable lines out of the mess** —
the content that survives the playability reduction (§12) and forms rhythmically
coherent voices (§13). The reduction+voice layers *are* the "pick out piano"
step, operating on organization and playability rather than on the impossible
audio isolation. That is the honest, buildable reading of the intent.

### 13.6 — Where this sits in the slice plan

Unchanged order, now with voices located precisely:

- **Slice 0** — playability *checker* (§12.6). Diagnostic, first, risk-free.
- **Slice 1** — reduce + register-split to a braced grand staff (§11.5), voicing
  each staff with the *existing* register-continuity voicer, capped at 4, as the
  baseline.
- **Slice 2** — replace that baseline with the **rhythmic-independence voice
  assignment** of §13.4 (≤4 coherent voices per staff). This is the research-grade
  lever, but now scoped to *one playable staff at a time* instead of the whole
  junk drawer — which is the only reason it is approachable at all.

The through-line: we do not make `voice_continuity` smart enough to voice the
cacophony. We shrink the cacophony first (reduce + split), then voice what's left
under the constraint the medium actually imposes (≤4). Chords and voices both fall
out of one rhythmic-independence test, and syncopation finally gets its own stems
instead of being crushed into a block.

---

## 14. Verdict on the pasted "strategic roadmap" (keep / steal / trash)

A roadmap was proposed: (1) replace `stream_into_lines` with JPDA/MHT
*immediately*, (2) build a synthetic injection–recovery harness, (3) re-run the
776-voice test targeting 2–4 voices. Adjudicated against the measured reality of
§11–§13:

**STEAL (genuinely useful, fold in):**
- **The harness discipline and its metric list.** Precision/recall for **voice
  attribution, tie recovery, tuplet detection** is a good concrete spec and is
  exactly the §7 / §IX.2 gate. Keep it. **Add** two metrics it omits:
  **playability pass-rate** (§12 — needs *no* ground truth, the cheapest real
  loss function we have) and, for the voice/staff axis, scoring against **real
  engraved corpora** (ASAP/PIG/MV2H, §7), which is stronger than synthetic-only
  for notation decisions.
- **"Voice split heuristics: fork-into-chord vs. independent-crossing."** This
  correctly names the gap `voice_continuity` has — it is the §13.1
  rhythmic-independence test. Worth building. (Already ours, but the roadmap
  points at the right hole.)
- **"Establish baselines before optimizing."** Correct and non-negotiable.

**TRASH (actively regressive):**
- **JPDA/MHT as the *immediate* target, on the raw stream.** This is the core
  error. §6.1 already ran voice separation on the full merged stem and got 776
  voices; §XIV measured that a fancier solver (path-cover) *floors at ~177* — it
  **cannot** produce 2–4 voices because the input **is not** 2–4 voices, it is a
  whole band summed together. A better tracker on that input re-learns §6.1 the
  expensive way. JPDA/MHT is **Slice 2**, and only after reduce+split (§13.4)
  shrink each staff into the ≤4 regime where it can succeed.
- **The dependency direction: "don't touch anything downstream until voice
  assignment is solid."** Backwards. The playability reduction (§12) and
  register split (§11.3.3) are *upstream* of the hard voice problem, cheaper,
  independently valuable, and they are **what makes voice separation tractable**.
  Gating them behind the research-grade step inverts the actual dependency.
- **Hoping 2–4 voices *emerges* as an output.** It must be *imposed* as a
  constraint (§13.4): ≤4 is the model, not a target to wish for.

**The reframed fork** (the roadmap's final either/or omits the real first move):
- Not "JPDA-first vs harness-first." The cheapest honest first step is the
  **playability *checker* (Slice 0, §12.6)** — diagnostic, needs no synthesis
  pipeline, runs on the existing `*.musicxml` today, and measures the objective
  function for the reduction work that *precedes* voice separation.
- The **full injection–recovery harness** the roadmap describes is real and
  needed — but its correct home is the **gate before Slice 2 (JPDA/MHT)**, not
  before the reduction. Build the checker now; build the full harness when voice
  separation is actually on the table.

---

## 15. Slice 1 result (2026-07-30): register split validated, voicer is the next wall

Built `tools/engraving_experiments.py`: extract the harmonic note stream from a
produced page once, re-engrave it three ways through the **same production
exporter** (only the staff assignment differs), and measure each identically.
`A_family` = today's brightness-family staves (control); `B_split_fixed` = two
staves split at middle C; `C_split_hyst` = two staves, boundary from a Viterbi
with a switch penalty (hysteresis).

| metric | Hopeful: A → C | You_Say: A → C |
|---|---|---|
| rest/note | 1.13 → **1.02** (−10%) | 0.89 → **0.79** (−11%) |
| worst-staff playability (by time) | 98% → **100%** | 75% → **99%** |
| staves produced | 4 → 3 | 5 → 4 |
| combined playability (raw content) | ~97% (flat) | ~88% (flat) |

Findings:

1. **The register split's clearest, largest win is per-staff playability** — the
   grand staff doing its literal job. You_Say's worst family staff was only 75%
   playable (a brightness bucket spanning too wide a register for one hand);
   after the split every hand is ~99–100% playable. Combined playability is flat
   because it measures the *same notes* regardless of staffing — staffing fixes
   *reachability per hand*, which is the point.
2. **Rest reduction is real but modest** (~10–11% both songs). The split does not
   fix horizontal fragmentation on its own (rest/note stays ~1.0).
3. **Hysteresis beats fixed middle-C** consistently and cheaply (Hopeful
   1.05→1.02, worst 98.5→100; You_Say 0.81→0.79, 98.5→99.2). The DP earns its
   place.
4. **It does NOT yet reach a clean two-staff grand staff, and the breakdown says
   exactly why:** each hand's primary staff **saturates all 4 voices** and spills
   the overflow (82 notes / 3% on Hopeful; 126 / 8% on You_Say) onto an extra
   staff. **The next bottleneck is the voicer, not the split.** The exporter's
   greedy register-continuity voicer over-produces voices; the §13
   rhythmic-independence rule (merge same-rhythm notes into *chord tones within a
   voice*) would absorb that overflow, cut rests further, and land the true
   two-staff grand staff.

**Verdict:** register split (with hysteresis) is validated — adopt it. The
measured hand-off is unambiguous: the **≤4 rhythmic-independence voicer (§13.4)**
is now the highest-value next build, and it is scoped to one playable staff at a
time exactly as §13.4 predicted.

---

## 16. Implemented and measured (2026-07-31): the clean two-staff grand staff

Both wins are now production code (opt-in, view-layer, frozen notes untouched):

- `output/notation_score.py` — `NotationNote` gained optional `voice_index`
  (None = unchanged legacy voicing).
- `output/piano_reduction.py` — `assign_hands` (hysteresis Viterbi split),
  `assign_voices` (the §13.4 ≤4 rhythmic-independence voicer: same-rhythm notes →
  chord tones of one voice; independent lines → separate voices; rare >4-overlap
  instants **merged into the nearest voice as chord tones**, never spilled),
  `build_grand_staff` (assembles the braced two staves).
- `output/musicxml_exporter.py` — honors an upstream `voice_index` verbatim
  (`_voices_from_index`) instead of its greedy fallback. Backward compatible:
  guarded by `voice_index is not None`, so family-mode is byte-for-byte unchanged
  (candidate A reproduced its prior numbers exactly — no regression).

Candidate **D** = the implemented path, measured against A/B/C on the same notes:

| | Hopeful: A → D | You_Say: A → D |
|---|---|---|
| **staves** | 4 → 3 → 3 → **2** | 5 → 4 → 4 → **2** |
| **rest/note** | 1.13 → 1.02 → **1.01** | 0.89 → 0.79 → **0.71** |
| worst-staff playability | 98.5% → … → **100%** | 74.9% → … → **99.2%** |
| max voices / staff | 4 | 4 |

Findings:

1. **The clean two-staff grand staff is achieved on both songs** — the exact
   thing §15 could not reach (it spilled to 3–5 staves). The ≤4 voicer +
   merge-overflow eliminates the spill entirely.
2. **Rests keep dropping.** On the busy song the full path is the best result by
   a clear margin: 0.89 → **0.71** rest/note (−20% vs. family, −10% vs. split-
   only). Sparse Hopeful was already near the floor (1.02 → 1.01).
3. **No note lost.** Overflow notes become chord tones, not casualties (note
   counts flat/within <1%); playability per hand stays ~99–100%.

**Status:** the modules are production-quality and proven, but **not wired into
the Conductor by default** (per the §6.1 discipline — the eye is the oracle for
the page). The `_EXP_D_grandstaff.musicxml` pages are openable now. Wiring a
`grand_staff` opt-in into the notation path is the next step, *after* an eyeball
confirms the rendered page reads as intended.

### 16.1 — De-chop pass (2026-07-31): read well, played choppy → fixed

User listened to `Hopeful …_EXP_D_grandstaff`: **readability improved, but the
performance was choppy/jittery** (and "stiff"). Diagnosed by measuring the D
page: **59%** of adjacent notes in a voice had a rest after them (median a
16th-rest), **52%** were same-pitch rearticulations (machine-gun), **39%** were
≤16th fragments. The grand-staff work was purely *vertical* (staff/voice); the
choppiness is the *horizontal* texture (§12.7), untouched.

("Stiff" is a separate thing: on-grid notation timing, which is correct for a
score — expressive timing lives in the performance-clock MIDI export, not the
page.)

Fix (`piano_reduction._smooth_slots`, on by default via `build_grand_staff(
smooth=True)`): within each monophonic voice, (1) **merge** same-pitch
rearticulations a short gap apart (note_consolidation's insight at the page),
and (2) **legato-fill only small gaps** (≤ a quarter note) — leaving real rests
as phrasing. Changes only the notation clock, never a frozen Note.

Measured, and the first attempt is instructive:

| variant | rest/note (Hopeful / You_Say) | playability (all) | gap>0 | same-pitch adj |
|---|---|---|---|---|
| D (no smooth) | 1.01 / 0.71 | 97% / 88% | 59% | 52% |
| E unbounded legato (fill *every* gap) | **0.08 / 0.02** | **33% / 34%** ✗ | — | — |
| E **bounded** legato (fill ≤ quarter) | 0.86 / **0.40** | **95% / 84%** ✓ | **34%** | **40%** |

Unbounded legato over-sustained — every voice rang continuously, so playability
cratered (a pedal reality the checker doesn't model, but also genuinely too
smeared). **Bounded** legato is the keeper: gaps between notes roughly halved
and mostly closed (median gap 16th → 0), rearticulation and fragments down,
rests cut — while playability held and real phrasing survived. `smooth=True` is
now the `build_grand_staff` default. Knobs: `merge_gap` (½ beat) and `fill_max`
(1 beat) in `_smooth_slots`. Openable: `_EXP_E_gs_smooth.musicxml`.

---

## 17. The standing baseline: raw stems → Basic Pitch (2026-07-31, user law)

**Rule (always, not a one-off):** the ground-truth baseline is the **raw stems
run through Basic Pitch with zero Grimlock** (`tools/naked_basic_pitch.py`).
Every build and every pipeline output is **juxtaposed against that naked-BP
baseline**. If our output deviates too far from what Basic Pitch actually heard,
we are off somewhere — we corrupted the transcription, not improved it.

Why this is the right anchor: BP-on-the-stem is the closest thing we have to
"what the audio contains." Our transforms (consolidation, register split,
voicing, legato) are meant to *reorganize and modestly clean*, never to invent or
drop musical content. §16.1's unbounded-legato failure is the textbook case: it
invented sustain BP never heard (rest/note → 0.02), which is exactly a large
deviation from baseline — and the guardrail below would have flagged it before
the ear did.

Wired into the harness (`tools/engraving_experiments.py --baseline <naked_bp>`):
each candidate now reports, next to its notation metrics, its deviation from the
BP baseline —

- **n/BP** — note-count ratio (added/dropped attacks; consolidation dips it
  modestly, over-merge tanks it),
- **sndT/BP** — total sounding-time ratio, the **sustain-invention guardrail**
  (legato that over-holds sends this far above 1; dropping notes sends it below),
- **pitchΔBP** — pitch-content L1 distance, timing-robust (0 = same pitches in
  the same proportions, 1 = disjoint).

Candidates that fly off (sndT/BP outside ~[0.5, 1.6] or pitchΔ > 0.35) are
flagged `<-- OFF`. The env runs BP straight from the global `python`; only
Hopeful currently has stems on disk, and the merged harmonic stem
(`Hopeful_MERGED_gtr_piano_other.wav`) is the correct baseline for the
junk-drawer reduction.

### 17.1 — First baseline run: the choppiness fix splits in two (2026-07-31)

Baseline: naked BP on the merged stem = **1438 notes, 370 sounding-sec**.

| candidate | rest/note | play | n/BP | sndT/BP |
|---|---|---|---|---|
| A_family (pipeline) | 1.13 | 97% | 1.66 | 1.53 |
| D_grandstaff (no smooth) | 1.01 | 97% | 1.66 | 1.54 |
| **G merge-only** | 1.12 | 97% | **1.35** | **1.57** |
| F legato ≤0.5 beat | 1.01 | 97% | 1.35 | 1.69 `OFF` |
| E legato ≤1.0 beat | 0.86 | 95% | 1.35 | 1.95 `OFF` |

The baseline localized the choppiness fix into two independent mechanisms:

- **pitch content is faithful throughout** (`pitchL1` ~0.08 for every candidate)
  — the reduction never invents/drops pitches.
- **same-pitch MERGE** (de-jitter) moves us *closer* to BP: n/BP 1.66 → **1.35**.
  Pure win, always on.
- **legato GAP-FILL** (de-chop) is the *only* thing that drifts: sndT/BP climbs
  1.57 → 1.69 → 1.95 as fill grows. It invents sustain BP never heard.

Decisive nuance: the pipeline's evidence-based `sustain_recovery` already
extended notes to their true acoustic ends (that is the 1.53×) and the page is
*still* gappy — so a real share of the "choppiness" is **faithful to the
source** (comping keys/guitar genuinely have gaps). Blind legato past that point
is prettier but less true.

**Decision (fidelity-first, per §17):** `build_grand_staff` defaults to
**merge-on, legato-off** (`fill_max_beats=0`) — the faithful setting (G), which
sits at baseline (1.57, unflagged) yet still beats the pipeline (2 staves not 4,
de-jittered, note count closer to BP). Legato is a documented dial
(`fill_max_beats`) for trading fidelity for smoothness by ear, not a default.
The genuinely honest de-chop is *more* posteriorgram-based sustain evidence, not
uniform gap-filling — a larger, separate lever.
