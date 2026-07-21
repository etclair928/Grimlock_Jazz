# Grimlock 6.0 — The NotationScore and the Notation Graph

*This is a rewrite of a pasted "Musical Knowledge Graph" proposal, reworked
until it is a thing I would actually build and defend. The original's core
instinct is right and is kept. Three of its load-bearing mechanisms are cut,
each against specific evidence — two of them measurements this project
generated in July 2026. What remains is smaller, buildable, and aimed at the
one problem the user has confirmed by ear is the frontier: the page, not the
sound.*

---

## 0. What I keep, what I cut, and why — up front

**Kept (the one true idea):** *Intent, Performance, and Representation are
three different things and must not live in one structure.* MIDI fails
because it tries to be all three. This separation is real, and it is the
whole case for a `NotationScore` object sitting between the pipeline's
evidence and its exporters.

**Kept and renamed:** the "graph." The annotations we already produce are a
graph waiting to be connected — but only a *small, notation-driven* subset of
edges earns its place. Not a general knowledge graph of everything.

**Cut — multiple hypotheses at the detection floor.** The proposal's
centerpiece example (one note carrying Beat 3 @0.61 / Beat 4 @0.54 / Triplet
@0.38, and by extension negotiable pitch) is the exact design that took 5.x
fidelity from **88% → 22%** (bible §8). We froze the `Note` on purpose. The
proof it was the right call is empirical and recent: **playback is faithful
and the page is a mess.** If premature commitment were the disease, the sound
would be wrong too. It isn't. Hypotheses belong to *interpretation* layers
(duration, timing, voice), never to detection.

**Cut — anchors feeding the tempo/meter resolver.** The proposal says "don't
detect beats, detect anchors; they are fixed, everything stretches around
them." We *have* an `Anchor` type with full alignment machinery. On
2026-07-20 I wired confirmed kick attacks in as anchors to the lattice tempo
witness and measured the result on two songs: **Hopeful 148→153.8, No Pasarán
132→136.4 — both wrong.** Anchoring made one witness more confident, which let
it survive the referee's exclusion filter and drag the weighted tempo toward
a known-biased reading. Anchors are not free. They belong to a consumer that
reads a grid *directly* (see §5), never to a contested vote the referee is
balancing.

**Cut — the global probabilistic constraint solver.** Maximum-weight clique,
simulated annealing over "all hypotheses," "find the world that violates the
fewest rules." This is a factor graph, it is real CS, and it is already
recorded as the **7.0 horizon** (open-problems §X.3). It cannot be built now
for one blunt reason the proposal never addresses: **you cannot tune
constraint weights without ground truth, and we have none.** Four separate
conclusions this session were overturned by a single measurement or by the
user's ear. A global optimizer scored against nothing is a confident-guess
generator — precisely the inherited reflex (bible §10) 6.0 exists to break.
The prerequisite for this layer is the injection–recovery harness (§IX.2),
which is unbuilt. Weights before the harness is cargo-culting.

**Cut — the full rewrite.** 6.0 was built to be *leaner* than 5.6.1. The
detection layer has now survived two hostile external architecture audits and
a whole-pipeline change (the harmonic-stem merge) with zero regression. You
do not rewrite what is load-bearing and working. You extend it at the one
seam that is genuinely missing.

That seam is `NotationScore`.

---

## 1. The separation, mapped to what already exists

The proposal's cleanest contribution is this table, so keep it — but grounded
in the actual codebase, which is already two-thirds of the way there:

| Layer | Stores | In Grimlock today |
|---|---|---|
| **Performance** | physical events | `Note` — frozen, `AnnotationStore` keyed by `note.id`. **Have it.** |
| **Intent** | musical relationships | scattered across annotations; **no object owns it.** The gap. |
| **Representation** | views | `scribe_engraver` → MIDI. One exporter, re-deriving structure ad hoc. |

We froze Performance (§2.2). We have a rich but *flat* interpretation layer.
What we lack is the middle object that turns interpretation into the
relationships a copyist actually uses — and a Representation layer that
*reads* that object instead of re-inferring measures and voices every export.

---

## 2. Why a graph at all — and which edges earn their place

A general "graph of all musical facts" is scope-creep. The disciplined
question is: **which relationships does a human copyist consult to lay out a
page?** Those, and only those, are the edges worth materializing, because
those are the edges that fix the problem we actually have.

| Edge | Drives on the page | Do we compute the input today? |
|---|---|---|
| note → **voice** | which staff, stem direction, beaming group | **Computed, then dropped.** `voice_continuity.stream_into_lines` produces `VoiceLine`s each with a stable `line_id`. But the Conductor uses those lines *only* to read back a family label for logging (`conductor.py:290`) and never writes `line_id` as a note annotation — so the engraver, which reads only annotations, can group solely by `instrument_family` (shared across all of an instrument's voices). That collapse is the guitar mush. |
| voice → **staff** | staff layout, treble/bass split | No — but trivial once voices survive. |
| notes → **beam group** | beaming, sub-beat grouping | Partially — beat grid exists; grouping rule does not. |
| note → **tie** | tied notation vs. re-attack | **Yes** — `tie_reconstruction` writes `tie_candidate` annotations. |
| notes → **tuplet** | triplet/tuplet brackets | Partially — `rhythm_inference` has ternary support. |
| note → **harmony** | accidental *spelling* (G♯ vs A♭) | `key_intelligence` gives a key; harmonic function is not tracked. |
| detections → **gesture** | one trill/gliss/roll = one symbol, not N notes | **No.** The open trouble (§XIII.3): nothing groups N detections into one notated object. |

Every "Yes" and "Partially" is evidence the raw material exists and is being
*discarded at the engraver* rather than missing. That is the whole opportunity
in one sentence: **we already infer most of what a copyist needs, then flatten
it on the way out.**

---

## 3. `NotationScore` — the object that owns the middle

One object, built by reading annotations + the edges above, consumed by every
exporter:

```
annotations + edges  ──►  NotationScore  ──►  { MIDI, MusicXML, LilyPond, PianoRoll }
```

`NotationScore` owns exactly what MuseScore currently has to *guess* because
MIDI can't carry it:

- **measures** (from the resolved tempo/meter + beat grid)
- **voices**, each assigned to a **staff** (from `VoiceLine`s, no longer
  flattened)
- **ties, tuplets, beams** (from existing annotations, made explicit)
- **spelling** (from key; harmonic function later)
- **gesture groups** (the trill/gliss/roll level above the note — §XIII.3)

Critically, it is built with the **confidences we already have**, and it is
**deterministic and inspectable** — an ordered set of engraving decisions,
each traceable to its evidence, not a black-box annealer. If a decision is
wrong, you can see which annotation drove it. That is the same discipline as
the rest of 6.0: witnesses testify, one place decides, the decision is
logged.

This is not new scope. It is already the **#1 lever** in the open-problems
doc (§XIII.6, §XIII.8). This document is the design for it.

---

## 4. What the exporters become

Views. The proposal is right about this and it costs nothing to state
plainly: once `NotationScore` exists, MIDI stops being the truth and becomes
one serializer among several. MIDI serializes the performance clock;
MusicXML/LilyPond serialize the notation clock; a piano-roll view serializes
either. None of them re-derive measures or voices — they read them.

This also retires a real bug we already hit: the notation-timing snap is
currently computed in the Conductor from *unrefined* onsets, so notation mode
inherits a loose base. With `NotationScore` owning the symbolic layer, snapping
happens once, in one place, on the corrected timeline.

---

## 5. Timing: anchors and warp, placed honestly

The proposal's marching-band model — fixed anchors, interior events stretching
between them — is musically correct and I keep the *intuition*. But this
session pinned down exactly where it may and may not act:

- **Onset refinement (shipped, user-confirmed by ear, 2026-07-19)** is the
  first working instance of "align interior events without moving the pulse."
  It corrects where a note *began* against its own stem's transients,
  annotation-only, bounded, declining when witnesses disagree. Vocals — the
  slowest-speaking stem — showed a real systematic −15.8 ms late bias; it was
  corrected; the lines sit better. That is the anchor/warp idea working, in
  its safe form.
- **Anchors into the resolver (measured, reverted, 2026-07-20)** is the same
  idea in its *unsafe* form. Feeding anchors into the tempo vote moved a
  correct answer wrong. The lesson is precise: anchors may inform a consumer
  that reads a grid **directly** — a future notation-grid phase-lock inside
  `NotationScore`, where "put the kicks on the grid" is the entire job and
  there is no witness reconciliation to skew — but never a contested scalar.

So the warp function belongs *inside* `NotationScore`'s barline placement,
anchored on confirmed kicks/downbeats, correcting the notation clock only —
with the performance clock (frozen `Note.start_ms`) untouched, exactly as the
three-timeline law already requires. That is the owned home for the
proposal's best timing idea.

---

## 6. First slice — small, measurable, aimed at a named failure

Not "Phase 1: build the graph foundation (weeks 1–2)." One concrete cut that
uses data we already compute and targets a page-problem the user has named
("guitar is a mess of polyphony... it's still polyphony that is a trouble
spot"):

**Persist `line_id`, then promote voices to staves in a MusicXML export.**

The gap is one missing annotation, not a rewrite:

1. `voice_continuity` already assigns every note to a `VoiceLine` with a
   stable `line_id`. **Write that `line_id` as a `voice` annotation** — the
   one line the Conductor is currently missing (it computes the lines and
   uses them only for family logging).
2. Build a minimal `NotationScore` that reads the `voice` annotation and
   carries voice→staff assignment.
3. Emit MusicXML with each voice on its own staff/voice.
4. **Measure the two axes separately** (§XIII.4): does it still *sound* right
   (performance clock unchanged — it must), and does the guitar staff stop
   reading as mush (notation clock)? The user's eye is the metric; there is no
   other ground truth yet.

The point of starting here: the expensive part (separating the voices) is
already done and *discarded*. The first slice is mostly plumbing a value
that already exists to a consumer that can use it — the cheapest possible
proof that `NotationScore` earns its keep.

### 6.1 — Slice 1 result (2026-07-20): the layer works, the premise doesn't

Built and run on Hopeful's merged harmonic stem. `NotationScore`, the `voice`
annotation, and the MusicXML serializer all work: each voice became its own
staff exactly as designed. But the result **falsified the premise** in the
most useful way:

- **776 voices for 2,183 notes** — ~2.8 notes each; **46% are single-note
  "voices."** Splitting these onto 776 staves is *worse* than the 3 dense
  family staves it replaced.
- **Root cause, one stage upstream:** `stream_into_lines` continues a line
  only when the next note does not overlap the last (monophonic by
  construction). On dense polyphony — guitar + piano + other all sounding at
  once — nearly every note overlaps its predecessor, so nearly every note
  *starts a new line*. Greedy monophonic streaming **shatters polyphony into
  confetti.**

So the premise "voice separation is computed and merely discarded" is only
half true: it is computed, and on this input it is *garbage*. The
`NotationScore` layer is validated; the blocker moved to voice separation
itself. **Voice-split is deliberately NOT wired into the pipeline** — with
these voices it degrades the page.

This is the empirical confirmation of §XIII.5: voice tracking is a
**data-association** problem, and `voice_continuity`'s greedy pitch-proximity
streaming is "a crude first-order version of it." It breaks exactly where the
theory predicted — dense, overlapping, ambiguous assignment. The real unlock
before the voice edge pays off is **replacing greedy streaming with proper
data association** (JPDA/MHT-style: overlap does not force a new line; a line
holds multiple candidate continuations; commitment is deferred). That is the
next slice, and it is upstream of `NotationScore`, not inside it.

If it reads better, `NotationScore` has earned its first edge and we add the
next (ties → tuplets → spelling → gestures), one measured slice at a time. If
it doesn't, we've spent a day and learned the voice lines aren't as separable
as `voice_continuity`'s confidence suggests — which is itself worth knowing
before building anything larger.

---

## 7. Where the graph is allowed to grow — and the gate

The proposal's Layers 4–6 (constraint engine, probability graph, world model)
are not wrong as a *destination*. They are wrong as a *starting point*,
because every one of them needs weights, and weights need a scoreable target.

The gate is explicit and non-negotiable: **no global constraint solver until
the injection–recovery harness (§IX.2) exists.** That harness — synthesize
audio from a known score, run the full pipeline, score the recovered
`NotationScore` against the truth on both the perceptual and the notation
axis — is the only thing that turns "this constraint weight feels right" into
"this constraint weight measurably improves recovery." Until it exists, every
weight is a guess, and a system of guesses that never crashes is the failure
mode we already named.

Build the harness. Then, and only then, does the world-model layer become
engineering instead of astronomy.

---

## 8. The owned thesis

The pasted document ends: *"The next step is not a better quantizer. It is a
Musical Knowledge Graph."*

Here is the version I'll defend:

> The next step is a **`NotationScore`** — one symbolic object that stops
> discarding the voice, tie, and phrase relationships we already infer, and
> lets every exporter read structure instead of re-guessing it. It is a graph,
> but a small one, and every edge in it must earn its place by changing what
> lands on the page. The detection floor stays frozen — that is why the music
> sounds right. The world-model cathedral stays on the horizon, gated behind a
> ground-truth harness we have not built. What we build now is the missing
> middle, and we build it one measured slice at a time, starting with voices.

The problem was never that Grimlock lacked a grand architecture. It's that the
relationships a copyist needs are computed and then thrown away one layer too
early. Catch them in one object, and the page follows.
