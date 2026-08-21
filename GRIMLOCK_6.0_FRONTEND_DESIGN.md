# Grimlock Jazz 6.0 — Front End Design

Status: **design, not built.** Decisions below were taken with the user on
2026-08-20. Nothing here has shipped; this document exists to be argued with
before any of it is written.

---

## 1. The three facts that shape everything

Every UI decision in this document falls out of one of these. They are
measured, not assumed.

**A run costs 30–60 minutes. A re-export costs seconds.** The full pipeline on
the 411-second Chopin recording took 2105s. Re-exporting notation from the
saved `.pkl` takes about five. That is a factor of four hundred, and it is the
shape of the application: *analyse once, re-notate many times.* Any design
that makes a user wait forty minutes to discover the meter is wrong is the
wrong design.

**A thirty-second probe decides whether to spend thirty minutes.** The solo
detector reads the master audio before separation and answers "one instrument
or many". On the Chopin recording, skipping Demucs on that verdict moved F1
from 0.507 to 0.603 — the largest single gain in the engine's recent history,
and it came from *not running a model*. That probe is a confirm-before-commit
gate, and it belongs in front of the user rather than buried in a log.

**The engine can already explain itself per note.** `AnnotationStore.for_note(id)`
returns the complete ordered history of every opinion any pass ever formed
about one note. `MusicBox` carries the same thing at stage level, with a
`reasoning` string on every record. No other part of this project is as
unusual, and a front end that ignores it is wasting the only thing that is
hard to rebuild elsewhere.

---

## 2. Decisions taken

| Question | Decision | Consequence |
|---|---|---|
| Purpose | **Two explicit modes** — Transcribe and Lab | Neither job compromises the other; more surface to build |
| Platform | **Desktop app** | Native file handling, no server, no browser |
| Notation | **Hand off to MuseScore** | The app never renders MusicXML |
| Toolkit | **tkinter** | Only GUI toolkit present; see below |

**Why tkinter, plainly.** It is the only GUI toolkit in the environment. Qt is
absent, and the `.venv` is shared with Symphony and layered over a fragile
global torch/librosa stack that the README explicitly warns against
disturbing. Installing PySide6 into it to get prettier widgets is a poor
trade.

It is also a smaller compromise than it sounds, *because* notation is handed
off. A notation renderer would have made toolkit choice critical. A control
panel, a progress feed, some tables and a diagnostics dashboard are things
tkinter does adequately.

**What the notation decision costs, stated honestly.** With no rendered score
there is no clicking a notehead to see its history. That capability does not
disappear, it changes shape: it becomes a *filterable note table* — "notes
flagged as octave stacks", "notes where the tempo witnesses disagreed", "the
four bars where key stability dropped". Same information, less immediate, and
its job becomes pointing you at a bar number to open in MuseScore.

---

## 3. Transcribe mode

The path for "I have audio, I want a chart."

```
  1  PICK        drag audio in, or browse
        |
  2  PROBE       ~30s, no separation. Reports:
        |          - solo / ensemble, with the three numbers that decided it
        |          - a fast tempo + meter estimate
        |        User CONFIRMS or OVERRIDES: solo piano / solo guitar / full band
        |
  3  OPTIONS     notation timing, consolidation, university mode,
        |        drop purge candidates, guided tempo/meter/key
        |
  4  RUN         live stage feed, tailed from the MusicBox JSONL
        |
  5  RESULT      the health panel (section 5), then:
                   [Open in MuseScore]  [Show folder]  [Re-export notation]
```

**Step 2 is the point of the whole mode.** It is where a user gets to disagree
with the machine before paying for the disagreement. The probe shows its
working — kit HF fraction, crescendo fraction, low-frequency fraction — because
a verdict nobody can audit is a verdict nobody should act on, and this one
skips an entire stage of the pipeline.

**The override matters as much as the verdict.** The solo detector is
deliberately asymmetric: it says ENSEMBLE unless the evidence is clear, because
a false "solo" silently discards real instruments while a false "ensemble"
only costs time. A user who *knows* the recording is solo piano should be able
to say so and reclaim the thirty minutes. Equally, the instrument label
(piano vs guitar) rests on an uncalibrated boundary — there is no clean solo
guitar recording in the corpus — so it is presented as a guess and is directly
editable.

**Step 5 leads with trust, not with statistics.** See section 5.

---

## 4. Lab mode

The path for "is the engine getting better, and where is it wrong."

- **Runs list** — every `.pkl` in `transcriptions/`, with its tempo, meter,
  key, note count and date. This is the session's working set.
- **Re-export panel** — change notation options, re-export, see the health
  panel update in seconds. The fast loop that makes iteration possible at all.
- **Compare** — two runs side by side on every health metric. Built for the
  question this session kept asking: *did that change actually move anything?*
- **Score against an answer key** — precision, recall, F1, timing moments.
  Wraps `tools/score_vs_answer_key.py`, which already exposes library
  functions rather than only a `main()`.
- **Note table** — filter by annotation kind, jump to a bar number.
- **Decision log** — the MusicBox ledger, filterable by stage, with the
  `reasoning` string shown in full.

**One rule for this mode.** Every number it shows names the tool that produced
it. This session reversed three verdicts by re-measuring on better data; a
dashboard that displays numbers without provenance would have made that
harder, not easier.

---

## 5. The health panel

The shared component, and the most valuable thing here. Everything in it is
already computable today.

| Check | Source | Reads as |
|---|---|---|
| Barline crossings | probe of the MusicXML | **must be 0** — hard rule |
| Off-grid onsets | probe of the MusicXML | **must be 0** — hard rule |
| Junk tuplet ratios | `tools/tuplet_audit.py` | count + the ratios themselves |
| Tuplets off the beat | `tools/tuplet_audit.py` | count |
| Key + stability | `key_intelligence.key_stability` | key, agreement %, modulation flag, window strip |
| Playability | `output.playability` | % impossible, against the edition's 0.1% floor |
| Octave stacks | `acoustic_witness.octave_stack` | count, with its "only helps on stem-bleed" caveat |
| Notes by stem | `PipelineResult` | the phantom-stem smell test |

Two design rules for it:

**Hard rules render as pass/fail, everything else as a number with a
reference.** "Barline crossings: 0" is a rule being met. "Playability: 0.3%"
means nothing without "published edition: 0.1%" beside it. A dashboard of
bare numbers invites false confidence.

**The key window strip is a picture, not a value.** Rendering Chopin's
`B D#m D#m D#m Ab Eb Eb D#m B B B B` as twelve coloured cells communicates
"opens in B, wanders, returns to B" instantly, where "key: B, confidence 0.38"
communicates almost nothing. It is one of the few places a graphic genuinely
beats a number here.

---

## 6. Process model

The hard part, and the reason this is not a weekend of widgets.

**The pipeline blocks for the better part of an hour, prints to stdout, and
loads multi-gigabyte models.** It cannot run on the UI thread and should not
run in the UI process at all — a Demucs OOM should not take the window with
it.

```
  tkinter process                     worker subprocess
  ---------------                     -----------------
  launch run  ------------------->    tools/run_full.py-equivalent
                                        |
  tail JSONL  <-------------------    MusicBox(log_path=..., buffer_size=1)
  poll exit code <----------------    writes .mid / .musicxml / .pkl
  read .pkl for diagnostics
```

**Live progress needs no new plumbing.** `MusicBox` already appends
`ForensicRecord`s and flushes them to JSONL. Construct it with
`buffer_size=1` and every decision lands on disk the moment it is made; the UI
tails the file. The stage feed is then not a progress bar someone invented —
it is the engine's own audit trail, shown live.

**Cancellation is killing the subprocess.** Honest and simple. Partial output
is discarded rather than presented, because a half-finished transcription that
looks finished is worse than none.

---

## 7. What the codebase needs first — **DONE**

Three gaps, all now closed, plus a fourth the user added.

1. ~~`transcribe_file` cannot be given a MusicBox.~~ **Done.** It now takes
   `music_box`, so a front end can hand in
   `MusicBox(log_path=..., buffer_size=1)` and tail the ledger live.

2. ~~No probe-only entry point.~~ **Done** — `app/probe.py`. Returns the solo
   verdict with the three numbers behind it, a rough tempo/meter, and
   `plan()`: one sentence saying what a run *will* do, phrased as a prediction
   rather than a recommendation, because the user is confirming a plan and not
   grading a classifier.

3. ~~The diagnostics are CLI scripts.~~ **Done** — `app/diagnostics.py`
   returns the whole health panel as data.

4. **GUIDED SEPARATION** (added by the user). `transcribe_file` takes
   `guided_separation`: `None` for auto, or `solo_piano` / `solo_guitar` /
   `ensemble` as a hard lock. It follows §2.7 exactly — a guided value SKIPS
   the detector rather than out-voting it, because a verdict that cannot
   change the outcome is not worth computing. The vocabulary is deliberately
   the same words the probe returns, so an overriding user is speaking the
   machine's own language rather than a parallel set of flags.

### One thing this shook out, worth recording

The first draft of `check_tuplets` reimplemented the anchoring rule and
reported **44 off-beat groups on a page that has 9**. It demanded a whole-beat
anchor — which the published edition disproves, since 4:3 and 8:2 groups
legitimately anchor to the second eighth of a beat — and it read
offset-within-part where the rule is about beat-within-measure.

`tools/tuplet_audit.py` already had the calibrated version, having had three
bugs beaten out of it by being made to pass the edition first. The fix was to
split that tool into `measure_tuplets()` (returns) and `audit()` (prints), and
have the panel call the former. A test now asserts the two agree exactly, and
a second asserts the printer contains no rules of its own so they cannot drift.

This is the whole argument for section 4's rule — *every number names the tool
that produced it* — arriving one hour after the rule was written.

---

## 8. Build order

Each slice is useful on its own, which is deliberate: if the project stops
after any one of them, what exists still earns its place.

1. **`app/diagnostics.py`** — one library module returning the health panel's
   numbers for a given `.pkl` + `.musicxml`. No UI. Immediately replaces four
   scratch scripts.
2. **Probe entry point** + the `music_box` parameter. Still no UI.
3. **Transcribe mode, single window** — pick, probe, confirm, run, tail, health
   panel, open in MuseScore. This is the first thing that is an *application*.
4. **Lab mode** — runs list, re-export loop, compare.
5. **Note table and decision log** — the explain-yourself surface.

Slice 1 is the recommended start. It is the only one that is pure gain with no
UI risk attached, and everything after it consumes it.
