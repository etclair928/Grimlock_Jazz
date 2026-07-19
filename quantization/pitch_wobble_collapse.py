# =================================================================
# MODULE: quantization/pitch_wobble_collapse.py
# NOT WIRED INTO THE PIPELINE. Built and saved per explicit user
# request during a design discussion about Symphony's old
# PitchWobbleCollapse (which caused real problems in 5.6.1 and was
# declined - see the "skip PitchWobbleCollapse" project memory). This
# is a ground-up redesign, not a port, scoped to the one case the user
# identified as actually real rather than hypothetical: a vocalist
# singing genuinely between the cracks of 12-TET (a blues neutral
# third, a scoop that settles just shy of the target semitone, etc.).
#
# THE PROBLEM THIS TARGETS (narrow, not general jitter/ornament
# discrimination): Basic Pitch's note posteriorgram is a fixed 88-bin,
# one-bin-per-semitone grid (see pitch_engine.basic_pitch_engine's
# BASIC_PITCH_POSTERIORGRAM_BASE_MIDI_PITCH/_N_BINS) - it has no bin
# for a pitch that genuinely, physically sits between two semitones.
# A singer sustaining exactly such a pitch gives the model no home bin;
# frame to frame it has to pick one of the two adjacent bins it
# straddles, and can flicker between them for the whole sustained note.
# That produces a stutter of short, near-pitch (usually 1 semitone
# apart), low-gap notes that LOOK identical in symbolic shape to a fast
# real chromatic run or ornamental figure - the two are only
# distinguishable by evidence Basic Pitch's own note-level output
# doesn't carry.
#
# WHY THIS VERSION IS SAFE TO EVENTUALLY WIRE IN (the general EWAD
# proposal this was workshopped from was not, for two reasons the user
# and I agreed on):
#   1. It never merges or mutates a Note. find_wobble_groups() only
#      returns candidate groupings; a caller (whenever this gets wired
#      in) would write a `wobble_group` Annotation per the established
#      pattern (sustain_recovery.py, micro_note_purge.py,
#      tie_reconstruction.py) - annotation, never mutation (SS2.2).
#   2. It does NOT invent subsystems Jazz doesn't have (no harmonic
#      map, no per-chain instrument classifier, no 5-6-factor weighted
#      probabilistic blend with unvalidated magic numbers). It uses
#      only: (a) structural gates computable from Note fields alone,
#      and (b) an OPTIONAL continuous-pitch (non-quantized) witness
#      the caller supplies - meant to eventually be CREPE run against
#      the vocals stem specifically (Jazz already runs CREPE for bass;
#      running it for vocals too is the natural, not-yet-done,
#      prerequisite for actually using this module for real).
#
# GATES (kept from the workshopped design, the ones that survived
# critique - the harmonic-alignment gate, instrument-specific threshold
# table, and heavy weighted blend were all cut as premature/unfounded):
#
#   Gate 1 - Global anchor radius: fixes 5.6.1's real chaining-drift
#   bug (pairwise-only comparison let 4 steps of 0.8 semitones each
#   drift the group 3.2 semitones off its start, erasing a real minor
#   third). Every candidate group is tested against its OWN highest-
#   confidence member (the anchor), re-evaluated as the group grows,
#   not against whichever note happened to be added last.
#
#   Gate 2 - Tempo-adaptive duration/gap ceilings: a flat 150ms cutoff
#   deletes real sixteenth-note lines at high tempo (83ms/note at
#   180bpm). Ceilings scale down from the resolved tempo, never up
#   past the original flat constants.
#
#   Gate 3 - Directional structure: a real run is monotonic (pitch
#   differences share a sign); genuine two-bin quantization flicker is
#   not. A LOCAL/global split additionally protects ornaments (turns,
#   enclosures) that are globally non-monotonic but locally structured
#   in each sub-window - those are deliberate figures, not tracking
#   noise, and must not be flagged even though a naive global-only
#   monotonicity test would miss them.
#
#   Gate 4 - Continuous-pitch corroboration (required, not optional,
#   for an actual collapse verdict): the caller-supplied continuous f0
#   trace across the candidate span must be genuinely flat (small
#   spread in semitones) - i.e. one real physical pitch sitting near a
#   bin boundary, not a real pitch change. Structural evidence alone
#   (Gates 1-3) is NEVER enough on its own to flag a group; absent a
#   continuous-pitch reading, find_wobble_groups reports no groups
#   rather than guessing - the same "insufficient evidence, do nothing"
#   discipline the rest of quantization/ already follows.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from core import Note

PITCH_WOBBLE_ANNOTATION_KIND = "wobble_group"

PITCH_WOBBLE_ANCHOR_RADIUS_SEMITONES = 1.2     # Gate 1: max distance any member may sit from the group's highest-confidence anchor
PITCH_WOBBLE_MIN_STEP_SEMITONES = 0.01         # Below this it's the same pitch - VelocityMerge's job, not this pass's
PITCH_WOBBLE_MAX_MEMBER_DURATION_MS = 150.0    # Gate 2 ceiling before tempo scaling
PITCH_WOBBLE_MAX_GAP_MS = 30.0                 # Gate 2 ceiling before tempo scaling
PITCH_WOBBLE_SIXTEENTH_DURATION_FRACTION = 0.85  # Fraction of a sixteenth-note's duration used as the tempo-scaled member-duration ceiling
PITCH_WOBBLE_LOCAL_WINDOW = 3                  # Gate 3 sliding-window size (in pitch differences) for the ornament check
PITCH_WOBBLE_LOCAL_MONOTONICITY_MIN = 0.8      # Gate 3: local windows this monotonic or higher mark a deliberate ornament
PITCH_WOBBLE_GLOBAL_MONOTONICITY_MAX = 0.5     # Gate 3: global monotonicity at or above this marks a real directional run
PITCH_WOBBLE_MAX_CONTINUOUS_PITCH_SPREAD_SEMITONES = 0.6  # Gate 4: continuous f0 spread must stay within this to count as "flat"


@dataclass(frozen=True)
class WobbleGroup:
    """A candidate cluster of Basic-Pitch note fragments that Gate 4's
    continuous-pitch evidence confirms are one physical, sustained,
    off-grid pitch rather than distinct notes. `member_note_ids` is in
    time order; `anchor_note_id` is the highest-confidence member -
    whatever eventual collapse a caller performs should adopt ITS pitch,
    not an average. Nothing in this module writes this as an
    Annotation or touches a Note - that is left to whoever wires this
    in later."""
    member_note_ids: Tuple[str, ...]
    anchor_note_id: str
    anchor_pitch: int
    continuous_pitch_spread_semitones: float
    reason: str


def _tempo_adaptive_ceilings(tempo_bpm: float) -> Tuple[float, float]:
    """Returns (max_member_duration_ms, max_gap_ms), scaled down from
    the flat constants once a sixteenth note at this tempo would be
    shorter than the flat ceiling - never scaled UP past the flat
    constants (a slow ballad shouldn't suddenly treat a genuinely long
    note as wobble-eligible)."""
    if tempo_bpm <= 0:
        return PITCH_WOBBLE_MAX_MEMBER_DURATION_MS, PITCH_WOBBLE_MAX_GAP_MS
    sixteenth_ms = 60000.0 / (tempo_bpm * 4.0)
    d_max = min(PITCH_WOBBLE_MAX_MEMBER_DURATION_MS, PITCH_WOBBLE_SIXTEENTH_DURATION_FRACTION * sixteenth_ms)
    g_max = min(PITCH_WOBBLE_MAX_GAP_MS, 0.5 * sixteenth_ms)
    return d_max, g_max


def _group_candidates(
        notes: Sequence[Note], d_max_ms: float, g_max_ms: float,
) -> List[List[Note]]:
    """Gate 1 + Gate 2's structural chaining: greedily grows a group
    from a seed note, re-testing the ANCHOR RADIUS against the group's
    current highest-confidence member every time a new note is
    considered - not just against the immediately preceding note, which
    is what let 5.6.1's version drift arbitrarily far over a long
    chain. Returns only groups with 2+ members; singletons are dropped
    since a lone note is never a wobble candidate."""
    sorted_notes = sorted(notes, key=lambda n: n.start_ms)
    groups: List[List[Note]] = []
    i = 0
    n = len(sorted_notes)
    while i < n:
        seed = sorted_notes[i]
        if seed.duration_ms >= d_max_ms:
            i += 1
            continue

        group = [seed]
        j = i + 1
        while j < n:
            candidate = sorted_notes[j]
            prev = group[-1]
            gap = candidate.start_ms - prev.end_ms
            if not (0.0 <= gap <= g_max_ms):
                break
            if candidate.duration_ms >= d_max_ms:
                break
            # A step below the min threshold means "same pitch" - not this
            # pass's job (VelocityMerge's), so don't chain it in.
            step = abs(candidate.pitch - prev.pitch)
            if step < PITCH_WOBBLE_MIN_STEP_SEMITONES:
                break

            tentative = group + [candidate]
            anchor = max(tentative, key=lambda note: note.confidence)
            if any(abs(member.pitch - anchor.pitch) > PITCH_WOBBLE_ANCHOR_RADIUS_SEMITONES for member in tentative):
                break

            group = tentative
            j += 1

        if len(group) >= 2:
            groups.append(group)
            i = j
        else:
            i += 1

    return groups


def _monotonicity(pitches: np.ndarray) -> float:
    """|sum(d_i)| / sum(|d_i|) - 1.0 is perfectly directional (a real
    run or slide), 0.0 is pure back-and-forth oscillation (jitter or
    vibrato). Returns 1.0 for an all-zero-diff sequence (nothing to
    call oscillatory about a flat line)."""
    diffs = np.diff(pitches)
    total_abs = np.sum(np.abs(diffs))
    if total_abs == 0:
        return 1.0
    return float(np.abs(np.sum(diffs)) / total_abs)


def _looks_like_structured_motion(pitches: np.ndarray) -> bool:
    """Gate 3: True if this group's pitch trajectory looks like a real
    musical figure (a directional run, or a locally-structured ornament
    like a turn/enclosure that isn't globally monotonic) rather than
    unstructured flicker. Structured motion must NEVER be flagged as a
    wobble group, regardless of what Gate 4 says."""
    if len(pitches) < 2:
        return True  # nothing to evaluate; don't flag

    global_mono = _monotonicity(pitches)
    if global_mono >= PITCH_WOBBLE_GLOBAL_MONOTONICITY_MAX:
        return True  # a real directional run

    diffs = np.diff(pitches)
    window = PITCH_WOBBLE_LOCAL_WINDOW - 1
    if len(diffs) < window:
        return False  # too short to have a distinguishable local pattern

    local_scores = []
    for k in range(len(diffs) - window + 1):
        segment = diffs[k:k + window]
        segment_abs = np.sum(np.abs(segment))
        if segment_abs > 0:
            local_scores.append(float(np.abs(np.sum(segment)) / segment_abs))

    avg_local = float(np.mean(local_scores)) if local_scores else 0.0
    return avg_local >= PITCH_WOBBLE_LOCAL_MONOTONICITY_MIN


def _continuous_pitch_flatness(f0_hz: np.ndarray) -> Tuple[bool, float]:
    """Gate 4: converts the caller-supplied continuous f0 trace to
    semitone units and checks its spread. A small spread means the
    singer held one real physical pitch that just happens to straddle
    a semitone bin boundary - the whole justification for treating the
    Basic Pitch fragments as one note. Returns (is_flat, spread) - if
    there's not enough voiced signal to judge, is_flat is False (no
    evidence, no verdict, per this module's law)."""
    voiced = f0_hz[f0_hz > 0]
    if len(voiced) < 2:
        return False, 0.0
    cents = 69.0 + 12.0 * np.log2(voiced / 440.0)
    spread = float(np.max(cents) - np.min(cents))
    return spread <= PITCH_WOBBLE_MAX_CONTINUOUS_PITCH_SPREAD_SEMITONES, spread


def find_wobble_groups(
        notes: List[Note],
        tempo_bpm: float,
        sample_continuous_pitch_hz: Optional[Callable[[float, float], np.ndarray]] = None,
) -> List[WobbleGroup]:
    """One stem's (intended for vocals - see module docstring) notes,
    scanned for candidate quantization-boundary wobble groups. Gates 1
    and 2 do the structural chaining/duration filtering; Gate 3 rejects
    anything that looks like a real musical figure; Gate 4 - which
    REQUIRES `sample_continuous_pitch_hz`, a callable returning a raw
    f0-in-Hz array for a given [start_ms, end_ms) span (meant to be
    backed by CREPE once it's run against the vocals stem, which Jazz
    does not yet do) - is the only thing that can actually confirm a
    group. Without it, this returns no groups at all: structural
    plausibility alone is not sufficient evidence to flag two notes as
    one, matching the caution this pass's history earned."""
    if sample_continuous_pitch_hz is None or not notes:
        return []

    d_max_ms, g_max_ms = _tempo_adaptive_ceilings(tempo_bpm)
    candidate_groups = _group_candidates(notes, d_max_ms, g_max_ms)

    wobble_groups: List[WobbleGroup] = []
    for group in candidate_groups:
        pitches = np.array([n.pitch for n in group], dtype=np.float64)
        if _looks_like_structured_motion(pitches):
            continue

        f0_hz = sample_continuous_pitch_hz(group[0].start_ms, group[-1].end_ms)
        is_flat, spread = _continuous_pitch_flatness(f0_hz)
        if not is_flat:
            continue

        anchor = max(group, key=lambda n: n.confidence)
        wobble_groups.append(WobbleGroup(
            member_note_ids=tuple(n.id for n in group),
            anchor_note_id=anchor.id,
            anchor_pitch=anchor.pitch,
            continuous_pitch_spread_semitones=spread,
            reason=(f"{len(group)} Basic-Pitch fragments spanning "
                    f"{group[-1].end_ms - group[0].start_ms:.0f}ms, continuous pitch spread "
                    f"{spread:.2f} semitones (quantization-boundary flicker, not a real pitch change)"),
        ))

    return wobble_groups


__all__ = [
    "WobbleGroup",
    "find_wobble_groups",
    "PITCH_WOBBLE_ANNOTATION_KIND",
    "PITCH_WOBBLE_ANCHOR_RADIUS_SEMITONES",
    "PITCH_WOBBLE_MIN_STEP_SEMITONES",
    "PITCH_WOBBLE_MAX_MEMBER_DURATION_MS",
    "PITCH_WOBBLE_MAX_GAP_MS",
    "PITCH_WOBBLE_MAX_CONTINUOUS_PITCH_SPREAD_SEMITONES",
]
