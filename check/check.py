# =================================================================
# MODULE: check/check.py
# "Check" - the form/self-similarity layer (the top rung of the
# perception ladder). A song repeats its ideas (A A B A; a riff; a
# lick), but the transcriber makes its note/rhythm decisions
# INDEPENDENTLY at each repeat, with independent noise - so the same
# idea gets written several slightly-different ways. Check finds the
# repeats (sections via check/structure.detect_form, motifs via
# find_motifs), then MEASURES how much the repeats disagree, so a
# consistent reading can be preferred where the evidence supports it.
#
# ANNOTATION-ONLY, like the other §7-adjacent passes. Check never
# mutates a Note - it stamps:
#   "section"       : which form label (A/B/C...) a note falls in
#   "repeat_group"  : set only when that section recurs (>=2 times)
#   "motif"         : index of a recurring motif the note participates in
# and returns a CheckResult with per-repeat-group consistency scores.
# The scores are the evidence a later reconciliation step (or the
# engraver) would use to prefer the consistent reading; forcing repeats
# identical is NOT done here (real music varies - the last chorus is
# bigger, a repeat has a fill), so this stays soft and opt-in.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core import Annotation, AnnotationStore, Note, Provenance
from check.structure import Section, Motif, detect_form, find_motifs

SECTION_KIND = "section"
REPEAT_GROUP_KIND = "repeat_group"
MOTIF_KIND = "motif"
DRIFT_KIND = "repeat_drift"

# A repeat group whose instances agree at least this well is "consistent" -
# no drift worth flagging. Below it, the instance farthest from the group's
# consensus profile is the one the transcriber read differently.
DRIFT_CONSISTENCY_THRESHOLD = 0.92


@dataclass
class CheckResult:
    sections: List[Section]
    repeat_groups: Dict[str, int]            # label -> how many times it recurs
    group_consistency: Dict[str, float]      # label -> mean pairwise pitch-class similarity (0-1)
    motifs: List[Motif]
    notes_labeled: int
    drift_flagged: int = 0                    # notes in an outlier repeat instance (reconciliation evidence)


def _section_of(t_ms: float, sections: List[Section]) -> Optional[Section]:
    for s in sections:
        if s.start_ms <= t_ms < s.end_ms:
            return s
    return None


def _pitch_class_hist(notes: List[Note]) -> np.ndarray:
    """Duration-weighted pitch-class profile, L2-normalized. Two instances
    of the same section should have near-identical profiles; the cosine
    similarity between them is how much they agree harmonically/melodically."""
    h = np.zeros(12, dtype=np.float64)
    for n in notes:
        h[n.pitch % 12] += max(n.duration_ms, 1.0)
    nrm = np.linalg.norm(h)
    return h / nrm if nrm > 0 else h


def run_check(engine, track, notes: List[Note], annotations: AnnotationStore) -> CheckResult:
    """Detect form + motifs on `track`/`notes`, label the notes, and score
    how consistently each repeated section is transcribed. Annotation-only."""
    sections = detect_form(engine, track)

    label_count: Dict[str, int] = {}
    for s in sections:
        label_count[s.label] = label_count.get(s.label, 0) + 1

    notes_labeled = 0
    for n in notes:
        s = _section_of(n.start_ms, sections)
        if s is None:
            continue
        annotations.add(Annotation(note_id=n.id, kind=SECTION_KIND, value=s.label,
                                   source=Provenance.CHECK, confidence=1.0))
        notes_labeled += 1
        if label_count.get(s.label, 0) >= 2:
            annotations.add(Annotation(note_id=n.id, kind=REPEAT_GROUP_KIND, value=s.label,
                                       source=Provenance.CHECK, confidence=1.0))

    # per-repeat-group consistency (pitch-class profile agreement across instances)
    by_label: Dict[str, List[Section]] = {}
    for s in sections:
        by_label.setdefault(s.label, []).append(s)
    group_consistency: Dict[str, float] = {}
    drift_flagged = 0
    for label, instances in by_label.items():
        if len(instances) < 2:
            continue
        inst_notes = [
            [n for n in notes
             if s.start_ms <= n.start_ms < s.end_ms
             and getattr(n.stem, "value", str(n.stem)) != "drums"]
            for s in instances
        ]
        hists = [(_pitch_class_hist(ns), ns) for ns in inst_notes if ns]
        if len(hists) < 2:
            continue
        profiles = [h for h, _ in hists]
        sims = [float(profiles[i] @ profiles[j])
                for i in range(len(profiles)) for j in range(i + 1, len(profiles))]
        group_consistency[label] = float(np.mean(sims)) if sims else 0.0

        # Reconciliation EVIDENCE (annotation-only; the note-rewrite half stays
        # deferred - real music varies). When a group's instances disagree,
        # find the OUTLIER instance (lowest mean similarity to the others = the
        # consensus reading it departs from) and flag its notes so the
        # disagreement is localized to a specific repeat, not just a group score.
        if group_consistency[label] < DRIFT_CONSISTENCY_THRESHOLD and len(profiles) >= 3:
            mean_sim_to_others = [
                float(np.mean([profiles[i] @ profiles[j] for j in range(len(profiles)) if j != i]))
                for i in range(len(profiles))
            ]
            outlier = int(np.argmin(mean_sim_to_others))
            for n in hists[outlier][1]:
                annotations.add(Annotation(
                    note_id=n.id, kind=DRIFT_KIND,
                    value={"group": label, "instance": outlier,
                           "similarity_to_consensus": round(mean_sim_to_others[outlier], 3)},
                    source=Provenance.CHECK,
                    confidence=float(1.0 - mean_sim_to_others[outlier]), contested=True,
                ))
                drift_flagged += 1

    # motifs, and tag participating notes (first note of each occurrence)
    motifs = find_motifs(notes)
    by_stem: Dict[str, List[Note]] = {}
    for n in notes:
        by_stem.setdefault(getattr(n.stem, "value", str(n.stem)), []).append(n)
    for stem, ns in by_stem.items():
        ns.sort(key=lambda n: n.start_ms)
    for m_idx, m in enumerate(motifs[:20]):
        cand = sorted(by_stem.get(m.stem, []), key=lambda n: n.start_ms)
        for occ_ms in m.occurrences_ms:
            hit = min(cand, key=lambda n: abs(n.start_ms - occ_ms), default=None)
            if hit is not None and abs(hit.start_ms - occ_ms) < 60.0:
                annotations.add(Annotation(note_id=hit.id, kind=MOTIF_KIND, value=m_idx,
                                           source=Provenance.CHECK, confidence=float(min(1.0, m.count / 5.0))))

    return CheckResult(
        sections=sections,
        repeat_groups={k: v for k, v in label_count.items() if v >= 2},
        group_consistency=group_consistency,
        motifs=motifs,
        notes_labeled=notes_labeled,
        drift_flagged=drift_flagged,
    )


__all__ = ["run_check", "CheckResult", "SECTION_KIND", "REPEAT_GROUP_KIND", "MOTIF_KIND", "DRIFT_KIND"]
