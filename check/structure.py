# =================================================================
# MODULE: check/structure.py
# The two detectors "Check" is built on: macro structure (repeated
# SECTIONS) and micro motifs (repeated MELODIC FRAGMENTS). Both find the
# same thing at different scales - places where the music repeats an idea.
# Detection only; the reconciliation ("write the repeats consistently")
# lives in check/check.py and is annotation-only.
# =================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

STRUCTURE_SR = 22050


@dataclass(frozen=True)
class Section:
    """One labeled span of the form. Same `label` => the same section
    recurring (A ... A ... B ...)."""
    label: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class Motif:
    """A recurring melodic fragment, stored transposition-invariantly as
    an interval sequence (so the same shape in any key still matches)."""
    intervals: Tuple[int, ...]
    count: int
    occurrences_ms: Tuple[float, ...]   # start time of each occurrence
    stem: str


def detect_form(engine, track, min_section_s: float = 4.0, k: int = 5) -> List[Section]:
    """Macro form via McFee/Ellis Laplacian structural segmentation.

    WHAT IT RETURNS: a time-ordered list of CONTIGUOUS, non-overlapping
    `Section`s covering the track - e.g. A B C B D C A ... Same `label` means
    the same section RECURRING, so labels repeat while spans do not. This is
    the song's form as a timeline, and it is the thing to read when asking
    "where are the sections?".

    HOW: beat-synchronous CQT gives the REPETITION structure (a recurrence
    matrix - which beats sound like which other beats), MFCC gives local timbre
    CONTINUITY (consecutive-beat similarity). Those two graphs are combined
    with a degree-balanced weight `mu`, and the normalized Laplacian's leading
    eigenvectors are k-means clustered: beats that both recur together AND flow
    together fall in one cluster. Consecutive same-cluster beats collapse into
    a segment; sub-minimum segments merge into a neighbour (see below); labels
    are then assigned A, B, C... by first appearance.

    Pure DSP + spectral clustering - no learned weights, nothing mutated.

    CAVEATS worth knowing before trusting the output:
      - `k` (default 5) is the number of distinct section TYPES, fixed
        regardless of song length. A song with 7 real sections gets 5.
      - It segments TIMBRE + REPETITION, not harmony or phrase. A section
        boundary here means "the texture changed", which usually but not
        always coincides with a musical section.
      - Boundaries land on beat times from librosa's own beat tracker, so they
        inherit its errors and are only as precise as +/- one beat.
      - Returns [] for audio under 8 s or with fewer than k+2 beats.

    NOTE ON CONSUMING IT: `check.run_check` writes each note's section as an
    annotation carrying the label AND the instance index and boundaries. Read
    the index when you need to tell one occurrence of A from another - the bare
    label alone cannot distinguish them, which previously made the form
    impossible to reconstruct from annotations (§XVIII).
    """
    import librosa
    import scipy.ndimage
    import scipy.linalg
    import scipy.sparse.csgraph
    from sklearn.cluster import KMeans

    y = np.asarray(engine.view(track, STRUCTURE_SR).samples, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0)
    if len(y) < STRUCTURE_SR * 8:
        return []

    _tempo, beats = librosa.beat.beat_track(y=y, sr=STRUCTURE_SR, trim=False)
    if len(beats) < k + 2:
        return []
    btimes = librosa.frames_to_time(beats, sr=STRUCTURE_SR)

    cqt = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=STRUCTURE_SR)), ref=np.max)
    csync = librosa.util.sync(cqt, beats, aggregate=np.median)
    rec = librosa.segment.recurrence_matrix(csync, width=3, mode="affinity", sym=True)
    diag_filter = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    rec = diag_filter(rec, size=(1, 7))

    mfcc = librosa.feature.mfcc(y=y, sr=STRUCTURE_SR)
    msync = librosa.util.sync(mfcc, beats)
    path_d = np.sum(np.diff(msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_d) + 1e-9
    path_sim = np.exp(-path_d / sigma)
    rpath = np.diag(path_sim, 1) + np.diag(path_sim, -1)

    deg_p = rpath.sum(axis=1)
    deg_r = rec.sum(axis=1)
    mu = deg_p.dot(deg_p + deg_r) / (np.sum((deg_p + deg_r) ** 2) + 1e-9)
    graph = mu * rec + (1.0 - mu) * rpath

    lap = scipy.sparse.csgraph.laplacian(graph, normed=True)
    _evals, evecs = scipy.linalg.eigh(lap)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

    kk = int(min(k, evecs.shape[1]))
    embed = evecs[:, :kk] / (cnorm[:, kk - 1:kk] + 1e-9)
    ids = KMeans(n_clusters=kk, n_init=10, random_state=0).fit_predict(embed)
    ids = scipy.ndimage.median_filter(ids, size=9)

    # collapse consecutive beats into segments. NB: librosa.util.sync
    # yields one more column than there are beat times (it also aggregates
    # the tail after the last beat), so `ids` can be one longer than
    # `btimes` - clamp the index rather than overrun.
    nb = len(btimes)
    raw: List[List] = []
    for i, lab in enumerate(ids):
        a = float(btimes[i]) if i < nb else float(btimes[-1])
        b = float(btimes[i + 1]) if i + 1 < nb else a + 0.5
        if not raw or raw[-1][0] != lab:
            raw.append([lab, a, b])
        else:
            raw[-1][2] = b
    # Merge sub-minimum segments into a NEIGHBOUR, choosing which one on
    # evidence rather than always folding into whatever came before.
    #
    # The old rule was `merged[-1][2] = seg[2]` unconditionally: a too-short
    # segment was always absorbed by its predecessor and its own label thrown
    # away. Two consequences, both wrong:
    #   - a brief return of A between two B sections got absorbed into B,
    #     erasing the recurrence the whole module exists to find;
    #   - the FIRST segment could never merge (nothing precedes it), so a short
    #     opening survived while an identical short segment mid-song did not -
    #     the rule was asymmetric in time.
    # Now: prefer the neighbour that SHARES the short segment's label (a brief
    # A between Bs rejoins A), else the longer neighbour, and a short leading
    # segment merges forward. Iterated shortest-first so one merge can enable
    # the next.
    merged: List[List] = [list(s) for s in raw]
    if merged:
        while len(merged) > 1:
            durations = [s[2] - s[1] for s in merged]
            i = int(np.argmin(durations))
            if durations[i] >= min_section_s:
                break
            prev_i, next_i = i - 1, i + 1
            if prev_i < 0:
                target = next_i
            elif next_i >= len(merged):
                target = prev_i
            elif merged[prev_i][0] == merged[i][0]:
                target = prev_i
            elif merged[next_i][0] == merged[i][0]:
                target = next_i
            else:
                target = (prev_i if (merged[prev_i][2] - merged[prev_i][1])
                          >= (merged[next_i][2] - merged[next_i][1]) else next_i)
            merged[target][1] = min(merged[target][1], merged[i][1])
            merged[target][2] = max(merged[target][2], merged[i][2])
            merged.pop(i)
        # a merge can leave two adjacent segments sharing a label - fuse them
        fused: List[List] = []
        for seg in merged:
            if fused and fused[-1][0] == seg[0]:
                fused[-1][2] = seg[2]
            else:
                fused.append(list(seg))
        merged = fused
    # relabel by first appearance -> A, B, C ...
    order: dict = {}
    sections: List[Section] = []
    for lab, a, b in merged:
        if lab not in order:
            order[lab] = chr(65 + len(order))
        sections.append(Section(order[lab], a * 1000.0, b * 1000.0))
    return sections


def find_motifs(notes, min_count: int = 3, lengths: Tuple[int, ...] = (4, 5, 6),
                max_gap_ms: float = 250.0) -> List[Motif]:
    """Micro motifs: per pitched stem, consolidate time-contiguous
    same-pitch fragments into one note (defusing over-detection first -
    the session keystone), reduce to an interval sequence, and count
    repeated n-grams. Drums are skipped (no melodic interval)."""
    from collections import defaultdict

    by_stem: dict = defaultdict(list)
    for n in notes:
        by_stem[n.stem].append(n)

    out: List[Motif] = []
    for stem, ns in by_stem.items():
        if getattr(stem, "value", str(stem)) == "drums":
            continue
        ns = sorted(ns, key=lambda n: n.start_ms)
        seq: List[dict] = []
        for n in ns:
            if seq and seq[-1]["pitch"] == n.pitch and n.start_ms - seq[-1]["end"] < max_gap_ms:
                seq[-1]["end"] = n.end_ms
            else:
                seq.append({"pitch": n.pitch, "start": n.start_ms, "end": n.end_ms})
        if len(seq) < max(lengths) + 2:
            continue
        pitches = [s["pitch"] for s in seq]
        starts = [s["start"] for s in seq]
        iv = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
        for L in lengths:
            grams: dict = defaultdict(list)
            for i in range(len(iv) - L):
                g = tuple(iv[i:i + L])
                if any(x != 0 for x in g):
                    grams[g].append(starts[i])
            for g, occ in grams.items():
                if len(occ) >= min_count:
                    out.append(Motif(intervals=g, count=len(occ),
                                     occurrences_ms=tuple(occ),
                                     stem=getattr(stem, "value", str(stem))))
    out.sort(key=lambda m: (-m.count, -len(m.intervals)))
    return out


__all__ = ["Section", "Motif", "detect_form", "find_motifs", "STRUCTURE_SR"]
