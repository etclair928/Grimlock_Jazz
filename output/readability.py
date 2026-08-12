# =================================================================
# MODULE: output/readability.py
# THE READABILITY OBJECTIVE (GRIMLOCK_6.0_OPEN_PROBLEMS.md §XVI.9).
#
# §XIII.4 said evaluation is two axes - perceptual (does it sound right)
# and notational (does it read right) - and we only ever built metrics
# for the first. Everything we optimise is fidelity (baseline ratios) or
# validity (null models). No page-quality target was ever written down,
# so nothing ever moved toward one, and we got steadily more CORRECT
# rather than more READABLE.
#
# This is that target. Every axis here is computed from the emitted
# MusicXML - the actual artifact a human opens - and every one of them
# has a measured reference point from the Klangio head-to-head (§XVII),
# so "good" is benchmarked rather than invented.
#
# THIS IS NOT A LOSS FUNCTION FOR MUSICALITY. The ear remains the oracle
# for whether the transcription is right (§VI). This measures only
# whether the PAGE communicates it - which Klangio proved is a separate,
# measurable thing we were losing badly on while winning on content.
# =================================================================

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Reference points measured from real scores (§XVII.4). Klangio is a
# shipping commercial engraver, so its numbers are an existence proof of
# what is achievable - not a ceiling, and explicitly not a fidelity target.
BENCHMARK = {
    "rest_ratio":        {"klangio": 0.08, "ours_2026_08_06": 0.79, "good": 0.25},
    "chord_share":       {"klangio": 0.45, "ours_2026_08_06": 0.13, "good": 0.35},
    "max_voices":        {"klangio": 2,    "ours_2026_08_06": 4,    "good": 2},
    "tempo_marks":       {"klangio": 1,    "ours_2026_08_06": 5,    "good": 1},
    "unpitched_drums":   {"klangio": True, "ours_2026_08_06": False, "good": True},
    # COHERENCE (added 2026-08-07 after an external critique caught a regression
    # this scoreboard could not see). Mean intra-voice pitch jump: how far a
    # voice leaps between consecutive events. LOW = the voice is a real melodic
    # line; HIGH = it is a bin that unrelated material was crammed into.
    #
    # This axis exists because rest_ratio and coherence TRADE AGAINST EACH
    # OTHER, and optimizing rest_ratio alone silently degraded the music:
    # capping voices 4 -> 2 cut rests 30-38% while raising mean jump 6.29 ->
    # 7.67 semitones (+22%) and cramming 196 -> 352 notes per voice. §XV had
    # already measured the same tension from the other side (register-continuity
    # voicing improved jump 8.2 -> 2.5 while RAISING rest ratio 1.10 -> 1.24).
    # A page score that reports only clutter will keep recommending the trade.
    "mean_voice_jump":   {"reference_first_free": 8.2, "reference_register": 2.5, "good": 5.0},
    "singleton_voices":  {"good": 0},
    # TOP-LINE STABILITY (added 2026-08-08). What fraction of onsets keep the
    # highest-sounding note in the SAME voice as the previous onset. This is the
    # "can I read the melody off one voice" axis, and mean_voice_jump is blind to
    # it: measured on Federal Blvd our three voices had jumps 5.74/5.59/4.94 -
    # all respectable - while the top line changed voice on 52% of onsets. Every
    # voice looked fine precisely BECAUSE none of them was the melody; the line
    # was smeared evenly across all three. Klangio on the same audio holds the
    # top line in one voice 86% of the time and puts it in voice 1 for 77.6% of
    # onsets, which is why its page reads as music and ours reads as texture.
    #
    # This is the second time a coherence defect hid behind an aggregate (see
    # mean_voice_jump's note). Aggregates over voices cannot see WHICH voice
    # carries the line - that needs a per-onset identity test.
    "top_line_stability": {"klangio": 0.86, "ours_2026_08_08": 0.48, "good": 0.80},
    "voice1_is_top":      {"klangio": 0.776, "ours_2026_08_08": 0.388, "good": 0.70},
}


@dataclass
class StaffReadability:
    name: str
    notes: int
    rests: int
    voices: int
    chord_tones: int
    is_drum_like: bool = False

    @property
    def rest_ratio(self) -> float:
        return self.rests / self.notes if self.notes else 0.0

    @property
    def chord_share(self) -> float:
        return self.chord_tones / self.notes if self.notes else 0.0


@dataclass
class ReadabilityReport:
    staves: List[StaffReadability] = field(default_factory=list)
    tempo_marks: int = 0
    overfull_voice_measures: int = 0
    total_voice_measures: int = 0
    drums_unpitched: Optional[bool] = None
    ties: int = 0
    voice_jumps: List[int] = field(default_factory=list)   # intra-voice pitch distances
    singleton_voices: int = 0                              # voices holding <=2 events
    top_line_onsets: int = 0        # onsets on multi-voice staves
    top_line_stable: int = 0        # ...where the top note kept the previous onset's voice
    top_line_in_voice1: int = 0     # ...where the top note was in voice "1"

    @property
    def mean_voice_jump(self) -> float:
        return sum(self.voice_jumps) / len(self.voice_jumps) if self.voice_jumps else 0.0

    @property
    def top_line_stability(self) -> float:
        return self.top_line_stable / (self.top_line_onsets - 1) if self.top_line_onsets > 1 else 1.0

    @property
    def voice1_is_top(self) -> float:
        return self.top_line_in_voice1 / self.top_line_onsets if self.top_line_onsets else 1.0

    @property
    def notes(self) -> int:
        return sum(s.notes for s in self.staves)

    @property
    def rests(self) -> int:
        return sum(s.rests for s in self.staves)

    @property
    def rest_ratio(self) -> float:
        return self.rests / self.notes if self.notes else 0.0

    @property
    def chord_share(self) -> float:
        ct = sum(s.chord_tones for s in self.staves)
        return ct / self.notes if self.notes else 0.0

    @property
    def max_voices(self) -> int:
        return max((s.voices for s in self.staves), default=0)

    @property
    def overfull_rate(self) -> float:
        return (self.overfull_voice_measures / self.total_voice_measures
                if self.total_voice_measures else 0.0)

    def score(self) -> Dict[str, object]:
        """A flat dict of the axes, each with its benchmark verdict. Deliberately
        NOT collapsed into a single number: these trade against each other, and
        a scalar would hide which one moved."""
        def verdict(key, value, lower_is_better=True):
            good = BENCHMARK[key]["good"]
            ok = value <= good if lower_is_better else value >= good
            return {"value": value, "good": good, "ok": bool(ok)}
        return {
            "rest_ratio":  verdict("rest_ratio", round(self.rest_ratio, 3)),
            "chord_share": verdict("chord_share", round(self.chord_share, 3), lower_is_better=False),
            "max_voices":  verdict("max_voices", self.max_voices),
            "tempo_marks": verdict("tempo_marks", self.tempo_marks),
            "unpitched_drums": {"value": self.drums_unpitched,
                                "good": True, "ok": self.drums_unpitched is not False},
            "mean_voice_jump": verdict("mean_voice_jump", round(self.mean_voice_jump, 2)),
            "singleton_voices": verdict("singleton_voices", self.singleton_voices),
            "top_line_stability": verdict("top_line_stability", round(self.top_line_stability, 3),
                                          lower_is_better=False),
            "voice1_is_top": verdict("voice1_is_top", round(self.voice1_is_top, 3),
                                     lower_is_better=False),
            "overfull_measures": {"value": self.overfull_voice_measures,
                                  "good": 0, "ok": self.overfull_voice_measures == 0},
            "staves": len(self.staves),
            "notes": self.notes,
            "ties": self.ties,
        }


def _part_names(root) -> Dict[str, str]:
    return {sp.get("id"): (sp.find("part-name").text if sp.find("part-name") is not None else "?")
            for sp in root.findall(".//score-part")}


def measure_readability(path: str) -> ReadabilityReport:
    """Score an emitted MusicXML file on the readability axes."""
    root = ET.parse(path).getroot()
    names = _part_names(root)
    rep = ReadabilityReport()

    divisions = None
    for d in root.iter("divisions"):
        divisions = int(d.text); break
    beats = beat_type = None
    for t in root.iter("time"):
        b = t.find("beats"); bt = t.find("beat-type")
        if b is not None and bt is not None:
            beats, beat_type = int(b.text), int(bt.text)
        break
    expected = (divisions * beats * 4 // beat_type) if (divisions and beats and beat_type) else None

    any_drums_seen = False
    for part in root.findall(".//part"):
        pid = part.get("id"); name = names.get(pid, "?")
        notes = part.findall(".//note")
        pitched = [n for n in notes if n.find("rest") is None]
        rests = [n for n in notes if n.find("rest") is not None]
        unpitched = [n for n in notes if n.find("unpitched") is not None]
        chord_tones = sum(1 for n in pitched if n.find("chord") is not None)
        voices = len(set(v.text for v in part.iter("voice") if v.text)) or 1
        drum_like = "drum" in name.lower() or "perc" in name.lower()
        if drum_like:
            any_drums_seen = True
            rep.drums_unpitched = len(unpitched) > 0
        rep.staves.append(StaffReadability(
            name=name, notes=len(pitched) + len(unpitched), rests=len(rests),
            voices=voices, chord_tones=chord_tones, is_drum_like=drum_like))
        rep.tempo_marks += sum(1 for d in part.findall(".//direction")
                               if d.find(".//metronome") is not None)

        # COHERENCE: walk each voice in time order and measure how far it leaps
        # between consecutive events. A voice that is a real line steps; a voice
        # that is a bin leaps. Chord members are skipped (they share an onset,
        # so the "jump" between them is vertical, not melodic), and drums are
        # excluded (percussion staff positions are not pitches).
        if not drum_like:
            _STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
            per_voice: Dict[str, List[int]] = {}
            for n in part.findall(".//note"):
                if n.find("rest") is not None or n.find("chord") is not None:
                    continue
                p_el = n.find("pitch")
                if p_el is None:
                    continue
                v_el = n.find("voice")
                key = v_el.text if v_el is not None else "1"
                octave = int(p_el.find("octave").text)
                step = _STEP[p_el.find("step").text]
                alter = int(p_el.find("alter").text) if p_el.find("alter") is not None else 0
                per_voice.setdefault(key, []).append((octave + 1) * 12 + step + alter)
            for seq in per_voice.values():
                if len(seq) <= 2:
                    rep.singleton_voices += 1
                for i in range(len(seq) - 1):
                    rep.voice_jumps.append(abs(seq[i + 1] - seq[i]))
            # TOP-LINE STABILITY. Walk the part in time and ask, at each onset,
            # which voice holds the highest sounding note. A readable score keeps
            # the melody in one voice; ours was handing it to whichever voice
            # happened to be free. Needs real onset times, so we run a per-voice
            # duration cursor through each measure (chord members share the
            # onset of the note they attach to, and do not advance it).
            per_measure_top: List[tuple] = []
            for m in part.findall("measure"):
                cursor: Dict[str, int] = {}
                last_onset: Dict[str, int] = {}
                for n in m.findall("note"):
                    if n.find("grace") is not None:
                        continue
                    v_el = n.find("voice")
                    key = v_el.text if v_el is not None else "1"
                    dur_el = n.find("duration")
                    dur = int(dur_el.text) if dur_el is not None else 0
                    if n.find("chord") is not None:
                        onset = last_onset.get(key, cursor.get(key, 0))
                    else:
                        onset = cursor.get(key, 0)
                        last_onset[key] = onset
                        cursor[key] = onset + dur
                    p_el = n.find("pitch")
                    if n.find("rest") is not None or p_el is None:
                        continue
                    octave = int(p_el.find("octave").text)
                    step = _STEP[p_el.find("step").text]
                    alter = int(p_el.find("alter").text) if p_el.find("alter") is not None else 0
                    per_measure_top.append((len(per_measure_top), onset,
                                            (octave + 1) * 12 + step + alter, key))
                # collapse to one (top pitch, voice) per distinct onset in this measure
                by_onset: Dict[int, tuple] = {}
                for _, onset, midi, key in per_measure_top:
                    cur = by_onset.get(onset)
                    if cur is None or midi > cur[0]:
                        by_onset[onset] = (midi, key)
                if len(set(v.text for v in m.iter("voice") if v.text)) > 1:
                    prev = None
                    for onset in sorted(by_onset):
                        _, key = by_onset[onset]
                        rep.top_line_onsets += 1
                        if key == "1":
                            rep.top_line_in_voice1 += 1
                        if prev is not None and key == prev:
                            rep.top_line_stable += 1
                        prev = key
                per_measure_top.clear()
        rep.ties += len(part.findall(".//tie"))

        # measure completeness per voice (the "corrupted file" signal)
        if expected:
            for m in part.findall("measure"):
                per_voice: Dict[str, int] = {}
                for n in m.findall("note"):
                    if n.find("grace") is not None or n.find("chord") is not None:
                        continue
                    dur = n.find("duration")
                    v = n.find("voice")
                    key = v.text if v is not None else "1"
                    if dur is not None:
                        per_voice[key] = per_voice.get(key, 0) + int(dur.text)
                for total in per_voice.values():
                    rep.total_voice_measures += 1
                    if total != expected:
                        rep.overfull_voice_measures += 1
    if not any_drums_seen:
        rep.drums_unpitched = None
    return rep


def format_report(path: str, rep: ReadabilityReport) -> str:
    import os
    s = rep.score()
    lines = [f"{os.path.basename(path)}"]
    lines.append(f"  staves={s['staves']} notes={s['notes']} ties={s['ties']}")
    for k in ("rest_ratio", "chord_share", "max_voices", "mean_voice_jump",
              "top_line_stability", "voice1_is_top",
              "singleton_voices", "tempo_marks", "unpitched_drums", "overfull_measures"):
        v = s[k]
        mark = "ok " if v["ok"] else "XX "
        lines.append(f"  {mark}{k:18s} {str(v['value']):>8s}   (target {v['good']})")
    return "\n".join(lines)


__all__ = ["measure_readability", "format_report", "ReadabilityReport",
           "StaffReadability", "BENCHMARK"]
