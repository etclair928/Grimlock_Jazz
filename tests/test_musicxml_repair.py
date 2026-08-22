# =================================================================
# MODULE: tests/test_musicxml_repair.py
# Pins output/musicxml_repair.py - the post-write pass.
#
# It edits the bytes of a finished score, which is the most dangerous place in
# this codebase to be wrong: nothing downstream re-checks it. So the tests are
# weighted toward what it must NOT do - never lengthen, never move music,
# never approximate a duration it cannot express exactly, and never touch a
# note.
# =================================================================

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output.musicxml_repair import (  # noqa: E402
    _bar_divisions, _unambiguous_rest_pieces, repair_measures,
)

D = 10080  # divisions per quarter, as music21 writes them


def test_bar_length_from_a_time_signature():
    assert _bar_divisions(D, 4, 4) == 4 * D
    assert _bar_divisions(D, 6, 4) == 6 * D
    assert _bar_divisions(D, 6, 8) == 3 * D
    assert _bar_divisions(D, 7, 8) == int(round(3.5 * D))


# --- the decomposition ----------------------------------------------------

def test_four_quarters_becomes_something_unambiguous():
    """A whole rest in a 6/4 bar reads as a BAR rest to any conventional
    reader, so four beats of silence must be written another way."""
    pieces = _unambiguous_rest_pieces(4 * D, D)
    assert pieces
    assert sum(ticks for _n, _d, ticks in pieces) == 4 * D
    assert all(name != "whole" for name, _d, _t in pieces)


def test_the_pieces_always_sum_exactly():
    for quarters in (1, 2, 3, 4, 5, 6, 7, 8):
        pieces = _unambiguous_rest_pieces(quarters * D, D)
        assert sum(t for _n, _d, t in pieces) == quarters * D, quarters


def test_a_duration_it_cannot_express_is_refused_not_approximated():
    """A rest of the wrong length is worse than an ambiguous one. Returning []
    leaves the original alone, which is the honest failure."""
    assert _unambiguous_rest_pieces(int(D / 3), D) == []
    assert _unambiguous_rest_pieces(1, D) == []


def test_nothing_is_ever_lengthened():
    """The rule this codebase keeps having to relearn: do not force a value
    into a space it cannot fit."""
    for ticks in (D, 2 * D, 3 * D, 4 * D, 5 * D, 7 * D):
        pieces = _unambiguous_rest_pieces(ticks, D)
        assert sum(t for _n, _d, t in pieces) <= ticks


# --- the pass over a real document ---------------------------------------

def _score(measure_body: str, beats: int = 6) -> str:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>t</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>{D}</divisions>
        <time><beats>{beats}</beats><beat-type>4</beat-type></time>
      </attributes>
      {measure_body}
    </measure>
  </part>
</score-partwise>
"""
    path = os.path.join(tempfile.mkdtemp(), "s.musicxml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return path


def _rest(ticks: int, rtype: str, measure_attr: str = "") -> str:
    return (f'<note><rest{measure_attr} /><duration>{ticks}</duration>'
            f'<voice>1</voice><type>{rtype}</type></note>')


def test_an_ambiguous_whole_rest_is_rewritten():
    path = _score(_rest(4 * D, "whole") + _rest(2 * D, "half"))
    stats = repair_measures(path)
    assert stats["whole_rests_disambiguated"] == 1
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "<type>whole</type>" not in text


def test_a_real_bar_rest_is_left_alone():
    """In 4/4 a whole rest IS the bar. Rewriting it would be wrong."""
    path = _score(_rest(4 * D, "whole"), beats=4)
    stats = repair_measures(path)
    assert stats["whole_rests_disambiguated"] == 0


def test_a_rest_marked_measure_yes_is_left_alone():
    """It already says what it means; the attribute is unambiguous."""
    path = _score(_rest(6 * D, "whole", ' measure="yes"'))
    assert repair_measures(path)["whole_rests_disambiguated"] == 0


def test_notes_are_never_touched():
    """A note crossing a barline is an ENGRAVER defect and must be fixed where
    it can be split and tied. Trimming it here would hide it."""
    note = (f'<note><pitch><step>C</step><octave>4</octave></pitch>'
            f'<duration>{8 * D}</duration><voice>1</voice><type>whole</type></note>')
    path = _score(note)
    stats = repair_measures(path)
    assert stats["notes_overrunning"] == 1
    assert stats["rests_trimmed"] == 0
    with open(path, encoding="utf-8") as fh:
        assert f"<duration>{8 * D}</duration>" in fh.read()


def test_chord_members_do_not_advance_the_cursor():
    """Treating a chord as sequential would make every chord look like an
    overrun and trigger spurious repairs."""
    body = ""
    for i in range(6):
        chord = "<chord />" if i else ""
        body += (f'<note>{chord}<pitch><step>C</step><octave>4</octave></pitch>'
                 f'<duration>{6 * D}</duration><voice>1</voice><type>whole</type></note>')
    stats = repair_measures(_score(body))
    assert stats["notes_overrunning"] == 0


def test_a_file_needing_nothing_is_not_rewritten():
    path = _score(_rest(6 * D, "whole", ' measure="yes"'))
    before = os.path.getmtime(path)
    repair_measures(path)
    assert os.path.getmtime(path) == before
