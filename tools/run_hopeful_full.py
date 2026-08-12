# Full end-to-end pipeline run on Hopeful (separation + all), MIDI + MusicXML.
# Standalone runner; not part of the pipeline.
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import scipy.signal as _sps
    if not hasattr(_sps, "gaussian"):
        from scipy.signal.windows import gaussian as _g
        _sps.gaussian = _g
except Exception:
    pass

from orchestration.conductor import transcribe_file

AUDIO = "Input/Hopeful/Hopeful.mp3"
MIDI = "transcriptions/Hopeful_FULLRUN.mid"
MXL = "transcriptions/Hopeful_FULLRUN.musicxml"
PKL = "transcriptions/Hopeful_FULLRUN.pkl"

t0 = time.time()
res = transcribe_file(
    AUDIO, MIDI,
    output_musicxml_path=MXL,
    use_notation_timing=True,
    use_consolidated_timing=True,
    save_intermediate_path=PKL,
)
print("=" * 60)
print(f"DONE in {time.time()-t0:.0f}s")
print(f"  separation_model: {res.separation_model}")
print(f"  tempo: {res.tempo_bpm:.1f}bpm  meter: {res.time_signature}")
print(f"  notes by stem: {res.note_counts_by_stem}")
print(f"  total notes: {res.total_notes_exported}")
print(f"  MIDI:  {MIDI}")
print(f"  MXL:   {MXL}")
