# Full end-to-end pipeline on any audio file (separation + all), MIDI +
# routed MusicXML + reusable intermediate pickle.
#   python tools/run_full.py <audio> [out_stem]
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

audio = sys.argv[1]
stem = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(audio))[0]
uni  = sys.argv[3] if len(sys.argv) > 3 else "off"   # off | study | apply
# Use cached stems when the song folder has them (Input/<song>/stems/), which
# skips Demucs entirely. Same model+seed as a live run, so results are
# equivalent - and identical between runs, which live separation is not.
_songdir = os.path.dirname(os.path.abspath(audio))
_cache = os.path.join(_songdir, "stems")
stem_cache = _cache if os.path.isdir(_cache) else None
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transcriptions")
os.makedirs(outdir, exist_ok=True)
midi = os.path.join(outdir, f"{stem}_FULLRUN.mid")
mxl = os.path.join(outdir, f"{stem}_FULLRUN.musicxml")
pkl = os.path.join(outdir, f"{stem}_FULLRUN.pkl")
corpus = os.path.join(outdir, f"{stem}_university.json")

t0 = time.time()
res = transcribe_file(
    audio, midi,
    output_musicxml_path=mxl,
    use_notation_timing=True,
    use_consolidated_timing=True,
    save_intermediate_path=pkl,
    university_mode=uni,
    university_corpus_path=(corpus if uni != "off" else None),
    stem_cache_dir=stem_cache,
)
print("=" * 60)
print(f"DONE in {time.time()-t0:.0f}s  ({os.path.basename(audio)})")
print(f"  separation_model: {res.separation_model}")
print(f"  tempo: {res.tempo_bpm:.1f}bpm  meter: {res.time_signature}")
print(f"  notes by stem: {res.note_counts_by_stem}")
print(f"  total notes: {res.total_notes_exported}")
print(f"  MIDI:  {midi}")
print(f"  MXL:   {mxl}")
print(f"  PKL:   {pkl}")
print(f"  STEMS: {'CACHED (Demucs skipped)' if stem_cache else 'separated live'}")
print(f"  UNIVERSITY: mode={uni}" + (f"  corpus={corpus}" if uni!="off" else ""))
