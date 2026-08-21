# =================================================================
# MODULE: app/ui.py
# Grimlock Jazz - Transcribe mode (GRIMLOCK_6.0_FRONTEND_DESIGN.md §3).
#
# The window is deliberately a thin thing. Every question it asks is answered
# by app/probe.py, app/runner.py or app/diagnostics.py, none of which import a
# UI toolkit; this file arranges widgets and calls them. That is why the
# library layer was built first, and it is what keeps any of this testable.
#
# THE SHAPE OF THE SCREEN FOLLOWS THE SHAPE OF THE COST.
#
#   PROBE is its own step, with its own button, because it costs seconds and
#   the decision it informs costs half an hour. It shows the three numbers
#   behind its verdict - a verdict that skips a whole pipeline stage is not
#   one a user should have to take on trust.
#
#   THE PLAN LINE is the most important text on the screen. It says what the
#   run WILL do, in a sentence, and it updates the moment the guided override
#   changes. A user should never press Run without knowing whether they are
#   about to spend thirty minutes on Demucs.
#
#   THE FEED is the engine's own narration, not a progress bar. Window Pane
#   carries every decision the conductor logs, with its reasoning, and that is
#   what scrolls past. A percentage would be an invention; this is a fact.
#
#   THE HEALTH PANEL leads with rules, then measurements. A rule that fails is
#   a defect. A measurement without its reference is a number that invites
#   false confidence, so every one of them is shown beside the value it should
#   be compared against.
#
# Run it:  python -m app.ui
# =================================================================

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.diagnostics import FAIL, INFO, PASS, diagnose               # noqa: E402
from app.runner import JAZZ_ROOT, RunConfig, TranscriptionRunner     # noqa: E402

POLL_MS = 300

AUTO = "auto (let the probe decide)"
SEPARATION_CHOICES = {
    AUTO: None,
    "solo piano - skip Demucs": "solo_piano",
    "solo guitar - skip Demucs": "solo_guitar",
    "full band - always separate": "ensemble",
}


class TranscribeWindow(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master, padding=10)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=3)
        self.rowconfigure(6, weight=2)

        self.audio_path = tk.StringVar()
        self.out_stem = tk.StringVar()
        self.separation = tk.StringVar(value=AUTO)
        self.plan_text = tk.StringVar(
            value="Choose a recording, then Probe to see what a run would do.")
        self.probe_text = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="idle")

        self.guided_tempo = tk.StringVar()
        self.guided_meter = tk.StringVar()
        self.guided_key = tk.StringVar()

        self._probe = None
        self._runner: Optional[TranscriptionRunner] = None

        self._build_source()
        self._build_probe()
        self._build_guided()
        self._build_actions()
        self._build_feed()
        self._build_health()

        ttk.Label(self, textvariable=self.status_text,
                  foreground="#555").grid(row=7, column=0, sticky="w", pady=(6, 0))

    # ------------------------------------------------------------- widgets
    def _build_source(self) -> None:
        box = ttk.LabelFrame(self, text="1  Recording", padding=8)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Audio").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.audio_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(box, text="Browse...", command=self._browse).grid(row=0, column=2)
        ttk.Label(box, text="Name").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(box, textvariable=self.out_stem).grid(row=1, column=1, sticky="ew",
                                                        padx=6, pady=(6, 0))

    def _build_probe(self) -> None:
        box = ttk.LabelFrame(self, text="2  What is this? (seconds, no separation)", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(1, weight=1)
        ttk.Button(box, text="Probe", command=self._start_probe).grid(row=0, column=0, sticky="w")
        ttk.Label(box, textvariable=self.probe_text, justify="left",
                  foreground="#333").grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(box, text="Separation").grid(row=1, column=0, sticky="w", pady=(8, 0))
        combo = ttk.Combobox(box, textvariable=self.separation, state="readonly",
                             values=list(SEPARATION_CHOICES))
        combo.grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_plan())

        plan = ttk.Label(box, textvariable=self.plan_text, wraplength=760,
                         justify="left", foreground="#0a4")
        plan.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _build_guided(self) -> None:
        box = ttk.LabelFrame(
            self, text="3  What you already know (guided mode - a hard lock, not a vote)",
            padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for i, (label, var, hint) in enumerate((
                ("Tempo (bpm)", self.guided_tempo, "blank = detect"),
                ("Meter (e.g. 6/4)", self.guided_meter, "blank = detect"),
                ("Key (e.g. Bbm)", self.guided_key, "blank = detect"))):
            ttk.Label(box, text=label).grid(row=0, column=i * 3, sticky="w", padx=(0 if i == 0 else 12, 0))
            ttk.Entry(box, textvariable=var, width=10).grid(row=0, column=i * 3 + 1, padx=4)
            ttk.Label(box, text=hint, foreground="#888").grid(row=0, column=i * 3 + 2, sticky="w")

    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.run_button = ttk.Button(bar, text="Run transcription", command=self._start_run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(bar, text="Cancel", command=self._cancel,
                                        state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        ttk.Button(bar, text="Open in MuseScore",
                   command=self._open_score).pack(side="left", padx=6)
        ttk.Button(bar, text="Show folder", command=self._open_folder).pack(side="left")

    def _build_feed(self) -> None:
        box = ttk.LabelFrame(self, text="4  What the engine is doing", padding=6)
        box.grid(row=4, column=0, sticky="nsew", pady=(10, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.feed = tk.Text(box, height=12, wrap="word", state="disabled")
        self.feed.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(box, orient="vertical", command=self.feed.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.feed.configure(yscrollcommand=bar.set)
        self.feed.tag_configure("stage", foreground="#036")
        self.feed.tag_configure("why", foreground="#555")

    def _build_health(self) -> None:
        box = ttk.LabelFrame(self, text="5  Is this transcription trustworthy?", padding=6)
        box.grid(row=6, column=0, sticky="nsew", pady=(10, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        cols = ("check", "value", "reference", "detail")
        self.health = ttk.Treeview(box, columns=cols, show="headings", height=9)
        for col, width in zip(cols, (200, 130, 130, 520)):
            self.health.heading(col, text=col.upper())
            self.health.column(col, width=width, anchor="w")
        self.health.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(box, orient="vertical", command=self.health.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.health.configure(yscrollcommand=bar.set)
        self.health.tag_configure("fail", background="#ffe0e0")
        self.health.tag_configure("pass", background="#e6f7e6")

    # -------------------------------------------------------------- actions
    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a recording",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.aiff"), ("All", "*.*")],
            initialdir=os.path.join(JAZZ_ROOT, "Input"))
        if not path:
            return
        self.audio_path.set(path)
        if not self.out_stem.get():
            self.out_stem.set(os.path.splitext(os.path.basename(path))[0])
        self.probe_text.set("")
        self._probe = None
        self._refresh_plan()

    def _start_probe(self) -> None:
        path = self.audio_path.get().strip()
        if not path or not os.path.exists(path):
            self.status_text.set("pick a recording first")
            return
        self.status_text.set("probing...")
        self.probe_text.set("reading the audio...")

        def work():
            try:
                from app.probe import probe_audio
                result = probe_audio(path)
            except Exception as exc:
                self.after(0, lambda: self.status_text.set(f"probe failed: {exc}"))
                return
            self.after(0, lambda: self._probe_done(result))

        threading.Thread(target=work, daemon=True).start()

    def _probe_done(self, result) -> None:
        self._probe = result
        solo = result.solo
        verdict = solo.verdict if solo else "unknown"
        lines = [f"{verdict}  ({result.elapsed_seconds:.0f}s for "
                 f"{result.duration_seconds:.0f}s of audio)"]
        if solo is not None:
            # The working, shown. A verdict that skips a pipeline stage is not
            # one anybody should take on trust.
            lines.append(f"kit above 8kHz {solo.kit_hf_fraction:.3f}   "
                         f"sustained frames rising {solo.crescendo_fraction:.1%}   "
                         f"energy below 78Hz {solo.low_energy_fraction:.1%}")
            lines.append(f"sure it is solo: {solo.confidence:.2f}   "
                         f"sure which instrument: {solo.instrument_confidence:.2f}")
        if result.tempo_bpm:
            lines.append(f"rough reading: {result.tempo_bpm:.0f}bpm, "
                         f"{result.time_signature[0]}/{result.time_signature[1]} "
                         f"(a preview - the run decides properly)")
        self.probe_text.set("\n".join(lines))
        self.status_text.set("probed")
        self._refresh_plan()

    def _refresh_plan(self) -> None:
        guided = SEPARATION_CHOICES.get(self.separation.get())
        if self._probe is not None:
            self.plan_text.set(self._probe.plan(guided))
        elif guided is not None:
            from app.probe import ProbeResult
            self.plan_text.set(ProbeResult(audio_path="", duration_seconds=0.0).plan(guided))
        else:
            self.plan_text.set("Probe to see what a run would do, or set "
                               "Separation yourself to decide it outright.")

    def _config(self) -> Optional[RunConfig]:
        path = self.audio_path.get().strip()
        if not path or not os.path.exists(path):
            self.status_text.set("pick a recording first")
            return None
        stem = self.out_stem.get().strip() or os.path.splitext(os.path.basename(path))[0]

        meter = None
        if self.guided_meter.get().strip():
            try:
                num, den = self.guided_meter.get().strip().split("/")
                meter = [int(num), int(den)]
            except ValueError:
                self.status_text.set("meter must look like 6/4")
                return None
        tempo = None
        if self.guided_tempo.get().strip():
            try:
                tempo = float(self.guided_tempo.get().strip())
            except ValueError:
                self.status_text.set("tempo must be a number")
                return None

        return RunConfig(
            audio_path=path, out_stem=stem,
            guided_separation=SEPARATION_CHOICES.get(self.separation.get()),
            guided_tempo_bpm=tempo, guided_time_signature=meter,
            guided_key=self.guided_key.get().strip() or None,
        )

    def _start_run(self) -> None:
        config = self._config()
        if config is None:
            return
        self._clear_feed()
        self._append("run", f"{os.path.basename(config.audio_path)}\n")
        self._append("why", self.plan_text.get() + "\n\n")

        self._runner = TranscriptionRunner(config)
        self._runner.start(python_executable=sys.executable)
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_text.set("running - this takes 30-60 minutes unless separation is skipped")
        self.after(POLL_MS, self._pump)

    def _pump(self) -> None:
        runner = self._runner
        if runner is None:
            return
        for event in runner.poll():
            self._show_event(event)
        if runner.status.running:
            self.after(POLL_MS, self._pump)
            return

        self.run_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if runner.status.cancelled:
            self.status_text.set("cancelled - partial output discarded")
            return
        if runner.status.error:
            self.status_text.set(runner.status.error)
            self._append("why", "\n" + runner.worker_log(30))
            return
        self.status_text.set(
            f"done in {runner.status.elapsed_seconds/60:.1f} min")
        self._load_health(runner.config)

    def _cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()

    # ------------------------------------------------------------- feed/health
    def _clear_feed(self) -> None:
        self.feed.configure(state="normal")
        self.feed.delete("1.0", "end")
        self.feed.configure(state="disabled")

    def _append(self, tag: str, text: str) -> None:
        self.feed.configure(state="normal")
        self.feed.insert("end", text, tag)
        self.feed.see("end")
        self.feed.configure(state="disabled")

    def _show_event(self, event: dict) -> None:
        kind = event.get("event_type", "")
        stage = event.get("stage", "")
        payload = event.get("payload", {}) or {}
        if kind == "decision":
            self._append("stage", f"[{stage}] {payload.get('decision_type','')}\n")
            why = payload.get("reasoning")
            if why:
                self._append("why", f"    {why}\n")
        elif kind in ("run_start", "run_end"):
            self._append("stage", f"--- {kind} {payload}\n")

    def _load_health(self, config: RunConfig) -> None:
        paths = config.paths()
        report = diagnose(paths["pkl"], paths["musicxml"])
        for row in self.health.get_children():
            self.health.delete(row)
        # Rules first: a failed invariant is a defect, and it should not be
        # sitting below a row of statistics.
        for check in sorted(report.checks, key=lambda c: (not c.is_rule, c.name)):
            tag = check.status if check.is_rule else ""
            value = str(check.value)
            if len(value) > 60:
                value = value[:57] + "..."
            self.health.insert("", "end", tags=(tag,), values=(
                check.name, value,
                "" if check.reference is None else str(check.reference),
                check.detail))
        for err in report.errors:
            self.health.insert("", "end", values=("(error)", "", "", err))

    # -------------------------------------------------------------- handoff
    def _score_path(self) -> Optional[str]:
        config = self._config()
        if config is None:
            return None
        path = config.paths()["musicxml"]
        return path if os.path.exists(path) else None

    def _open_score(self) -> None:
        path = self._score_path()
        if path is None:
            self.status_text.set("no engraved score for that name yet")
            return
        try:
            os.startfile(path)                       # noqa: S606 - Windows handoff
        except AttributeError:
            subprocess.Popen(["xdg-open", path])
        self.status_text.set(f"opened {os.path.basename(path)}")

    def _open_folder(self) -> None:
        config = self._config()
        folder = config.out_dir if config else os.path.join(JAZZ_ROOT, "transcriptions")
        try:
            os.startfile(folder)                     # noqa: S606
        except AttributeError:
            subprocess.Popen(["xdg-open", folder])


def main() -> None:
    root = tk.Tk()
    root.title("Grimlock Jazz 6.0 - Transcribe")
    root.geometry("1040x900")
    TranscribeWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
