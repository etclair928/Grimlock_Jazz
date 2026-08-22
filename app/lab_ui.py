# =================================================================
# MODULE: app/lab_ui.py
# Lab mode's window (GRIMLOCK_6.0_FRONTEND_DESIGN.md §4).
#
# The question this half of the app exists to answer is not "is this chart
# good" but "did that change actually move anything" - and that question was
# answered by hand five times in one working session and got the wrong answer
# three of them. The cause was identical every time: a measurement path
# quietly differing from the shipping path. Re-exports measured on a flat
# clock, diagnostics run over parts the pipeline never grand-staffs, an audit
# that re-derived the rule it was auditing.
#
# So this window has one rule beyond arranging widgets: every number comes
# from app/lab.py, which comes from app/diagnostics.py, which delegates to the
# shipped functions. Nothing here re-derives anything, and a comparison across
# two different notation paths SAYS SO in amber rather than looking tidy.
# =================================================================

from __future__ import annotations

import os
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk

from app.diagnostics import FAIL, PASS
from app.lab import compare, health_of, list_runs, reexport, score_against
from app.runner import JAZZ_ROOT


class LabWindow(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master, padding=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=2)
        self.rowconfigure(3, weight=3)
        self.status = tk.StringVar(value="")
        self._runs = []
        self._build_runs()
        self._build_actions()
        self._build_detail()
        ttk.Label(self, textvariable=self.status, foreground="#555",
                  wraplength=1000, justify="left").grid(
            row=4, column=0, sticky="w", pady=(6, 0))
        self.refresh()

    # -------------------------------------------------------------- widgets
    def _build_runs(self) -> None:
        ttk.Label(self, text="Runs  (select one, or two to compare)").grid(
            row=0, column=0, sticky="w")
        box = ttk.Frame(self)
        box.grid(row=1, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        cols = ("stem", "notes", "bpm", "meter", "key", "score", "reproducible")
        self.runs = ttk.Treeview(box, columns=cols, show="headings",
                                 selectmode="extended", height=10)
        for col, width in zip(cols, (270, 70, 60, 70, 60, 60, 120)):
            self.runs.heading(col, text=col.upper())
            self.runs.column(col, width=width, anchor="w")
        self.runs.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(box, orient="vertical", command=self.runs.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.runs.configure(yscrollcommand=bar.set)
        # Amber, not hidden. An intermediate that cannot reproduce its own
        # score is still worth listing - it is just not comparable with one
        # that can, and the colour is the only warning that arrives before
        # someone draws a conclusion from it.
        self.runs.tag_configure("stale", foreground="#a66000")

    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for text, command in (("Refresh", self.refresh),
                              ("Health", self._health),
                              ("Compare two", self._compare),
                              ("Re-export notation", self._reexport),
                              ("Score vs answer key", self._score),
                              ("Open in MuseScore", self._open)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=(0, 6))

    def _build_detail(self) -> None:
        box = ttk.Frame(self)
        box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.detail = tk.Text(box, height=16, wrap="word", state="disabled")
        self.detail.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(box, orient="vertical", command=self.detail.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.detail.configure(yscrollcommand=bar.set)
        self.detail.tag_configure("head", foreground="#003366")
        self.detail.tag_configure("bad", foreground="#aa0000")
        self.detail.tag_configure("warn", foreground="#a66000")

    # -------------------------------------------------------------- helpers
    def _say(self, text: str, tag: str = "") -> None:
        self.detail.configure(state="normal")
        self.detail.insert("end", text, tag)
        self.detail.see("end")
        self.detail.configure(state="disabled")

    def _clear(self) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.configure(state="disabled")

    def _selected(self):
        return [self._runs[int(i)] for i in self.runs.selection()]

    def refresh(self) -> None:
        self._runs = list_runs()
        for row in self.runs.get_children():
            self.runs.delete(row)
        for i, record in enumerate(self._runs):
            self.runs.insert(
                "", "end", iid=str(i),
                tags=("" if record.reproducible else "stale",),
                values=(record.stem, record.notes, f"{record.tempo_bpm:.0f}",
                        f"{record.time_signature[0]}/{record.time_signature[1]}",
                        record.key, "yes" if record.has_score else "-",
                        "yes" if record.reproducible else "flat clock"))
        self.status.set(
            f"{len(self._runs)} runs. Amber means the intermediate predates "
            f"beat_times_ms: a re-export of it engraves on a flat clock and is "
            f"NOT the notation the pipeline shipped, so notation numbers from "
            f"it are not comparable with a newer run's.")

    # -------------------------------------------------------------- actions
    def _health(self) -> None:
        selection = self._selected()
        if not selection:
            self.status.set("select a run first")
            return
        self._clear()
        for record in selection[:2]:
            report = health_of(record)
            self._say(f"{record.stem}    rules hold: {report.rules_hold}\n", "head")
            for check in sorted(report.checks, key=lambda c: (not c.is_rule, c.name)):
                mark = {PASS: "PASS", FAIL: "FAIL"}.get(check.status, "    ")
                ref = f"   ref {check.reference}" if check.reference is not None else ""
                self._say(f"  {mark:4} {check.name:26} {str(check.value)[:46]}{ref}\n",
                          "bad" if check.status == FAIL else "")
            for err in report.errors:
                self._say(f"  ERROR {err}\n", "bad")
            self._say("\n")

    def _compare(self) -> None:
        selection = self._selected()
        if len(selection) != 2:
            self.status.set("select exactly two runs to compare")
            return
        self._clear()
        result = compare(selection[0], selection[1])
        self._say(f"{result.left}   ->   {result.right}\n", "head")
        for warning in result.warnings:
            self._say(f"  WARNING: {warning}\n", "warn")
        if not result.changed:
            self._say("  nothing moved.\n")
        for delta in result.changed:
            arrow = (f"{delta.numeric_delta:+g}" if delta.numeric_delta is not None
                     else "changed")
            self._say(f"  {delta.name:26} {str(delta.left)[:22]:>22} -> "
                      f"{str(delta.right)[:22]:<22} {arrow}\n")

    def _reexport(self) -> None:
        selection = self._selected()
        if not selection:
            self.status.set("select a run first")
            return
        self._clear()
        for record in selection[:2]:
            try:
                path, warns = reexport(record)
            except Exception as exc:
                self._say(f"{record.stem}: {type(exc).__name__}: {exc}\n", "bad")
                continue
            for warning in warns:
                self._say(f"  WARNING: {warning}\n", "warn")
            self._say(f"{record.stem}: re-exported -> {os.path.basename(path)}\n", "head")
        self.refresh()

    def _score(self) -> None:
        selection = self._selected()
        if not selection:
            self.status.set("select a run first")
            return
        key = filedialog.askopenfilename(
            title="Answer key (MusicXML)",
            initialdir=os.path.join(JAZZ_ROOT, "transcriptions"),
            filetypes=[("MusicXML", "*.musicxml *.xml *.mxl"), ("All", "*.*")])
        if not key:
            return
        self._clear()
        for record in selection[:2]:
            try:
                result = score_against(record, key)
            except Exception as exc:
                self._say(f"{record.stem}: {type(exc).__name__}: {exc}\n", "bad")
                continue
            if "error" in result:
                self._say(f"{record.stem}: {result['error']}\n", "bad")
                continue
            self._say(f"{record.stem}  vs  {os.path.basename(key)}\n", "head")
            self._say(f"  precision {result['precision']:.3f}    "
                      f"recall {result['recall']:.3f}    F1 {result['f1']:.3f}\n")
            self._say(f"  {result['hits']} hits, {result['ours']} emitted, "
                      f"{result['reference']} in the reference\n\n")

    def _open(self) -> None:
        selection = self._selected()
        if not selection or not selection[0].has_score:
            self.status.set("no engraved score for that run yet")
            return
        path = selection[0].musicxml_path
        try:
            os.startfile(path)                      # noqa: S606 - Windows handoff
        except AttributeError:
            subprocess.Popen(["xdg-open", path])
        self.status.set(f"opened {os.path.basename(path)}")


__all__ = ["LabWindow"]
