#!/usr/bin/env python3
# =================================================================
# TOOL: tools/code2prompt.py
# Dumps the Jazz source tree into one markdown file for pasting into an AI.
#
# EXCLUDES, deliberately:
#   Input/ and transcriptions/   audio and generated scores - megabytes of
#                                content that says nothing about the code
#   .venv/ __pycache__/ .git/    not source
#   previous dumps               so a dump never contains a dump
#   binaries                     wav/mp3/mid/musicxml/pkl/pdf/zip
#
#   python tools/code2prompt.py [-o jazz_code2prompt2.md] [--no-docs]
# =================================================================

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

JAZZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", ".venv", "__pycache__", ".idea", ".pytest_cache",
             "Input", "transcriptions", "debug", "node_modules", ".claude"}
SKIP_FILES = {"code2prompt_jazz.md", "jazz_code2prompt.md", "jazz_code2prompt2.md"}
CODE_EXT = {".py"}
DOC_EXT = {".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml"}
SKIP_EXT = {".wav", ".mp3", ".mid", ".midi", ".musicxml", ".mxl", ".pkl",
            ".pdf", ".zip", ".png", ".jpg", ".npz", ".pyc"}
LANG = {".py": "python", ".md": "markdown", ".json": "json",
        ".toml": "toml", ".yaml": "yaml", ".yml": "yaml"}


def collect(include_docs: bool) -> List[Tuple[str, str]]:
    out = []
    for root, dirs, files in os.walk(JAZZ):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name in SKIP_FILES:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in SKIP_EXT:
                continue
            if ext not in CODE_EXT and not (include_docs and ext in DOC_EXT):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, JAZZ).replace("\\", "/")
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    out.append((rel, fh.read()))
            except OSError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(JAZZ, "jazz_code2prompt2.md"))
    ap.add_argument("--no-docs", action="store_true",
                    help="source only - omit the .md design/open-problems docs")
    args = ap.parse_args()

    files = collect(not args.no_docs)
    code = [f for f in files if f[0].endswith(".py")]
    docs = [f for f in files if not f[0].endswith(".py")]

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Grimlock Jazz 6.0 - source dump\n\n")
        fh.write(f"{len(code)} python files, {len(docs)} other. "
                 "Input/, transcriptions/, .venv/ and binaries omitted.\n\n")
        fh.write("## Tree\n\n```\n")
        for rel, _ in files:
            fh.write(rel + "\n")
        fh.write("```\n\n")
        for rel, text in files:
            lang = LANG.get(os.path.splitext(rel)[1].lower(), "")
            fh.write(f"\n---\n\n## `{rel}`\n\n```{lang}\n{text.rstrip()}\n```\n")

    size = os.path.getsize(args.out)
    lines = sum(t.count("\n") for _r, t in files)
    print(f"wrote {args.out}")
    print(f"  {len(code)} .py + {len(docs)} other = {len(files)} files")
    print(f"  {lines:,} source lines, {size/1024:.0f} KB "
          f"(~{size//4:,} tokens rough)")


if __name__ == "__main__":
    main()
