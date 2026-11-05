"""One call that does a whole session: analysis, references, fixes, and rendered outputs.

Both the CLI (analyze.py) and the Streamlit app call run().
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .coaching import Fix, Reference, choose_references, make_fixes
from .pipeline import AnalysisError, SessionResult, analyze_video

ProgressFn = Callable[[float, str], None]


@dataclass
class Report:
    session: SessionResult
    references: dict[str, Reference]
    fixes: list[Fix]
    fixes_note: str | None
    notes: list[str]
    reference_session: SessionResult | None = None
    outputs: dict[str, str] = field(default_factory=dict)  # name -> file path

    def to_dict(self) -> dict:
        return {
            **self.session.to_dict(),
            "references": {k: r.to_dict() for k, r in self.references.items()},
            "fixes": [f.to_dict() for f in self.fixes],
            "fixes_note": self.fixes_note,
            "notes": self.notes,
            "reference_clip": None if self.reference_session is None else self.reference_session.to_dict(),
            "outputs": self.outputs,
        }


def run(video: str | Path, handedness: str, out_dir: str | Path, fast: bool = False,
        reference: str | Path | None = None, reference_handedness: str | None = None,
        render_videos: bool = True, progress: ProgressFn | None = None, use_cache: bool = True) -> Report:
    progress = progress or (lambda f, m: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    has_ref = reference is not None
    span = (0.0, 0.55 if has_ref else 0.75) if render_videos else (0.0, 0.8 if has_ref else 1.0)

    def sub(lo: float, hi: float) -> ProgressFn:
        return lambda f, m: progress(lo + (hi - lo) * f, m)

    session = analyze_video(video, handedness, fast=fast, progress=sub(*span), use_cache=use_cache)
    ref_session = None
    notes: list[str] = []
    if has_ref:
        lo = span[1]
        hi = lo + (0.2 if render_videos else 1.0 - lo)
        try:
            ref_session = analyze_video(reference, reference_handedness or handedness, fast=fast,
                                        progress=sub(lo, hi), use_cache=use_cache)
        except AnalysisError as e:
            notes.append(f"Reference clip not used: {e}")
        span = (lo, hi)
    refs, ref_notes = choose_references(session, ref_session)
    notes += ref_notes
    fixes, fixes_note = make_fixes(session, refs)
    report = Report(session=session, references=refs, fixes=fixes, fixes_note=fixes_note, notes=notes,
                    reference_session=ref_session)

    if render_videos:
        from . import render
        t0 = time.time()
        report.outputs.update(render.render_outputs(report, out, progress=sub(span[1], 1.0)))
        session.timings_s["render"] = time.time() - t0

    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2, default=float), encoding="utf-8")
    report.outputs["report"] = str(out / "report.json")
    progress(1.0, "Done")
    return report
