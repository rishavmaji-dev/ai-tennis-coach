"""Drive the running Streamlit app with Playwright: upload a sample, analyze, screenshot, and check
that every <video> on the page actually decodes in a real browser.

    .venv\\Scripts\\python -m streamlit run app.py          (in another terminal)
    .venv\\Scripts\\python scripts\\screenshot_ui.py samples\\forehand.mp4 --hand right
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent

VIDEO_CHECK = """async () => {
  const vids = [...document.querySelectorAll('video')];
  const out = [];
  for (const v of vids) {
    v.muted = true;
    try { await v.play(); } catch (e) {}
    const t0 = Date.now();
    while (v.readyState < 2 && !v.error && Date.now() - t0 < 15000) { await new Promise(r => setTimeout(r, 200)); }
    out.push({src: (v.currentSrc || '').slice(-60), readyState: v.readyState, width: v.videoWidth,
              height: v.videoHeight, duration: v.duration, error: v.error ? v.error.code : null,
              canPlayH264: v.canPlayType('video/mp4; codecs="avc1.640028"')});
  }
  return out;
}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    ap.add_argument("--channel", default="chrome", help="chrome / msedge / chromium (bundled)")
    ap.add_argument("--timeout-min", type=float, default=15)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if args.channel == "chromium" else args.channel)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(args.url)
        page.get_by_text("Tennis stroke analyzer").first.wait_for(timeout=60_000)
        page.locator('input[type="file"]').first.set_input_files(str(args.video.resolve()))
        # wait until the uploader shows the file name
        page.get_by_text(args.video.name).first.wait_for(timeout=10 * 60_000)
        page.get_by_role("radiogroup", name="Is the player right- or left-handed?").get_by_role(
            "radio", name="Right-handed" if args.hand == "right" else "Left-handed").click()
        page.screenshot(path=str(args.out / "01_ready.png"))
        page.get_by_role("button", name="Analyze").click()
        page.get_by_role("heading", name=re.compile(r"^(Your \d+ fix(es)?|Fixes)$")).wait_for(
            timeout=int(args.timeout_min * 60_000))
        page.wait_for_timeout(3000)
        page.screenshot(path=str(args.out / "02_results_top.png"))

        checks = page.evaluate(VIDEO_CHECK)
        print(json.dumps(checks, indent=2))

        # Full results: grow the viewport to the app's scroll height so one image holds everything.
        height = page.evaluate("""() => {
            const el = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
            return Math.max(el.scrollHeight, document.body.scrollHeight); }""")
        page.set_viewport_size({"width": 1440, "height": min(int(height) + 200, 16000)})
        page.wait_for_timeout(2000)
        page.screenshot(path=str(args.out / "03_results_full.png"), full_page=True)
        browser.close()

    bad = [c for c in checks if c["readyState"] < 2 or c["error"] or not c["width"]]
    print(f"{len(checks)} video element(s); {len(checks) - len(bad)} decoding in {args.channel}.")
    print(f"Screenshots in {args.out}")
    return 1 if bad or not checks else 0


if __name__ == "__main__":
    sys.exit(main())
