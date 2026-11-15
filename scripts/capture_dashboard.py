#!/usr/bin/env python3
"""Start the Streamlit dashboard headlessly and save a screenshot of every tab.

Optional helper used to produce docs/screenshots/. Requires Playwright with Chromium:
    pip install playwright && python -m playwright install chromium

Usage:
    python scripts/capture_dashboard.py [--port 8599] [--out docs/screenshots]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABS = ["Overview", "Violation records", "Annotated frames", "Model metrics"]


def wait_for(url: str, timeout: float = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"{url} did not come up within {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--chromium", default=None, help="path to a Chromium executable (optional)")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
                             "--server.headless=true", f"--server.port={args.port}",
                             "--browser.gatherUsageStats=false"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://localhost:{args.port}"
        wait_for(url + "/_stcore/health")
        with sync_playwright() as p:
            kw = {"executable_path": args.chromium} if args.chromium else {}
            browser = p.chromium.launch(**kw)
            page = browser.new_page(viewport={"width": 1500, "height": 2300})
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("text=Manufacturing Safety Analytics", timeout=60000)
            for i, tab in enumerate(TABS, 1):
                page.get_by_role("tab", name=tab).click()
                time.sleep(4)  # let plotly charts render
                path = out / f"{i:02d}_{tab.lower().replace(' ', '_')}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"saved {path}")
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=20)


if __name__ == "__main__":
    main()
