"""Headless-Chrome screenshot tool for UI iteration.

Usage:
  python scripts/shot.py out.png [--js "js code to run before shot"] [--wait 3]
                                 [--w 1680] [--h 945] [--tab temperature]
Captures console errors to stderr so we can fix things silently failing.
"""
import argparse
import sys
import time

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--js", default="")
ap.add_argument("--wait", type=float, default=3.0)
ap.add_argument("--w", type=int, default=1680)
ap.add_argument("--h", type=int, default=945)
ap.add_argument("--tab", default="")
ap.add_argument("--select", default="")
args = ap.parse_args()

errors, messages = [], []
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
                                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": args.w, "height": args.h})
    page.on("console", lambda m: messages.append(f"[{m.type}] {m.text}"))
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("response", lambda r: messages.append(f"[http {r.status}] {r.url}") if r.status >= 400 else None)
    page.goto("http://127.0.0.1:8501/", wait_until="networkidle")
    page.wait_for_selector("#catalog li.selected", timeout=20000)   # boot complete
    if args.select:
        page.evaluate(f"""(() => {{
            const items = document.querySelectorAll('#catalog li');
            for (const li of items) if (li.dataset.name === {args.select!r}) li.click();
        }})()""")
        time.sleep(1.2)
    if args.tab:
        page.evaluate(f"""(() => {{
            for (const b of document.querySelectorAll('.ptab'))
                if (b.dataset.tab === {args.tab!r}) b.click();
        }})()""")
        time.sleep(0.8)
    if args.js:
        page.evaluate(f"""() => new Promise((resolve) => {{
            const t0 = Date.now();
            const tick = () => {{
                const el = document.querySelector({args.js!r});
                if (el) {{ el.click(); return resolve(); }}
                if (Date.now() - t0 > 8000) return resolve();
                setTimeout(tick, 200);
            }};
            tick();
        }})""")
    time.sleep(args.wait)
    page.screenshot(path=args.out)
    browser.close()

print(f"saved {args.out}")
for e in errors:
    print("PAGEERROR:", e, file=sys.stderr)
bad = [m for m in messages if m.startswith("[error]")]
for m in bad[:10]:
    print("CONSOLE:", m, file=sys.stderr)
