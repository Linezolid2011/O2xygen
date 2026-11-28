"""Isolate which scene object fills the viewport by toggling visibility.

Current scene graph (post ADR-0010/0016/0017): body spheres + star glow
sprites + smoke-trail lines. The ecliptic reference grid was retired in
branch 2; the surface overlay (graticule + observation dots) is a per-body
mesh child, not a scene-level object.
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

VARIANTS = [
    ("all-on", ""),
    ("no-glow", "s.traverse(o=>{if(o.isSprite)o.visible=false;})"),
    ("no-trails", "Object.values(w.trailByName).forEach(t=>t.line.visible=false)"),
    ("overlay-on", "w.setSurfaceOverlay(true)"),
    ("bodies-only", "s.traverse(o=>{if(o.isSprite)o.visible=false;}); Object.values(w.trailByName).forEach(t=>t.line.visible=false)"),
]

OUT = Path(__file__).resolve().parents[1] / "shots"

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True,
                          args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1200, "height": 700})
    pg.goto("http://127.0.0.1:8501/", wait_until="networkidle")
    time.sleep(3)
    for name, js in VARIANTS:
        pg.evaluate(f"""() => {{
            const w = window.__scene, s = w.scene;
            s.traverse(o => o.visible = true);
            w.setSurfaceOverlay(false);
            Object.values(w.trailByName).forEach(t => t.line.visible = true);
            {js}
            w.render();
        }}""")
        time.sleep(0.3)
        pg.screenshot(path=str(OUT / f"dbg-{name}.png"))
        print("saved", name)
    b.close()
