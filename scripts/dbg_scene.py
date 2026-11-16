"""Isolate which scene object fills the viewport by toggling visibility.

Current scene graph (post ADR-0010/0016/0017): body spheres + star glow
sprites + smoke-trail lines + (optional) reference grid. No starfield, no
orbit lines - those are gone.
"""
import time
from playwright.sync_api import sync_playwright

VARIANTS = [
    ("all-on", ""),
    ("no-glow", "s.traverse(o=>{if(o.isSprite)o.visible=false;})"),
    ("no-trails", "Object.values(w.trailByName).forEach(t=>t.line.visible=false)"),
    ("grid-on", "w.grid.visible=true"),
    ("bodies-only", "s.traverse(o=>{if(o.isSprite)o.visible=false;}); Object.values(w.trailByName).forEach(t=>t.line.visible=false); w.grid.visible=false"),
]

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
            w.grid.visible = false;
            Object.values(w.trailByName).forEach(t => t.line.visible = true);
            {js}
            w.render();
        }}""")
        time.sleep(0.3)
        pg.screenshot(path=rf"D:\test3\o2xygen\shots\dbg-{name}.png")
        print("saved", name)
    b.close()
