"""Isolate which scene object fills the viewport by toggling visibility."""
import time
from playwright.sync_api import sync_playwright

VARIANTS = [
    ("all-on", ""),
    ("no-sprites", "s.traverse(o=>{if(o.isSprite)o.visible=false;})"),
    ("no-starfield", "w.starfield.visible=false"),
    ("no-glow-no-starfield", "s.traverse(o=>{if(o.isSprite)o.visible=false;}); w.starfield.visible=false"),
    ("no-grid", "w.grid.visible=false"),
    ("basic-only", "s.traverse(o=>{if(o.isSprite||o.isPoints)o.visible=false;}); w.grid.visible=false; w.orbitLines.forEach(l=>l.visible=false)"),
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
            w.orbitLines.forEach(l => l.visible = true);
            {js}
            w.render();
        }}""")
        time.sleep(0.3)
        pg.screenshot(path=rf"D:\test3\o2xygen\shots\dbg-{name}.png")
        print("saved", name)
    b.close()
