"""Real Chromium/WebGL2 live-mode acceptance. Read-only for the world save.

Run the project server first, then:
  python scripts/verify_live.py
Playwright is a development-only dependency, not a project runtime dependency.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
(root / "tmp").mkdir(exist_ok=True)
errors, logs = [], []
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
        args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1600, "height": 960})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
    page.goto("http://127.0.0.1:8501/", wait_until="load", timeout=30000)
    page.wait_for_selector("#catalog li.selected", timeout=30000)
    page.locator('[data-app-mode="live"]').click()
    page.wait_for_function("window.__surface?.active && window.__surface?.data", timeout=30000)
    page.wait_for_function("window.__surface?.atmosphere", timeout=120000)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(root / "shots" / "live-zenith.png"))
    zenith = page.evaluate("({mode:document.body.className,point:__surface.data.point,multi:__surface.atmosphere.multiple.length,stars:__surface.stars.geometry.attributes.position.count,camera:__surface.camera.position.toArray()})")
    pixels = page.evaluate("""() => {const s=__surface;s.render();const gl=s.renderer.getContext(),a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);let n=0,max=0;for(let i=0;i<a.length;i+=4){const v=a[i]+a[i+1]+a[i+2];if(v>0)n++;max=Math.max(max,v);}return {nonzero:n,max};}""")
    assert pixels["nonzero"] > 100, pixels
    page.evaluate("__surface.pitch=-1.30")
    page.wait_for_timeout(700)
    page.screenshot(path=str(root / "shots" / "live-horizon.png"))
    page.keyboard.press("Home")
    page.wait_for_timeout(300)
    home = page.evaluate("({yaw:__surface.yaw,pitch:__surface.pitch,time:__surface.data.t_days})")
    assert home == {"yaw": 0, "pitch": 0, "time": 0}, home
    page.locator('[data-app-mode="edit"]').click()
    page.wait_for_selector("body.mode-edit", timeout=30000)
    page.evaluate("window.__scene.cb.onSelect('Νεφέλη')")
    page.wait_for_selector("#nonstar-editor:not([hidden])", timeout=30000)
    page.locator("#ns-preview").click()
    page.wait_for_function("document.querySelector('#ns-tau').textContent !== '—'", timeout=10000)
    editor = page.evaluate("({period:document.querySelector('#ns-period').value,phase:document.querySelector('#ns-phase').value,atm:document.querySelector('#ns-has-atm').checked})")
    assert editor["atm"] is True
    page.screenshot(path=str(root / "shots" / "body-editor.png"))
    # Regression: the non-star overlay must not cover a subsequently selected
    # star editor. Exercise both directions through the real selection callback.
    page.evaluate("window.__scene.cb.onSelect('Αστραία')")
    page.wait_for_selector("#star-editor:not([hidden])", timeout=30000)
    assert page.evaluate("document.querySelector('#nonstar-editor').hidden") is True
    page.evaluate("window.__scene.cb.onSelect('Νεφέλη')")
    page.wait_for_selector("#nonstar-editor:not([hidden])", timeout=30000)
    editor_switch = page.evaluate("({starHidden:document.querySelector('#star-editor').hidden,bodyHidden:document.querySelector('#nonstar-editor').hidden})")
    assert editor_switch == {"starHidden": True, "bodyHidden": False}, editor_switch
    page.locator("#ns-close").click()
    page.locator('[data-app-mode="view"]').click()
    page.wait_for_selector("body.mode-view", timeout=30000)
    restored = page.evaluate("({panel:!document.querySelector('#panel-body').hidden,title:document.querySelector('#catalog-head-title').textContent,evolveDisabled:document.querySelector('#btn-evolve').disabled})")
    print(json.dumps({"zenith": zenith, "pixels": pixels, "home": home,
                      "editor": editor, "editor_switch": editor_switch,
                      "restored": restored}, ensure_ascii=False, indent=2))
    print("ERRORS", json.dumps(errors, ensure_ascii=False))
    bad = [x for x in logs if x.startswith("[error]")]
    print("LOGS", json.dumps(bad, ensure_ascii=False))
    assert not errors and not bad, (errors, bad)
    browser.close()
