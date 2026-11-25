"""Synthetic daylight rendering control; no save and no server-state mutation.

Only the in-browser observation's light direction is replaced. This is a
renderer regression test, NOT a claim that the current real point is in daylight.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
(root / "tmp").mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
        args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1200, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto("http://127.0.0.1:8501/", wait_until="load")
    page.wait_for_selector("#catalog li.selected")
    page.locator('[data-app-mode="live"]').click()
    page.wait_for_function("__surface.atmosphere", timeout=120000)
    result = page.evaluate("""() => {
      const s=__surface,d=structuredClone(s.data),stars=d.bodies.filter(b=>b.kind==='star');
      stars.forEach((b,i)=>{b.direction_enu=i?[0,0,-1]:[0,0,1];b.illumination_wm2=i?0:b.illumination_wm2;b.above_horizon=!i;b.occluded_by=[];});
      d.point.name='合成昼侧散射测试（不入档）';s.setData(d);s.render();
      const gl=s.renderer.getContext();function pixel(){const a=new Uint8Array(4);gl.readPixels((gl.drawingBufferWidth*.25)|0,(gl.drawingBufferHeight/2)|0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,a);return [...a];}
      const total=pixel(),old=s.uniforms.uMultiple.value,tex=old.clone();
      tex.image={...old.image,data:new Float32Array(old.image.data.length)};tex.needsUpdate=true;
      s.uniforms.uMultiple.value=tex;s.render();const single=pixel();
      s.uniforms.uMultiple.value=old;s.render();return {total,single,time:s.data.t_days};
    }""")
    page.screenshot(path=str(root / "shots" / "atmosphere-scattering-check.png"))
    print(json.dumps({"synthetic_daylight": result, "errors": errors},
                     ensure_ascii=False, indent=2))
    assert result["total"][2] > 20, result
    assert sum(abs(a-b) for a,b in zip(result["total"], result["single"])) > 2, result
    assert result["time"] == 0, result
    assert not errors, errors
    browser.close()
