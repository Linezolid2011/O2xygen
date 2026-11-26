"""Isolated 0.4.1 appearance/editing acceptance on a temporary world copy.

Never saves the user's world. Starts an in-process HTTP server on a random port,
copies the current save to a TemporaryDirectory, and closes both server/browser.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
(root / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="o2xygen-appearance-") as temp:
    world_dir = Path(temp) / "白玫瑰"
    world_dir.mkdir()
    shutil.copy2(root / "data/worlds/白玫瑰/白玫瑰.oxygen", world_dir / "白玫瑰.oxygen")
    os.environ["O2YGEN_WORLDS_DIR"] = str(Path(temp))
    import sys
    sys.path.insert(0, str(root / "server"))
    import app
    server = app.Server(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True,
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1500, "height": 900})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(url, wait_until="load")
            page.wait_for_selector("#catalog li.selected", timeout=30000)
            page.locator('[data-app-mode="edit"]').click()
            page.wait_for_selector("body.mode-edit")
            page.evaluate("window.__scene.cb.onSelect('Νεφέλη')")
            page.wait_for_selector("#nonstar-editor:not([hidden])")
            page.locator("#ns-mass").fill("2")
            page.locator("#ns-albedo").fill("0.31")
            for channel, value in zip("rgb", [204, 102, 51]):
                page.locator(f"#ns-color-{channel}").fill(str(value))
            page.locator("#ns-save").click()
            page.wait_for_selector("#nonstar-editor[hidden]", state="attached", timeout=30000)
            saved = app.world_mod.load_world(Path(temp) / "白玫瑰" / "白玫瑰.oxygen")
            b = saved.bodies["Νεφέλη"]
            assert abs(b.mass_kg / 5.9722e24 - 2.0) < 1e-12
            assert b.albedo == 0.31
            assert b.extra["color_rgb"] == [0.8, 0.4, 0.2]
            sync = page.evaluate("""async()=>{
              const {bodyHex}=await import('/js/color.js');
              const {bodyDisplayColor}=await import('/js/scene3d.js');
              const s=__scene,b=s.system.bodies.find(b=>b.name==='Νεφέλη');
              const dot=document.querySelector('#catalog li[data-name="Νεφέλη"] .dot');
              const expected=bodyDisplayColor(b).toArray();
              const actual=s.bodyMeshes['Νεφέλη'].material.uniforms.uBodyColor.value.toArray();
              const trail=s.trailByName['Νεφέλη'].baseColor.toArray();
              const star=s.system.bodies.find(b=>b.name==='Εἴδωλον');
              const starDot=document.querySelector('#catalog li[data-name="Εἴδωλον"] .dot');
              return {bodyHex:bodyHex(b),dot:dot.style.background,
                sphereError:Math.max(...actual.map((x,i)=>Math.abs(x-expected[i]))),
                trailError:Math.max(...trail.map((x,i)=>Math.abs(x-expected[i]))),
                starHex:bodyHex(star),starDot:starDot.style.background,
                starColor:s.bodyMeshes['Εἴδωλον'].material.color.toArray(),
                starExpected:bodyDisplayColor(star).toArray(),
                ambient:s.bodyMeshes['Νεφέλη'].material.uniforms.uAmbient.value,
                albedo:s.bodyMeshes['Νεφέλη'].material.uniforms.uAlbedo.value};
            }""")
            assert sync["bodyHex"] == "#cc6633" and sync["dot"] == "rgb(204, 102, 51)", sync
            assert sync["sphereError"] < 1e-12 and sync["trailError"] < 1e-12, sync
            expected_star_rgb = tuple(int(sync["starHex"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            actual_star_rgb = tuple(int(x) for x in sync["starDot"]
                                    .removeprefix("rgb(").removesuffix(")").split(","))
            assert expected_star_rgb == actual_star_rgb, sync
            # GPU/Three.js colors pass through float32; sub-ulp equality is not expected.
            assert max(abs(a-b) for a,b in zip(sync["starColor"], sync["starExpected"])) < 1e-9, sync
            assert sync["ambient"] == 0.015 and sync["albedo"] == 0.31
            page.wait_for_function("Object.values(__scene.snap?.stars ?? {}).some(s=>s.luminosity_w>0)", timeout=30000)
            contrast = page.evaluate("""()=>{
              const s=__scene,g=s.bodyGroups['Νεφέλη'],mesh=s.bodyMeshes['Νεφέλη'];
              const target=g.position.clone(),light=mesh.material.uniforms.uStarDirection.value[0].clone();
              const radius=s.displayRadii['Νεφέλη'];
              const sample=(side)=>{
                s.controls.target.copy(target);
                s.camera.position.copy(target).addScaledVector(light,side*radius*4);
                s.camera.near=radius*0.01;s.camera.updateProjectionMatrix();s.controls.update();
                s.update(0);s.render();
                const gl=s.renderer.getContext(),a=new Uint8Array(4);
                gl.readPixels((gl.drawingBufferWidth/2)|0,(gl.drawingBufferHeight/2)|0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,a);
                return [...a];
              };
              return {day:sample(1),night:sample(-1)};
            }""")
            assert sum(contrast["day"][:3]) > 3 * max(sum(contrast["night"][:3]), 1), contrast
            page.evaluate("""()=>{
              const s=__scene,g=s.bodyGroups['Νεφέλη'];
              const target=g.position.clone(),light=s.bodyMeshes['Νεφέλη'].material.uniforms.uStarDirection.value[0].clone();
              const R=s.displayRadii['Νεφέλη'];s.controls.target.copy(target);
              s.camera.position.copy(target).addScaledVector(light,R*4);
              s.camera.near=R*0.01;s.camera.updateProjectionMatrix();s.controls.update();s.update(0);s.render();
            }""")
            page.screenshot(path=str(root / "shots" / "nonstar-day.png"))
            page.evaluate("""()=>{
              const s=__scene,g=s.bodyGroups['Νεφέλη'];
              const target=g.position.clone(),light=s.bodyMeshes['Νεφέλη'].material.uniforms.uStarDirection.value[0].clone();
              const R=s.displayRadii['Νεφέλη'];s.controls.target.copy(target);
              s.camera.position.copy(target).addScaledVector(light,-R*4);
              s.camera.near=R*0.01;s.camera.updateProjectionMatrix();s.controls.update();s.update(0);s.render();
            }""")
            page.screenshot(path=str(root / "shots" / "nonstar-night.png"))
            browser.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
    print(json.dumps({"sync": sync, "contrast": contrast, "errors": errors},
                     ensure_ascii=False, indent=2))
    assert not errors, errors
