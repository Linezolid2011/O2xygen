"""Read-only photometry/color acceptance for both sky modes.

The night case is explicitly an in-browser rendering control; no observation,
world state, or save file is changed.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
(root / "tmp").mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
        args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1600, "height": 960})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto("http://127.0.0.1:8501/", wait_until="load")
    page.wait_for_selector("#catalog li.selected")
    page.locator('[data-app-mode="live"]').click()
    page.wait_for_function("__surface.atmosphere && __surface.data?.point?.latitude_deg===0", timeout=120000)
    result = page.evaluate("""() => {
      const s=__surface,star=s.data.bodies.find(b=>b.name==='Αστραία');
      const target=s.up.clone().set(...star.direction_enu).normalize();
      s.yaw=-Math.atan2(target.x,target.y);s.pitch=-Math.acos(target.z);
      s.camera.fov=8;s.camera.updateProjectionMatrix();s.render();
      const gl=s.renderer.getContext();
      const pixel=(x,y)=>{const a=new Uint8Array(4);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,a);return [...a];};
      const cx=(gl.drawingBufferWidth/2)|0,cy=(gl.drawingBufferHeight/2)|0;
      const disk=pixel(cx,cy),background=pixel((cx+100)|0,cy);
      const read=()=>{const a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
        gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);return a;};
      const before=read();s.stars.visible=false;s.render();const after=read();s.stars.visible=true;s.render();
      let sum=0,large=0,changed=0;
      for(let i=0;i<before.length;i+=4){const d=(Math.abs(before[i]-after[i])+Math.abs(before[i+1]-after[i+1])+Math.abs(before[i+2]-after[i+2]))/3;
        sum+=d;if(d>0.5)changed++;if(d>=10)large++;}
      const a=__scene.skyData,b=s.data.sky,colors=[];
      const other=new Map(b.hip.map((h,i)=>[h,i]));
      for(let i=0;i<a.hip.length && colors.length<200;i++){
        const j=other.get(a.hip[i]);if(j==null)continue;
        const ca=__scene.skyPoints.geometry.attributes.aColor,cb=s.stars.geometry.attributes.aColor;
        colors.push(Math.abs(ca.getX(i)-cb.getX(j))+Math.abs(ca.getY(i)-cb.getY(j))+Math.abs(ca.getZ(i)-cb.getZ(j)));
      }
      return {disk,background,dayExposure:s.exposure,skyLevel:s.skyLevel,
        starOffMeanDelta:sum/(before.length/4),starOffChanged:changed,starOffLarge:large,
        sharedColorCount:colors.length,sharedColorMaxError:Math.max(...colors),
        diskColor:s.worldScene.children.find(c=>c.userData.bodyName==='Αστραία').material.uniforms.uColor.value.toArray()};
    }""")
    page.screenshot(path=str(root / "shots" / "astraia-photometric.png"))
    page.evaluate("window.__live.refresh=()=>Promise.resolve(null)")
    page.wait_for_timeout(1500)  # let any already in-flight request finish
    night = page.evaluate("""() => {
      const s=__surface,d=structuredClone(s.data),label='夜间颜色验证（不入档）';
      const original=s.setData.bind(s);
      s.setData=(incoming)=>{if(incoming.point?.name===label)original(incoming);};
      for(const b of d.bodies.filter(b=>b.kind==='star')){
        b.direction_enu=[0,0,-1];b.above_horizon=false;b.occluded_by=[];
      }
      d.point.name='夜间颜色验证（不入档）';s.setData(d);
      s.camera.fov=60;s.camera.updateProjectionMatrix();s.render();
      const gl=s.renderer.getContext(),a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
      gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);
      let warm=0,cool=0,bright=0;
      for(let i=0;i<a.length;i+=4){const r=a[i],g=a[i+1],b=a[i+2];
        if(r+g+b<100)continue;bright++;if(r>b+18)warm++;if(b>r+18)cool++;}
      return {exposure:s.exposure,skyLevel:s.skyLevel,warm,cool,bright,time:s.data.t_days,point:s.data.point.name};
    }""")
    page.screenshot(path=str(root / "shots" / "live-night-colors.png"))
    blue = page.evaluate("""() => {
      const s=__surface,d=s.data,o=d.observer,sky=d.sky,candidates=[];
      for(let i=0;i<sky.hip.length;i++){
        if(sky.bv[i]==null || sky.bv[i]>0.35)continue;
        const w=[sky.dir_x[i],sky.dir_y[i],sky.dir_z[i]];
        const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
        const v=[dot(w,o.east_world),dot(w,o.north_world),dot(w,o.zenith_world)];
        if(v[2]>0.65)candidates.push({i,v,mag:sky.mag[i],bv:sky.bv[i],hip:sky.hip[i]});
      }
      candidates.sort((a,b)=>a.mag-b.mag);
      const c=candidates[0];if(!c)return null;
      s.yaw=-Math.atan2(c.v[0],c.v[1]);s.pitch=-Math.acos(c.v[2]);
      s.camera.fov=12;s.camera.updateProjectionMatrix();s.render();
      const gl=s.renderer.getContext(),a=new Uint8Array(4);
      gl.readPixels((gl.drawingBufferWidth/2)|0,(gl.drawingBufferHeight/2)|0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,a);
      return {...c,pixel:[...a]};
    }""")
    if blue:
        page.screenshot(path=str(root / "shots" / "live-blue-star.png"))
    print(json.dumps({"day": result, "synthetic_night": night, "blue_star": blue, "errors": errors}, ensure_ascii=False, indent=2))
    assert result["disk"][0] > result["disk"][2] + 5, result
    assert result["background"][2] > result["background"][0] + 10, result
    assert result["sharedColorCount"] > 100 and result["sharedColorMaxError"] < 1e-6, result
    assert result["starOffMeanDelta"] < 0.1, result
    assert night["exposure"] > result["dayExposure"] * 10, (result, night)
    assert night["warm"] > 0, night
    assert blue and blue["pixel"][2] > blue["pixel"][0] + 3, blue
    assert night["time"] == 0 and night["point"].startswith("夜间颜色验证"), night
    assert not errors, errors
    browser.close()
