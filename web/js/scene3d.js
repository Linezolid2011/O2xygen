// Three.js scene: emissive stars by Teff, integrated-path smoke trails, hierarchy, labels.
import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";
import { NBodyPropagator, AU, DAY } from "./kepler.js";

const texLoaderCache = {};
const _followDelta = new THREE.Vector3();
function glowTexture() {
  if (texLoaderCache.glow) return texLoaderCache.glow;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,0.9)");
  grad.addColorStop(0.25, "rgba(255,244,230,0.35)");
  grad.addColorStop(1, "rgba(255,240,220,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  texLoaderCache.glow = new THREE.CanvasTexture(c);
  return texLoaderCache.glow;
}

// blackbody-ish colour by Teff
export function teffColor(teff) {
  const t = Math.min(Math.max((teff - 2500) / 5000, 0), 1);
  const r = 1.0;
  const g = Math.min(0.45 + 0.55 * t, 1);
  const b = Math.min(Math.max(0.15 + 0.85 * (t - 0.35) * 1.6, 0.15), 1);
  return new THREE.Color(r, g, b);
}

function styleRadius(body) {
  // ADR-0017: DRAWN TO SCALE. The rendered sphere IS the physical radius.
  // Planets shrink to dots at system zoom - that is the honest view.
  return Math.max(body.radius_m, 1e6);     // floor only against degenerate 0
}

export class OrbitScene {
  constructor(canvas, system, callbacks = {}) {
    this.system = system;
    this.cb = callbacks;
    this.snap = null;             // latest server N-body snapshot (ADR-0015)
    this.selected = null;
    this.hover = null;
    this.is2D = false;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0c11);
    this.scene.fog = null;

    this.camera = new THREE.PerspectiveCamera(50, 1, AU * 0.001, AU * 4000);
    this.camera.position.set(0, AU * 1.4, AU * 2.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // ambient + point light at primary star
    this.scene.add(new THREE.AmbientLight(0x334, 0.6));
    this.starLight = new THREE.PointLight(0xfff2e0, 4, 0, 1.4);
    this.scene.add(this.starLight);

    // NOTE: no procedural starfield by design — a real all-sky renderer
    // (positioned relative to the Solar System) will be integrated later.
    this.buildBodies();
    this.buildTrails();
    this.buildGrid();
    this.labelsEl = document.getElementById("body-labels");
    this.buildLabels();

    // picking
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.meshes = Object.values(this.pickProxies);   // raycast vs pick proxies
    this.bindPicking(canvas);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.resize();

    this.gizmoCanvas = document.getElementById("gizmo-canvas");
    this.gizmoCtx = this.gizmoCanvas.getContext("2d");
  }

  buildBodies() {
    this.bodyMeshes = {};
    this.bodyGroups = {};
    this.pickProxies = {};
    // ADR-0017: display radius = physical radius, no caps, no baselines.
    this.displayRadii = {};
    for (const b of this.system.bodies) {
      this.displayRadii[b.name] = styleRadius(b);
    }

    for (const b of this.system.bodies) {
      const group = new THREE.Group();
      const R = this.displayRadii[b.name];
      let mesh;
      if (b.kind === "star") {
        const color = teffColor(b.extra?.teff_k ?? 5772);
        mesh = new THREE.Mesh(
          new THREE.SphereGeometry(R, 40, 28),
          new THREE.MeshBasicMaterial({ color })
        );
        const glow = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTexture(), color, transparent: true, opacity: 0.55,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }));
        // glow is a light halo around the star, not a size claim (ADR-0017:
        // the sphere itself is the size claim): a small floor keeps stars
        // findable at system zoom; the 2.6x factor is deliberate and keeps
        // the halo visually anchored to the star's own scale
        glow.scale.setScalar(Math.max(R * 2.6, AU * 0.0006));
        group.add(glow);
      } else {
        const baseHue = 0.58;                 // flat world: one non-star kind
        const color = new THREE.Color().setHSL(baseHue, 0.35, 0.62);
        mesh = new THREE.Mesh(
          new THREE.SphereGeometry(R, 36, 24),
          new THREE.MeshStandardMaterial({
            color, roughness: 0.85, metalness: 0.05,
            emissive: color, emissiveIntensity: 0.22,   // never fully black
          })
        );
      }
      mesh.userData.bodyName = b.name;
      group.add(mesh);

      // invisible pick proxy: true-scale spheres are too small to click at
      // system zoom; the proxy is a USABILITY aid, not a size claim
      const proxy = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(R * 1.1, AU * 0.0015), 8, 6),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      proxy.userData.bodyName = b.name;
      group.add(proxy);
      this.pickProxies[b.name] = proxy;

      if (b.extra?.ring) {
        const outer = Math.min(R * 2.3, AU * 0.5);
        const inner = outer * 0.63;
        const ringGeo = new THREE.RingGeometry(inner, outer, 96);
        // fix UVs so the ring shades flat
        const posAttr = ringGeo.attributes.position;
        const uv = ringGeo.attributes.uv;
        for (let i = 0; i < posAttr.count; i++) {
          const d = Math.hypot(posAttr.getX(i), posAttr.getY(i));
          uv.setXY(i, (d - inner) / (outer - inner), 0.5);
        }
        const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({
          color: 0xc8b18a, side: THREE.DoubleSide, transparent: true, opacity: 0.35,
        }));
        ring.rotation.x = Math.PI / 2 - 0.35;
        group.add(ring);
      }
      this.scene.add(group);
      this.bodyMeshes[b.name] = mesh;
      this.bodyGroups[b.name] = group;
    }
  }

  buildTrails() {
    /** Smoke trails: server snapshots seed a dense per-frame history
     *  (trailHist); rendering resamples 91 uniform sim-times across the
     *  window with linear interpolation, so BOTH the head (body) and the
     *  tail glide continuously instead of hopping in stride-sized steps. */
    this.trailByName = {};
    this.trailHist = {};
    this.trailWindow = {};
    for (const b of this.system.bodies) {
      if (b.name === this.system.origin) continue;      // origin defines the frame
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(3), 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(3), 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.9,
      }));
      this.scene.add(line);
      this.trailByName[b.name] = { line, baseColor: this._trailColor(b) };
    }
  }

  _trailColor(b) {
    if (b.kind === "star") return new THREE.Color(0xd8b090);
    return new THREE.Color(0xaab3c8);
  }

  /** Latest server snapshot: live positions/velocities + trails + elements. */
  setSnapshot(snap) {
    this.snap = snap;
    this.propagator = new NBodyPropagator(snap, snap.dt_s);
    this._starBase = this._starBase ?? {};      // first snapshot = scale baseline
    this.updateStars(snap);
  }

  /** Stellar evolution made visible: swell + redden stars relative to the
   *  first snapshot's MIST radius/Teff (display-scale factor and colour). */
  updateStars(snap) {
    if (!snap.stars) return;
    // the point light is positioned at the PRIMARY star (see update()) - its
    // colour must track that same star, whatever the world origin happens to be
    const primary = this.system.bodies.find((b) => b.kind === "star");
    for (const b of this.system.bodies) {
      if (b.kind !== "star") continue;
      const st = snap.stars[b.name];
      if (!st) continue;
      const base = this._starBase[b.name] ?? (this._starBase[b.name] = st.radius_m);
      const g = this.bodyGroups[b.name];
      if (g) g.scale.setScalar(Math.max(st.radius_m / base, 1e-3));
      const c = teffColor(st.teff_k ?? 5772);
      const mesh = this.bodyMeshes[b.name];
      if (mesh?.material?.color) mesh.material.color.copy(c);
      const glow = g?.children.find((ch) => ch.isSprite);
      if (glow) glow.material.color.copy(c);
      if (primary && b.name === primary.name) this.starLight.color.copy(c);
    }
  }

  /** Fade-gradient line geometry from oldest->newest points. */
  _trailGeometry(points, baseColor) {
    const n = points.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const bg = new THREE.Color(0x0a0c11);              // scene background
    for (let i = 0; i < n; i++) {
      pos[i * 3] = points[i][0]; pos[i * 3 + 1] = points[i][1]; pos[i * 3 + 2] = points[i][2];
      const k = Math.pow(i / (n - 1), 1.6);            // newest = full colour
      const c = bg.clone().lerp(baseColor, k);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    return geo;
  }

  /** Anchor the smoke to the server's integrated paths (per snapshot):
   *  seeds the dense history with the exact server points at known times. */
  updateTrails(snap = this.snap) {
    if (!snap?.trails) return;
    for (const [name, t] of Object.entries(this.trailByName)) {
      const data = snap.trails[name];
      if (!data || data.points.length < 2) {
        t.line.visible = false;
        this.trailHist[name] = [];
        continue;
      }
      t.line.visible = true;
      const existing = this.trailHist[name];
      const anchorStride = data.window_days / (data.points.length - 1);
      if (existing && existing.length > 2 &&
          existing[existing.length - 1][0] >= snap.t_days - anchorStride) {
        // our client history already reaches into the anchor's newest
        // stride: reseeding would swap the window mid-stream and show as
        // a one-frame tail jump (measured: a reseed every poll -> a jump
        // every ~800 ms). Keep the continuous client history; only a
        // genuine client lag (> one stride behind) reseeds.
        this.trailWindow[name] = data.window_days;
        continue;
      }
      const n = data.points.length;
      const stride = data.window_days / (n - 1);
      const pts = data.points.slice().reverse();       // oldest -> newest
      const hist = [];
      for (let i = 0; i < n; i++) {
        hist.push([snap.t_days - (n - 1 - i) * stride,
                   pts[i][0], pts[i][1], pts[i][2]]);
      }
      this.trailHist[name] = hist;
      this.trailWindow[name] = data.window_days;
      t.line.geometry.dispose();
      t.line.geometry = this._trailGeometry(pts, t.baseColor);
      t.posArr = null;                   // invalidate: sampleTrails owns buffers
    }
  }

  _sampleHist(hist, tt) {
    const first = hist[0], last = hist[hist.length - 1];
    if (tt <= first[0]) return [first[1], first[2], first[3]];
    if (tt >= last[0]) return [last[1], last[2], last[3]];
    for (let i = hist.length - 2; i >= 0; i--) {       // tt is usually recent
      if (hist[i][0] <= tt) {
        const a = hist[i], b = hist[i + 1];
        const f = (tt - a[0]) / Math.max(1e-12, b[0] - a[0]);
        return [a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
                a[3] + (b[3] - a[3]) * f];
      }
    }
    return [first[1], first[2], first[3]];
  }

  /** Per-frame smoke update: append the current render-clock sample to the
   *  history (while evolving), then rebuild the visible polyline by
   *  CONTINUOUS resampling - the tail slides out smoothly instead of
   *  hopping one stride per buffer shift. */
  sampleTrails(renderDays, active) {
    if (!this.propagator || !this.system.origin) return;
    const o = this.propagator.pos(this.system.origin);
    for (const [name, t] of Object.entries(this.trailByName)) {
      const hist = this.trailHist?.[name];
      const windowDays = this.trailWindow?.[name];
      if (!hist || !windowDays) continue;
      if (active) {
        // append at ~stride density (window/90) - the continuous TIME-
        // uniform resample below supplies the smoothness, so the history
        // only needs to be dense enough to interpolate (a per-frame append
        // made _sampleHist O(N) x 91 per body per frame = slideshow)
        const stride = windowDays / 90;
        const last = hist[hist.length - 1];
        if (!last || renderDays >= last[0] + stride) {
          const p = this.propagator.pos(name);
          hist.push([renderDays, p.x - o.x, p.y - o.y, p.z - o.z]);
          const cutoff = renderDays - windowDays - stride;
          while (hist.length > 2 && hist[0][0] < cutoff) hist.shift();
          if (hist.length > 400) hist.splice(0, hist.length - 400);
        }
      }
      if (!active) continue;               // paused/idle: keep last geometry
      if (hist.length < 2) continue;
      // reuse buffers: colours are index-static (fade), only positions move
      const n = 91;
      if (!t.posArr || t.posArr.length !== n * 3) {
        const geo = this._trailGeometry(new Array(n).fill([0, 0, 0]), t.baseColor);
        t.line.geometry.dispose();
        t.line.geometry = geo;
        t.posArr = geo.attributes.position.array;
      }
      for (let j = 0; j < n; j++) {
        const tt = renderDays - windowDays * (1 - j / (n - 1));
        const pt = this._sampleHist(hist, tt);
        t.posArr[j * 3] = pt[0]; t.posArr[j * 3 + 1] = pt[1]; t.posArr[j * 3 + 2] = pt[2];
      }
      // pin the head to the body: history appends are stride-gated, so the
      // freshest stored point lags; the body itself is drawn from the
      // propagator and the smoke must attach to it exactly
      const hp = this.propagator.pos(name);
      t.posArr[(n - 1) * 3] = hp.x - o.x;
      t.posArr[(n - 1) * 3 + 1] = hp.y - o.y;
      t.posArr[(n - 1) * 3 + 2] = hp.z - o.z;
      t.line.geometry.attributes.position.needsUpdate = true;
    }
  }

  /** Position of a body at render time: client-side N-body propagation from
   *  the latest snapshot (true curved motion between polls, ADR-0015). */
  _posAt(name, tS) {
    if (this.propagator && this.snap?.bodies?.[name]) {
      if (tS < this.propagator.t) {
        this.propagator = new NBodyPropagator(this.snap, this.snap.dt_s);  // rewind: re-seed
      }
      this.propagator.propagateTo(tS);
      return this.propagator.pos(name);
    }
    return { x: 0, y: 0, z: 0 };   // pre-first-snapshot: bodies sit at the origin
  }

  buildGrid() {
    this.grid = new THREE.GridHelper(AU * 30, 30, 0x2a3040, 0x1a1e2a);
    this.grid.rotation.x = Math.PI / 2;      // ecliptic plane = XY
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.25;
    this.grid.visible = false;
    this.scene.add(this.grid);
  }

  buildLabels() {
    this.labelDivs = {};
    for (const b of this.system.bodies) {
      const div = document.createElement("div");
      div.className = "body-label";
      div.textContent = b.name;
      this.labelsEl.appendChild(div);
      this.labelDivs[b.name] = div;
    }
  }

  bindPicking(canvas) {
    let downAt = null;
    canvas.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
    canvas.addEventListener("pointerup", (e) => {
      if (!downAt) return;
      const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
      downAt = null;
      if (moved > 5) return;
      const hit = this.pick(e);
      if (hit) this.cb.onSelect?.(hit);
    });
    canvas.addEventListener("pointermove", (e) => {
      const hit = this.pick(e);
      this.hover = hit;
      canvas.style.cursor = hit ? "pointer" : "default";
    });
  }

  pick(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1,
                     -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.meshes, false);
    return hits.length ? hits[0].object.userData.bodyName : null;
  }

  select(name) {
    this.selected = name;
    this._followLast = null;               // follow starts from the new body
    for (const [n, div] of Object.entries(this.labelDivs)) {
      div.classList.toggle("sel", n === name);
      // names are user-supplied and persisted: textContent, never innerHTML
      div.textContent = n;
      if (n === name) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = "已选择";
        div.appendChild(tag);
      }
    }
    this.focusOn(name);           // selection recenters the view
  }

  focusSelected() {
    if (this.selected) this.focusOn(this.selected);
  }

  /** Smoothly fly the camera so `name` sits at the viewport center. */
  focusOn(name) {
    const g = this.bodyGroups[name];
    if (!g) return;
    const target = g.position.clone();
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    let dist = this.camera.position.distanceTo(this.controls.target);
    dist = Math.min(Math.max(dist, AU * 0.0002), AU * 400);  // true-scale: allow close fly-ins
    const toPos = target.clone().add(dir.multiplyScalar(dist));
    this._flyTo(target, toPos, 600);
  }

  _flyTo(target, pos, ms = 600) {
    const fromT = this.controls.target.clone();
    const fromP = this.camera.position.clone();
    const t0 = performance.now();
    if (this._flyRaf) cancelAnimationFrame(this._flyRaf);
    const step = (now) => {
      const k = Math.min((now - t0) / ms, 1);
      const e = 1 - Math.pow(1 - k, 3);          // easeOutCubic
      this.controls.target.lerpVectors(fromT, target, e);
      this.camera.position.lerpVectors(fromP, pos, e);
      this._flyRaf = k < 1 ? requestAnimationFrame(step) : null;
    };
    this._flyRaf = requestAnimationFrame(step);
  }

  set2D(on) {
    this.is2D = on;
    this.controls.enableRotate = !on;
    if (on) {
      this._savedPos = this.camera.position.clone();
      this._savedTarget = this.controls.target.clone();
      this.camera.position.set(this.controls.target.x, this.controls.target.y, AU * 240);
      this.camera.up.set(0, 1, 0);
    } else if (this._savedPos) {
      this.camera.position.copy(this._savedPos);
      this.controls.target.copy(this._savedTarget);
      this.camera.up.set(0, 1, 0);
    }
    this.controls.update();
  }

  toggleGrid() { this.grid.visible = !this.grid.visible; }

  update(tDays) {
    const tS = tDays * DAY;
    // adaptive near plane: true-scale bodies are tiny, the camera must be
    // able to fly in close (fixed AU*0.001 near would clip a 4 R_E planet)
    const dist = this.camera.position.distanceTo(this.controls.target);
    const near = Math.min(Math.max(dist * 0.02, 1e4), AU * 0.001);
    if (Math.abs(near - this.camera.near) / this.camera.near > 0.2) {
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
    // ADR-0016: the render frame is ORIGIN-RELATIVE - everything is drawn at
    // (absolute position - origin body's position).
    const originName = this.system.origin;
    const op = originName ? this._posAt(originName, tS) : { x: 0, y: 0, z: 0 };
    for (const b of this.system.bodies) {
      const g = this.bodyGroups[b.name];
      if (!g) continue;
      const p = this._posAt(b.name, tS);
      g.position.set(p.x - op.x, p.y - op.y, p.z - op.z);
      // slow spin for non-stars
      if (b.kind !== "star" && b.rotation_period_s) {
        const m = this.bodyMeshes[b.name];
        m.rotation.y = (tS / b.rotation_period_s) * 2 * Math.PI;
      }
    }
    // star light follows the primary star (rendered position)
    const primary = this.system.bodies.find((b) => b.kind === "star");
    if (primary) {
      const p = this._posAt(primary.name, tS);
      this.starLight.position.set(p.x - op.x, p.y - op.y, p.z - op.z);
    }
  }

  drawLabels() {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    for (const [name, div] of Object.entries(this.labelDivs)) {
      const g = this.bodyGroups[name];
      if (!g) continue;
      v.copy(g.position).project(this.camera);
      if (v.z > 1) { div.style.display = "none"; continue; }
      div.style.display = "block";
      div.style.left = `${(v.x * 0.5 + 0.5) * rect.width}px`;
      div.style.top = `${(-v.y * 0.5 + 0.5) * rect.height}px`;
    }
  }

  drawGizmo() {
    const ctx = this.gizmoCtx;
    if (!ctx) return;                       // headless/diagnostic contexts may be lost
    try {
      ctx.clearRect(0, 0, 72, 72);
      const axes = [
        { d: new THREE.Vector3(1, 0, 0), c: "#e87a7a", t: "X" },
        { d: new THREE.Vector3(0, 1, 0), c: "#8fd0a0", t: "Y" },
        { d: new THREE.Vector3(0, 0, 1), c: "#7aa2e8", t: "Z" },
      ];
      for (const a of axes) {
        const p = a.d.clone().applyQuaternion(this.camera.quaternion);
        ctx.strokeStyle = a.c; ctx.fillStyle = a.c; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(36, 36);
        ctx.lineTo(36 + p.x * 26, 36 - p.y * 26);
        ctx.stroke();
        ctx.font = "10px sans-serif";
        ctx.fillText(a.t, 36 + p.x * 30 - 3, 36 - p.y * 30 + 3);
      }
    } catch { /* lost 2d context: ignore */ }
  }

  resize() {
    const el = this.renderer.domElement.parentElement;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Keep the camera glued to the selected body's motion: apply its
   *  frame-to-frame delta to both camera and target. Additive only - the
   *  user's own pan/zoom is untouched. Without this, selecting a body
   *  during evolution watches it drift out of frame. */
  _follow() {
    const g = this.selected ? this.bodyGroups[this.selected] : null;
    if (!g) { this._followLast = null; return; }
    const p = g.position;
    if (this._followLast) {
      _followDelta.copy(p).sub(this._followLast);
      if (_followDelta.lengthSq() > 0) {
        this.camera.position.add(_followDelta);
        this.controls.target.add(_followDelta);
      }
    }
    (this._followLast ??= new THREE.Vector3()).copy(p);
  }

  render() {
    this._follow();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.drawLabels();
    this.drawGizmo();
  }
}
