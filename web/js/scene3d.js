// Three.js scene: emissive stars by Teff, received-pose smoke trails, labels.
// ADR-0021/0034: the client computes NO physics. Trails are only a finite
// client-side history of authoritative positions; the server integrates none.
import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";
import { AU, DAY } from "./kepler.js";

const TRAIL_CAPACITY = 240;          // ~4 s at 60 Hz, fixed memory per body
const texLoaderCache = {};
const _followDelta = new THREE.Vector3();
const _Y_AXIS = new THREE.Vector3(0, 1, 0);
const _Z_AXIS = new THREE.Vector3(0, 0, 1);
const _spinAxis = new THREE.Vector3();
const _qAlign = new THREE.Quaternion();
const _qSpin = new THREE.Quaternion();
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
    this.snap = null;             // latest server meta snapshot (poll)
    this.pose = null;             // latest server pose (SSE or poll fallback)
    this.trailCount = 0;          // shared ring occupancy for every body
    this.trailHead = 0;           // next ring slot
    this.trailLastTime = null;    // pause/duplicate guard
    this.selected = null;
    this.hover = null;
    this.is2D = false;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0c11, 1);
    this.renderer.autoClear = false;             // sky pass, then world pass
    this.scene = new THREE.Scene();
    // the WORLD pass must not clear the sky pass: a scene.background Color
    // forces a color clear even with autoClear=false. Only the sky scene owns
    // the background color; world geometry then draws over the stars.
    this.scene.background = null;
    this.scene.fog = null;
    // Camera-independent celestial sphere: same orientation as the main
    // camera, but always centered on the observer - never on the fly-to
    // camera's world-space position (no fake parallax while panning).
    this.skyScene = new THREE.Scene();
    this.skyScene.background = new THREE.Color(0x0a0c11);
    this.skyCamera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    this.skyPoints = null;
    this.skyVisible = true;

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
    this.ringMeshes = {};
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
        // RingGeometry normal is +Z; update() aligns it to the live spin
        // axis every frame (equatorial rings, ADR-0023)
        group.add(ring);
        this.ringMeshes[b.name] = ring;
      }
      this.scene.add(group);
      this.bodyMeshes[b.name] = mesh;
      this.bodyGroups[b.name] = group;
    }
  }

  buildTrails() {
    /** ADR-0034: each line is ONLY the latest 240 received pose samples.
     *  Ring storage and GPU buffers are allocated once.  There is no initial
     *  backward integration, no server trail payload and no persistence. */
    this.trailByName = {};
    for (const b of this.system.bodies) {
      const positions = new Float32Array(TRAIL_CAPACITY * 3);
      const colors = new Float32Array(TRAIL_CAPACITY * 3);
      const posAttr = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
      const colAttr = new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", posAttr);
      geo.setAttribute("color", colAttr);
      geo.setDrawRange(0, 0);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.9,
      }));
      line.visible = false;
      line.frustumCulled = false;      // ring positions change every pose
      this.scene.add(line);
      this.trailByName[b.name] = {
        line, baseColor: this._trailColor(b), background: new THREE.Color(0x0a0c11),
        history: new Float32Array(TRAIL_CAPACITY * 3), positions, colors,
      };
    }
  }

  _trailColor(b) {
    if (b.kind === "star") return new THREE.Color(0xd8b090);
    return new THREE.Color(0xaab3c8);
  }

  /** Replace the real Hipparcos starfield. Server supplies observer-centric
   *  directions already rotated into world axes, apparent magnitudes and B-V. */
  setSky(data) {
    const n = data.hip.length;
    const pos = new Float32Array(n * 3);
    const mag = new Float32Array(n);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = data.dir_x[i] * 100;
      pos[i * 3 + 1] = data.dir_y[i] * 100;
      pos[i * 3 + 2] = data.dir_z[i] * 100;
      mag[i] = data.mag[i];
      let r = 1.0, g = 0.96, b = 0.90;          // default / Sun-like
      const bv = data.bv[i];
      if (i !== data.sun_index && bv != null) {
        const t = Math.max(0, Math.min(1, (bv + 0.25) / 2.0));
        r = 0.62 + 0.38 * t;                   // blue-white -> warm
        g = 0.72 + 0.22 * t;
        b = 1.0 - 0.42 * t;
      }
      col[i * 3] = r; col[i * 3 + 1] = g; col[i * 3 + 2] = b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aMag", new THREE.BufferAttribute(mag, 1));
    geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uPixelRatio: { value: this.renderer.getPixelRatio() } },
      vertexShader: `
        attribute float aMag;
        attribute vec3 aColor;
        varying float vMag;
        varying vec3 vColor;
        uniform float uPixelRatio;
        void main() {
          vMag = aMag; vColor = aColor;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          float bright = clamp(7.0 - aMag, 0.0, 10.0);
          gl_PointSize = (1.1 + 0.48 * bright) * uPixelRatio;
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: `
        varying float vMag;
        varying vec3 vColor;
        void main() {
          float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
          if (r > 1.0) discard;
          float alpha = smoothstep(1.0, 0.12, r);
          float strength = clamp(1.25 - 0.10 * vMag, 0.22, 1.35);
          gl_FragColor = vec4(vColor * strength, alpha);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    if (this.skyPoints) {
      this.skyPoints.geometry.dispose();
      this.skyPoints.material.dispose();
      this.skyScene.remove(this.skyPoints);
    }
    this.skyPoints = new THREE.Points(geo, mat);
    this.skyPoints.visible = this.skyVisible;
    this.skyScene.add(this.skyPoints);
    this.skyCount = n;
    this.skyData = data;
  }

  toggleStars() {
    this.skyVisible = !this.skyVisible;
    if (this.skyPoints) this.skyPoints.visible = this.skyVisible;
  }

  /** Latest server meta snapshot (poll): star display props + fallback pose. */
  setSnapshot(snap, allowPose = false) {
    this.snap = snap;
    if (!this.pose || allowPose) {
      const p = {};
      for (const [n, b] of Object.entries(snap.bodies ?? {})) p[n] = b.p;
      this.setPose(p, snap.t_days, this._bcmOf(p));
    }
    this._starBase = this._starBase ?? {};      // first snapshot = scale baseline
    this.updateStars(snap);
  }

  /** Barycenter of a {name: [x,y,z]} position map, using the masses from
   *  the latest meta snapshot (degraded path only; SSE poses carry bcm). */
  _bcmOf(p) {
    let mx = 0, my = 0, mz = 0, mt = 0;
    for (const [n, q] of Object.entries(p)) {
      const m = this.snap?.bodies?.[n]?.m ?? 0;
      mx += m * q[0]; my += m * q[1]; mz += m * q[2]; mt += m;
    }
    return mt > 0 ? [mx / mt, my / mt, mz / mt] : [0, 0, 0];
  }

  /** Append one authoritative pose to the fixed-size client history. */
  setPose(p, tDays, bcm) {
    if (this.pose && tDays < this.pose.t_days - 1e-9) {
      this.clearTrails();            // rollback: discard the cancelled future
    }
    const origin = bcm ?? this._bcmOf(p);
    this.pose = { p, t_days: tDays, bcm: origin };
    if (this.trailLastTime !== null && tDays === this.trailLastTime) return;

    const slot = this.trailHead;
    for (const [name, t] of Object.entries(this.trailByName)) {
      const q = p[name];
      if (!q) continue;
      const j = slot * 3;
      t.history[j] = q[0] - origin[0];
      t.history[j + 1] = q[1] - origin[1];
      t.history[j + 2] = q[2] - origin[2];
    }
    this.trailHead = (slot + 1) % TRAIL_CAPACITY;
    this.trailCount = Math.min(this.trailCount + 1, TRAIL_CAPACITY);
    this.trailLastTime = tDays;
    this._refreshTrails();
  }

  clearTrails() {
    this.trailCount = 0;
    this.trailHead = 0;
    this.trailLastTime = null;
    for (const t of Object.values(this.trailByName)) {
      t.line.visible = false;
      t.line.geometry.setDrawRange(0, 0);
    }
  }

  /** Copy ring order into preallocated GPU arrays only when a pose arrives. */
  _refreshTrails() {
    const n = this.trailCount;
    if (n < 2) return;
    const start = (this.trailHead - n + TRAIL_CAPACITY) % TRAIL_CAPACITY;
    for (const t of Object.values(this.trailByName)) {
      for (let i = 0; i < n; i++) {
        const src = ((start + i) % TRAIL_CAPACITY) * 3;
        const dst = i * 3;
        t.positions[dst] = t.history[src];
        t.positions[dst + 1] = t.history[src + 1];
        t.positions[dst + 2] = t.history[src + 2];
        const k = Math.pow(i / (n - 1), 1.6);  // oldest faint, newest full
        t.colors[dst] = t.background.r + (t.baseColor.r - t.background.r) * k;
        t.colors[dst + 1] = t.background.g + (t.baseColor.g - t.background.g) * k;
        t.colors[dst + 2] = t.background.b + (t.baseColor.b - t.background.b) * k;
      }
      t.line.geometry.setDrawRange(0, n);
      t.line.geometry.attributes.position.needsUpdate = true;
      t.line.geometry.attributes.color.needsUpdate = true;
      t.line.visible = true;
    }
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
      if (!st || !(st.radius_m > 0) || !Number.isFinite(st.radius_m)) continue;
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

  /** Absolute position of a body from the latest server pose (SSE), falling
   *  back to the latest poll snapshot before the first pose arrives. */
  _posOf(name) {
    const p = this.pose?.p?.[name] ?? this.snap?.bodies?.[name]?.p;
    return p ?? [0, 0, 0];
  }

  update(tDays) {
    // adaptive near plane: true-scale bodies are tiny, the camera must be
    // able to fly in close (fixed AU*0.001 near would clip a 4 R_E planet)
    const dist = this.camera.position.distanceTo(this.controls.target);
    const near = Math.min(Math.max(dist * 0.02, 1e4), AU * 0.001);
    if (Math.abs(near - this.camera.near) / this.camera.near > 0.2) {
      this.camera.near = near;
      this.camera.updateProjectionMatrix();
    }
    // ADR-0022: the render frame is BARYCENTRIC - everything is drawn at
    // (absolute position - system barycenter), all from server poses.
    const op = this.pose?.bcm ?? [0, 0, 0];
    for (const b of this.system.bodies) {
      const g = this.bodyGroups[b.name];
      if (!g) continue;
      const p = this._posOf(b.name);
      g.position.set(p[0] - op[0], p[1] - op[1], p[2] - op[2]);
      // server spin state: axis + rate + phase at the snapshot's clock;
      // only the display phase advances between meta polls (ADR-0023)
      const sp = this.snap?.spins?.[b.name];
      const axis = sp?.axis ?? b.spin_axis ?? [0, 0, 1];
      const rate = sp?.rate_rad_s ?? b.spin_rate_rad_s ??
        (b.rotation_period_s ? 2 * Math.PI / b.rotation_period_s : 0);
      const phase0 = sp?.phase_rad ?? b.spin_phase_rad ?? 0;
      const phase = phase0 + rate * (tDays - (this.snap?.t_days ?? tDays)) * DAY;
      _spinAxis.set(axis[0], axis[1], axis[2]);
      if (_spinAxis.lengthSq() < 1e-30) _spinAxis.copy(_Y_AXIS);
      else _spinAxis.normalize();
      _qAlign.setFromUnitVectors(_Y_AXIS, _spinAxis);
      _qSpin.setFromAxisAngle(_Y_AXIS, phase);
      const m = this.bodyMeshes[b.name];
      m.quaternion.copy(_qAlign).multiply(_qSpin);
      const ring = this.ringMeshes[b.name];
      if (ring) ring.quaternion.setFromUnitVectors(_Z_AXIS, _spinAxis);
    }
    // star light follows the primary star (rendered position)
    const primary = this.system.bodies.find((b) => b.kind === "star");
    if (primary) {
      const p = this._posOf(primary.name);
      this.starLight.position.set(p[0] - op[0], p[1] - op[1], p[2] - op[2]);
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
    // sky pass: observer-centered celestial sphere, then the AU-scale world
    this.renderer.clear();
    if (this.skyVisible && this.skyPoints) {
      this.skyCamera.position.set(0, 0, 0);
      this.skyCamera.quaternion.copy(this.camera.quaternion);
      this.skyCamera.fov = this.camera.fov;
      this.skyCamera.aspect = this.camera.aspect;
      this.skyCamera.updateProjectionMatrix();
      this.renderer.render(this.skyScene, this.skyCamera);
      this.renderer.clearDepth();
    }
    this.renderer.render(this.scene, this.camera);
    this.drawLabels();
    this.drawGizmo();
  }
}
