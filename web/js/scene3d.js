// Three.js scene: emissive stars by Teff, received-pose smoke trails, labels.
// ADR-0021/0034: the client computes NO physics. Trails are only a finite
// client-side history of authoritative positions; the server integrates none.
import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";
import { AU, DAY } from "./kepler.js";
import { blackbodyInfo, bvColor, bodySrgb, STAR_FIELD_VERTEX,
  STAR_FIELD_FRAGMENT, STAR_RADIANCE_UNIT } from "./color.js";

const TRAIL_CAPACITY = 240;          // ~4 s at 60 Hz, fixed memory per body
const texLoaderCache = {};
const _followDelta = new THREE.Vector3();
const _Y_AXIS = new THREE.Vector3(0, 1, 0);
const _Z_AXIS = new THREE.Vector3(0, 0, 1);
const _spinAxis = new THREE.Vector3();
const _qAlign = new THREE.Quaternion();
const _qSpin = new THREE.Quaternion();
const _vRef = new THREE.Vector3();      // surface-overlay alignment scratch
const _qPre = new THREE.Quaternion();
const _qInv = new THREE.Quaternion();
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

// Soft round sprite for the screen-space observation-point markers.
function circleTexture() {
  if (texLoaderCache.dot) return texLoaderCache.dot;
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(16, 16, 0, 16, 16, 16);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.62, "rgba(255,255,255,1)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 32, 32);
  texLoaderCache.dot = new THREE.CanvasTexture(c);
  return texLoaderCache.dot;
}

// Shared physical blackbody-to-display conversion (see color.js).
export function teffColor(teff) {
  return new THREE.Color(...blackbodyInfo(teff).rgb);
}

export function bodyDisplayColor(body) {
  return new THREE.Color().setRGB(...bodySrgb(body), THREE.SRGBColorSpace);
}

// logarithmicDepthBuffer is ON (AU-scale depth range): custom shaders MUST
// include the logdepthbuf chunks, otherwise their gl_Depth values live on
// the standard-projection scale while every built-in material writes the
// logarithmic scale. The mismatch made distant star spheres and ring far
// sides draw straight through a planet's night side (0.4.1 regression:
// dark side looked semi-transparent / fully transparent at some angles).
const BODY_VERTEX = `
#include <common>
#include <logdepthbuf_pars_vertex>
varying vec3 vWorldNormal;
void main() {
  vWorldNormal = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  #include <logdepthbuf_vertex>
}`;

function bodyFragment(starCount) {
  return `
#include <logdepthbuf_pars_fragment>
uniform vec3 uBodyColor;
uniform vec3 uStarDirection[${Math.max(1, starCount)}];
uniform vec3 uStarColor[${Math.max(1, starCount)}];
uniform float uStarWeight[${Math.max(1, starCount)}];
uniform float uAmbient;
uniform float uAlbedo;
varying vec3 vWorldNormal;
void main() {
  #include <logdepthbuf_fragment>
  vec3 n = normalize(vWorldNormal);
  vec3 light = vec3(0.0);
  for (int i=0; i<${starCount}; i++) {
    light += uStarColor[i] * uStarWeight[i] * max(dot(n, uStarDirection[i]), 0.0);
  }
  gl_FragColor = vec4(uBodyColor * uAlbedo * (uAmbient + light), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;
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
    this.pickEnabled = true;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false,
      logarithmicDepthBuffer: true });
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
    // branch 2: surface overlay (real body-fixed lat/lon graticule +
    // observation-point dots) replaces the retired ecliptic reference grid.
    this.surfaceOverlayVisible = false;
    this.labelsVisible = true;
    this.overlayGroups = {};      // body name -> overlay group (child of the body MESH)
    this.overlayAxisCache = {};   // body name -> spin axis the overlay was aligned to
    this.pointDots = {};          // point name -> dot mesh
    this.pointLabelDivs = {};     // point name -> label div
    this.buildSurfaceOverlay();
    this.labelsEl = document.getElementById("body-labels");
    this.pointLabelsEl = document.getElementById("point-labels");
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
        const stars = this.system.bodies.filter((s) => s.kind === "star");
        const count = Math.max(1, stars.length);
        mesh = new THREE.Mesh(
          new THREE.SphereGeometry(R, 64, 40),
          new THREE.ShaderMaterial({
            uniforms: {
              uBodyColor: { value: bodyDisplayColor(b) },
              uStarDirection: { value: Array.from({ length: count }, () => new THREE.Vector3()) },
              uStarColor: { value: Array.from({ length: count }, () => new THREE.Color(1, 1, 1)) },
              uStarWeight: { value: new Float32Array(count) },
              uAmbient: { value: 0.015 },
              uAlbedo: { value: b.albedo },
            },
            vertexShader: BODY_VERTEX,
            fragmentShader: bodyFragment(stars.length),
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
        history: new Float64Array(TRAIL_CAPACITY * 3), positions, colors,
      };
    }
  }

  _trailColor(b) {
    return bodyDisplayColor(b);
  }

  /** Apply edited/persisted appearance to the sphere and its trail together. */
  refreshBodyAppearance(name) {
    const b = this.system.bodies.find((body) => body.name === name);
    if (!b) return;
    const mesh = this.bodyMeshes[name];
    const color = bodyDisplayColor(b);
    if (mesh?.material?.color) mesh.material.color.copy(color);
    mesh?.material?.uniforms?.uBodyColor?.value.copy(color);
    if (mesh?.material?.uniforms?.uAlbedo) mesh.material.uniforms.uAlbedo.value = b.albedo;
    const glow = this.bodyGroups[name]?.children.find((ch) => ch.isSprite);
    if (glow) glow.material.color.copy(color);
    this.trailByName[name]?.baseColor.copy(color);
    if (this.trailCount >= 2) this._refreshTrails();
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
      const rgb = bvColor(data.bv[i]);
      col[i * 3] = rgb[0]; col[i * 3 + 1] = rgb[1]; col[i * 3 + 2] = rgb[2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aMag", new THREE.BufferAttribute(mag, 1));
    geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uPixelRatio: { value: this.renderer.getPixelRatio() },
        uRadianceUnit: { value: STAR_RADIANCE_UNIT },
        uExposure: { value: 16000 },
        uRaw: { value: false },
      },
      vertexShader: STAR_FIELD_VERTEX,
      fragmentShader: STAR_FIELD_FRAGMENT,
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
      this.setPose(p, snap.t_days);
    }
    this._starBase = this._starBase ?? {};      // first snapshot = scale baseline
    this.updateStars(snap);
  }

  /** Append one ABSOLUTE-origin pose to the fixed-size client history. */
  setPose(p, tDays) {
    if (this.pose && tDays < this.pose.t_days - 1e-9) {
      this.clearTrails();            // rollback: discard the cancelled future
    }
    this.pose = { p, t_days: tDays };
    if (this.trailLastTime !== null && tDays === this.trailLastTime) return;

    const slot = this.trailHead;
    for (const [name, t] of Object.entries(this.trailByName)) {
      const q = p[name];
      if (!q) continue;
      const j = slot * 3;
      t.history[j] = q[0];
      t.history[j + 1] = q[1];
      t.history[j + 2] = q[2];
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

  /** Copy ring order into preallocated LOCAL-position GPU arrays.
   *  ADR-0035 keeps absolute world positions in history; the newest point is
   *  only a numerical render anchor.  At hundreds of AU, feeding absolute
   *  coordinates to Float32 loses ~0.004 AU of subpixel motion and makes the
   *  line visibly staircase.  Subtracting the newest point before upload keeps
   *  all small relative motion while preserving the exact same world geometry. */
  _refreshTrails() {
    const n = this.trailCount;
    if (n < 2) return;
    const start = (this.trailHead - n + TRAIL_CAPACITY) % TRAIL_CAPACITY;
    const newest = ((this.trailHead - 1 + TRAIL_CAPACITY) % TRAIL_CAPACITY) * 3;
    for (const t of Object.values(this.trailByName)) {
      t.line.position.set(t.history[newest], t.history[newest + 1],
                          t.history[newest + 2]);
      for (let i = 0; i < n; i++) {
        const src = ((start + i) % TRAIL_CAPACITY) * 3;
        const dst = i * 3;
        t.positions[dst] = t.history[src] - t.history[newest];
        t.positions[dst + 1] = t.history[src + 1] - t.history[newest + 1];
        t.positions[dst + 2] = t.history[src + 2] - t.history[newest + 2];
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
    let appearanceChanged = false;
    for (const b of this.system.bodies) {
      if (b.kind !== "star") continue;
      const st = snap.stars[b.name];
      if (!st || !(st.radius_m > 0) || !Number.isFinite(st.radius_m)) continue;
      const base = this._starBase[b.name] ?? (this._starBase[b.name] = st.radius_m);
      const g = this.bodyGroups[b.name];
      if (g) g.scale.setScalar(Math.max(st.radius_m / base, 1e-3));
      const c = teffColor(st.teff_k ?? 5772);
      const previousTeff = b.extra?.teff_k;
      b.extra = { ...(b.extra ?? {}), teff_k: st.teff_k };
      if (previousTeff !== st.teff_k) {
        appearanceChanged = true;
        this.cb.onStarProps?.();
      }
      const mesh = this.bodyMeshes[b.name];
      if (mesh?.material?.color) mesh.material.color.copy(c);
      const glow = g?.children.find((ch) => ch.isSprite);
      if (glow) glow.material.color.copy(c);
      this.trailByName[b.name]?.baseColor.copy(c);
      if (primary && b.name === primary.name) this.starLight.color.copy(c);
    }
    if (appearanceChanged && this.trailCount >= 2) this._refreshTrails();
  }

  /** branch 2: one overlay group per non-stellar body, attached as a child of
   *  the body MESH so it inherits qAlign*qSpin — the graticule and dots are
   *  material surface features and rotate with the body (user ruling: the
   *  grid must be real and spin-locked). The retired ecliptic GridHelper is
   *  gone entirely. */
  buildSurfaceOverlay() {
    for (const b of this.system.bodies) {
      if (b.kind === "star") continue;
      const R = this.displayRadii[b.name];
      const g = new THREE.Group();
      g.visible = false;
      g.add(new THREE.LineSegments(
        this._graticuleGeometry(R * 1.002),
        new THREE.LineBasicMaterial({
          color: 0x93a4bd, transparent: true, opacity: 0.32, depthWrite: false,
        })));
      this.bodyMeshes[b.name].add(g);
      this.overlayGroups[b.name] = g;
    }
    // screen-space observation markers: one shared material, constant
    // apparent size at ANY zoom (user ruling, branch 2). size is in device
    // pixels, so scale it by the capped pixelRatio.
    this.dotMaterial = new THREE.PointsMaterial({
      color: 0xff5a5a,
      size: 7 * this.renderer.getPixelRatio(),
      sizeAttenuation: false,
      map: circleTexture(),
      transparent: true,
      alphaTest: 0.35,
      depthWrite: false,
    });
  }

  /** Meridians every 30°, parallels every 30° (poles excluded). Overlay-local
   *  frame: pole = +Y, longitude 0°E = +Z, 90°E = +X — a right-handed frame
   *  matching observation.py local_frame (east = axis × zenith). */
  _graticuleGeometry(R) {
    const pts = [];
    const push = (latDeg, lonDeg) => {
      const lat = latDeg * Math.PI / 180, lon = lonDeg * Math.PI / 180;
      const cl = Math.cos(lat);
      pts.push(R * cl * Math.sin(lon), R * Math.sin(lat), R * cl * Math.cos(lon));
    };
    const SEG = 96;
    for (let lon = 0; lon < 360; lon += 30) {
      for (let i = 0; i < SEG; i++) {
        push(-90 + 180 * i / SEG, lon);
        push(-90 + 180 * (i + 1) / SEG, lon);
      }
    }
    for (let lat = -60; lat <= 60; lat += 30) {
      for (let i = 0; i < SEG; i++) {
        push(lat, 360 * i / SEG);
        push(lat, 360 * (i + 1) / SEG);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return geo;
  }

  /** (Re)build observation-point dots + labels from [{name, body,
   *  latitude_deg, longitude_deg}] (GET /api/observations, schema v3). */
  setObservations(points) {
    for (const g of Object.values(this.overlayGroups)) {
      for (let i = g.children.length - 1; i >= 1; i--) g.remove(g.children[i]);
    }
    if (this.pointLabelsEl) this.pointLabelsEl.textContent = "";   // drop old divs
    this.pointDots = {};
    this.pointLabelDivs = {};
    const byBody = {};
    for (const p of points ?? []) (byBody[p.body] ??= []).push(p);
    for (const [body, list] of Object.entries(byBody)) {
      const g = this.overlayGroups[body];
      if (!g) continue;                 // points exist only on non-stellar bodies
      const R = this.displayRadii[body] * 1.002;   // just above the surface
      const local = [];
      for (const p of list) {
        const lat = p.latitude_deg * Math.PI / 180, lon = p.longitude_deg * Math.PI / 180;
        const cl = Math.cos(lat);
        // same overlay-local parametrization as the graticule above
        local.push(R * cl * Math.sin(lon), R * Math.sin(lat), R * cl * Math.cos(lon));
      }
      // markers are a usability aid, not a size claim (cf. the pick proxies):
      // THREE.Points with sizeAttenuation:false renders every dot at the
      // same screen size regardless of camera distance
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(local, 3));
      g.add(new THREE.Points(geo, this.dotMaterial));
      for (let i = 0; i < list.length; i++) {
        const p = list[i];
        const anchor = new THREE.Object3D();
        anchor.position.set(local[i * 3], local[i * 3 + 1], local[i * 3 + 2]);
        g.add(anchor);
        this.pointDots[p.name] = anchor;    // label anchor; pixels come from Points
        const div = document.createElement("div");
        div.className = "point-label";
        div.textContent = p.name;       // user-supplied names: textContent only
        this.pointLabelsEl?.appendChild(div);
        this.pointLabelDivs[p.name] = div;
      }
    }
  }

  /** Fixed per-body pre-rotation that turns the overlay's local +Z prime
   *  meridian onto the body's material 0°E direction: world +X projected
   *  onto the equatorial plane (fallback +Y when +X is the pole), exactly
   *  the reference construction of observation.py local_frame. The overlay
   *  is a MESH child (quaternion qAlign*qSpin); qPre also turns about the
   *  local pole, so qSpin and qPre commute and the product qAlign*qSpin*qPre
   *  matches the server frame at any spin phase. */
  _overlayPreQuaternion(axis) {
    _vRef.set(1, 0, 0).addScaledVector(axis, -axis.x);
    if (_vRef.lengthSq() < 1e-12) _vRef.set(0, 1, 0).addScaledVector(axis, -axis.y);
    _vRef.normalize();
    _qAlign.setFromUnitVectors(_Y_AXIS, axis);
    _vRef.applyQuaternion(_qInv.copy(_qAlign).invert());
    return _qPre.setFromAxisAngle(_Y_AXIS, Math.atan2(_vRef.x, _vRef.z));
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
    if (this.pickEnabled === false) return null;
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

  /** branch 2: the catalog re-click toggle dropped the selection — clear the
   *  highlight tag and fly the view back to the fixed archive origin. */
  clearSelection() {
    this.selected = null;
    this._followLast = null;
    for (const [n, div] of Object.entries(this.labelDivs)) {
      div.classList.remove("sel");
      div.textContent = n;
    }
    this.focusOrigin();
  }

  /** Fly the camera so the fixed world origin (0,0,0, ADR-0035) sits at the
   *  viewport center, keeping the current heading and distance. */
  focusOrigin() {
    const target = new THREE.Vector3(0, 0, 0);
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    const dist = this.camera.position.distanceTo(this.controls.target);
    const toPos = target.clone().addScaledVector(
      dir, Math.min(Math.max(dist, AU * 0.0002), AU * 400));
    this._flyTo(target, toPos, 600);
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

  /** branch 2: the old ecliptic grid is retired; this button now controls the
   *  spin-locked surface graticule + observation-point dots on non-stellar
   *  bodies (view and edit modes). */
  setSurfaceOverlay(on) {
    this.surfaceOverlayVisible = on;
    for (const g of Object.values(this.overlayGroups)) g.visible = on;
    return on;
  }

  toggleSurfaceOverlay() { return this.setSurfaceOverlay(!this.surfaceOverlayVisible); }

  /** branch 2: body labels and observation-point labels share one toggle. */
  setLabelsVisible(on) {
    this.labelsVisible = on;
    return on;
  }

  toggleLabels() { return this.setLabelsVisible(!this.labelsVisible); }

  /** Absolute position of a body from the latest server pose (SSE), falling
   *  back to the latest poll snapshot before the first pose arrives. */
  _posOf(name) {
    const p = this.pose?.p?.[name] ?? this.snap?.bodies?.[name]?.p;
    return p ?? [0, 0, 0];
  }

  _updateBodyLighting(body) {
    const uniforms = this.bodyMeshes[body.name]?.material?.uniforms;
    if (!uniforms?.uStarDirection) return;
    const stars = this.system.bodies.filter((b) => b.kind === "star");
    const p = this._posOf(body.name);
    const flux = stars.map((s) => {
      const q = this._posOf(s.name);
      const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      const lum = this.snap?.stars?.[s.name]?.luminosity_w ?? s.luminosity_w ?? 0;
      return d2 > 0 ? lum / d2 : 0;
    });
    const total = flux.reduce((a, v) => a + v, 0);
    stars.forEach((s, i) => {
      const q = this._posOf(s.name);
      uniforms.uStarDirection.value[i].set(q[0] - p[0], q[1] - p[1], q[2] - p[2]).normalize();
      const st = this.snap?.stars?.[s.name];
      uniforms.uStarColor.value[i].copy(teffColor(st?.teff_k ?? s.extra?.teff_k ?? 5772));
      // Geometry comes from the authoritative positions. Relative flux only
      // balances the visible lights; this is not an N-body or climate solver.
      uniforms.uStarWeight.value[i] = total > 0 ? flux[i] / total : 0;
    });
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
    // ADR-0035: draw absolute world coordinates.  The fixed origin O=(0,0,0)
    // never moves with mass edits, body membership, winds or CCSN.
    for (const b of this.system.bodies) {
      const g = this.bodyGroups[b.name];
      if (!g) continue;
      const p = this._posOf(b.name);
      g.position.set(p[0], p[1], p[2]);
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
      const ov = this.overlayGroups[b.name];
      if (ov) {
        // align the body-fixed lat/lon frame; only recompute when the spin
        // axis itself changes (edit-mode spin edits arrive via the meta poll)
        const ax = _spinAxis.toArray();
        const c = this.overlayAxisCache[b.name];
        if (!c || c[0] !== ax[0] || c[1] !== ax[1] || c[2] !== ax[2]) {
          this.overlayAxisCache[b.name] = ax;
          ov.quaternion.copy(this._overlayPreQuaternion(_spinAxis));
        }
      }
      const ring = this.ringMeshes[b.name];
      if (ring) ring.quaternion.setFromUnitVectors(_Z_AXIS, _spinAxis);
      if (b.kind !== "star") this._updateBodyLighting(b);
    }
    // star light follows the primary star (rendered position)
    const primary = this.system.bodies.find((b) => b.kind === "star");
    if (primary) {
      const p = this._posOf(primary.name);
      this.starLight.position.set(p[0], p[1], p[2]);
    }
  }

  drawLabels() {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    for (const [name, div] of Object.entries(this.labelDivs)) {
      const g = this.bodyGroups[name];
      if (!g || !this.labelsVisible) { div.style.display = "none"; continue; }
      v.copy(g.position).project(this.camera);
      if (v.z > 1) { div.style.display = "none"; continue; }
      div.style.display = "block";
      div.style.left = `${(v.x * 0.5 + 0.5) * rect.width}px`;
      div.style.top = `${(-v.y * 0.5 + 0.5) * rect.height}px`;
    }
  }

  /** Observation-point name labels (branch 2): projected from each dot's
   *  world position; visible only while both the surface overlay and the
   *  label toggle are on. */
  drawPointLabels() {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    for (const [name, div] of Object.entries(this.pointLabelDivs)) {
      const dot = this.pointDots[name];
      if (!dot || !this.surfaceOverlayVisible || !this.labelsVisible) {
        div.style.display = "none";
        continue;
      }
      dot.getWorldPosition(v);
      v.project(this.camera);
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
    this.drawPointLabels();
    this.drawGizmo();
  }
}
