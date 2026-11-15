// Client-side Kepler propagation — viz layer only (server physics is source of truth).
const DAY = 86400.0;
const AU = 1.495978707e11;
const G = 6.67430e-11;
const M_SUN = 1.98847e30;

export function gm(massKg) { return G * massKg; }

export function solveKepler(M, e) {
  M = ((M % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  let E = e < 0.8 ? M : Math.PI;
  for (let i = 0; i < 60; i++) {
    const dE = -(E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E += dE;
    if (Math.abs(dE) < 1e-13) break;
  }
  return E;
}

// position (m) relative to parent, from elements
export function localPos(el, mu, tS) {
  const n = Math.sqrt(mu / el.a_m ** 3);
  const E = solveKepler(el.m0 + n * tS, el.e);
  const r = el.a_m * (1 - el.e * Math.cos(E));
  const xp = el.a_m * (Math.cos(E) - el.e);
  const yp = el.a_m * Math.sqrt(1 - el.e * el.e) * Math.sin(E);
  const ci = Math.cos(el.inc), si = Math.sin(el.inc);
  const co = Math.cos(el.raan), so = Math.sin(el.raan);
  const cw = Math.cos(el.argp), sw = Math.sin(el.argp);
  return {
    x: (co * cw - so * sw * ci) * xp + (-co * sw - so * cw * ci) * yp,
    y: (so * cw + co * sw * ci) * xp + (-so * sw + co * cw * ci) * yp,
    z: sw * si * xp + cw * si * yp,
    r,
  };
}

export function makePropagator(system) {
  // system: {bodies: [{name, kind, parent, mass_kg, elements}]} — elements may be null
  const byName = {};
  for (const b of system.bodies) byName[b.name] = b;

  function absPos(name, tS) {
    const b = byName[name];
    if (!b) return { x: 0, y: 0, z: 0 };
    if (!b.parent || !b.elements) return { x: 0, y: 0, z: 0 };
    const parent = absPos(b.parent, tS);
    const mu = gm(byName[b.parent].mass_kg);
    const local = localPos(b.elements, mu, tS);
    return { x: parent.x + local.x, y: parent.y + local.y, z: parent.z + local.z };
  }

  function orbitPath(name, nSamples = 256) {
    const b = byName[name];
    if (!b || !b.parent || !b.elements) return null;
    const mu = gm(byName[b.parent].mass_kg);
    const T = 2 * Math.PI * Math.sqrt(b.elements.a_m ** 3 / mu);
    const pts = [];
    for (let i = 0; i <= nSamples; i++) {
      const t = (T * i) / nSamples;
      const parent = absPos(b.parent, t);
      const local = localPos(b.elements, mu, t);
      pts.push([parent.x + local.x, parent.y + local.y, parent.z + local.z]);
    }
    return pts;
  }

  return { absPos, orbitPath, byName };
}

// Client-side N-body propagator: velocity Verlet on direct-summation gravity,
// seeded from a server snapshot (positions, velocities, masses of ALL bodies).
// This is the render layer between polls (ADR-0015) - unlike linear
// extrapolation it follows curved orbits, so labels and spheres stay on track
// even at high playback speeds. Small-N systems only; step = server dt.
export class NBodyPropagator {
  constructor(snap, stepS) {
    this.names = Object.keys(snap.bodies);
    this.m = this.names.map((n) => snap.bodies[n].m);
    this.p = this.names.map((n) => snap.bodies[n].p.slice());
    this.v = this.names.map((n) => snap.bodies[n].v.slice());
    this.index = Object.fromEntries(this.names.map((n, i) => [n, i]));
    this.t = snap.t_days * DAY;
    this.step = stepS || 3600;
    this.a = this._accel();
  }

  _accel() {
    const n = this.names.length;
    const a = Array.from({ length: n }, () => [0, 0, 0]);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = this.p[j][0] - this.p[i][0];
        const dy = this.p[j][1] - this.p[i][1];
        const dz = this.p[j][2] - this.p[i][2];
        const r2 = dx * dx + dy * dy + dz * dz;
        const inv = 1 / (r2 * Math.sqrt(r2));
        const f = G * this.m[j] * inv;
        a[i][0] += f * dx; a[i][1] += f * dy; a[i][2] += f * dz;
      }
    }
    return a;
  }

  propagateTo(tS) {
    let remaining = Math.abs(tS - this.t);
    const dir = tS >= this.t ? 1 : -1;
    let guard = 2_000_000;
    while (remaining > 1e-9 && guard-- > 0) {
      const h = Math.min(this.step, remaining) * dir;
      const n = this.names.length;
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) {
          this.p[i][k] += this.v[i][k] * h + 0.5 * this.a[i][k] * h * h;
        }
      }
      const aNew = this._accel();
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) {
          this.v[i][k] += 0.5 * (this.a[i][k] + aNew[i][k]) * h;
        }
      }
      this.a = aNew;
      remaining -= Math.abs(h);
    }
    this.t = tS;
  }

  pos(name) {
    const p = this.p[this.index[name]];
    return { x: p[0], y: p[1], z: p[2] };
  }
}

export function periodDays(body, parentMassKg) {
  if (!body.elements || !parentMassKg) return null;
  const T = 2 * Math.PI * Math.sqrt(body.elements.a_m ** 3 / gm(parentMassKg));
  return T / DAY;
}

export { DAY, AU, M_SUN };
