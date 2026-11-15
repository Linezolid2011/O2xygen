// Client-side N-body propagator — viz layer only (server physics is source of truth).
// Velocity Verlet on direct-summation gravity, seeded from a server snapshot
// (positions, velocities, masses of ALL bodies). This is the render layer
// between polls (ADR-0015) - unlike linear extrapolation it follows curved
// orbits, so labels and spheres stay on track even at high playback speeds.
// Small-N systems only; step = server dt.
const DAY = 86400.0;
const AU = 1.495978707e11;
const G = 6.67430e-11;

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
        // epsilon floor: coincident bodies must not yield 1/0 -> NaN/Infinity
        // (1 m^2 is invisible at AU scales but keeps the integrator finite)
        const r2f = r2 + 1.0;
        const inv = 1 / (r2f * Math.sqrt(r2f));
        const f = G * this.m[j] * inv;
        a[i][0] += f * dx; a[i][1] += f * dy; a[i][2] += f * dz;
      }
    }
    return a;
  }

  propagateTo(tS) {
    let remaining = Math.abs(tS - this.t);
    const dir = tS >= this.t ? 1 : -1;
    let advanced = 0;
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
      advanced += Math.abs(h);
    }
    // only claim the target time if we actually reached it; if the step
    // budget ran out, report the time truly integrated to (callers re-seed)
    this.t = remaining <= 1e-9 ? tS : this.t + dir * advanced;
  }

  pos(name) {
    const p = this.p[this.index[name]];
    return { x: p[0], y: p[1], z: p[2] };
  }
}

export { DAY, AU };
