"""Spherical-shell atmospheric optics, derived from CalcMySky's optical model.

The save contains physical inputs, not an Earth texture or a sky colour.  The
exponential profiles are a deliberately small, inspectable subset of
CalcMySky's arbitrary GLSL profiles.  RGB channels approximate wavelength
bands; all extinction coefficients are per metre at the reference surface.
"""
from __future__ import annotations

import math
import numpy as np

MODEL = "spherical_exponential"
OPTICS_VERSION = 3  # exact segment CDF, finite ground, grazing-refined TOA table


def default_atmosphere() -> dict:
    """Earth-like clear-air starting point, not a claim about a saved world."""
    return {
        "model": MODEL, "height_m": 100_000.0, "scale_height_m": 8_000.0,
        "surface_pressure_pa": 101_325.0, "surface_temperature_k": 288.15,
        "mean_molecular_weight": 28.97,
        "composition": {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004},
        "rayleigh_scattering_m_inv": [5.8e-6, 1.35e-5, 3.31e-5],
        "aerosol_scattering_m_inv": [1.0e-6, 1.0e-6, 1.0e-6],
        "absorption_m_inv": [0.0, 0.0, 0.0],
        "aerosol_scale_height_m": 1_200.0, "aerosol_anisotropy": 0.65,
        "ground_albedo": [0.18, 0.22, 0.16],
        "infrared_optical_depth": 0.8, "exposure": 1.0,
    }


def _float(data, key, default=None):
    value = data.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"atmosphere.{key} must be a number") from e
    if not math.isfinite(value):
        raise ValueError(f"atmosphere.{key} must be finite")
    return value


def _spectrum(data, key, default=None):
    value = data.get(key, default)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        value = [value] * 3
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"atmosphere.{key} must be a number or RGB triplet") from e
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"atmosphere.{key} must contain three finite numbers")
    return [float(x) for x in a]


def normalize_atmosphere(value, allow_none=True) -> dict | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError("missing atmosphere")
    if not isinstance(value, dict):
        raise ValueError("atmosphere must be an object")
    model = value.get("model", MODEL)
    if model != MODEL:
        raise ValueError(f"unsupported atmosphere model {model!r}")
    out = {"model": MODEL}
    for key in ("height_m", "scale_height_m", "surface_pressure_pa",
                "surface_temperature_k", "mean_molecular_weight",
                "aerosol_scale_height_m"):
        out[key] = _float(value, key)
        if out[key] <= 0.0:
            raise ValueError(f"atmosphere.{key} must be positive")
    for key in ("rayleigh_scattering_m_inv", "aerosol_scattering_m_inv",
                "absorption_m_inv"):
        out[key] = _spectrum(value, key)
        if min(out[key]) < 0.0:
            raise ValueError(f"atmosphere.{key} must be nonnegative")
    out["ground_albedo"] = _spectrum(value, "ground_albedo")
    if not all(0.0 <= x <= 1.0 for x in out["ground_albedo"]):
        raise ValueError("atmosphere.ground_albedo must be within 0..1")
    out["aerosol_anisotropy"] = _float(value, "aerosol_anisotropy")
    if not -0.95 <= out["aerosol_anisotropy"] <= 0.95:
        raise ValueError("atmosphere.aerosol_anisotropy must be within -0.95..0.95")
    out["infrared_optical_depth"] = _float(value, "infrared_optical_depth", 0.0)
    if out["infrared_optical_depth"] < 0.0:
        raise ValueError("atmosphere.infrared_optical_depth must be nonnegative")
    out["exposure"] = _float(value, "exposure", 1.0)
    if out["exposure"] <= 0.0:
        raise ValueError("atmosphere.exposure must be positive")
    surface_model = value.get("surface_model", "solid")
    if surface_model not in ("solid", "cloud_deck"):
        raise ValueError("atmosphere.surface_model must be solid or cloud_deck")
    out["surface_model"] = surface_model
    if max(out["rayleigh_scattering_m_inv"]
           + out["aerosol_scattering_m_inv"] + out["absorption_m_inv"]) <= 0.0:
        raise ValueError("an atmosphere needs a nonzero optical coefficient")
    comp = value.get("composition", {})
    if not isinstance(comp, dict) or len(comp) > 64:
        raise ValueError("atmosphere.composition must be an object with at most 64 species")
    out["composition"] = {}
    for name, fraction in comp.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("invalid atmospheric species name")
        f = _float({"fraction": fraction}, "fraction")
        if f < 0.0:
            raise ValueError("atmospheric composition fractions must be nonnegative")
        out["composition"][name.strip()] = f
    if sum(out["composition"].values()) > 1.000001:
        raise ValueError("atmospheric composition fractions must sum to at most one")
    return out


class SphericalAtmosphere:
    """Numerically integrated optical paths in a spherical exponential shell."""
    def __init__(self, radius_m: float, atmosphere: dict, integration_points=48):
        if not (math.isfinite(radius_m) and radius_m > 0.0):
            raise ValueError("atmosphere requires a positive planetary radius")
        self.radius_m = float(radius_m)
        self.a = normalize_atmosphere(atmosphere, allow_none=False)
        if self.a["height_m"] > 0.5 * self.radius_m:
            raise ValueError("atmosphere height must be at most half the planetary radius")
        self.rayleigh = np.asarray(self.a["rayleigh_scattering_m_inv"])
        self.aerosol = np.asarray(self.a["aerosol_scattering_m_inv"])
        self.absorption = np.asarray(self.a["absorption_m_inv"])
        self.ground_albedo = np.asarray(self.a["ground_albedo"])
        self.top_m = self.radius_m + self.a["height_m"]
        self._gx, self._gw = np.polynomial.legendre.leggauss(integration_points)

    def coefficients(self, position) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.asarray(position, dtype=float)
        h = np.maximum(np.linalg.norm(r, axis=-1) - self.radius_m, 0.0)
        molecular = np.exp(-h / self.a["scale_height_m"])[..., None]
        aerosol = np.exp(-h / self.a["aerosol_scale_height_m"])[..., None]
        return molecular * self.rayleigh, aerosol * self.aerosol, molecular * self.absorption

    def ray_length(self, position, direction, ground=True) -> np.ndarray:
        p = np.asarray(position, dtype=float)
        d = np.asarray(direction, dtype=float)
        if p.ndim > d.ndim:
            d = d.reshape(d.shape[:-1] + (1,) * (p.ndim-d.ndim) + (3,))
        pv = np.sum(p * d, axis=-1)
        pp = np.sum(p * p, axis=-1)
        if ground:
            disc = pv * pv - pp + self.radius_m ** 2
            tolerance = 1e-12 * (np.abs(pv*pv) + np.abs(pp) + self.radius_m**2)
            hit = disc >= -tolerance
            dg = -pv - np.sqrt(np.maximum(disc, 0.0))
            hit &= dg >= -1e-5
        else:
            hit = np.zeros(pv.shape, dtype=bool)
            dg = np.full(pv.shape, np.inf)
        da = -pv + np.sqrt(np.maximum(pv * pv - pp + self.top_m ** 2, 0.0))
        return np.maximum(np.where(hit, np.minimum(dg, da), da), 0.0)

    def optical_depth(self, position, direction, length=None, ground=True):
        p = np.asarray(position, dtype=float)
        d = np.asarray(direction, dtype=float)
        if p.ndim > d.ndim:
            d = np.broadcast_to(d.reshape(d.shape[:-1]
                + (1,) * (p.ndim-d.ndim) + (3,)), p.shape)
        if length is None:
            length = self.ray_length(p, d, ground=ground)
        length = np.maximum(np.asarray(length, dtype=float), 0.0)
        mid = 0.5 * length[..., None]
        distances = mid * (self._gx + 1.0)
        # A (ray, primary-sample) array of partial lengths adds its own
        # quadrature axis. Reserve matching singleton axes for p/d.
        extra = max(0, distances.ndim - p.ndim + 1)
        shape = tuple(p.shape[:-1]) + (1,) * extra + (3,)
        samples = p.reshape(shape) + distances[..., None] * d.reshape(shape)
        ray, mie, absorb = self.coefficients(samples)
        extinction = ray + mie + absorb
        return np.sum(extinction * (0.5 * self._gw)[..., None], axis=-2) * length[..., None]

    def transmittance(self, position, direction, length=None, ground=True):
        return np.exp(-np.minimum(self.optical_depth(position, direction, length, ground), 700.0))

    def sun_transmittance(self, position, direction):
        """Star transmittance with explicit finite-planet occultation.

        No local-zenith cosine is applied to volume scattering. A point above
        the surface can see a star slightly below its local horizontal plane.
        """
        p = np.asarray(position, dtype=float)
        d = np.asarray(direction, dtype=float)
        da = self.ray_length(p, d, ground=False)
        dg = self.ray_length(p, d, ground=True)
        blocked = (dg > 1e-5) & (dg < da - 1e-5)
        trans = self.transmittance(p, d, ground=False)
        if trans.ndim > blocked.ndim:
            blocked = blocked[..., None]
        return np.where(blocked, 0.0, trans)

    def phase(self, cos_angle, anisotropic=True):
        c = np.clip(np.asarray(cos_angle, dtype=float), -1.0, 1.0)
        ray = 3.0 / (16.0 * np.pi) * (1.0 + c * c)
        g = self.a["aerosol_anisotropy"]
        hg = (1.0 - g * g) / (4.0 * np.pi * np.maximum(1.0 + g * g - 2.0 * g * c, 1e-8) ** 1.5)
        return (ray, hg) if anisotropic else ray

    def single_scattering(self, position, direction, sun_direction, ground=True):
        """In-scattered direct sunlight along one ray (CalcMySky order 1).

        Returns fractional RGB radiance for unit isotropic stellar irradiance;
        the caller supplies actual spectrum/flux.  This is an integral of the
        physical phase functions and both transmittances, not a sky gradient.
        """
        p = np.asarray(position, dtype=float)
        v = np.asarray(direction, dtype=float)
        s = np.asarray(sun_direction, dtype=float)
        length = self.ray_length(p, v, ground=ground)
        distances = 0.5 * length[..., None] * (self._gx + 1.0)
        points = p[..., None, :] + distances[..., None] * v[..., None, :]
        ray, mie, _ = self.coefficients(points)
        tau_view = self.optical_depth(p, v, distances, ground=ground)
        tau_sun = -np.log(np.maximum(self.sun_transmittance(points, s), 1e-300))
        pr, pm = self.phase(np.sum(v * s, axis=-1))
        pr = np.asarray(pr); pm = np.asarray(pm)
        if points.ndim == 3:
            pr, pm = pr[..., None, None], pm[..., None, None]
        source = ray * pr + mie * pm
        axis = 1 if points.ndim == 3 else 0
        weights = 0.5 * self._gw
        if axis == 1:
            weights = weights[None, :, None]
        else:
            weights = weights[:, None]
        radiance = np.sum(source * np.exp(-np.minimum(tau_view + tau_sun, 700.0))
                          * weights, axis=axis) * length[..., None]
        return radiance

    def vertical_optical_depth(self, altitude_m=0.0):
        mr = 1.0 - math.exp(-max(0.0, self.a["height_m"] - altitude_m)
                            / self.a["scale_height_m"])
        ma = 1.0 - math.exp(-max(0.0, self.a["height_m"] - altitude_m)
                            / self.a["aerosol_scale_height_m"])
        return (self.rayleigh + self.absorption) * self.a["scale_height_m"] * mr \
            + self.aerosol * self.a["aerosol_scale_height_m"] * ma


def make_optical_table(model: SphericalAtmosphere, mu_count=129, height_count=48):
    """tau to the top ignoring the solid planet; solar occultation is separate.

    Segment tau before a ground intersection can still be obtained by
    subtracting the endpoint's continuation: the common continuation cancels.
    """
    # Cluster samples around grazing incidence (mu=0), where tau varies most
    # rapidly. Unlike a free-path CDF, this table is only evaluated directly.
    mu_grid = np.linspace(-1.0, 1.0, mu_count)
    mu = np.sign(mu_grid) * np.abs(mu_grid)**3
    # More altitude resolution close to the surface, like CalcMySky's
    # distance-to-horizon coordinate, without importing its binary format.
    h = model.a["height_m"] * np.linspace(0.0, 1.0, height_count)**2
    hh, mm = np.meshgrid(h, mu, indexing="ij")
    p = np.stack((np.zeros(hh.size), np.zeros(hh.size), model.radius_m + hh.ravel()), axis=1)
    d = np.stack((np.sqrt(1-mm.ravel()**2), np.zeros(mm.size), mm.ravel()), axis=1)
    tau = model.optical_depth(p, d, ground=False).reshape(height_count, mu_count, 3)
    return {"mu": mu, "height_m": h, "tau": tau}


def sample_optical_table(model, table, position, direction):
    p, d = np.asarray(position, float), np.asarray(direction, float)
    if p.ndim > d.ndim:
        d = np.broadcast_to(d.reshape(d.shape[:-1] + (1,)*(p.ndim-d.ndim) + (3,)), p.shape)
    shape = p.shape[:-1]
    p, d = p.reshape(-1, 3), np.broadcast_to(d, p.shape).reshape(-1, 3)
    r = np.linalg.norm(p, axis=1)
    h = np.clip(r-model.radius_m, 0.0, model.a["height_m"])
    mu = np.clip(np.sum(p*d, axis=1)/np.maximum(r, 1e-30), -1.0, 1.0)
    h, mu = np.sqrt(h/model.a["height_m"]), mu
    hh, mm = table["height_m"], table["mu"]
    hi = np.clip(np.searchsorted(hh, h*h*model.a["height_m"]), 1, len(hh)-1)
    mi = np.clip(np.searchsorted(mm, mu), 1, len(mm)-1)
    hf = (h*h*model.a["height_m"]-hh[hi-1])/np.maximum(hh[hi]-hh[hi-1], 1e-30)
    mf = (mu-mm[mi-1])/(mm[mi]-mm[mi-1])
    out = []
    for c in range(3):
        tau = table["tau"][..., c]
        out.append((tau[hi-1, mi-1]*(1-hf)*(1-mf) + tau[hi, mi-1]*hf*(1-mf)
                    + tau[hi-1, mi]*(1-hf)*mf + tau[hi, mi]*hf*mf))
    return np.stack(out, axis=1).reshape(shape+(3,))


def _sample_rayleigh(d, rng):
    """Rayleigh phase sampling by bounded rejection (mean 1.5 trials)."""
    count = len(d)
    z = np.zeros(count)
    remaining = np.arange(count)
    while len(remaining):
        trial = rng.uniform(-1.0, 1.0, len(remaining))
        accept = rng.random(len(remaining)) < 0.5 * (1.0 + trial * trial)
        z[remaining[accept]] = trial[accept]
        remaining = remaining[~accept]
    az = rng.uniform(0.0, 2.0 * np.pi, count)
    ref = np.tile([0.0, 0.0, 1.0], (count, 1))
    ref[np.abs(d[:, 2]) > 0.95] = [0.0, 1.0, 0.0]
    e1 = np.cross(d, ref)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1)[:, None], 1e-30)
    e2 = np.cross(d, e1)
    return (np.sqrt(1-z*z)[:, None] * (np.cos(az)[:, None]*e1
            + np.sin(az)[:, None]*e2) + z[:, None] * d)


def multiple_scattering(model: SphericalAtmosphere, view, sun, samples=256,
                        orders=4, seed=20261125, replace_first=True) -> dict:
    """Per-channel Monte-Carlo Neumann radiative transfer at the surface.

    Each RGB path samples its OWN extinction CDF. Therefore collision density
    already contains the path transmittance; multiplying T again would bias
    it. Volume next-event estimation has no ground cosine, and a Lambertian
    bounce uses a cosine-weighted hemisphere. Direct ground, volume orders,
    and later ground feedback are separately reported for validation.
    """
    v, s = np.atleast_2d(np.asarray(view, float)), np.atleast_2d(np.asarray(sun, float))
    if len(s) == 1 and len(v) > 1:
        s = np.broadcast_to(s, v.shape)
    if v.shape != s.shape or v.shape[1] != 3:
        raise ValueError("view/sun must have matching (N,3) shapes")
    v = v / np.linalg.norm(v, axis=1)[:, None]
    s = s / np.linalg.norm(s, axis=1)[:, None]
    n_rays = len(v)
    table = make_optical_table(model)
    def tau_at(p, d):
        return sample_optical_table(model, table, p, d)
    def sun_t(p, d):
        da = model.ray_length(p, d, ground=False)
        dg = model.ray_length(p, d, ground=True)
        blocked = (dg > 1e-5) & (dg < da - 1e-5)
        return np.where(blocked[:, None], 0.0, np.exp(-np.minimum(tau_at(p, d), 700.0)))
    p0 = np.tile([0.0, 0.0, model.radius_m + 0.01], (n_rays, 1))
    first = model.single_scattering(p0, v, s)
    rng = np.random.default_rng(seed)
    count = n_rays * samples * 3
    p = np.tile([0.0, 0.0, model.radius_m + 0.01], (count, 1))
    d = np.repeat(np.repeat(v, samples, axis=0), 3, axis=0)
    s = np.repeat(np.repeat(s, samples, axis=0), 3, axis=0)
    channel = np.tile(np.arange(3), n_rays * samples)
    weight = np.ones(count)
    active = np.ones(count, dtype=bool)
    volume = np.zeros((orders, count))
    ground = np.zeros((orders, count))
    for order in range(orders):
        boundary = model.ray_length(p, d, ground=True)
        target = -np.log(np.maximum(rng.random(count), 1e-15))
        # Never difference two interpolated TOA depths here: near the horizon
        # that loses the small segment depth (and can even make it negative).
        ids = np.flatnonzero(active)
        tau_end = np.zeros(count)
        if len(ids):
            tau_end[ids] = model.optical_depth(p[ids], d[ids], boundary[ids])[
                np.arange(len(ids)), channel[ids]]
        collision = active & (tau_end > target)
        da = model.ray_length(p, d, ground=False)
        dg = model.ray_length(p, d, ground=True)
        ground_hit = active & ~collision & (dg > 1e-6) & (dg < da - 1e-6)
        qg = p + boundary[:, None] * d
        tg = sun_t(qg, s)[np.arange(count), channel]
        mu_g = np.clip(np.sum(qg * s, axis=1) /
                       np.maximum(np.linalg.norm(qg, axis=1), 1e-30), -1.0, 1.0)
        direct = weight * model.ground_albedo[channel] / np.pi * tg * np.maximum(mu_g, 0.0)
        ground[order] = np.where(ground_hit, direct, 0.0)
        # Invert the channel's optical-depth CDF with bisection. No current
        # segment T is added to weight: the sampled density includes it.
        lo, hi = np.zeros(count), boundary.copy()
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            ids = np.flatnonzero(collision)
            tau = np.zeros(count)
            if len(ids):
                tau[ids] = model.optical_depth(p[ids], d[ids], mid[ids])[
                    np.arange(len(ids)), channel[ids]]
            over = tau > target
            hi, lo = np.where(over, mid, hi), np.where(over, lo, mid)
        q = p + (0.5*(lo+hi))[:, None] * d
        ts = sun_t(q, s)[np.arange(count), channel]
        ray, mie, absorb = model.coefficients(q)
        sc = ray + mie
        beta_t = sc[np.arange(count), channel] + absorb[np.arange(count), channel]
        pr, pm = model.phase(np.sum(d * s, axis=1))
        beta_source = ray[np.arange(count), channel] * pr + mie[np.arange(count), channel] * pm
        contribution = weight * beta_source / np.maximum(beta_t, 1e-30) * ts
        volume[order] = np.where(collision, contribution, 0.0)
        # Ground and volume states are disjoint and BOTH survive this order.
        normal = qg / np.maximum(np.linalg.norm(qg, axis=1)[:, None], 1e-30)
        z = np.sqrt(rng.random(count))
        az = rng.uniform(0.0, 2*np.pi, count)
        e1 = np.cross(normal, np.tile([1.0, 0.0, 0.0], (count, 1)))
        close = np.linalg.norm(e1, axis=1) < 1e-8
        e1[close] = np.cross(normal[close], np.tile([0.0, 1.0, 0.0], (int(close.sum()), 1)))
        e1 /= np.maximum(np.linalg.norm(e1, axis=1)[:, None], 1e-30)
        e2 = np.cross(normal, e1)
        bounce = np.sqrt(1-z*z)[:, None]*(np.cos(az)[:, None]*e1 + np.sin(az)[:, None]*e2) \
            + z[:, None]*normal
        weight_ground = weight * model.ground_albedo[channel]
        p_ground, d_ground = qg + normal*0.02, bounce
        weight_volume = weight * sc[np.arange(count), channel] / np.maximum(beta_t, 1e-30)
        choose_mie = rng.random(count) < (
            mie[np.arange(count), channel] / np.maximum(sc[np.arange(count), channel], 1e-30))
        ray_d = _sample_rayleigh(d, rng)
        g = model.a["aerosol_anisotropy"]
        if abs(g) > 1e-6:
            z = (1 + g*g - ((1-g*g) /
                 np.maximum(1-g+2*g*rng.random(count), 1e-8))**2) / (2*g)
        else:
            z = rng.uniform(-1.0, 1.0, count)
        z = np.clip(z, -1.0, 1.0)
        az = rng.uniform(0.0, 2*np.pi, count)
        ref = np.tile([0.0, 0.0, 1.0], (count, 1))
        ref[np.abs(d[:, 2]) > 0.95] = [0.0, 1.0, 0.0]
        e1 = np.cross(d, ref)
        e1 /= np.maximum(np.linalg.norm(e1, axis=1)[:, None], 1e-30)
        e2 = np.cross(d, e1)
        mie_d = np.sqrt(1-z*z)[:, None]*(np.cos(az)[:, None]*e1 + np.sin(az)[:, None]*e2) \
            + z[:, None]*d
        p = np.where(collision[:, None], q, np.where(ground_hit[:, None], p_ground, p))
        d = np.where(collision[:, None], np.where(choose_mie[:, None], mie_d, ray_d),
                     np.where(ground_hit[:, None], d_ground, d))
        weight = np.where(collision, weight_volume, np.where(ground_hit, weight_ground, 0.0))
        active = collision | ground_hit
        if not np.any(active) or order + 1 >= orders:
            break
    def collect(a):
        return a.reshape(orders, n_rays, samples, 3).mean(axis=2)
    vo, gr = collect(volume), collect(ground)
    # Production replaces the sampled first order with deterministic
    # quadrature. Tests may retain it for an independent MC convergence check.
    if replace_first:
        vo[0] = first
    out = {"volume_orders": vo, "ground_orders": gr,
           "single": first, "ground": gr[0],
           "multiple": vo[1:].sum(axis=0) + gr[1:].sum(axis=0),
           "samples_per_channel": samples, "truncation_order": orders}
    out["orders"] = (vo + gr).tolist()
    return out
