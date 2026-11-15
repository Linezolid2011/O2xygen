"""End-to-end demo: build a world, integrate, compute the 温度表.

Flat world (ADR-0016): bodies are placed with the element->state placement
tool, then the world remembers only Cartesian initial conditions. Run:

    .venv/Scripts/python examples/demo_temperature_table.py
"""
import numpy as np

from o2xygen.constants import AU, DAY, YEAR, M_SUN, M_EARTH, L_SUN, G
from o2xygen.system import System, Body, STAR, BODY
from o2xygen.kepler import Elements, elements_to_state
from o2xygen.nbody import NBodySimulation
from o2xygen.temperature import equilibrium_temperature


def place(sys, name, kind, around, a_m, e=0.0, inc=0.0, mass=0.0, m0=0.0, **kw):
    ref = sys.bodies[around]
    mu = G * (ref.mass_kg + mass)
    r_rel, v_rel = elements_to_state(Elements(a_m=a_m, e=e, inc=inc, m0=m0), mu, 0.0)
    b = Body(name, kind, mass_kg=mass, pos0=ref.pos0 + r_rel, vel0=ref.vel0 + v_rel, **kw)
    sys.add(b)
    return b


SYS = System("demo", origin="star")
SYS.add(Body("star", STAR, mass_kg=M_SUN, luminosity_w=L_SUN))
place(SYS, "planet", BODY, "star", a_m=1.2 * AU, e=0.05, mass=20 * M_EARTH, albedo=0.09)
place(SYS, "moon", BODY, "planet", a_m=0.01 * AU, e=0.15, mass=7.3e22,
      albedo=0.09, m0=1.0)

SPAN_DAYS = 10 * 365.25
N_SAMPLES = 13
REGIME = "dayside"


def solve_temperature_table(sim, body, n=N_SAMPLES, span_days=SPAN_DAYS):
    rows = []
    for i in range(n):
        t = span_days * i / (n - 1) * DAY
        sim.integrate_to(t)
        states = sim.live_state()
        pos = states[body][0]
        flux = sum(s.luminosity_w / (4 * np.pi * np.linalg.norm(pos - states[s.name][0])**2)
                   for s in SYS.stars())
        rows.append((t / DAY, flux, equilibrium_temperature(flux, albedo=0.09, regime=REGIME)))
    return rows


if __name__ == "__main__":
    nb = NBodySimulation(SYS)
    print(f"{'t (days)':>12} {'flux (W/m^2)':>14} {'T (K)':>10}")
    for d, S, T in solve_temperature_table(nb, "moon"):
        print(f"{d:>12.2f} {S:>14.2f} {T:>10.2f}")
