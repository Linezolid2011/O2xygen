"""End-to-end P1 demo: build a system, propagate, compute the 温度表.

Mimics his 一键求解 step ③ "温度表: N 个采样点" across the timeline
(13 samples / 10 years in his screenshots) for a moon on an eccentric orbit
around a planet around a star. Run from repo root:

    .venv/Scripts/python examples/demo_temperature_table.py
"""
import numpy as np

from o2xygen.constants import AU, DAY, YEAR, M_SUN, M_EARTH, L_SUN
from o2xygen.system import System, Body, STAR, PLANET, MOON
from o2xygen.kepler import Elements
from o2xygen.temperature import equilibrium_temperature

# --- a "白玫瑰"-like toy system ------------------------------------------------
SYS = System("demo")
SYS.add(Body("Αστραία", STAR, mass_kg=M_SUN, luminosity_w=L_SUN))
SYS.add(Body("Νεφέλη", PLANET, parent="Αστραία",
             elements=Elements(a_m=1.2 * AU, e=0.05, m0=0.0),
             mass_kg=20.0 * M_EARTH))
SYS.add(Body("Αύρα", MOON, parent="Νεφέλη",
             elements=Elements(a_m=0.01 * AU, e=0.15, m0=1.0),
             mass_kg=7.3e22, albedo=0.09))

SPAN_DAYS = 10 * 365.25
N_SAMPLES = 13          # his 温度表 sampling count from the progress dialog
REGIME = "dayside"      # calibration result from tests: S=861.46 <-> 288.24 K

def solve_temperature_table(system, body, n=N_SAMPLES, span_days=SPAN_DAYS):
    """Pipeline stage ③: sample equilibrium temperature across the timeline."""
    rows = []
    for i in range(n):
        t = span_days * i / (n - 1) * DAY
        ill = system.illumination(body, t)
        T = equilibrium_temperature(ill["total"], albedo=system.bodies[body].albedo,
                                    regime=REGIME)
        rows.append((system.days_since_epoch(t), ill["total"], T))
    return rows

if __name__ == "__main__":
    print(f"{'距初始时刻 (天)':>16} {'总辐照 (W/m^2)':>15} {'温度 (K)':>10}")
    for d, S, T in solve_temperature_table(SYS, "Αύρα"):
        print(f"{d:>16.4f} {S:>15.3f} {T:>10.3f}")

    # temperature-curve property check: T^4 tracks flux at fine sampling
    ts = np.linspace(0, 30 * DAY, 400)
    flux = np.array([SYS.illumination("Αύρα", t)["total"] for t in ts])
    T4 = ((1 - 0.09) * flux / (2 * 5.670374419e-8))
    print("\ncurve check: std(T^4 / S) over 30 d =",
          float(np.std(T4 / flux)), "(should be ~0 by construction)")
