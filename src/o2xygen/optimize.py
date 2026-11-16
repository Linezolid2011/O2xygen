"""Hand-written scalar root-finding and minimisation (stdlib only).

Replaces the two scipy calls the project ever used:
  * scipy.optimize.brentq           -> brentq        (Brent's method)
  * scipy.optimize.minimize_scalar  -> minimize_scalar_bounded (golden section)

Both are textbook algorithms (Numerical Recipes); the tolerances match the
old call sites (xtol=1e-14, xatol=1e-4).
"""
from __future__ import annotations

__all__ = ["brentq", "minimize_scalar_bounded"]


def brentq(f, a: float, b: float, xtol: float = 1e-12,
           max_iter: int = 100) -> float:
    """Root of f in [a, b] (f(a), f(b) must bracket it) by Brent's method."""
    fa, fb = float(f(a)), float(f(b))
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0.0:
        raise ValueError(f"f({a})={fa} and f({b})={fb} do not bracket a root")
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * 5e-324 * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return float(b)
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0.0 else -tol)
        fb = float(f(b))
    raise RuntimeError(f"brentq did not converge in {max_iter} iterations")


def minimize_scalar_bounded(f, a: float, b: float, xatol: float = 1e-5,
                            max_iter: int = 200) -> float:
    """Minimise f on [a, b] by golden-section search; returns the argmin.

    Equivalent to scipy.optimize.minimize_scalar(method="bounded") for the
    unimodal objectives it is used on here."""
    invphi = (5.0 ** 0.5 - 1.0) / 2.0          # 0.618...
    x1 = b - invphi * (b - a)
    x2 = a + invphi * (b - a)
    f1, f2 = float(f(x1)), float(f(x2))
    for _ in range(max_iter):
        if abs(b - a) < xatol:
            break
        if f1 > f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + invphi * (b - a)
            f2 = float(f(x2))
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - invphi * (b - a)
            f1 = float(f(x1))
    return 0.5 * (a + b)
