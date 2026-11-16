"""o2xygen — a real-time N-body worldbuilding engine (worlds are files)."""
from . import constants, kepler, system, temperature  # noqa: F401
from . import mass_radius, structure, world, optimize  # noqa: F401
from . import nbody  # noqa: F401
try:  # star module needs downloaded MIST grids at runtime, not import time
    from . import stars, star_evolution  # noqa: F401
except Exception:  # pragma: no cover
    pass
