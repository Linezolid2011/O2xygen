"""o2xygen - clean-room reconstruction of the ORBIT worldbuilding physics engine."""
from . import constants, kepler, system, temperature  # noqa: F401
from . import mass_radius, structure  # noqa: F401
try:  # star module needs downloaded MIST grids at runtime, not import time
    from . import stars, star_evolution  # noqa: F401
except Exception:  # pragma: no cover
    pass
