"""E-commerce experimentation and propensity platform."""

from ecom.env import configure_local_dirs

__version__ = "0.1.0"

# Route temp/spill/cache directories into the project before any heavy library loads.
configure_local_dirs()
