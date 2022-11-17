"""Utility for simulations."""

from numpy.random import Generator, default_rng


def validate_rng(rng=None):
    """Validate Random Number Generator."""
    if isinstance(rng, int):
        return default_rng(rng)
    elif rng is None:
        return default_rng()
    elif isinstance(rng, Generator):
        return rng
    else:
        raise ValueError("rng shoud be a numpy Generator, None or an integer seed.")
