"""Rotation functions."""

import numpy as np
from numpy.random import Generator
from scipy.spatial.transform import Rotation


def uniform_random_unit_quaternion(
    rng: Generator | None = None,
) -> tuple[float, float, float, float]:
    """Generate a uniform random quaternion.

    Parameters
    ----------
        Random number generator.

    Returns
    -------
        A tuple representing a quaternion (q0, q1, q2, q3).
    """
    rng = np.random.default_rng() if rng is None else rng

    u1 = rng.uniform(0.0, 1.0)
    u2 = rng.uniform(0.0, 1.0)
    u3 = rng.uniform(0.0, 1.0)

    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return (q0, q1, q2, q3)


def uniform_random_rotation(rng: Generator | None = None) -> Rotation:
    """Generate a uniform random rotation matrix.

    Parameters
    ----------
        Random number generator.

    Returns
    -------
        A 3x3 rotation matrix.
    """
    q = uniform_random_unit_quaternion(rng=rng)
    return Rotation.from_quat(q)
