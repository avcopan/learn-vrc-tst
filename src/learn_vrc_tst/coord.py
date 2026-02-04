"""Cartesian coordinate functions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def closest_unit_perpendicular(
    coords: ArrayLike, *, away: bool = False
) -> NDArray[np.float64]:
    """Get the closest unit perpendicular vector to a set of coordinates.

    Parameters
    ----------
    coords
        Cartesian coordinates.
    away
        If True, return perpendicular coordinate pointing away from coordinates.

    Returns
    -------
        Closest unit perpendicular coordinate.
    """
    coords = np.asarray(coords)
    _, _, vt = np.linalg.svd(coords, full_matrices=False)
    perp = vt[-1]

    is_away = np.sum(np.matmul(coords, perp)) < 0
    if is_away ^ away:
        perp = -perp

    return perp
