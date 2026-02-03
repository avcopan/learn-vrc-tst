"""Geometry functions."""

import numpy as np
from automol import Geometry
from numpy.typing import ArrayLike


def translate(geo: Geometry, arr: ArrayLike, *, in_place: bool = False) -> Geometry:
    """Translate a geometry by a given vector.

    Parameters
    ----------
    geo
        Molecular geometry.
    arr
        Translation vector or matrix.

    Returns
    -------
        Molecular geometry.
    """
    geo = geo if in_place else geo.model_copy()
    geo.coordinates = np.add(geo.coordinates, arr)
    return geo


def xyz_string(geo: Geometry) -> str:
    """Get an XYZ string from a geometry.

    Parameters
    ----------
    geo
        Molecular geometry.

    Returns
    -------
    str
        XYZ string representation of the geometry.
    """
    num = len(geo.symbols)
    return f"{num}\n\n" + "\n".join(
        f"{s} {x:10.6f} {y:10.6f} {z:10.6f}"
        for s, (x, y, z) in zip(geo.symbols, geo.coordinates, strict=True)
    )
