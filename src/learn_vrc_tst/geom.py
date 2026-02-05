"""Geometry functions."""

import itertools

import numpy as np
import py3Dmol
from automol import Geometry
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


def concat(geos: list[Geometry]) -> Geometry:
    """Concatenate geometries.

    Parameters
    ----------
    geos
        List of geometries.

    Returns
    -------
        Geometry.
    """
    symbols = list(itertools.chain.from_iterable(geo.symbols for geo in geos))
    coordinates = np.vstack([geo.coordinates for geo in geos])
    charge = sum(geo.charge for geo in geos)
    spin = sum(geo.spin for geo in geos)
    return Geometry(symbols=symbols, coordinates=coordinates, charge=charge, spin=spin)


def translate(geo: Geometry, arr: ArrayLike, *, in_place: bool = False) -> Geometry:
    """Translate geometry.

    Parameters
    ----------
    geo
        Geometry.
    arr
        Translation vector or matrix.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy()
    geo.coordinates = np.add(geo.coordinates, arr)
    return geo


def rotate(geo: Geometry, rot: Rotation, *, in_place: bool = False) -> Geometry:
    """Rotate geometry.

    Parameters
    ----------
    geo
        Geometry.
    rot
        Rotation object.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy()
    geo.coordinates = rot.apply(geo.coordinates)
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


def view(geo: Geometry, *, view: py3Dmol.view | None = None) -> py3Dmol.view:
    """View a geometry with py3Dmol.

    Parameters
    ----------
    geo
        Geometry.
    view
        py3Dmol view.

    Returns
    -------
        py3Dmol view.
    """
    view = py3Dmol.view(width=400, height=400) if view is None else view
    xyz_str = xyz_string(geo)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    return view
