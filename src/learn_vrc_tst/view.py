"""View functions."""

import py3Dmol
from automol import Geometry

from .geom import xyz_string

View = py3Dmol.view


def empty(width: int = 400, height: int | None = None) -> View:
    """Create empty view.

    Parameters
    ----------
    width
        Width of the image
    height
        Height of the image

    Returns
    -------
        Empty py3Dmol view object
    """
    height = width if height is None else height
    return py3Dmol.view(width=width, height=height)


def geometry(geo: Geometry, view: View | None = None) -> View:
    """Create a view with a geometry.

    Parameters
    ----------
    geo
        The molecular geometry.
    view
        The py3Dmol view object to add the geometry to.

    Returns
    -------
        py3Dmol view object with the geometry added
    """
    view = empty() if view is None else view
    xyz_str = xyz_string(geo)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.zoomTo()
    return view
