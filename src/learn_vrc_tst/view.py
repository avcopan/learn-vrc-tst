"""View functions."""

import numpy as np
import py3Dmol
from automol import Geometry
from numpy.typing import ArrayLike

from .geom import xyz_string


def empty(width: int = 400, height: int | None = None) -> py3Dmol.view:
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


def geometry(geo: Geometry, *, view: py3Dmol.view | None = None) -> py3Dmol.view:
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


def arrow(
    coord: ArrayLike,
    start_coord: ArrayLike = (0, 0, 0),
    *,
    direction: bool = False,
    color: str = "black",
    view: py3Dmol.view | None = None,
) -> py3Dmol.view:
    """Create a view with an arrow.

    Parameters
    ----------
    coord
        The arrow tip coordinates.
    start_coord
        The arrow start coordinates.
    direction
        If True, coord is treated as a direction vector from start_coord.
    color
        The arrow color.

    Returns
    -------
        py3Dmol view object with the arrow added
    """
    if direction:
        coord = np.add(coord, start_coord)

    start = np.asarray(start_coord).tolist()
    end = np.asarray(coord).tolist()

    arrow_spec = {
        "start": {"x": start[0], "y": start[1], "z": start[2]},
        "end": {"x": end[0], "y": end[1], "z": end[2]},
        "color": color,
    }
    view = empty() if view is None else view
    view.addArrow(arrow_spec)
    view.zoomTo()
    return view


class View:
    """Class for creating and displaying 3D molecular views."""

    def __init__(self, width: int = 400, height: int | None = None) -> None:
        """Initialize the View object."""
        self._view = empty(width=width, height=height)

    def add_geometry(self, geo: Geometry) -> None:
        """Add geometry to view."""
        self._view = geometry(geo, view=self._view)

    def add_arrow(
        self,
        coord: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        color: str = "black",
    ) -> None:
        """Add arrow to view."""
        self._view = arrow(coord, start_coord=start_coord, color=color, view=self._view)

    def show(self) -> None:
        """Show view."""
        self._view.show()
