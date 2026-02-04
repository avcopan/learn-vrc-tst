"""View functions."""

import numpy as np
import py3Dmol
from automol import Geometry
from numpy.typing import ArrayLike

from . import geom


class View(py3Dmol.view):
    """Class for creating and displaying 3D molecular views."""

    def add_geometry(self, geo: Geometry) -> None:
        """Add geometry to view.

        Parameters
        ----------
        geo
            Geometry.
        """
        geom.view(geo, view=self)

    def add_arrow(
        self,
        coord: ArrayLike,
        start_coord: ArrayLike = (0, 0, 0),
        *,
        direction: bool = False,
        color: str = "black",
    ) -> None:
        """Add arrow to view.

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
        self.addArrow(arrow_spec)
