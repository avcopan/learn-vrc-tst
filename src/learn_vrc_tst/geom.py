"""Geometry functions."""

import itertools
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pint
import py3Dmol
import pyparsing as pp
from automol import Geometry
from automol.geom import center_of_mass
from numpy.typing import ArrayLike
from pyparsing import pyparsing_common as ppc
from scipy.spatial.transform import Rotation

RADIANS_TO_DEGREES = pint.Quantity("radian").m_as("degree")
DEGREES_TO_RADIANS = 1 / RADIANS_TO_DEGREES


def inertia_tensor(geo: Geometry) -> np.ndarray:
    """Calculate the inertia tensor of a geometry.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Inertia tensor.
    """
    masses = geo.masses
    coords = geo.coordinates - center_of_mass(geo)
    return sum(
        m * (np.vdot(r, r) * np.eye(3) - np.outer(r, r))
        for (r, m) in zip(coords, masses, strict=True)
    )


def rotational_analysis(geo: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Calculate rotational analysis of a geometry.

    Parameters
    ----------
    geo
        Geometry.
    drop_null
        Whether to drop null eigenvalues.

    Returns
    -------
        Eigenvalues and eigenvectors of the inertia tensor.
    """
    inert = inertia_tensor(geo)
    evals, evecs = np.linalg.eigh(inert)
    # Ensure right-handed coordinate system
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1  # flip one eigenvector
    return evals, evecs


def rotation_to_inertial_frame(geo: Geometry) -> Rotation:
    """Return a rotation that aligns the geometry with its principal axes.

    Parameters
    ----------
    geo
        Geometry.

    Returns
    -------
        Rotation object.
    """
    _, evecs = rotational_analysis(geo)
    return Rotation.from_matrix(evecs.T)


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


def translate(
    geo: Geometry,
    arr: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
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
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = np.add(geo.coordinates[mask], arr)
    return geo


def reflect(
    geo: Geometry,
    normal: ArrayLike,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Reflect geometry across a plane.

    Parameters
    ----------
    geo
        Geometry.
    normal
        Normal vector of the reflection plane.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    normal = np.asarray(normal, dtype=float)
    proj = np.outer(normal, normal) / np.dot(normal, normal)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = geo.coordinates[mask] - 2 * geo.coordinates[mask] @ proj
    return geo


def rotate(
    geo: Geometry,
    rot: Rotation,
    *,
    keys: Collection[int] | None = None,
    in_place: bool = False,
) -> Geometry:
    """Rotate geometry.

    Parameters
    ----------
    geo
        Geometry.
    rot
        Rotation object.
    keys
        Atoms to rotate. If None, rotate all atoms.
    in_place
        Whether to rotate in place or return a new geometry.

    Returns
    -------
        Geometry.
    """
    geo = geo if in_place else geo.model_copy(deep=True)
    mask = slice(None) if keys is None else list(keys)
    geo.coordinates[mask] = rot.apply(geo.coordinates[mask])
    return geo


def dihedral_angle(
    geo: Geometry, keys: Sequence[int], *, degrees: bool = False
) -> float:
    """Calculate the dihedral angle defined by four atoms.

    Parameters
    ----------
    geo
        Geometry.
    keys
        Indices of the four atoms defining the dihedral angle.
    degrees
        Whether to return the angle in degrees or radians.

    Returns
    -------
        Dihedral angle.
    """
    coords = geo.coordinates[list(keys)]
    if len(coords) != 4:  # noqa: PLR2004
        msg = "Exactly four atoms must be specified for dihedral angle."
        raise ValueError(msg)

    # Determine bond vectors and 1-2-3 plane normal
    r1, r2, r3, r4 = coords
    r12 = r2 - r1
    r23 = r3 - r2
    r34 = r4 - r3
    n123 = np.cross(r12, r23)

    # Form coordinate system with x upward in plane, y along plane normal, and z
    # away along central bond:
    #
    #     x
    #     ^
    #     1
    #     |
    #     2/3   > y
    #      \
    #       4
    #
    z = r23 / np.linalg.norm(r23)
    y = n123 / np.linalg.norm(n123)
    x = np.cross(y, z)

    # Determine components of 3-4 bond along x and y and calculate angle from arctan
    v = r34 / np.linalg.norm(r34)
    vx = np.dot(v, x)
    vy = np.dot(v, y)
    angle = np.arctan2(vy, vx)
    return angle * RADIANS_TO_DEGREES if degrees else angle


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


CHAR = pp.Char(pp.alphas)
SYMBOL = pp.Combine(CHAR + pp.Opt(CHAR))
XYZ_LINE = SYMBOL + pp.Group(ppc.fnumber * 3) + pp.Suppress(... + pp.LineEnd())


def from_xyz_string(geo_str: str) -> Geometry:
    """Parse an XYZ string into a geometry.

    Parameters
    ----------
    xyz_str
        XYZ string.

    Returns
    -------
        Geometry.
    """
    geo_str = geo_str.strip()
    lines = geo_str.splitlines()[2:]
    if not lines:
        return Geometry(symbols=[], coordinates=[])  # ty:ignore[invalid-argument-type]

    symbs, coords = zip(
        *[XYZ_LINE.parse_string(line).as_list() for line in lines], strict=True
    )
    return Geometry(symbols=symbs, coordinates=np.array(coords))


def read_xyz_file(path: str | Path) -> Geometry:
    """Read a geometry from an XYZ file.

    Parameters
    ----------
    path
        Path to XYZ file.

    Returns
    -------
        Geometry.
    """
    path = path if isinstance(path, Path) else Path(path)
    return from_xyz_string(path.read_text())


def write_xyz_file(geo: Geometry, path: str | Path) -> None:
    """Write a geometry to an XYZ file.

    Parameters
    ----------
    geo
        Geometry.
    path
        Path to XYZ file.
    """
    path = path if isinstance(path, Path) else Path(path)
    path.write_text(xyz_string(geo))


def view(
    geo: Geometry, *, view: py3Dmol.view | None = None, label: bool = False
) -> py3Dmol.view:
    """View a geometry with py3Dmol.

    Parameters
    ----------
    geo
        Geometry.
    view
        py3Dmol view.
    label
        Whether to add atom labels to the view.

    Returns
    -------
        py3Dmol view.
    """
    view = py3Dmol.view(width=400, height=400) if view is None else view
    xyz_str = xyz_string(geo)
    view.addModel(xyz_str, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    if label:
        for key in range(len(geo.symbols)):
            view.addLabel(
                key,
                {
                    "backgroundOpacity": 0.0,
                    "fontColor": "black",
                    "alignment": "center",
                    "inFront": True,
                },
                {"index": key},
            )
    return view
