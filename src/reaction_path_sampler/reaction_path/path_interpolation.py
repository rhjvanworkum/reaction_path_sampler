from typing import Any

import numpy as np
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.interpolation import redistribute


def interpolate_geodesic(
    symbols: list[str], rc_coordinates: np.ndarray, pc_coordinates: np.ndarray, settings: Any
) -> Geodesic:
    X = [rc_coordinates, pc_coordinates]
    raw = redistribute(symbols, X, settings["nimages"], tol=settings["tol"])
    smoother = Geodesic(
        symbols,
        raw,
        settings["scaling"],
        threshold=settings["dist_cutoff"],
        friction=settings["friction"],
    )
    try:
        smoother.smooth(tol=settings["tol"], max_iter=settings["maxiter"])
    except Exception as e:
        print(e)

    return smoother
