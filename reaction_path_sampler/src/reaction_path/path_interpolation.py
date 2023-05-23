import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any

from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic

def interpolate_geodesic(
    symbols: List[str],
    rc_coordinates: np.ndarray,
    pc_coordinates: np.ndarray,
    settings: Any
) -> Geodesic:
    X = [rc_coordinates, pc_coordinates]
    raw = redistribute(symbols, X, settings['nimages'], tol=settings['tol'])
    smoother = Geodesic(symbols, raw, settings['scaling'], threshold=settings['dist_cutoff'], friction=settings['friction'])
    try:
        smoother.smooth(tol=settings['tol'], max_iter=settings['maxiter'])
    except Exception as e:
        logging.debug(e)

    return smoother
