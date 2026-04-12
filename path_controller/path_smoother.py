"""
Path Smoother Module
Takes discrete 2D waypoints and returns a smooth continuous path
using cubic spline interpolation.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def smooth_path(waypoints: list, num_points: int = 500) -> np.ndarray:
    """
    Smooths a list of (x, y) waypoints using cubic spline interpolation.

    Args:
        waypoints: List of (x, y) tuples
        num_points: How many points to sample on the smooth path

    Returns:
        numpy array of shape (num_points, 2)
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints to smooth a path.")

    waypoints = np.array(waypoints)

    # Chord-length parameterization: use distance along path as parameter t
    diffs = np.diff(waypoints, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(distances)])

    # Fit cubic splines independently for x and y
    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])

    # Sample at evenly spaced parameter values
    t_fine = np.linspace(0, t[-1], num_points)
    x_smooth = cs_x(t_fine)
    y_smooth = cs_y(t_fine)

    return np.column_stack([x_smooth, y_smooth])
