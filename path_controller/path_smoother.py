# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""
Path Smoother Module
====================
Converts a coarse list of 2D waypoints into a dense, smooth, continuous
curve using cubic spline interpolation with chord-length parameterisation.

Why cubic splines?
------------------
Cubic splines guarantee C2 continuity — position, velocity, AND curvature
are all smooth.  This matters for a differential drive robot because
discontinuous curvature would require instantaneous changes in angular
velocity, which is physically impossible.

Why chord-length parameterisation?
-----------------------------------
The naive approach indexes waypoints uniformly (0, 1, 2, ...).  This causes
the spline to overshoot wildly when waypoints are unevenly spaced because it
treats a 0.1 m gap and a 5.0 m gap as equal steps.  Chord-length
parameterisation uses the actual physical arc-length between waypoints as
the parameter, so the spline behaves correctly regardless of spacing.

This module has no ROS2 dependency — it is pure Python/NumPy so it can be
unit-tested without a running ROS2 environment.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def smooth_path(waypoints: list, num_points: int = 500) -> np.ndarray:
    """
    Smooth a list of (x, y) waypoints into a dense continuous path.

    The function fits two independent cubic splines — one for x(t) and one
    for y(t) — where t is the cumulative chord-length distance along the
    waypoint sequence.  It then samples both splines at `num_points` evenly
    spaced values of t to produce the output array.

    Args:
        waypoints:  List of (x, y) tuples in metres.  Must contain at least
                    two points.  Points may be unevenly spaced.
        num_points: Number of output points to sample along the smooth curve.
                    Default 500 gives ~0.04 m spacing for a 20 m path, which
                    is sufficient for a 20 Hz control loop at 0.4 m/s.

    Returns:
        numpy array of shape (num_points, 2).  Row i is (x_i, y_i) in metres.
        The first row exactly equals waypoints[0]; the last row exactly equals
        waypoints[-1].

    Raises:
        ValueError: If fewer than two waypoints are provided.

    Example:
        >>> wps = [(0, 0), (1, 1), (2, 0), (3, 1)]
        >>> path = smooth_path(wps, num_points=200)
        >>> path.shape
        (200, 2)
    """
    if len(waypoints) < 2:
        raise ValueError(
            f'smooth_path requires at least 2 waypoints, got {len(waypoints)}.'
        )

    waypoints = np.array(waypoints, dtype=float)

    # --- Chord-length parameterisation ---
    # Compute the Euclidean distance between each consecutive pair of
    # waypoints and accumulate them into a monotone parameter array t.
    # t[0] = 0.0 (start), t[-1] = total arc length of the waypoint polyline.
    diffs = np.diff(waypoints, axis=0)                        # shape (n-1, 2)
    distances = np.sqrt((diffs ** 2).sum(axis=1))             # shape (n-1,)
    t = np.concatenate([[0.0], np.cumsum(distances)])         # shape (n,)

    # --- Fit independent splines for x(t) and y(t) ---
    # CubicSpline uses not-a-knot boundary conditions by default, which
    # gives the smoothest possible curve at the endpoints.
    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])

    # --- Sample at evenly spaced parameter values ---
    # linspace gives uniform spacing in arc-length, NOT in Cartesian space.
    # This means output points are approximately equidistant along the curve,
    # which is what the trajectory generator and controller expect.
    t_fine = np.linspace(0.0, t[-1], num_points)
    x_smooth = cs_x(t_fine)
    y_smooth = cs_y(t_fine)

    return np.column_stack([x_smooth, y_smooth])
