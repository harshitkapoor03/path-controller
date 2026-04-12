"""
Trajectory Generator Module
Converts a smooth path into a time-parameterized trajectory.
Supports constant velocity and trapezoidal velocity profiles.
"""

import numpy as np


def generate_trajectory(
    smooth_path: np.ndarray,
    velocity: float = 0.3,
    profile: str = "trapezoidal"
) -> list:
    """
    Assigns timestamps to each point on the smooth path.

    Args:
        smooth_path: numpy array of (x, y) points
        velocity: target cruise velocity in m/s
        profile: 'constant' or 'trapezoidal'

    Returns:
        List of (x, y, t) tuples
    """
    diffs = np.diff(smooth_path, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))

    if profile == "trapezoidal":
        timestamps = _trapezoidal_profile(segment_lengths, velocity)
    else:
        cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
        timestamps = cumulative / velocity

    trajectory = [
        (float(smooth_path[i, 0]), float(smooth_path[i, 1]), float(timestamps[i]))
        for i in range(len(smooth_path))
    ]
    return trajectory


def _trapezoidal_profile(segment_lengths: np.ndarray, v_max: float) -> np.ndarray:
    """
    Trapezoidal velocity profile: accelerate -> cruise -> decelerate.
    Robot smoothly speeds up and slows down instead of instant jumps.
    """
    accel = 0.5  # m/s^2
    total = segment_lengths.sum()
    accel_dist = min(0.2 * total, (v_max ** 2) / (2 * accel))

    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    timestamps = np.zeros(len(cumulative))

    for i in range(1, len(cumulative)):
        seg = cumulative[i - 1]

        if seg < accel_dist:
            v = max(0.05, np.sqrt(2 * accel * seg)) if seg > 0 else 0.05
        elif seg > total - accel_dist:
            remaining = total - seg
            v = max(0.05, np.sqrt(2 * accel * remaining))
        else:
            v = v_max

        dt = segment_lengths[i - 1] / v
        timestamps[i] = timestamps[i - 1] + dt

    return timestamps
