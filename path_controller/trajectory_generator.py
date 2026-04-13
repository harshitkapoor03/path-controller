# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""
Trajectory Generator Module
============================
Converts a smooth spatial path (numpy array of x, y points) into a
time-parameterised trajectory — a list of (x, y, t) tuples — by assigning
a timestamp to every point.

Two velocity profiles are supported:

  constant    — robot moves at a fixed speed throughout.  Simple and fast
                to compute.  Useful for benchmarking and unit tests.

  trapezoidal — robot accelerates from rest, cruises at v_max, then
                decelerates to a stop.  This is the physically correct
                profile for a real differential drive robot: it avoids
                instantaneous velocity jumps that would cause wheel slip
                and stress the motor drivers.  This is the default.

The module has no ROS2 dependency and can be tested in isolation.
"""

import numpy as np


def generate_trajectory(
    smooth_path: np.ndarray,
    velocity: float = 0.3,
    profile: str = 'trapezoidal',
) -> list:
    """
    Assign timestamps to every point on a smooth path.

    Args:
        smooth_path: numpy array of shape (N, 2) — the output of smooth_path().
        velocity:    Target cruise speed in m/s.  For 'constant' this is the
                     fixed speed throughout.  For 'trapezoidal' this is the
                     peak cruise speed; actual speed is lower during ramps.
        profile:     'trapezoidal' (default) or 'constant'.  Any other string
                     silently falls back to 'constant' for robustness.

    Returns:
        List of N tuples (x, y, t) where t is the elapsed time in seconds
        at which the robot should be at (x, y).  t[0] is always 0.0 and
        the sequence is strictly monotone increasing.

    Note:
        The (x, y) coordinates are copied directly from smooth_path — this
        function only adds timing, it never modifies geometry.
    """
    # Compute the Euclidean distance between consecutive path points.
    # These segment lengths are the foundation for all time calculations.
    diffs = np.diff(smooth_path, axis=0)                        # shape (N-1, 2)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))         # shape (N-1,)

    if profile == 'trapezoidal':
        timestamps = _trapezoidal_profile(segment_lengths, velocity)
    else:
        # Constant velocity: t = cumulative_arc_length / v
        # t[0] = 0.0 because cumulative starts at 0.
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        timestamps = cumulative / velocity

    # Package each path point with its timestamp into a (x, y, t) tuple.
    trajectory = [
        (float(smooth_path[i, 0]), float(smooth_path[i, 1]), float(timestamps[i]))
        for i in range(len(smooth_path))
    ]
    return trajectory


def _trapezoidal_profile(segment_lengths: np.ndarray, v_max: float) -> np.ndarray:
    """
    Compute timestamps for a trapezoidal velocity profile.

    The profile has three phases:
      1. Acceleration — robot speeds up from ~0 to v_max over the first
         accel_dist metres using constant acceleration.
      2. Cruise        — robot moves at v_max.
      3. Deceleration  — robot slows from v_max to ~0 over the last
         accel_dist metres (same distance as acceleration, by symmetry).

    The acceleration distance is capped at 20% of total path length so
    that even short paths get a sensible ramp without the phases overlapping.

    Args:
        segment_lengths: 1-D array of distances between consecutive path points.
        v_max:           Target cruise speed in m/s.

    Returns:
        1-D array of timestamps, same length as segment_lengths + 1.
        timestamps[0] is always 0.0.

    Physics:
        Under constant acceleration a from rest:
            v(d) = sqrt(2 * a * d)
            t(d) = v(d) / a  (not used directly; we use dt = ds / v)
        We evaluate v at the START of each segment and compute dt = ds / v.
        A floor of 0.05 m/s prevents division-by-zero at d=0.
    """
    accel = 0.5          # m/s² — constant acceleration / deceleration rate
    total = segment_lengths.sum()

    # Acceleration distance: use kinematic formula v² = 2*a*d → d = v²/(2a),
    # but cap at 20% of total path so short paths don't have overlapping phases.
    accel_dist = min(0.2 * total, (v_max ** 2) / (2.0 * accel))

    # cumulative[i] is the arc-length from the start to the i-th path point.
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    timestamps = np.zeros(len(cumulative))

    for i in range(1, len(cumulative)):
        seg = cumulative[i - 1]   # arc-length at the START of this segment

        if seg < accel_dist:
            # Acceleration phase: v grows with sqrt(2*a*d).
            # Floor at 0.05 m/s to avoid zero velocity at d=0.
            v = max(0.05, np.sqrt(2.0 * accel * seg)) if seg > 0 else 0.05

        elif seg > total - accel_dist:
            # Deceleration phase: symmetric with acceleration.
            # remaining = distance left to travel.
            remaining = total - seg
            v = max(0.05, np.sqrt(2.0 * accel * remaining))

        else:
            # Cruise phase: constant speed.
            v = v_max

        # Time to traverse this segment at the local velocity.
        dt = segment_lengths[i - 1] / v
        timestamps[i] = timestamps[i - 1] + dt

    return timestamps
