# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""Tests for path_controller.path_smoother.

WHAT is being tested
--------------------
smooth_path() takes a list of raw (x, y) waypoints and returns a dense
numpy array of smoothly interpolated points using cubic spline interpolation
with chord-length parameterisation.

WHY these tests exist
---------------------
If the smoother produces wrong shapes, NaNs, misses waypoints, or overshoots
on uneven input spacing, every downstream module silently operates on garbage.
These tests catch all of those failures before they propagate.

HOW the tests are organised
----------------------------
Four test classes, each targeting one specific contract:
  TC-SM-1  Output contract  (shape, type, no NaN/Inf)
  TC-SM-2  Interpolation accuracy  (passes through waypoints)
  TC-SM-3  Geometric properties  (straight lines, continuity, no overshoot)
  TC-SM-4  Error handling  (bad input raises loudly)
"""

import numpy as np
import pytest

from path_controller.path_smoother import smooth_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_waypoints():
    """Gentle S-curve — the standard happy-path input."""
    return [
        (0.0, 0.0), (2.0, 1.0), (4.0, 0.0),
        (6.0, 1.0), (8.0, 0.0), (10.0, 1.0), (12.0, 0.0),
    ]


@pytest.fixture
def collinear_waypoints():
    """Three points on a straight horizontal line."""
    return [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]


@pytest.fixture
def clustered_waypoints():
    """Dense cluster at start, sparse middle, dense cluster at end.

    This is the hardest input for a naive uniform-parameter spline because
    it overshoots badly at the dense clusters.  Chord-length parameterisation
    must prevent that.
    """
    return [
        (0.0, 0.0), (0.2, 0.1), (0.5, 0.3), (1.0, 1.0),
        (5.0, 1.5), (9.0, 1.0),
        (9.5, 1.3), (9.8, 1.7), (10.0, 3.0),
    ]


@pytest.fixture
def smooth_sine(sine_waypoints):
    """Pre-computed 500-point smooth path from the sine waypoints."""
    return smooth_path(sine_waypoints)


# ---------------------------------------------------------------------------
# TC-SM-1  Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    """smooth_path must return the exact array shape and type promised."""

    def test_default_shape_is_500_by_2(self, smooth_sine):
        """Default num_points=500 must produce shape (500, 2).

        HOW: call with default args, check .shape.
        WHY: downstream modules assume exactly this shape; a wrong shape
        crashes the trajectory generator immediately.
        """
        assert smooth_sine.shape == (500, 2), \
            f'Expected (500, 2), got {smooth_sine.shape}'

    def test_custom_num_points_respected(self, sine_waypoints):
        """Any num_points value must be reflected exactly in output length.

        HOW: call with several values (50, 100, 250, 750, 1000), check shape.
        WHY: the ROS2 node passes num_points=500 explicitly; if the function
        ignores it the controller gets a path of unexpected density.
        """
        for n in [50, 100, 250, 750, 1000]:
            result = smooth_path(sine_waypoints, num_points=n)
            assert result.shape == (n, 2), \
                f'num_points={n} produced shape {result.shape}'

    def test_output_is_numpy_array(self, smooth_sine):
        """Return type must be ndarray, not list or tuple.

        HOW: isinstance check.
        WHY: trajectory_generator calls np.diff() on the result directly;
        a Python list would cause an AttributeError.
        """
        assert isinstance(smooth_sine, np.ndarray)

    def test_no_nan_in_output(self, smooth_sine):
        """Output must contain no NaN values.

        HOW: np.any(np.isnan(...)).
        WHY: NaN poisons every arithmetic operation — the controller would
        compute NaN velocities and the robot would never move.
        """
        assert not np.any(np.isnan(smooth_sine)), \
            'Smooth path contains NaN'

    def test_no_inf_in_output(self, smooth_sine):
        """Output must contain no Inf values.

        HOW: np.any(np.isinf(...)).
        WHY: Same reason as NaN — Inf in a position corrupts all geometry.
        """
        assert not np.any(np.isinf(smooth_sine)), \
            'Smooth path contains Inf'


# ---------------------------------------------------------------------------
# TC-SM-2  Interpolation accuracy
# ---------------------------------------------------------------------------

class TestInterpolationAccuracy:
    """The spline must pass through or very near every input waypoint."""

    def test_first_point_matches_first_waypoint(self, smooth_sine, sine_waypoints):
        """First output point must exactly coincide with first waypoint.

        HOW: np.allclose with atol=1e-6.
        WHY: the simulator starts at waypoints[0]; if the path starts
        somewhere else the robot begins off-track.
        """
        assert np.allclose(smooth_sine[0], np.array(sine_waypoints[0]), atol=1e-6)

    def test_last_point_matches_last_waypoint(self, smooth_sine, sine_waypoints):
        """Last output point must exactly coincide with last waypoint.

        HOW: np.allclose with atol=1e-6.
        WHY: goal arming checks distance to path_points[-1]; if that point
        is wrong the robot stops at the wrong location.
        """
        assert np.allclose(smooth_sine[-1], np.array(sine_waypoints[-1]), atol=1e-6)

    def test_all_waypoints_within_5cm_of_path(self, sine_waypoints):
        """Every input waypoint must lie within 0.05 m of some smooth-path point.

        HOW: for each waypoint compute min distance to all 500 path points.
        WHY: if an intermediate waypoint is not represented, the robot may
        miss critical turns or goal locations entirely.
        """
        result = smooth_path(sine_waypoints, num_points=500)
        for wp in sine_waypoints:
            wp_arr = np.array(wp)
            min_dist = np.linalg.norm(result - wp_arr, axis=1).min()
            assert min_dist < 0.05, \
                f'Waypoint {wp} is {min_dist:.4f} m from nearest path point'


# ---------------------------------------------------------------------------
# TC-SM-3  Geometric properties
# ---------------------------------------------------------------------------

class TestGeometricProperties:
    """The path must obey geometric constraints enforced by chord-length splines."""

    def test_straight_line_no_lateral_deviation(self, collinear_waypoints):
        """On collinear waypoints the path must not deviate sideways.

        HOW: smooth three y=0 waypoints, check that all y values are ~0.
        WHY: a uniform-parameter spline on unevenly-spaced collinear points
        can produce a non-zero lateral arc.  Chord-length parameterisation
        prevents this.  If it regresses the controller steers sideways for
        no reason.
        """
        result = smooth_path(collinear_waypoints, num_points=200)
        max_lateral = np.abs(result[:, 1]).max()
        assert max_lateral < 1e-4, \
            f'Lateral deviation {max_lateral:.6f} m on straight-line input'

    def test_straight_line_x_is_monotone(self, collinear_waypoints):
        """X must increase monotonically on a left-to-right straight path.

        HOW: np.diff on the x column, check all values >= 0.
        WHY: a regressing x means the spline doubled back, which would make
        the controller drive the robot backwards.
        """
        result = smooth_path(collinear_waypoints, num_points=200)
        diffs = np.diff(result[:, 0])
        assert np.all(diffs >= -1e-9), \
            f'X went backwards at index {int(np.argmin(diffs))}'

    def test_path_is_spatially_continuous(self, smooth_sine):
        """No two consecutive path points may be more than 0.1 m apart.

        HOW: np.diff on the path array, take norms, check max.
        WHY: a large gap means the spline was evaluated at non-contiguous
        parameter values — a teleport.  The controller's lookahead search
        would skip the gap and mis-target.
        """
        gaps = np.linalg.norm(np.diff(smooth_sine, axis=0), axis=1)
        assert gaps.max() < 0.1, \
            f'Max consecutive gap {gaps.max():.4f} m — teleport detected'

    def test_chord_length_prevents_overshoot_on_clustered_input(
        self, clustered_waypoints
    ):
        """Chord-length parameterisation keeps the path inside a 2 m bounding box.

        HOW: smooth clustered waypoints, check that the path bounding box
        is no more than 2 m wider than the waypoint bounding box.
        WHY: with uniform parameterisation the spline overshoots wildly at
        the dense clusters because it treats all gaps as equal length.
        Chord-length scaling by actual physical distance prevents this.
        This is the key reason chord-length was chosen over uniform indexing.
        """
        wps = np.array(clustered_waypoints)
        result = smooth_path(clustered_waypoints, num_points=500)
        margin = 2.0
        assert result[:, 0].min() >= wps[:, 0].min() - margin
        assert result[:, 0].max() <= wps[:, 0].max() + margin
        assert result[:, 1].min() >= wps[:, 1].min() - margin
        assert result[:, 1].max() <= wps[:, 1].max() + margin


# ---------------------------------------------------------------------------
# TC-SM-4  Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """The function must fail loudly on invalid input rather than silently."""

    def test_raises_on_single_waypoint(self):
        """A single waypoint provides nothing to interpolate — must raise.

        HOW: call with one-element list, expect any Exception.
        WHY: better to crash early with a clear error than to produce a
        degenerate path silently and crash the robot 10 seconds later.
        """
        with pytest.raises(Exception):
            smooth_path([(0.0, 0.0)])

    def test_raises_on_empty_input(self):
        """Empty input must raise rather than return empty silently.

        HOW: call with [], expect any Exception.
        WHY: same as above — fail fast, fail loud.
        """
        with pytest.raises(Exception):
            smooth_path([])

    def test_two_waypoints_is_minimum_valid(self):
        """Two waypoints is the minimum legal call and must succeed.

        HOW: call with exactly two points, check output shape.
        WHY: the path from A to B with only two waypoints is a valid
        input (a straight line in parameter space).
        """
        result = smooth_path([(0.0, 0.0), (5.0, 5.0)], num_points=100)
        assert result.shape == (100, 2)
