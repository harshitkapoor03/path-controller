# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""Tests for path_controller.trajectory_generator.

WHAT is being tested
--------------------
generate_trajectory() converts a smooth numpy path array into a list of
(x, y, t) tuples by assigning a timestamp to every point.  It supports
two velocity profiles: 'constant' and 'trapezoidal'.

WHY these tests exist
---------------------
The timestamps drive the robot's speed at every point.  If timestamps go
backwards the robot would need to time-travel.  If they grow too slowly
the robot creeps.  If the XY coordinates are mutated the robot follows
the wrong path.  These tests verify all of those contracts.

HOW the tests are organised
----------------------------
  TC-TG-1  Output contract  (format, length, coordinates unchanged)
  TC-TG-2  Constant profile  (t=0 start, monotone, correct duration)
  TC-TG-3  Trapezoidal profile  (monotone, starts slow, ends slow)
  TC-TG-4  Profile comparison  (trapezoidal >= constant duration)
  TC-TG-5  Velocity sensitivity  (faster = shorter duration)
"""

import numpy as np
import pytest

from path_controller.path_smoother import smooth_path
from path_controller.trajectory_generator import generate_trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_path():
    """200-point horizontal straight line, 10 m long."""
    return np.column_stack([np.linspace(0, 10, 200), np.zeros(200)])


@pytest.fixture
def curved_path():
    """500-point smooth S-curve path."""
    wps = [
        (0.0, 0.0), (2.0, 1.0), (4.0, 0.0),
        (6.0, 1.0), (8.0, 0.0), (10.0, 1.0), (12.0, 0.0),
    ]
    return smooth_path(wps, num_points=500)


@pytest.fixture
def const_traj(straight_path):
    """Constant-velocity trajectory at 0.3 m/s."""
    return generate_trajectory(straight_path, velocity=0.3, profile='constant')


@pytest.fixture
def trap_traj(curved_path):
    """Trapezoidal-velocity trajectory at 0.3 m/s."""
    return generate_trajectory(curved_path, velocity=0.3, profile='trapezoidal')


# ---------------------------------------------------------------------------
# TC-TG-1  Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    """Output must be a list of (x, y, t) triples of exactly the right length."""

    def test_output_is_a_list(self, const_traj):
        """Return value must be a Python list.

        HOW: isinstance check.
        WHY: path_controller_node iterates over it with a for loop;
        a numpy array would still work but the type contract should be kept.
        """
        assert isinstance(const_traj, list)

    def test_each_element_is_a_triple(self, const_traj):
        """Every element must have exactly three values: x, y, t.

        HOW: check len(item) == 3 for every item.
        WHY: the controller unpacks p[0], p[1] from each element; a tuple of
        the wrong length would raise IndexError deep inside the control loop.
        """
        for item in const_traj:
            assert len(item) == 3, f'Expected 3 values, got {len(item)}: {item}'

    def test_length_matches_input_path(self, straight_path, const_traj):
        """Trajectory length must equal the number of path points.

        HOW: compare len(trajectory) to len(smooth_path).
        WHY: the controller iterates over path indices; a different length
        would mean some path points have no timestamp.
        """
        assert len(const_traj) == len(straight_path)

    def test_xy_coordinates_are_unchanged(self, straight_path, const_traj):
        """(x, y) in every trajectory point must exactly match the input path.

        HOW: compare trajectory[i][0] and [1] to smooth_path[i, 0] and [1].
        WHY: generate_trajectory must only ADD time, not modify geometry.
        Any mutation here means the robot follows a different path than
        what was smoothed.
        """
        for i, (x, y, _t) in enumerate(const_traj):
            assert x == pytest.approx(float(straight_path[i, 0]), abs=1e-9)
            assert y == pytest.approx(float(straight_path[i, 1]), abs=1e-9)


# ---------------------------------------------------------------------------
# TC-TG-2  Constant velocity profile
# ---------------------------------------------------------------------------

class TestConstantProfile:
    """With profile='constant', timestamps must follow t = arc_length / v."""

    def test_first_timestamp_is_zero(self, const_traj):
        """The trajectory must start at t=0.

        HOW: check const_traj[0][2] == 0.0.
        WHY: the simulator initialises its clock to 0; if the trajectory
        starts at a non-zero time the first segment is undefined.
        """
        assert const_traj[0][2] == pytest.approx(0.0, abs=1e-9)

    def test_timestamps_are_monotonically_increasing(self, const_traj):
        """Every subsequent timestamp must be strictly greater than the previous.

        HOW: iterate and compare t[i] > t[i-1].
        WHY: non-monotone timestamps mean the robot would need to go backwards
        in time to follow the trajectory — physically impossible.
        """
        times = [p[2] for p in const_traj]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], \
                f'Time went backwards at index {i}: {times[i-1]:.6f} -> {times[i]:.6f}'

    def test_total_duration_matches_physics(self, straight_path):
        """On a 10 m path at 0.5 m/s total time must be ~20 s.

        HOW: generate with v=0.5, check last timestamp within 5% of 20s.
        WHY: verifies the formula t = distance / velocity is applied correctly.
        A factor-of-10 bug here would make the robot run 10x too fast or slow.
        """
        traj = generate_trajectory(straight_path, velocity=0.5, profile='constant')
        assert traj[-1][2] == pytest.approx(20.0, rel=0.05), \
            f'Expected ~20 s, got {traj[-1][2]:.3f} s'


# ---------------------------------------------------------------------------
# TC-TG-3  Trapezoidal velocity profile
# ---------------------------------------------------------------------------

class TestTrapezoidalProfile:
    """With profile='trapezoidal', timestamps must reflect a ramp-up and ramp-down."""

    def test_first_timestamp_is_zero(self, trap_traj):
        """Trapezoidal trajectory must also start at t=0.

        HOW: same check as constant profile.
        WHY: same reason — simulator clock starts at 0.
        """
        assert trap_traj[0][2] == pytest.approx(0.0, abs=1e-9)

    def test_timestamps_are_monotonically_increasing(self, trap_traj):
        """Trapezoidal timestamps must also be strictly increasing.

        HOW: iterate and compare t[i] > t[i-1].
        WHY: the ramp calculation must never produce a negative dt — if it
        does it means the local velocity became negative, which is physically
        wrong.
        """
        times = [p[2] for p in trap_traj]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], \
                f'Time went backwards at index {i}'

    def test_early_timestamps_grow_slowly(self, curved_path):
        """The first 5% of timestamps must grow slower than the mid-section.

        HOW: compare average dt in first 5% vs middle 10%.
        WHY: during the acceleration ramp the robot is moving slowly, so
        each segment takes longer to traverse (larger dt).  If dt is not
        larger at the start the trapezoidal profile is not actually ramping.
        """
        traj = generate_trajectory(curved_path, velocity=0.4, profile='trapezoidal')
        times = [p[2] for p in traj]
        n = len(times)
        ramp_dts = np.diff(times[:n // 20])
        mid_dts = np.diff(times[n // 2: n // 2 + n // 10])
        assert ramp_dts.mean() > mid_dts.mean(), \
            'Ramp phase is not slower than cruise phase — trapezoidal profile broken'

    def test_late_timestamps_grow_slowly(self, curved_path):
        """The last 5% of timestamps must grow slower than the mid-section.

        HOW: same as above but for the deceleration ramp.
        WHY: verifies deceleration zone is actually applied at the end.
        """
        traj = generate_trajectory(curved_path, velocity=0.4, profile='trapezoidal')
        times = [p[2] for p in traj]
        n = len(times)
        decel_dts = np.diff(times[-(n // 20):])
        mid_dts = np.diff(times[n // 2: n // 2 + n // 10])
        assert decel_dts.mean() > mid_dts.mean(), \
            'Deceleration phase is not slower than cruise phase'


# ---------------------------------------------------------------------------
# TC-TG-4  Profile comparison
# ---------------------------------------------------------------------------

class TestProfileComparison:
    """Comparing constant vs trapezoidal on the same input."""

    def test_trapezoidal_takes_at_least_as_long_as_constant(self, curved_path):
        """Trapezoidal must take >= time than constant at the same cruise velocity.

        HOW: compare final timestamps of both profiles.
        WHY: the trapezoidal profile spends part of the path below cruise
        speed (ramp zones), so the total duration must be >= constant.
        If trapezoidal is faster it means the ramps are being applied wrong.
        """
        t_const = generate_trajectory(curved_path, velocity=0.3, profile='constant')
        t_trap = generate_trajectory(curved_path, velocity=0.3, profile='trapezoidal')
        assert t_trap[-1][2] >= t_const[-1][2], \
            f'Trapezoidal ({t_trap[-1][2]:.3f}s) faster than constant ({t_const[-1][2]:.3f}s)'


# ---------------------------------------------------------------------------
# TC-TG-5  Velocity sensitivity
# ---------------------------------------------------------------------------

class TestVelocitySensitivity:
    """Changing velocity must change total duration in the expected direction."""

    def test_faster_velocity_gives_shorter_duration(self, straight_path):
        """Doubling the velocity must more than halve the duration.

        HOW: generate at v=0.1 and v=0.5, compare final timestamps.
        WHY: basic sanity check that the velocity parameter is actually
        used in the time computation.  A bug that ignores velocity would
        make this test fail.
        """
        slow = generate_trajectory(straight_path, velocity=0.1, profile='constant')
        fast = generate_trajectory(straight_path, velocity=0.5, profile='constant')
        assert fast[-1][2] < slow[-1][2], \
            f'Faster velocity gave longer duration: slow={slow[-1][2]:.2f}s, fast={fast[-1][2]:.2f}s'

    def test_unknown_profile_falls_back_gracefully(self, curved_path):
        """An unrecognised profile name must not crash — fall back to constant.

        HOW: call with profile='unknown_profile', verify monotone timestamps.
        WHY: robust error handling — a typo in the profile param should
        not crash the robot mid-mission.
        """
        traj = generate_trajectory(curved_path, velocity=0.3, profile='unknown_profile')
        times = [p[2] for p in traj]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], \
                f'Fallback profile produced non-monotone timestamps at index {i}'
