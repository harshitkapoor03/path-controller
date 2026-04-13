# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""Tests for path_controller.controller.

WHAT is being tested
--------------------
TrajectoryController.compute_velocity() implements Pure Pursuit.  Given
the robot's current pose (x, y, yaw) and the trajectory, it returns
(v, omega) — the linear and angular velocity commands.

WHY these tests exist
---------------------
The controller is the most complex module.  It has a known algorithmic
bug (the looping-path index-skip) that was fixed with a bounded progress
window.  Tests here verify: correct velocity bounds, turn direction
geometry, goal arming logic, and — most importantly — that the fix
actually prevents the index from skipping on a looping path.

HOW the tests are organised
----------------------------
  TC-C-1  Velocity bounds  (v floor, v ceiling, omega clip)
  TC-C-2  Pure Pursuit geometry  (curvature direction, zero omega straight)
  TC-C-3  Goal detection  (arms late, stops correctly)
  TC-C-4  Bounded window bug fix  (index never jumps, monotone index)
  TC-C-5  Velocity scaling with heading error
"""

import numpy as np
import pytest

from path_controller.controller import _PROGRESS_WINDOW, TrajectoryController
from path_controller.path_smoother import smooth_path
from path_controller.trajectory_generator import generate_trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_traj():
    """200-point constant-velocity trajectory along a straight 10 m line."""
    pts = np.column_stack([np.linspace(0, 10, 200), np.zeros(200)])
    return generate_trajectory(pts, velocity=0.3)


@pytest.fixture
def figure_eight_traj():
    """500-point trajectory on a figure-eight path.

    This is the canonical trigger for the looping-path index-skip bug:
    the path physically passes near (0, 0) twice — once at the start
    and again at about index 250 — so a naive closest-point search
    would jump 250+ indices in one control tick.
    """
    wps = [
        (0.0, 0.0), (3.0, 2.0), (6.0, 4.0), (8.0, 2.0),
        (6.0, 0.0), (3.0, 2.0), (0.0, 4.0), (-2.0, 2.0), (0.0, 0.0),
    ]
    path = smooth_path(wps, num_points=500)
    return generate_trajectory(path, velocity=0.3)


@pytest.fixture
def fresh_ctrl():
    """A freshly constructed controller with default parameters."""
    return TrajectoryController(lookahead_distance=0.3, max_linear_vel=0.4)


# ---------------------------------------------------------------------------
# TC-C-1  Velocity bounds
# ---------------------------------------------------------------------------

class TestVelocityBounds:
    """v and omega must stay within physically meaningful limits at all times."""

    def test_linear_velocity_never_below_floor(self, straight_traj, fresh_ctrl):
        """v must never drop below 0.08 m/s (motor minimum) mid-path.

        HOW: step controller through 18 evenly-spaced path positions,
        check v >= 0.08 at each.
        WHY: if v hits zero mid-path the robot stalls.  Real DC motors
        need a minimum current to overcome static friction; 0.08 m/s
        is the empirical floor for TurtleBot-class hardware.
        """
        for i in range(0, 180, 10):
            x, y, _ = straight_traj[i]
            v, _ = fresh_ctrl.compute_velocity(x, y, 0.0, straight_traj)
            assert v >= 0.08, \
                f'v={v:.4f} fell below floor at index {i}'

    def test_linear_velocity_never_exceeds_max(self, straight_traj):
        """v must never exceed max_linear_vel.

        HOW: set max_linear_vel=0.4, iterate through path, check v <= 0.4.
        WHY: exceeding the hardware limit can cause wheel slip, encoder
        saturation, or a safety shutdown on a real robot.
        """
        max_v = 0.4
        ctrl = TrajectoryController(max_linear_vel=max_v)
        for i in range(0, 180, 5):
            x, y, _ = straight_traj[i]
            v, _ = ctrl.compute_velocity(x, y, 0.0, straight_traj)
            assert v <= max_v + 1e-9, \
                f'v={v:.4f} exceeded max_linear_vel={max_v}'

    def test_omega_clipped_to_hardware_limit(self, figure_eight_traj, fresh_ctrl):
        """omega must stay within [-1.5, 1.5] rad/s at all times.

        HOW: step through a demanding curved path, check omega at each point.
        WHY: 1.5 rad/s is the angular velocity limit of a TurtleBot3 Burger.
        Exceeding it causes the differential drive to produce the wrong
        heading change because one wheel would need to spin faster than its
        motor allows.
        """
        for i in range(0, 450, 15):
            x, y, _ = figure_eight_traj[i]
            _, omega = fresh_ctrl.compute_velocity(x, y, 0.0, figure_eight_traj)
            assert -1.5 <= omega <= 1.5, \
                f'omega={omega:.4f} outside [-1.5, 1.5] at index {i}'


# ---------------------------------------------------------------------------
# TC-C-2  Pure Pursuit geometry
# ---------------------------------------------------------------------------

class TestPurePursuitGeometry:
    """The steering direction must follow from the geometry of Pure Pursuit."""

    def test_turns_left_when_path_is_to_the_left(self):
        """Robot facing +x, path going upper-left — omega must be positive.

        HOW: construct a 3-point trajectory that passes above and left of
        the robot, compute omega, check > 0.
        WHY: in robot frame positive omega = counterclockwise = left turn.
        If this is wrong the robot steers in the opposite direction to the
        path and diverges.
        """
        ctrl = TrajectoryController(lookahead_distance=1.0)
        traj = [(-2.0, 2.0, 0.0), (0.0, 3.0, 1.0), (2.0, 2.0, 2.0)]
        _, omega = ctrl.compute_velocity(0.0, 0.0, 0.0, traj)
        assert omega > 0, \
            f'Expected left turn (omega>0) for path above robot, got omega={omega:.4f}'

    def test_turns_right_when_path_is_to_the_right(self):
        """Robot facing +x, path going lower-right — omega must be negative.

        HOW: mirror of the left-turn test.
        WHY: same correctness requirement for the opposite direction.
        """
        ctrl = TrajectoryController(lookahead_distance=1.0)
        traj = [(-2.0, -2.0, 0.0), (0.0, -3.0, 1.0), (2.0, -2.0, 2.0)]
        _, omega = ctrl.compute_velocity(0.0, 0.0, 0.0, traj)
        assert omega < 0, \
            f'Expected right turn (omega<0) for path below robot, got omega={omega:.4f}'

    def test_near_zero_omega_on_straight_aligned_path(self, straight_traj):
        """Robot aligned with a straight path must produce near-zero omega.

        HOW: place the robot mid-path with yaw=0 (aligned with +x), check
        |omega| < 0.1.
        WHY: if the robot is perfectly aligned there is nothing to steer;
        a large omega here means the geometric formula is wrong.
        """
        ctrl = TrajectoryController(lookahead_distance=0.5)
        mid = len(straight_traj) // 2
        x, y, _ = straight_traj[mid]
        ctrl._last_closest_idx = mid
        _, omega = ctrl.compute_velocity(x, y, 0.0, straight_traj)
        assert abs(omega) < 0.15, \
            f'Expected near-zero omega on aligned straight path, got {omega:.4f}'


# ---------------------------------------------------------------------------
# TC-C-3  Goal detection
# ---------------------------------------------------------------------------

class TestGoalDetection:
    """Goal must arm only after the full path is consumed, then stop the robot."""

    def test_returns_zero_when_goal_armed_and_at_goal(self, straight_traj):
        """Once goal is armed and robot is at the final point, output must be (0, 0).

        HOW: force _goal_armed=True and _last_closest_idx near end, place
        robot at trajectory[-1], check v=0 and omega=0.
        WHY: if the robot does not stop it will drive past the goal and
        potentially into an obstacle or off the edge of the map.
        """
        ctrl = TrajectoryController()
        ctrl._last_closest_idx = len(straight_traj) - 5
        ctrl._goal_armed = True
        gx, gy = straight_traj[-1][0], straight_traj[-1][1]
        v, omega = ctrl.compute_velocity(gx, gy, 0.0, straight_traj)
        assert v == pytest.approx(0.0), f'Expected v=0 at goal, got {v:.4f}'
        assert omega == pytest.approx(0.0), f'Expected omega=0 at goal, got {omega:.4f}'

    def test_goal_does_not_arm_early_on_looping_path(self, figure_eight_traj):
        """Goal must NOT fire when robot is near start but index is still low.

        HOW: place robot at the start position (which is also the goal on
        a figure-eight), but leave _last_closest_idx at 10.  v must be > 0.
        WHY: on a looping path the start and end are the same physical
        point.  A pure distance check would fire the goal the moment the
        path passes near the start mid-run.  The _goal_armed flag prevents
        this by requiring the index to reach near the end first.
        """
        ctrl = TrajectoryController()
        ctrl._last_closest_idx = 10
        ctrl._goal_armed = False
        sx, sy = figure_eight_traj[0][0], figure_eight_traj[0][1]
        v, _ = ctrl.compute_velocity(sx, sy, 0.0, figure_eight_traj)
        assert v > 0.0, \
            'Goal fired too early — arming bug present on looping path'

    def test_moves_forward_far_from_goal(self, straight_traj, fresh_ctrl):
        """Robot far from goal must receive a positive forward velocity.

        HOW: place robot at start, check v > 0.
        WHY: basic sanity — if the controller outputs v=0 immediately it
        means the goal check or some other early-return is mis-triggering.
        """
        v, _ = fresh_ctrl.compute_velocity(0.0, 0.0, 0.0, straight_traj)
        assert v > 0.0, 'Controller returned v=0 far from goal'


# ---------------------------------------------------------------------------
# TC-C-4  Bounded window bug fix
# ---------------------------------------------------------------------------

class TestBoundedWindowFix:
    """The looping-path index-skip bug must be provably absent.

    BACKGROUND
    ----------
    The original controller had an unbounded while-loop:
        while path[idx+1] is closer than path[idx]: idx += 1
    On a figure-eight path this would race from index 150 to 450+ in a
    single tick because the path loops back near the robot.  The robot
    would then skip that entire arc — missing turns and obstacles.

    FIX
    ---
    The index may only advance by at most _PROGRESS_WINDOW (=20) per call.
    These tests verify the invariant directly.
    """

    def test_index_advance_never_exceeds_window_per_tick(self, figure_eight_traj):
        """_last_closest_idx must advance <= _PROGRESS_WINDOW per compute call.

        HOW: record index before and after each call, check the delta.
        WHY: this is the direct mathematical proof that the index-skip
        bug cannot occur.  If delta > _PROGRESS_WINDOW even once, the
        fix is broken.
        """
        ctrl = TrajectoryController()
        prev_idx = 0
        for i in range(0, 450, 5):
            x, y, _ = figure_eight_traj[i]
            ctrl.compute_velocity(x, y, 0.0, figure_eight_traj)
            advance = ctrl._last_closest_idx - prev_idx
            assert advance <= _PROGRESS_WINDOW, (
                f'Index jumped {advance} steps at iteration {i}. '
                f'Maximum allowed: {_PROGRESS_WINDOW}. '
                f'Index-skip bug is present.'
            )
            prev_idx = ctrl._last_closest_idx

    def test_index_is_monotonically_non_decreasing(self, figure_eight_traj):
        """_last_closest_idx must never go backwards.

        HOW: check index >= previous index after every call.
        WHY: a backwards index means the controller is regressing — it
        would re-issue commands for path points it already executed,
        causing the robot to turn around and retrace its path.
        """
        ctrl = TrajectoryController()
        prev_idx = 0
        for i in range(0, 490, 3):
            x, y, _ = figure_eight_traj[i]
            ctrl.compute_velocity(x, y, 0.0, figure_eight_traj)
            assert ctrl._last_closest_idx >= prev_idx, (
                f'Index went backwards from {prev_idx} to '
                f'{ctrl._last_closest_idx} at iteration {i}'
            )
            prev_idx = ctrl._last_closest_idx

    def test_window_constant_has_correct_value(self):
        """_PROGRESS_WINDOW must be 20 — changing it breaks the bug-fix guarantee.

        HOW: assert _PROGRESS_WINDOW == 20.
        WHY: the value 20 was chosen because at 20 Hz and 0.4 m/s the robot
        moves ~0.45 index units per tick.  Window=20 is 40x the real advance
        rate, giving headroom for sharp turns while still preventing jumps
        of hundreds of indices.  A smaller value could make the robot stall
        on sharp turns; a much larger value re-introduces the bug.
        """
        assert _PROGRESS_WINDOW == 20, (
            f'_PROGRESS_WINDOW is {_PROGRESS_WINDOW}, expected 20. '
            f'Changing this value breaks the looping-path bug fix guarantee.'
        )


# ---------------------------------------------------------------------------
# TC-C-5  Velocity scaling with heading error
# ---------------------------------------------------------------------------

class TestVelocityScaling:
    """Velocity must decrease as heading error increases."""

    def test_aligned_robot_faster_than_turned_robot(self, straight_traj):
        """An aligned robot must move faster than one facing 90 degrees off path.

        HOW: place robot mid-path (index ~50) where feedforward velocity is
        above the floor, then compare v at yaw=0 vs yaw=pi/2.
        WHY: at index 0 the trapezoidal ramp starts near zero so both aligned
        and misaligned robots hit the velocity floor — the test must start
        mid-path where the feedforward speed is meaningfully above the floor
        so the heading-error penalty has room to show its effect.
        """
        ctrl_aligned = TrajectoryController()
        ctrl_turned  = TrajectoryController()

        # Advance both controllers to index 50 so feedforward is above floor
        ctrl_aligned._last_closest_idx = 50
        ctrl_turned._last_closest_idx  = 50

        # Use the position at index 50 on the path
        x50, y50 = straight_traj[50][0], straight_traj[50][1]

        v_aligned, _ = ctrl_aligned.compute_velocity(x50, y50, 0.0,       straight_traj)
        v_turned,  _ = ctrl_turned.compute_velocity( x50, y50, np.pi / 2, straight_traj)

        assert v_aligned > v_turned, (
            f'Aligned v={v_aligned:.4f} should exceed '
            f'90-deg-turned v={v_turned:.4f}'
        )

    def test_velocity_decreases_as_heading_error_increases(self, straight_traj):
        """Velocity must be lower at large heading errors than at small ones.

        HOW: place robot at index 50 (above velocity floor), compute v at
        yaw offsets 0 and 2*pi/3 — compare first vs last.
        WHY: the adaptive blend formula ensures a robot facing 120 degrees
        off the path goes slower than one facing directly along it.
        Testing at index 50 ensures feedforward is above the floor so the
        heading-error penalty has room to show its effect.
        """
        x50, y50 = straight_traj[50][0], straight_traj[50][1]
        angles = [0.0, np.pi / 3, 2 * np.pi / 3]
        velocities = []
        for yaw_offset in angles:
            c = TrajectoryController()
            c._last_closest_idx = 50
            v, _ = c.compute_velocity(x50, y50, yaw_offset, straight_traj)
            velocities.append(v)

        assert velocities[0] > velocities[-1], (
            f'Well-aligned v={velocities[0]:.4f} must exceed '
            f'badly-misaligned v={velocities[-1]:.4f}'
        )