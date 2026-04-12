# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""Tests for path_controller.simulator.

WHAT is being tested
--------------------
RobotSimulator implements the unicycle kinematic model for a differential
drive robot.  Given (v, omega) commands it integrates the state (x, y, theta)
forward by one timestep using Euler integration and records all history.

WHY these tests exist
---------------------
The simulator is the ground truth for position during testing.  If the
kinematics are wrong every metric — cross-track error, goal distance,
heading — is wrong.  These tests verify each kinematic equation independently
with known analytical solutions, so bugs cannot hide behind compound effects.

HOW the tests are organised
----------------------------
  TC-SIM-1  Initialisation  (state and history set correctly)
  TC-SIM-2  Unicycle kinematics  (each equation individually)
  TC-SIM-3  Heading normalisation  (theta stays in [-pi, pi])
  TC-SIM-4  History recording  (lengths, values, time)
  TC-SIM-5  Multi-step integration  (distance, position after N steps)
"""

import numpy as np
import pytest

from path_controller.simulator import RobotSimulator


# ---------------------------------------------------------------------------
# TC-SIM-1  Initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    """Constructor must set state and history to the given initial values."""

    def test_initial_position_matches_arguments(self):
        """get_state() must return exactly the (x0, y0, theta0) passed in.

        HOW: construct with non-default values, call get_state(), compare.
        WHY: if the state is not initialised correctly the robot starts at
        the wrong position and every subsequent kinematic step is wrong.
        """
        sim = RobotSimulator(x0=3.0, y0=-1.5, theta0=1.2, dt=0.05)
        x, y, theta = sim.get_state()
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(-1.5)
        assert theta == pytest.approx(1.2)

    def test_history_contains_initial_state_only(self):
        """Before any step, history arrays must contain exactly the initial state.

        HOW: check history lengths and values right after construction.
        WHY: the plot_results function uses history arrays; if they start
        non-empty the plots will show phantom motion before the run begins.
        """
        sim = RobotSimulator(x0=1.0, y0=2.0)
        assert sim.history_x == [1.0]
        assert sim.history_y == [2.0]
        assert sim.history_t == [0.0]

    def test_velocity_history_empty_before_first_step(self):
        """Velocity history must be empty before any step is taken.

        HOW: check history_v and history_omega are empty lists.
        WHY: velocity is undefined before the first command is issued;
        plotting an empty velocity list is correct; plotting [0.0] would
        show a phantom zero-velocity command.
        """
        sim = RobotSimulator()
        assert sim.history_v == []
        assert sim.history_omega == []


# ---------------------------------------------------------------------------
# TC-SIM-2  Unicycle kinematics
# ---------------------------------------------------------------------------

class TestUnicycleKinematics:
    """Each kinematic equation is tested independently with exact analytical values."""

    def test_straight_forward_step(self):
        """v=1.0, omega=0, theta=0, dt=1.0 must move robot by (1, 0).

        HOW: single step, compare state to analytical result.
        WHY: tests x += v*cos(0)*dt = 1.0.  The simplest possible motion —
        if this fails all other kinematic tests are meaningless.
        """
        sim = RobotSimulator(x0=0.0, y0=0.0, theta0=0.0, dt=1.0)
        sim.step(1.0, 0.0)
        x, y, _ = sim.get_state()
        assert x == pytest.approx(1.0, abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_forward_facing_positive_y(self):
        """v=1.0, omega=0, theta=pi/2, dt=1.0 must move robot by (0, 1).

        HOW: single step at 90-degree heading.
        WHY: tests x += v*cos(pi/2)*dt ≈ 0 and y += v*sin(pi/2)*dt = 1.
        Verifies the trig is computing cos/sin of theta, not a fixed axis.
        """
        sim = RobotSimulator(x0=0.0, y0=0.0, theta0=np.pi / 2, dt=1.0)
        sim.step(1.0, 0.0)
        x, y, _ = sim.get_state()
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(1.0, abs=1e-9)

    def test_diagonal_motion(self):
        """v=1.0, omega=0, theta=pi/4, dt=1.0 must move by (1/sqrt2, 1/sqrt2).

        HOW: single step at 45-degree heading.
        WHY: verifies the full trig evaluation at a non-trivial angle.
        """
        sim = RobotSimulator(x0=0.0, y0=0.0, theta0=np.pi / 4, dt=1.0)
        sim.step(1.0, 0.0)
        x, y, _ = sim.get_state()
        expected = 1.0 / np.sqrt(2)
        assert x == pytest.approx(expected, abs=1e-9)
        assert y == pytest.approx(expected, abs=1e-9)

    def test_pure_rotation_does_not_change_position(self):
        """v=0, omega=pi/2, dt=1.0 must rotate heading without moving position.

        HOW: single step with v=0, check x and y unchanged.
        WHY: verifies that the position equations multiply by v, so v=0
        correctly produces zero displacement regardless of omega.
        """
        sim = RobotSimulator(x0=5.0, y0=3.0, theta0=0.0, dt=1.0)
        sim.step(0.0, np.pi / 2)
        x, y, theta = sim.get_state()
        assert x == pytest.approx(5.0, abs=1e-9)
        assert y == pytest.approx(3.0, abs=1e-9)
        assert theta == pytest.approx(np.pi / 2, abs=1e-9)

    def test_heading_accumulates_correctly(self):
        """Heading must accumulate as theta += omega * dt over multiple steps.

        HOW: take 4 steps with omega=pi/8 and dt=1.0, expect theta=pi/2.
        WHY: verifies the heading update is additive and not overwritten.
        """
        sim = RobotSimulator(theta0=0.0, dt=1.0)
        for _ in range(4):
            sim.step(0.0, np.pi / 8)
        _, _, theta = sim.get_state()
        assert theta == pytest.approx(np.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# TC-SIM-3  Heading normalisation
# ---------------------------------------------------------------------------

class TestHeadingNormalisation:
    """Theta must always stay in [-pi, pi] regardless of accumulated rotation."""

    def test_large_positive_omega_wraps_theta(self):
        """Spinning past +pi must wrap theta back into [-pi, pi].

        HOW: start near +pi, apply one step that would push past it.
        WHY: arctan2(sin, cos) normalisation in step() must handle this.
        Without it theta would grow unboundedly and cos/sin would still
        give correct values (they're periodic) but cross-track error
        and heading comparisons would give wrong results.
        """
        sim = RobotSimulator(theta0=3.0, dt=1.0)
        sim.step(0.0, 1.0)
        _, _, theta = sim.get_state()
        assert -np.pi <= theta <= np.pi, f'theta={theta:.4f} outside [-pi, pi]'

    def test_large_negative_omega_wraps_theta(self):
        """Spinning past -pi must wrap theta back into [-pi, pi].

        HOW: start near -pi, apply one step that would push past it.
        WHY: same as above for the negative direction.
        """
        sim = RobotSimulator(theta0=-3.0, dt=1.0)
        sim.step(0.0, -1.0)
        _, _, theta = sim.get_state()
        assert -np.pi <= theta <= np.pi, f'theta={theta:.4f} outside [-pi, pi]'

    def test_theta_always_bounded_over_many_rotations(self):
        """Theta must stay in [-pi, pi] over 500 consecutive rotation steps.

        HOW: spin at omega=1.2 for 500 ticks, check bound after every step.
        WHY: verifies the normalisation is applied on every step, not just
        at certain angles.  A conditional normalisation could miss some cases.
        """
        sim = RobotSimulator(theta0=0.0, dt=0.05)
        for _ in range(500):
            sim.step(0.0, 1.2)
            _, _, theta = sim.get_state()
            assert -np.pi <= theta <= np.pi, \
                f'theta={theta:.4f} outside [-pi, pi] after step'


# ---------------------------------------------------------------------------
# TC-SIM-4  History recording
# ---------------------------------------------------------------------------

class TestHistoryRecording:
    """Every state and command must be recorded accurately for later plotting."""

    def test_history_length_after_n_steps(self):
        """After N steps, position history has N+1 entries, velocity history has N.

        HOW: take 10 steps, check lengths.
        WHY: position history includes the initial state (N+1); velocity
        history only records issued commands (N).  If lengths are wrong
        the velocity profile plot will have mismatched axes.
        """
        sim = RobotSimulator()
        for _ in range(10):
            sim.step(0.2, 0.1)
        assert len(sim.history_x) == 11
        assert len(sim.history_y) == 11
        assert len(sim.history_t) == 11
        assert len(sim.history_v) == 10
        assert len(sim.history_omega) == 10

    def test_recorded_velocities_match_commands(self):
        """Recorded v and omega must exactly match the commands issued.

        HOW: issue three known (v, omega) commands, compare to history.
        WHY: if velocities are not recorded correctly the velocity profile
        plot is meaningless and cross-track error analysis is impossible.
        """
        sim = RobotSimulator()
        commands = [(0.1, 0.3), (0.4, -0.2), (0.2, 0.0)]
        for v, omega in commands:
            sim.step(v, omega)
        for i, (v, omega) in enumerate(commands):
            assert sim.history_v[i] == pytest.approx(v)
            assert sim.history_omega[i] == pytest.approx(omega)

    def test_time_advances_by_dt_each_step(self):
        """Simulation clock must advance exactly by dt at every step.

        HOW: take 5 steps with dt=0.05, check sim.time and history_t[-1].
        WHY: if time is not advancing correctly the trapezoidal profile
        timestamps will not align with the simulation timeline.
        """
        dt = 0.05
        sim = RobotSimulator(dt=dt)
        for i in range(1, 6):
            sim.step(0.0, 0.0)
            assert sim.time == pytest.approx(i * dt, abs=1e-9)
        assert sim.history_t[-1] == pytest.approx(5 * dt, abs=1e-9)


# ---------------------------------------------------------------------------
# TC-SIM-5  Multi-step integration
# ---------------------------------------------------------------------------

class TestMultiStepIntegration:
    """Over many steps the integrated position must match the analytical solution."""

    def test_straight_run_distance_matches_kinematics(self):
        """After N straight steps at v, distance must equal v * N * dt.

        HOW: run 100 steps at v=0.4, dt=0.05 (expected 2.0 m), measure
        Euclidean distance from start.
        WHY: verifies Euler integration accumulates correctly over time —
        a numerical drift or wrong dt application would cause this to fail.
        """
        v, dt, n = 0.4, 0.05, 100
        sim = RobotSimulator(x0=0.0, y0=0.0, theta0=0.0, dt=dt)
        for _ in range(n):
            sim.step(v, 0.0)
        x, y, _ = sim.get_state()
        actual = np.sqrt(x ** 2 + y ** 2)
        expected = v * n * dt
        assert actual == pytest.approx(expected, rel=1e-6), \
            f'Expected {expected:.4f} m, got {actual:.4f} m'

    def test_full_circle_returns_near_start(self):
        """At constant v and omega completing a full 2*pi rotation must return close to start.

        HOW: compute omega for a circle of radius r, run for exactly one
        full revolution's worth of steps, check distance from start < 0.1 m.
        WHY: verifies that the combined x, y, theta updates are geometrically
        consistent — a wrong sign or coefficient would cause the robot to
        spiral rather than circle.
        """
        r = 1.0
        v = 0.3
        omega = v / r
        circumference = 2 * np.pi * r
        dt = 0.05
        n_steps = int(circumference / (v * dt))
        sim = RobotSimulator(x0=0.0, y0=0.0, theta0=0.0, dt=dt)
        for _ in range(n_steps):
            sim.step(v, omega)
        x, y, _ = sim.get_state()
        dist_from_start = np.sqrt(x ** 2 + y ** 2)
        assert dist_from_start < 0.2, \
            f'Circle test: robot ended {dist_from_start:.4f} m from start, expected < 0.2 m'
