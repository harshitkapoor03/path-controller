# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""
2D Differential Drive Robot Simulator
======================================
Implements the unicycle kinematic model for a differential drive robot
and records full state history for post-run plotting and analysis.

Kinematic model (Euler integration at fixed timestep dt):
    x(t+dt)     = x(t)     + v * cos(theta(t)) * dt
    y(t+dt)     = y(t)     + v * sin(theta(t)) * dt
    theta(t+dt) = theta(t) + omega * dt

where:
    v     — linear  velocity command (m/s), forward positive
    omega — angular velocity command (rad/s), counter-clockwise positive
    theta — heading angle in radians, measured from the positive x-axis

Heading is normalised to (-pi, pi] after every step using arctan2(sin, cos)
so that heading comparisons and cross-track error calculations are correct
regardless of how many full rotations the robot has made.

This model exactly matches the TurtleBot3 burger kinematics used in Gazebo.
Replacing get_state() / step() with ROS2 odometry callbacks is the only
change needed to run on real hardware.

This module has no ROS2 dependency and can be unit-tested in isolation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — write PNG, never open a window.
import matplotlib.pyplot as plt


class RobotSimulator:
    """
    Simulate a differential drive robot and record its trajectory.

    The simulator maintains the robot's current state (x, y, theta) and
    a complete history of all states and velocity commands issued since
    construction.  The history is used by plot_results() to generate the
    three-panel results figure saved at the end of every run.
    """

    def __init__(self, x0: float = 0.0, y0: float = 0.0,
                 theta0: float = 0.0, dt: float = 0.05):
        """
        Initialise the simulator at a given pose.

        Args:
            x0:     Initial x position in metres.
            y0:     Initial y position in metres.
            theta0: Initial heading in radians.  0 = facing right (+x axis).
                    Use np.arctan2(dy, dx) to point toward a known target.
            dt:     Simulation timestep in seconds.  Must match the ROS2
                    control loop period.  Default 0.05 s = 20 Hz.
        """
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.dt = dt
        self.time = 0.0

        # Full history — used for plotting and cross-track error analysis.
        # Position history starts with the initial state (length = steps + 1).
        # Velocity history starts empty (length = steps, one entry per step).
        self.history_x = [x0]
        self.history_y = [y0]
        self.history_v = []
        self.history_omega = []
        self.history_t = [0.0]

    def step(self, v: float, omega: float) -> None:
        """
        Advance the simulation by one timestep using Euler integration.

        Applies the unicycle kinematic equations and normalises heading.
        Records the new state and the velocity commands in history.

        Args:
            v:     Linear  velocity command in m/s.  Non-negative (forward only).
            omega: Angular velocity command in rad/s.
                   Positive = counter-clockwise (left turn).
                   Negative = clockwise (right turn).
        """
        # Euler integration of unicycle model
        self.x     += v * np.cos(self.theta) * self.dt
        self.y     += v * np.sin(self.theta) * self.dt
        self.theta += omega * self.dt

        # Normalise heading to (-pi, pi] — prevents unbounded growth and
        # keeps heading differences meaningful for error calculations.
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        self.time += self.dt

        # Append to history
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        self.history_v.append(v)
        self.history_omega.append(omega)
        self.history_t.append(self.time)

    def get_state(self) -> tuple:
        """
        Return the current robot pose.

        Returns:
            (x, y, theta): position in metres and heading in radians.
        """
        return self.x, self.y, self.theta

    def plot_results(self, trajectory: list, waypoints: list,
                     smooth_path: np.ndarray, save_path: str = None) -> None:
        """
        Generate and save the three-panel results figure.

        Panel 1 — Path comparison:
            Overlays the original waypoints, the smooth spline, and the
            robot's actual path.  Visually confirms tracking quality.

        Panel 2 — Velocity profile:
            Shows linear velocity (m/s) and angular velocity (rad/s) over
            time.  The trapezoidal ramp should be visible in linear velocity.
            Angular velocity should be smooth with no sudden spikes.

        Panel 3 — Cross-track error:
            For every recorded robot position, finds the nearest point on
            the reference trajectory and plots that distance over time.
            Reports max and mean error in the title.

        Args:
            trajectory:  List of (x, y, t) tuples — the reference trajectory.
            waypoints:   Original input waypoints — plotted as red squares.
            smooth_path: Dense smooth path array — plotted as a blue line.
            save_path:   Full file path for the output PNG.  If None the
                         figure is not saved (useful for interactive testing).
        """
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        wp_x   = [p[0] for p in waypoints]
        wp_y   = [p[1] for p in waypoints]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Path Smoothing and Trajectory Tracking Results', fontsize=14)

        # --- Panel 1: Path comparison ---
        ax1 = axes[0]
        ax1.plot(wp_x, wp_y, 'rs--', markersize=10,
                 label='Original Waypoints', zorder=5)
        ax1.plot(smooth_path[:, 0], smooth_path[:, 1], 'b-',
                 linewidth=1.5, label='Smooth Path', alpha=0.6)
        ax1.plot(self.history_x, self.history_y, 'g-',
                 linewidth=2, label='Robot Actual Path')
        ax1.plot(wp_x[0], wp_y[0], 'g^', markersize=12, label='Start')
        ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=12, label='Goal')
        ax1.set_title('Path Comparison')
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')

        # --- Panel 2: Velocity profile ---
        ax2 = axes[1]
        t_v = self.history_t[1:]   # velocity history is one shorter than position
        ax2.plot(t_v, self.history_v,     'b-', label='Linear Velocity (m/s)')
        ax2.plot(t_v, self.history_omega, 'r-', label='Angular Velocity (rad/s)')
        ax2.set_title('Velocity Profile')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Velocity')
        ax2.legend()
        ax2.grid(True)

        # --- Panel 3: Cross-track error ---
        # For each recorded robot position, compute its minimum distance
        # to any point on the reference trajectory.  This is the standard
        # cross-track error metric used in Nav2 and Apollo.
        ax3 = axes[2]
        robot_path = np.column_stack([self.history_x, self.history_y])
        ref_path   = np.column_stack([traj_x, traj_y])
        errors = [
            np.min(np.linalg.norm(ref_path - pt, axis=1))
            for pt in robot_path
        ]
        ax3.plot(self.history_t, errors, 'm-')
        ax3.set_title(
            'Cross-Track Error\n'
            f'Max: {max(errors):.3f}m  |  Mean: {np.mean(errors):.3f}m'
        )
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Error (meters)')
        ax3.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'\n  Plot saved to: {save_path}')
