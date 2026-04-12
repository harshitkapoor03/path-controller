"""
2D Differential Drive Robot Simulator
Simulates robot motion and records history for plotting.

Unicycle model equations:
    x     = x + v * cos(theta) * dt
    y     = y + v * sin(theta) * dt
    theta = theta + omega * dt
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Save to file instead of opening a window
import matplotlib.pyplot as plt


class RobotSimulator:
    """Simulates a differential drive robot."""

    def __init__(self, x0=0.0, y0=0.0, theta0=0.0, dt=0.05):
        """
        Args:
            x0, y0: Starting position in meters
            theta0: Starting heading in radians (0 = facing right)
            dt: Time step in seconds
        """
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.dt = dt
        self.time = 0.0

        # Record every state for plotting later
        self.history_x = [x0]
        self.history_y = [y0]
        self.history_v = []
        self.history_omega = []
        self.history_t = [0.0]

    def step(self, v: float, omega: float):
        """Move the robot forward by one time step."""
        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        self.theta += omega * self.dt
        # Keep theta in [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.time += self.dt

        self.history_x.append(self.x)
        self.history_y.append(self.y)
        self.history_v.append(v)
        self.history_omega.append(omega)
        self.history_t.append(self.time)

    def get_state(self):
        """Returns (x, y, theta) — current robot position and heading."""
        return self.x, self.y, self.theta

    def plot_results(self, trajectory: list, waypoints: list,
                     smooth_path: np.ndarray, save_path: str = None):
        """
        Creates and saves 3 plots:
          1. Path comparison (waypoints vs smooth path vs robot actual path)
          2. Velocity profile over time
          3. Cross-track error over time
        """
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        wp_x = [p[0] for p in waypoints]
        wp_y = [p[1] for p in waypoints]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Path Smoothing and Trajectory Tracking Results", fontsize=14)

        # --- Plot 1: Path Comparison ---
        ax1 = axes[0]
        ax1.plot(wp_x, wp_y, 'rs--', markersize=10,
                 label='Original Waypoints', zorder=5)
        ax1.plot(smooth_path[:, 0], smooth_path[:, 1], 'b-',
                 linewidth=1.5, label='Smooth Path', alpha=0.6)
        ax1.plot(self.history_x, self.history_y, 'g-',
                 linewidth=2, label='Robot Actual Path')
        ax1.plot(wp_x[0], wp_y[0], 'g^', markersize=12, label='Start')
        ax1.plot(wp_x[-1], wp_y[-1], 'r*', markersize=12, label='Goal')
        ax1.set_title("Path Comparison")
        ax1.set_xlabel("X (meters)")
        ax1.set_ylabel("Y (meters)")
        ax1.legend()
        ax1.grid(True)
        ax1.set_aspect('equal')

        # --- Plot 2: Velocity Profile ---
        ax2 = axes[1]
        t_v = self.history_t[1:]
        ax2.plot(t_v, self.history_v, 'b-', label='Linear Velocity (m/s)')
        ax2.plot(t_v, self.history_omega, 'r-', label='Angular Velocity (rad/s)')
        ax2.set_title("Velocity Profile")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Velocity")
        ax2.legend()
        ax2.grid(True)

        # --- Plot 3: Cross-Track Error ---
        ax3 = axes[2]
        robot_path = np.column_stack([self.history_x, self.history_y])
        ref_path = np.column_stack([traj_x, traj_y])
        errors = [
            np.min(np.linalg.norm(ref_path - pt, axis=1))
            for pt in robot_path
        ]
        ax3.plot(self.history_t, errors, 'm-')
        ax3.set_title(
            "Cross-Track Error\n"
            f"Max: {max(errors):.3f}m  |  Mean: {np.mean(errors):.3f}m"
        )
        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("Error (meters)")
        ax3.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved to: {save_path}")
