"""
Pure Pursuit Trajectory Tracking Controller
Outputs smooth linear and angular velocity commands.

Velocity scaling: inverse relationship with heading error —
never collapses to zero, always physically executable.

Goal detection: flag-based — only arms after the robot has
consumed the full path array, so looping paths never trigger early.

---- THE LOOPING-PATH BUG (and the fix) ----
Each call to _find_lookahead_point is only allowed to advance
_last_closest_idx by at most PROGRESS_WINDOW steps.  At 20 Hz and
0.4 m/s the robot travels ~0.02 m per tick; the path is sampled at
~500 points over ~22 m, so one real "step" corresponds to ~0.45
index units.  A window of 20 is therefore ~40× larger than needed,
giving plenty of headroom for sharp turns while making it physically
impossible to skip hundreds of indices in one call.
"""

import numpy as np


# Maximum number of path indices the tracker may advance in a single
# control tick.  Keep this comfortably above one real step's worth of
# indices (see module docstring) but well below the loop-back distance.
_PROGRESS_WINDOW = 20


class TrajectoryController:
    """Pure Pursuit controller for a differential drive robot."""

    def __init__(self, lookahead_distance: float = 0.3, max_linear_vel: float = 0.4):
        """
        Args:
            lookahead_distance: How far ahead (meters) the robot looks on the path.
            max_linear_vel:     Maximum forward speed in m/s.
        """
        self.lookahead_distance = lookahead_distance
        self.max_linear_vel     = max_linear_vel
        self.goal_threshold     = 0.10

        # Monotonically increasing pointer into the path array.
        # Never resets; only moves forward — this is what stops the robot
        # from triggering the goal check early on a looping path.
        self._last_closest_idx = 0
        self._goal_armed       = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_velocity(self, robot_x, robot_y, robot_yaw, trajectory):
      path_points = np.array([(p[0], p[1]) for p in trajectory])
      robot_pos = np.array([robot_x, robot_y])

      # Goal arming (unchanged)
      if self._last_closest_idx >= len(path_points) - 10:
        self._goal_armed = True
      if self._goal_armed and np.linalg.norm(path_points[-1] - robot_pos) < self.goal_threshold:
        return 0.0, 0.0

      # Find lookahead point (updates self._last_closest_idx)
      lookahead_point = self._find_lookahead_point(robot_pos, path_points)

      # ---- Pure Pursuit geometry (unchanged) ----
      dx = lookahead_point[0] - robot_x
      dy = lookahead_point[1] - robot_y
      angle_to_target = np.arctan2(dy, dx)
      alpha = np.arctan2(np.sin(angle_to_target - robot_yaw),
                       np.cos(angle_to_target - robot_yaw))
      L = max(np.linalg.norm(lookahead_point - robot_pos), 0.1)
      curvature = 2.0 * np.sin(alpha) / L

      # ---- Feedforward desired velocity from trajectory timestamps ----
      idx = self._last_closest_idx
      if idx < len(trajectory) - 1:
        dt = trajectory[idx+1][2] - trajectory[idx][2]          # time diff
        ds = np.linalg.norm(path_points[idx+1] - path_points[idx])  # distance
        v_desired = ds / dt if dt > 0 else self.max_linear_vel
      else:
        v_desired = 0.0
      v_desired = np.clip(v_desired, 0.08, self.max_linear_vel)

      # ---- Feedback velocity based on heading error (original formula) ----
      v_feedback = self.max_linear_vel / (1.0 + 2.5 * abs(alpha))
      v_feedback = max(0.08, v_feedback)

      # ---- Adaptive blending ratio ----
      # Smaller heading error -> trust feedforward more
      adaptive_ratio = 1.0 / (1.0 + 2.0 * abs(alpha))
      # Blend: v = ratio * v_desired + (1-ratio) * v_feedback
      v = adaptive_ratio * v_desired + (1 - adaptive_ratio) * v_feedback
      v = np.clip(v, 0.08, self.max_linear_vel)

      # Angular velocity (pure pursuit curvature times current speed)
      omega = float(np.clip(v * curvature, -1.5, 1.5))

      return float(v), omega

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_lookahead_point(
        self,
        robot_pos:   np.ndarray,
        path_points: np.ndarray,
    ) -> np.ndarray:
        """
        Return the first path point that lies at least lookahead_distance
        ahead of the robot, searching strictly forward from the last known
        closest index.

        KEY INVARIANT — bounded progress window
        ----------------------------------------
        We only allow _last_closest_idx to advance by at most
        _PROGRESS_WINDOW steps per call.  This prevents the unbounded
        while-loop that existed before from jumping hundreds of indices
        at once when the path loops back near the robot's position.

        Without the window, consider a path that passes near (x, y) at
        index 150 and again at index 555:

            Old code: while next closer → advance idx
                      → idx races to 555 in one tick
                      → for-loop starts at 555
                      → indices 150-554 are SKIPPED FOREVER

            New code: per-tick advance ≤ _PROGRESS_WINDOW (= 20)
                      → idx reaches 555 only after ~20 ticks
                      → all intermediate points are visited normally
        """
        last = len(path_points) - 1

        # --- Step 1: advance the closest-point index (bounded) ---
        # Find the closest point inside the window and move the index
        # there.  Searching inside a window rather than greedily
        # advancing until a local minimum ensures we cannot skip a
        # large arc even if the path doubles back.
        window_end = min(self._last_closest_idx + _PROGRESS_WINDOW, last)
        window_pts = path_points[self._last_closest_idx : window_end + 1]
        dists      = np.linalg.norm(window_pts - robot_pos, axis=1)
        local_best = int(np.argmin(dists))          # index within window
        self._last_closest_idx += local_best        # absolute index update

        # --- Step 2: find lookahead point (also bounded) ---
        # Search forward from the updated index for the first point
        # that is at least lookahead_distance away.  Cap the search at
        # _PROGRESS_WINDOW * 4 beyond the current index so we never
        # scan the entire remaining path (O(1) per call instead of O(n)).
        search_end = min(self._last_closest_idx + _PROGRESS_WINDOW * 4, last)
        for i in range(self._last_closest_idx, search_end + 1):
            if np.linalg.norm(path_points[i] - robot_pos) >= self.lookahead_distance:
                return path_points[i]

        # Fallback: end of the bounded search window (or final point).
        return path_points[search_end]