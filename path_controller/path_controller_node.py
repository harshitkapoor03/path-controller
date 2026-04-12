"""
ROS2 Path Controller Node - Main entry point.

This node:
  1. Generates a smooth trajectory from waypoints
  2. Runs a simulated robot that follows the trajectory
  3. Publishes velocity commands to /cmd_vel
  4. Saves result plots when the goal is reached
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import numpy as np
import os

from path_controller.path_smoother import smooth_path
from path_controller.trajectory_generator import generate_trajectory
from path_controller.controller import TrajectoryController
from path_controller.simulator import RobotSimulator


# The (x, y) waypoints the robot must navigate through, in meters
DEFAULT_WAYPOINTS =[
    (0.0, 0.0),
    (10.0, 0.0),   # long straight
    (10.0, 0.5),
    (10.0, 1.0),
    (9.5, 1.5),    # sharp 90° turn
    (9.0, 1.5),
    (8.0, 1.5),
]
class PathControllerNode(Node):

    def __init__(self):
        super().__init__('path_controller_node')

        # ROS2 parameters - change these from command line with --ros-args -p name:=value
        self.declare_parameter('velocity', 0.3)
        self.declare_parameter('lookahead_distance', 0.2)
        self.declare_parameter('velocity_profile', 'trapezoidal')
        self.declare_parameter('max_steps', 3000)

        v       = self.get_parameter('velocity').value
        ld      = self.get_parameter('lookahead_distance').value
        profile = self.get_parameter('velocity_profile').value
        self.max_steps = self.get_parameter('max_steps').value

        self.get_logger().info("=== Path Controller Node Starting ===")

        # Create ROS2 publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub    = self.create_publisher(Path, '/smooth_path', 10)

        # Build the smooth trajectory
        self.waypoints = DEFAULT_WAYPOINTS
        try:
            self.smooth = smooth_path(self.waypoints, num_points=500)
            self.trajectory = generate_trajectory(
                self.smooth, velocity=v, profile=profile
            )
            self.get_logger().info(
                f"Trajectory ready: {len(self.trajectory)} points, profile={profile}"
            )
        except ValueError as e:
            self.get_logger().error(f"Failed to generate trajectory: {e}")
            raise SystemExit(1)

        # Set up controller and simulator
        self.controller = TrajectoryController(
            lookahead_distance=ld,
            max_linear_vel=v + 0.1
        )

        # Find the true lookahead point on the spline from the start position
        # and use it to set the robot's initial heading — avoids startup swerve
        start_pos = np.array([self.waypoints[0][0], self.waypoints[0][1]])
        path_pts = np.array([(p[0], p[1]) for p in self.trajectory])

        initial_lookahead = path_pts[-1]
        for pt in path_pts:
            if np.linalg.norm(pt - start_pos) >= ld:
                initial_lookahead = pt
                break

        initial_heading = float(np.arctan2(
            initial_lookahead[1] - start_pos[1],
            initial_lookahead[0] - start_pos[0]
        ))

        self.sim = RobotSimulator(
            x0=self.waypoints[0][0],
            y0=self.waypoints[0][1],
            theta0=initial_heading,
            dt=0.05
        )
        self.step_count = 0
        self.done = False

        self._publish_path()

        # Run the control loop at 20 Hz (every 0.05 seconds)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Control loop running at 20 Hz...")

    def control_loop(self):
        """Called automatically every 0.05 seconds by ROS2 timer."""
        if self.done:
            return

        if self.step_count >= self.max_steps:
            self.get_logger().info("Max steps reached. Finishing.")
            self.done = True
            self._finish()
            return

        # Get current robot state
        x, y, theta = self.sim.get_state()

        # Compute velocity commands
        v, omega = self.controller.compute_velocity(x, y, theta, self.trajectory)

        # Publish to ROS2 topic
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)

        # Advance simulation
        self.sim.step(v, omega)
        self.step_count += 1

        # Log progress every 100 steps
        if self.step_count % 100 == 0:
            goal = self.trajectory[-1]
            dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            self.get_logger().info(
                f"Step {self.step_count} | "
                f"pos=({x:.2f}, {y:.2f}) | "
                f"dist_to_goal={dist:.2f}m"
            )

        # Goal check — mirrors the controller's flag so both stop together.
        # _goal_armed only becomes True after the full path array is consumed,
        # so this never fires early on looping paths.
        if self.controller._goal_armed:
            goal = self.trajectory[-1]
            dist_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            if dist_to_goal < 0.10:
                self.get_logger().info(
                    f"Goal reached in {self.step_count} steps!"
                )
                self.done = True
                self._finish()

    def _finish(self):
        """Stop the robot, save plots, and shut down the node."""
        self.cmd_vel_pub.publish(Twist())

        output_dir = os.path.expanduser("~/ros2_ws/results")
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, "tracking_results.png")

        self.sim.plot_results(
            self.trajectory,
            self.waypoints,
            self.smooth,
            save_path=plot_file
        )
        self.get_logger().info(f"Results saved to {output_dir}")
        rclpy.shutdown()

    def _publish_path(self):
        """Publish the smooth path as a ROS2 nav_msgs/Path message."""
        msg = Path()
        msg.header = Header()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y, _ in self.trajectory:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathControllerNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()