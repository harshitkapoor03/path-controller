# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""ROS2 integration tests for PathControllerNode.

WHAT is being tested
--------------------
PathControllerNode is the ROS2 entry point.  It wires all modules together,
declares parameters, publishes on /cmd_vel and /smooth_path, and runs the
control loop at 20 Hz via a ROS2 timer.

WHY these tests exist
---------------------
The unit tests (test_path_smoother, test_trajectory_generator, etc.) only
test the Python logic in isolation.  These tests verify the ROS2 interface:
that the node starts correctly, exposes the right parameters, publishes on
the correct topics with the correct message types, and that real Twist
messages arrive on /cmd_vel during a live control loop.

HOW the tests are organised
----------------------------
  TC-N-1  Node lifecycle  (name, no crash on init)
  TC-N-2  Parameters  (all four exist with sensible values)
  TC-N-3  Publishers  (correct topics and message types)
  TC-N-4  Topic output  (/smooth_path and /cmd_vel actually publish)
  TC-N-5  Error handling  (bad input triggers clean failure)

HOW to run
----------
Source your workspace, then:
    colcon test --packages-select path_controller
    colcon test-result --verbose

Or directly (workspace must be sourced):
    pytest test/test_node_ros.py -v
"""

import threading
import time
import unittest

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path


# ---------------------------------------------------------------------------
# Helper — lightweight subscriber node
# ---------------------------------------------------------------------------

class ProbeNode(Node):
    """Minimal node that collects messages published by PathControllerNode."""

    def __init__(self):
        """Initialise probe with subscribers on /cmd_vel and /smooth_path."""
        super().__init__('probe_node')
        self.received_twists = []
        self.received_paths = []
        self.create_subscription(
            Twist, '/cmd_vel',
            lambda msg: self.received_twists.append(msg), 10,
        )
        self.create_subscription(
            Path, '/smooth_path',
            lambda msg: self.received_paths.append(msg), 10,
        )

    def spin_for(self, seconds):
        """Spin the ROS2 event loop for a fixed duration."""
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        deadline = time.time() + seconds
        while time.time() < deadline:
            executor.spin_once(timeout_sec=0.05)
        executor.remove_node(self)
        executor.shutdown()


# ---------------------------------------------------------------------------
# TC-N-1  Node lifecycle
# ---------------------------------------------------------------------------

class TestNodeLifecycle(unittest.TestCase):
    """PathControllerNode must start without errors and expose the correct name."""

    @classmethod
    def setUpClass(cls):
        """Initialise ROS2 context once for this test class."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down ROS2 context after all tests in this class."""
        rclpy.shutdown()

    def test_node_starts_without_exception(self):
        """PathControllerNode.__init__ must not raise any exception.

        HOW: instantiate the node inside a try/except, call fail() if any
        exception is caught.
        WHY: __init__ runs smooth_path, generate_trajectory, and builds
        the controller and simulator.  Any import error, shape mismatch,
        or configuration issue surfaces here.
        """
        from path_controller.path_controller_node import PathControllerNode
        try:
            node = PathControllerNode()
            node.destroy_node()
        except Exception as e:
            self.fail(f'PathControllerNode raised on init: {e}')

    def test_node_name_is_path_controller_node(self):
        """Node must register under the name 'path_controller_node'.

        HOW: instantiate, call get_name(), compare.
        WHY: other nodes and tools (RViz, ros2 node list) discover this
        node by name.  A wrong name breaks tool integration.
        """
        from path_controller.path_controller_node import PathControllerNode
        node = PathControllerNode()
        self.assertEqual(node.get_name(), 'path_controller_node')
        node.destroy_node()


# ---------------------------------------------------------------------------
# TC-N-2  Parameters
# ---------------------------------------------------------------------------

class TestNodeParameters(unittest.TestCase):
    """All four ROS2 parameters must be declared with sensible default values."""

    @classmethod
    def setUpClass(cls):
        """Start ROS2 and create the node once for all parameter tests."""
        rclpy.init()
        from path_controller.path_controller_node import PathControllerNode
        cls.node = PathControllerNode()

    @classmethod
    def tearDownClass(cls):
        """Destroy node and shut down ROS2."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_velocity_parameter_is_positive_float(self):
        """'velocity' must be a positive float.

        HOW: get_parameter, check type and value > 0.
        WHY: velocity=0 would make the trajectory timestamps all identical
        (division by zero) and velocity<0 would make the robot drive backwards.
        """
        v = self.node.get_parameter('velocity').value
        self.assertIsInstance(v, float)
        self.assertGreater(v, 0.0)

    def test_lookahead_distance_is_positive_float(self):
        """'lookahead_distance' must be a positive float.

        HOW: get_parameter, check type and value > 0.
        WHY: lookahead_distance=0 makes the lookahead point coincide with the
        robot, causing curvature to blow up and omega to saturate.
        """
        ld = self.node.get_parameter('lookahead_distance').value
        self.assertIsInstance(ld, float)
        self.assertGreater(ld, 0.0)

    def test_velocity_profile_is_valid_string(self):
        """'velocity_profile' must be either 'constant' or 'trapezoidal'.

        HOW: get_parameter, check value is in the allowed set.
        WHY: any other value falls through to the constant profile silently;
        an invalid default would mislead users running with default settings.
        """
        profile = self.node.get_parameter('velocity_profile').value
        self.assertIn(profile, ['constant', 'trapezoidal'])

    def test_max_steps_is_positive_int(self):
        """'max_steps' must be a positive integer.

        HOW: get_parameter, check type and value > 0.
        WHY: max_steps=0 would make the node finish immediately without
        running a single control tick.
        """
        ms = self.node.get_parameter('max_steps').value
        self.assertIsInstance(ms, int)
        self.assertGreater(ms, 0)


# ---------------------------------------------------------------------------
# TC-N-3  Publishers
# ---------------------------------------------------------------------------

class TestNodePublishers(unittest.TestCase):
    """Both /cmd_vel and /smooth_path publishers must exist on the node."""

    @classmethod
    def setUpClass(cls):
        """Start ROS2 and create the node once for all publisher tests."""
        rclpy.init()
        from path_controller.path_controller_node import PathControllerNode
        cls.node = PathControllerNode()

    @classmethod
    def tearDownClass(cls):
        """Destroy node and shut down ROS2."""
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_cmd_vel_publisher_exists(self):
        """/cmd_vel publisher must be registered on the node.

        HOW: query get_publisher_names_and_types_by_node, check /cmd_vel present.
        WHY: if this publisher is missing the robot receives no velocity commands
        — it simply never moves regardless of what the controller outputs.
        """
        topic_names = [
            name for name, _ in
            self.node.get_publisher_names_and_types_by_node('path_controller_node', '/')
        ]
        self.assertIn('/cmd_vel', topic_names,
                      f'/cmd_vel not found in publishers: {topic_names}')

    def test_smooth_path_publisher_exists(self):
        """/smooth_path publisher must be registered on the node.

        HOW: same query, check /smooth_path present.
        WHY: if this publisher is missing the path cannot be visualised
        in RViz — important for the demo video.
        """
        topic_names = [
            name for name, _ in
            self.node.get_publisher_names_and_types_by_node('path_controller_node', '/')
        ]
        self.assertIn('/smooth_path', topic_names,
                      f'/smooth_path not found in publishers: {topic_names}')


# ---------------------------------------------------------------------------
# TC-N-4  Topic output
# ---------------------------------------------------------------------------

class TestNodeTopicOutput(unittest.TestCase):
    """Real messages must arrive on both topics within a reasonable timeout."""

    @classmethod
    def setUpClass(cls):
        """Start ROS2 context once for all output tests."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down ROS2 context."""
        rclpy.shutdown()

    def test_smooth_path_published_on_startup(self):
        """/smooth_path must receive at least one message within 3 s of startup.

        HOW: create probe node and controller node, spin probe for 3 s,
        check probe.received_paths is non-empty.
        WHY: _publish_path() is called in __init__; if it silently fails
        no path appears in RViz and the assignment demo shows nothing.
        """
        from path_controller.path_controller_node import PathControllerNode
        probe = ProbeNode()
        node = PathControllerNode()
        probe.spin_for(3.0)
        node.destroy_node()
        probe.destroy_node()
        self.assertGreater(
            len(probe.received_paths), 0,
            'No Path message on /smooth_path within 3 s',
        )

    def test_smooth_path_has_many_poses(self):
        """/smooth_path message must contain at least 100 poses.

        HOW: collect path message, check len(poses).
        WHY: the path was generated with 500 points; fewer than 100 would
        mean the trajectory was truncated or incorrectly serialised.
        """
        from path_controller.path_controller_node import PathControllerNode
        probe = ProbeNode()
        node = PathControllerNode()
        probe.spin_for(3.0)
        node.destroy_node()
        probe.destroy_node()
        self.assertGreater(len(probe.received_paths), 0, 'No path received')
        self.assertGreater(
            len(probe.received_paths[0].poses), 100,
            f'Path has only {len(probe.received_paths[0].poses)} poses',
        )

    def test_cmd_vel_published_during_control_loop(self):
        """/cmd_vel must receive messages while the control loop is running.

        HOW: spin the controller node in a background thread for 20 ticks
        while the probe node collects Twist messages.
        WHY: verifies the 20 Hz timer is actually firing and the controller
        is computing and publishing non-zero velocity commands.
        """
        from path_controller.path_controller_node import PathControllerNode
        probe = ProbeNode()
        node = PathControllerNode()

        def spin_node():
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(node)
            for _ in range(20):
                executor.spin_once(timeout_sec=0.05)
            executor.remove_node(node)
            executor.shutdown()

        t = threading.Thread(target=spin_node)
        t.start()
        probe.spin_for(1.5)
        t.join()
        node.destroy_node()
        probe.destroy_node()

        self.assertGreater(
            len(probe.received_twists), 0,
            'No Twist message on /cmd_vel — control loop not running',
        )

    def test_all_cmd_vel_linear_x_non_negative(self):
        """Every /cmd_vel message must have linear.x >= 0 (forward only).

        HOW: collect all Twist messages, check linear.x on each.
        WHY: the controller's velocity floor is 0.08 m/s and the robot
        has no reverse motion in this system.  A negative linear.x means
        the controller output is inverted.
        """
        from path_controller.path_controller_node import PathControllerNode
        probe = ProbeNode()
        node = PathControllerNode()

        def spin_node():
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(node)
            for _ in range(20):
                executor.spin_once(timeout_sec=0.05)
            executor.remove_node(node)
            executor.shutdown()

        t = threading.Thread(target=spin_node)
        t.start()
        probe.spin_for(1.5)
        t.join()
        node.destroy_node()
        probe.destroy_node()

        self.assertGreater(len(probe.received_twists), 0, 'No Twist messages')
        for msg in probe.received_twists:
            self.assertGreaterEqual(
                msg.linear.x, 0.0,
                f'linear.x={msg.linear.x:.4f} — robot commanded to go backwards',
            )


# ---------------------------------------------------------------------------
# TC-N-5  Error handling
# ---------------------------------------------------------------------------

class TestNodeErrorHandling(unittest.TestCase):
    """Bad configuration must produce a clean, loud failure rather than silent garbage."""

    @classmethod
    def setUpClass(cls):
        """Start ROS2 context once."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shut down ROS2 context."""
        rclpy.shutdown()

    def test_single_waypoint_triggers_system_exit(self):
        """DEFAULT_WAYPOINTS with one point must trigger SystemExit.

        HOW: monkey-patch DEFAULT_WAYPOINTS to a single point, attempt to
        construct the node, expect SystemExit.
        WHY: smooth_path raises ValueError on single-waypoint input.
        The node catches it and calls SystemExit(1) — this verifies that
        the error is not silently swallowed and the node does not start
        in a broken state.
        """
        import path_controller.path_controller_node as pcn
        original = pcn.DEFAULT_WAYPOINTS
        try:
            pcn.DEFAULT_WAYPOINTS = [(0.0, 0.0)]
            with self.assertRaises(SystemExit):
                from path_controller.path_controller_node import PathControllerNode
                PathControllerNode()
        finally:
            pcn.DEFAULT_WAYPOINTS = original


if __name__ == '__main__':
    unittest.main()
