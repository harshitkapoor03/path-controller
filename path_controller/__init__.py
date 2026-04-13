# Copyright 2024 Harshit Kapoor
#
# Licensed under the MIT License.

"""
path_controller — ROS2 path smoothing and trajectory tracking package.

Modules
-------
path_smoother          : Cubic spline smoothing with chord-length parameterisation.
trajectory_generator   : Time-stamped trajectory generation (constant + trapezoidal).
controller             : Pure Pursuit tracking controller with bounded progress window.
simulator              : Unicycle kinematic model with full history recording.
path_controller_node   : ROS2 node wiring all modules together at 20 Hz.
"""
