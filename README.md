# Path Smoothing & Trajectory Tracking — ROS2 Jazzy

A complete ROS2 Python package implementing path smoothing, time-parameterised
trajectory generation, and trajectory tracking control for a differential drive
robot. Built and tested on Ubuntu 24.04 with ROS2 Jazzy.

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Requirements](#requirements)
3. [Setup & Installation](#setup--installation)
4. [Running the Simulation](#running-the-simulation)
5. [Running the Tests](#running-the-tests)
6. [Design Choices & Architecture](#design-choices--architecture)
7. [Extending to a Real Robot](#extending-to-a-real-robot)
8. [Obstacle Avoidance — Extra Credit](#obstacle-avoidance--extra-credit)
9. [AI Tools Used](#ai-tools-used)

---

## Package Overview

The package is structured into five independent modules, each with a single
responsibility:
```
path_controller/
├── path_smoother.py          # Task 1 — cubic spline path smoothing
├── trajectory_generator.py   # Task 2 — time-stamped trajectory generation
├── controller.py             # Task 3 — Pure Pursuit tracking controller
├── simulator.py              # Unicycle robot simulator + result plots
└── path_controller_node.py   # ROS2 entry point — wires everything together

Test suite (`test/`):
test/
├── test_path_smoother.py       # 11 tests — smoother contracts and geometry
├── test_trajectory_generator.py # 10 tests — timestamp and profile correctness
├── test_simulator.py           # 16 tests — kinematics and history recording
├── test_controller.py          # 13 tests — Pure Pursuit and goal detection
└── test_node_ros.py            # 9 tests  — ROS2 interface integration tests
```
**59 unit tests + 13 integration tests = 72 tests total, all passing.**

---

## Requirements

- Ubuntu 24.04 (or WSL2 on Windows)
- ROS2 Jazzy
- Python 3.12+
- pip packages: `numpy scipy matplotlib pytest`

---

## Setup & Installation

```bash
# 1. Source ROS2
source /opt/ros/jazzy/setup.bash

# 2. Clone or copy the package into your workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
# (place path_controller/ here)

# 3. Install Python dependencies
pip3 install numpy scipy matplotlib pytest --break-system-packages

# 4. Build
cd ~/ros2_ws
colcon build --packages-select path_controller

# 5. Source the workspace
source install/setup.bash
```

---

## Running the Simulation

```bash
# Default run — trapezoidal velocity profile
ros2 run path_controller path_controller_node

# Constant velocity profile
ros2 run path_controller path_controller_node \
  --ros-args -p velocity_profile:=constant

# Custom speed and lookahead distance
ros2 run path_controller path_controller_node \
  --ros-args -p velocity:=0.4 -p lookahead_distance:=0.3

# All available parameters
#   velocity          (float, default 0.3)  — cruise speed in m/s
#   lookahead_distance (float, default 0.2) — Pure Pursuit lookahead in m
#   velocity_profile   (string, default 'trapezoidal') — 'trapezoidal' or 'constant'
#   max_steps          (int,   default 3000) — safety cutoff
```

The simulation runs at 20 Hz, prints progress every 100 steps, and saves a
three-panel results plot to:
~/ros2_ws/results/tracking_results.png

The plot shows: path comparison, velocity profile, and cross-track error.

---

## Running the Tests

```bash
cd ~/ros2_ws
python3 -m pytest src/path_controller/test/ -v
```

Expected output: **72 passed, 0 failed.**

To run a single test file:

```bash
python3 -m pytest src/path_controller/test/test_controller.py -v
```

---

## Design Choices & Architecture

### Why a single ROS2 node?

The pipeline is tightly sequential: waypoints → smoother → trajectory generator
→ controller → simulator. Every stage consumes the output of the previous one
synchronously, with no branching or parallel data flows.

Splitting this into multiple nodes would introduce unnecessary complexity: each
inter-node communication requires serialisation into a ROS2 message type,
publishing on a topic, subscribing and deserialising on the other end, and
handling the case where one node starts before another. For this system, that
overhead buys nothing. A single node with five clean internal modules gives
the same separation of concerns with none of the coordination cost.

This changes on a real robot, see the *Extending to a Real Robot* section for
how and why the architecture would expand.

### Path Smoothing : Cubic Spline with Chord-Length Parameterisation

The smoother fits two independent cubic splines : one for x(t) and one for y(t)
, where t is the cumulative arc-length distance along the waypoint polyline
(chord-length parameterisation).

The key design decision is chord-length over uniform indexing. With uniform
indexing (treating waypoint gaps as 0, 1, 2, ...), a spline overshoots badly
when waypoints are unevenly spaced because it treats a 0.1 m gap the same as a
5.0 m gap. Chord-length parameterisation uses actual physical distance as the
parameter, so the spline behaves correctly regardless of spacing. This is the
same parameterisation used in industrial robot path planning.

Cubic splines guarantee C2 continuity : position, velocity, and curvature are
all smooth. Discontinuous curvature would require instantaneous changes in
angular velocity, which is physically impossible on a differential drive robot.

### Trajectory Generation : Trapezoidal Velocity Profile

Two profiles are implemented and selectable at runtime:

**Constant velocity** : timestamps are simply `t = arc_length / v`. Simple and
fast to compute. Useful for benchmarking and as a baseline comparison.

**Trapezoidal velocity** (default) : the robot accelerates from rest, cruises
at the target speed, then decelerates to a stop. This is the physically correct
profile for real hardware: instantaneous velocity jumps would cause wheel slip,
stress motor drivers, and make the robot hard to control. The acceleration
distance is capped at 20% of total path length so short paths get a sensible
ramp without phases overlapping.

### Trajectory Tracking Controller : Pure Pursuit with Adaptive Velocity Blending

The controller uses the Pure Pursuit algorithm: at each tick, find a
*lookahead point* on the path ahead of the robot, compute the curvature needed
to reach it, and convert that to an angular velocity command. Pure Pursuit is
widely used in autonomous vehicles and ROS2 Nav2's Regulated Pure Pursuit
plugin. It has a single, physically intuitive tuning parameter : lookahead
distance , and is robust to noise in the robot's pose estimate.

**Velocity evolution : three iterations:**

*Iteration 1 — pure feedback.* Initial implementation used
`v = v_max / (1 + 2.5 * |alpha|)` where alpha is the heading error. Simple and
smooth, but it ignores the trajectory's own timing entirely. The robot would
slow down correctly on turns but had no awareness of whether it was ahead or
behind the planned schedule.

*Iteration 2 — feedforward from trajectory timestamps.* Added a desired
velocity computed directly from the trajectory: `v_desired = ds / dt` where ds
is the distance between consecutive trajectory points and dt is the time
difference. This made the robot respect the trapezoidal profile's acceleration
and deceleration ramps. However, at the very start of the trajectory, the
trapezoidal ramp begins near zero, so both the aligned and misaligned robots
floored at the minimum velocity — the heading error had no room to show its
effect.

*Iteration 3 — adaptive blending (current).* The final controller blends both
signals using a heading-error-dependent ratio:
adaptive_ratio = 1 / (1 + 2 * |alpha|)
v = adaptive_ratio * v_desired + (1 - adaptive_ratio) * v_feedback

When heading error is small (robot well aligned), the ratio is close to 1 and
the controller mostly follows the feedforward schedule — good for straight
sections where timing matters. When heading error is large (robot on a sharp
turn), the ratio drops toward 0 and the feedback term dominates — the robot
slows down to handle the turn safely. A velocity floor of 0.08 m/s prevents
stalling.

**The looping-path bug and the Progress Window fix:**

The original lookahead search was an unbounded loop that scanned forward from
the closest point until it found a point at least `lookahead_distance` away.
On a looping path , one that passes near the start coordinates again mid run ,
the search would jump hundreds of indices in a single control tick. The robot
would skip portions of the path if the path looped back close to a position.

The fix is `_PROGRESS_WINDOW = 20`. Each control tick is only allowed to
advance `_last_closest_idx` by at most 20 steps. At 0.3 m/s and 20 Hz the
robot moves about 0.015 m per tick. The path has 500 points over roughly 12 m,
so one real step is about 0.6 index units. A window of 20 is 30× larger than
needed : ample headroom for sharp turns , but makes it physically impossible
to skip a loop-back in one tick. Search complexity drops from O(n) to O(1).

**Flag-based goal detection:**

A naive distance check fires whenever the robot is within the goal threshold,
even if it is on the first pass of a path that happens to go near the goal
coordinates mid-run. The `_goal_armed` flag only flips to `True` after
`_last_closest_idx` reaches within 10 indices of the end of the full path
array , meaning the robot has physically consumed the entire trajectory. Only
then does proximity to the goal trigger a stop. This is the same two-condition
goal check used in Nav2's goal checker plugin.
This was implemented to avoid the case where robot would stop prematurely if the trajectory loops back to starting point and goes on.

**Initial heading from spline lookahead:**

Early versions set the robot's initial heading toward waypoint[1] , a straight
line approximation. The controller steers toward the lookahead point on the
curved spline, not toward waypoint[1]. The spline curves away immediately, so
there was always a heading error at t=0 causing an error spike. The fix
computes the actual first lookahead point on the spline before the robot spawns
and sets the initial heading exactly toward that point , minimizing heading
error for any waypoint set.

### Code Architecture

Each module has a single responsibility and no circular dependencies:
path_smoother        ← no dependencies
trajectory_generator ← depends on numpy only
controller           ← depends on numpy only
simulator            ← depends on numpy, matplotlib
path_controller_node ← depends on all four modules + rclpy

Every public function and class has a full docstring explaining what it does,
why it exists, and what its arguments and return values are. Critical
implementation decisions (the progress window, the blend formula, the
chord-length parameterisation) are explained inline with the code they affect.

---

## Extending to a Real Robot

On a real TurtleBot3 or similar platform, the single-node architecture would
expand into a proper multi-node ROS2 system. Here is what changes and why.

**Pose estimation : replace simulator with odometry subscriber:**

The simulator's `step()` and `get_state()` are replaced with a subscriber to
the `/odom` topic (nav_msgs/Odometry). For better accuracy, a separate
localisation node using the `robot_localization` package fuses wheel odometry
with IMU data using an Extended Kalman Filter, publishing a filtered pose on
`/odometry/filtered`. The controller node subscribes to this filtered topic.

**Why this needs a separate node:**

On real hardware, pose data arrives asynchronously from the odometry driver at
its own frequency (typically 50-100 Hz), independent of the control loop. A
separate localisation node handles sensor fusion, timestamp alignment, and
frame transforms without blocking the controller. This is the standard Nav2
architecture: `amcl` or `robot_localization` runs as its own node, the
controller subscribes to its output.

**tf2 transforms:**

The controller needs the robot's pose in the `map` frame, but odometry is
published in the `odom` frame. A tf2 `TransformListener` in the controller
node handles the `map → odom → base_link` transform chain. This also means
the path publisher must set `header.frame_id = 'map'`.

**Watchdog timer:**

A real robot must stop safely if the controller node crashes or loses its
position estimate. A watchdog timer publishes zero velocity on `/cmd_vel` if
no control command has been issued within 0.5 seconds. This is a mandatory
safety feature for any real deployment.

**Path planning integration:**

On a real robot, waypoints would come from a global planner (Nav2's NavFn or
Theta*) via an action server interface rather than being hardcoded. The
`path_controller_node` would be rewritten as a Nav2 BT plugin or a standalone
action server that accepts a `nav_msgs/Path` goal and reports progress.

**Summary of real-robot node graph:**
[sensor drivers]  →  [robot_localization]  →  [path_controller_node]
↓
[/cmd_vel topic]
↓
[motor controllers]

Lookahead distance tuning: start at 0.3 m and increase until the robot tracks
without oscillation. On real hardware, 0.4-0.6 m is typical for TurtleBot3
at 0.3 m/s.

---

## Obstacle Avoidance — Extra Credit

The current system is a pure path following stack with no awareness of the
environment. Here is how obstacle avoidance would be layered on top without
restructuring the existing modules.

**Architecture : two-layer planner:**

The system would be split into a global planner layer (runs once or on
replanning trigger) and a local planner layer (runs every control tick).

**Local avoidance : Dynamic Window Approach (DWA):**

A LaserScan subscriber on `/scan` would run in the controller node. Every
control tick, before computing the Pure Pursuit command, the node checks
whether any laser point falls within a configurable safety radius (e.g. 0.5 m)
around the robot.

If the path is clear, Pure Pursuit runs exactly as now.

If an obstacle is detected, control switches to DWA. DWA works by sampling a
grid of feasible (v, omega) pairs — constrained by the robot's current velocity
and acceleration limits — simulating each pair forward by a short time window
(~0.5 s), and scoring them on three criteria:

1. Distance to nearest obstacle (higher is better)
2. Progress toward the Pure Pursuit lookahead point (higher is better)
3. Forward speed (higher is better, weighted less than safety)

The (v, omega) pair with the highest combined score is sent as the command.
This gives smooth, real-time avoidance without touching the global path.

**Global replanning : RRT\*:**

If DWA cannot find any collision-free sample : a dead end , the node triggers
a global replan. RRT* samples random configurations in the free space, connects
them to the existing tree while rewiring to minimise path cost, and converges
to an asymptotically optimal collision-free path. The new path is immediately
re-smoothed using the same `smooth_path()` function and re-timed using
`generate_trajectory()` — no other changes needed.

**Why this architecture is clean:**

All avoidance logic lives in `controller.py` and in the node's LaserScan
callback. The path smoother, trajectory generator, and simulator are completely
untouched. This is possible because of the modular design , each module exposes
a clean interface and has no knowledge of the others.

**Costmap integration:**

For a production system, the LaserScan would feed into a `costmap_2d` occupancy
grid (from the Nav2 stack). DWA would score samples against the costmap rather
than raw laser points, giving inflation layers around obstacles and lettting the
planner treat narrow corridors conservatively.

---

## AI Tools Used

Three AI tools were used actively throughout development:

**Claude (Anthropic)** was the primary development tool. It was used for
architecture planning and the rationale behind the algorithms, debugging the
looping-path bug and designing the progress window fix, writing the test suite
including the ROS2 integration tests, and iterating on the adaptive velocity
blending formula.

**ChatGPT (OpenAI)** was used for cross-checking ROS2 Jazzy API differences
from Humble (particularly the executor API changes that caused the
`RuntimeError: Executor is already spinning` issue in the integration tests)
and for verifying the DWA scoring formula described in the extra credit section.

**DeepSeek** was used for sanity checking the Pure Pursuit curvature formula
derivation and confirming the chord length parameterisation approach against
academic references on spline path planning for mobile robots. It was also 
used in debugging when claudes limit ran out.

All three tools were used as accelerators and thinking partners complimenting 
the development of this product after thinking the entire flow through to the end.
