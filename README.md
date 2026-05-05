Humanoid Dev - Vishal

ROS 2 + MuJoCo based humanoid simulation with real-time computation of:
- Center of Mass (CoM)
- Zero Moment Point (ZMP)
- Support Polygon

---

## Prerequisites

- ROS 2 (Jazzy)
- Python 3.x
- MuJoCo
- colcon

---

## 🚀 Setup

```bash
git clone https://github.com/vishalnaik37/humanoid-dev-vishal.git
cd humanoid-dev-vishal
colcon build
source install/setup.bash

ros2 launch full_body_mujoco full_body_mujoco.launch.py
ros2 run full_body_mujoco zmp_calculation.py
ros2 run full_body_mujoco com_calculation.py
ros2 run full_body_mujoco support_polygon.py
ros2 run rviz2 rviz2

After launching Rviz, select frame as odom, then click add->By display type -> RobotModel -> select Description Topic from left
then Add -> By Topic -> click /com_marker,/zmp_marker, and vice versa one by one
