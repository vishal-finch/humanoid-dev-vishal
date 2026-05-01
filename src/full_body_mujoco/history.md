# Integration History: MuJoCo Force/Torque & ZMP System

This document summarizes the changes made to the `full_body_mujoco` package to support stable ZMP (Zero Moment Point) calculation from simulated contact sensors.

## 1. Physical Sensor Definition (MuJoCo)
- **File**: `config/angad_full_body.xml`
- **Change**: Added 8 `<torque>` sensors paired with the existing 8 `<force>` sensors at foot contact sites (`toe_1/2`, `heel_1/2` for both feet).
- **Purpose**: Provides full 6-DOF wrench data (3D Force + 3D Torque) at every foot contact point.

## 2. Hardware Interface (URDF/ros2_control)
- **File**: `urdf/full_body.ros2_control.xacro`
- **Change**: Defined 8 `<sensor>` tags with `force.x/y/z` and `torque.x/y/z` state interfaces.
- **Purpose**: Maps MuJoCo simulation data into the standard `ros2_control` hardware component.

## 3. Custom ROS 2 Messages
- **Files**: `msg/FootSensor.msg`, `msg/FootSensorArray.msg`
- **Change**: Updated the message structure to include `torque_x`, `torque_y`, and `torque_z` fields.
- **Purpose**: Enables high-efficiency transmission of packed sensor arrays specifically for bipedal control logic.

## 4. Controller Configuration
- **File**: `config/full_body_controllers.yaml`
- **Change**: Configured 8 `force_torque_sensor_broadcaster` instances in `sensor_name` mode.
- **Purpose**: Exposes raw MuJoCo wrench data to standard ROS 2 `/wrench` topics.

## 5. Signal Filtering & Aggregation
- **File**: `scripts/foot_force_array_publisher.py`
- **Change**: Implemented a first-order exponential low-pass filter ($\alpha = 0.16$, ~15 Hz cutoff).
- **Purpose**: Smooths high-frequency simulation noise and bundles the 8 raw topics into two convenient arrays (`/foot_force_left` and `/foot_force_right`).

## 6. Global ZMP Calculation
- **File**: `scripts/zmp_calculation.py`
- **Change**:
    - Implemented Real-time ZMP calculation: $ZMP_x = \frac{\sum{X F_z - T_y}}{\sum{F_z}}$, $ZMP_y = \frac{\sum{Y F_z + T_x}}{\sum{F_z}}$.
    - Added **Kinematic Leg Odometry**: Dynamically broadcasts an `odom -> base_link` transform by anchoring the robot's feet to the Z=0 plane.
- **Purpose**: Provides a physically consistent global Stability Point (published to `/zmp`) in a world-fixed frame.

## 7. Environment & Infrastructure
- **System**: Fixed FastDDS Shared Memory (`SHM`) lock conflicts by clearing `/dev/shm/fastrtps*`.
- **Build**: Updated `CMakeLists.txt` to install and register all Python executable nodes.

## how to run?
- ros2 launch full_body_mujoco full_body_mujoco.launch.py
- ros2 run full_body_mujoco zmp_calculation.py
- ros2 run rviz2 rviz2
- ros2 run full_body_mujoco com_calculation.py
