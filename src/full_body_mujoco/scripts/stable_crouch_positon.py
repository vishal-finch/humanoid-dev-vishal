#!/usr/bin/env python3
"""
Stable Crouch Position Controller — ROS 2
==========================================
Uses pre-verified IK angles (from scipy SLSQP) and a 50 Hz PD + balance
feedback loop to achieve a stable 12cm deep crouch.

Balance feedback reads pelvis orientation from TF tree (world→base_link)
and applies ankle pitch / hip roll corrections.

Usage:
    ros2 run full_body_mujoco stable_crouch_positon.py
    ros2 run full_body_mujoco stable_crouch_positon.py --ros-args -p depth_cm:=10
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros

# Local params module
from angad_robot_params import (
    LOWER_BODY_JOINTS,
    CROUCH_IK,
    RECOMMENDED_DEPTH_CM,
    BALANCE_GAINS,
    PD_GAINS,
    ANKLE_PITCH_SHARE,
    ANKLE_ROLL_SHARE,
    HIP_ROLL_SHARE,
    get_crouch_targets,
)


class StableCrouchController(Node):

    LOOP_HZ = 50
    LOOP_DT = 1.0 / LOOP_HZ
    RAMP_TIME = 3.0          # seconds to interpolate into crouch

    def __init__(self):
        super().__init__('stable_crouch_controller',
                         allow_undeclared_parameters=True,
                         parameter_overrides=[
                             Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('depth_cm', RECOMMENDED_DEPTH_CM)
        depth_cm = self.get_parameter('depth_cm').value

        if depth_cm not in CROUCH_IK:
            avail = sorted(CROUCH_IK.keys())
            self.get_logger().error(
                f'Depth {depth_cm}cm not in verified set {avail}. Aborting.')
            raise SystemExit(1)

        if not CROUCH_IK[depth_cm][1]:
            self.get_logger().warn(
                f'Depth {depth_cm}cm is marked UNSTABLE (exceeds torque). Proceed with caution.')

        # ── Target angles ─────────────────────────────────────────────
        self.target_positions = np.array(get_crouch_targets(depth_cm))
        self.standing_positions = np.zeros(12)  # all zeros = standing
        self.current_targets = np.copy(self.standing_positions)

        self.get_logger().info(
            f'Crouch depth = {depth_cm} cm')
        self.get_logger().info(
            f'Target angles: {np.round(self.target_positions, 4).tolist()}')

        # ── Ramp state ────────────────────────────────────────────────
        self.ramp_ticks = int(self.RAMP_TIME * self.LOOP_HZ)
        self.tick = 0
        self.settle_delay = int(2.0 * self.LOOP_HZ)  # 2s settling before ramp

        # ── Balance state ─────────────────────────────────────────────
        self.pitch_filter = 0.0
        self.roll_filter = 0.0
        self.prev_pitch = 0.0
        self.prev_roll = 0.0

        # ── Joint state tracking ──────────────────────────────────────
        self.joint_positions = {}
        self.joint_velocities = {}

        # ── TF Buffer ─────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/lower_body_controller/joint_trajectory', 10)

        # ── Main loop at 50 Hz ────────────────────────────────────────
        self.timer = self.create_timer(self.LOOP_DT, self.control_loop)

        self.get_logger().info(
            f'Controller ready. Settling for 2s, then ramping over {self.RAMP_TIME}s...')

    # ── Callbacks ─────────────────────────────────────────────────────

    def joint_state_cb(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    # ── Balance: read pelvis orientation from TF ──────────────────────

    def get_pelvis_orientation(self):
        """Return (pitch, roll) in radians from TF world→base_link."""
        try:
            t = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02))
            q = t.transform.rotation
            # Extract pitch and roll from quaternion
            # pitch = rotation about X (forward tilt)
            # roll  = rotation about Y (side tilt)
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(max(-1.0, min(1.0, sinp)))

            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            return pitch, roll
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return 0.0, 0.0

    # ── Main control loop ─────────────────────────────────────────────

    def control_loop(self):
        # Phase 0: Wait for sim to settle
        if self.settle_delay > 0:
            self.settle_delay -= 1
            if self.settle_delay == 0:
                self.get_logger().info('🔄 Starting crouch ramp...')
            return

        # ═══════════════════════════════════════════════════════════════
        # 1. RAMP: interpolate standing → crouch
        # ═══════════════════════════════════════════════════════════════
        if self.tick < self.ramp_ticks:
            alpha = (self.tick + 1) / self.ramp_ticks
            self.current_targets = (
                (1.0 - alpha) * self.standing_positions +
                alpha * self.target_positions)
            self.tick += 1
        else:
            self.current_targets = np.copy(self.target_positions)

        # ═══════════════════════════════════════════════════════════════
        # 2. BALANCE: read pelvis orientation, compute corrections
        # ═══════════════════════════════════════════════════════════════
        pitch, roll = self.get_pelvis_orientation()

        # Derivative (finite difference at 50 Hz)
        d_pitch = (pitch - self.prev_pitch) * self.LOOP_HZ
        d_roll = (roll - self.prev_roll) * self.LOOP_HZ
        self.prev_pitch = pitch
        self.prev_roll = roll

        # PD balance torque
        bal_pitch = -(BALANCE_GAINS['pitch']['kp'] * pitch +
                      BALANCE_GAINS['pitch']['kd'] * d_pitch)
        bal_roll = -(BALANCE_GAINS['roll']['kp'] * roll +
                     BALANCE_GAINS['roll']['kd'] * d_roll)

        # Low-pass filter (EMA α=0.1)
        self.pitch_filter += 0.1 * (bal_pitch - self.pitch_filter)
        self.roll_filter += 0.1 * (bal_roll - self.roll_filter)

        # Convert torque correction → position delta
        # delta = torque / Kp_joint
        ankle_pitch_delta = (self.pitch_filter * ANKLE_PITCH_SHARE
                             / PD_GAINS['ankle']['kp'])
        ankle_roll_delta = (self.roll_filter * ANKLE_ROLL_SHARE
                            / PD_GAINS['ankle']['kp'])
        hip_roll_delta = (self.roll_filter * HIP_ROLL_SHARE
                          / PD_GAINS['hip']['kp'])

        # Clamp corrections to safe range
        ankle_pitch_delta = np.clip(ankle_pitch_delta, -0.1, 0.1)
        ankle_roll_delta = np.clip(ankle_roll_delta, -0.05, 0.05)
        hip_roll_delta = np.clip(hip_roll_delta, -0.05, 0.05)

        # ═══════════════════════════════════════════════════════════════
        # 3. APPLY balance corrections to targets
        # ═══════════════════════════════════════════════════════════════
        final = np.copy(self.current_targets)

        # Joint indices in LOWER_BODY_JOINTS:
        # R: hip_pitch=0, hip_roll=1, thigh_yaw=2, knee=3, ankle_pitch=4, ankle_roll=5
        # L: hip_pitch=6, hip_roll=7, thigh_yaw=8, knee=9, ankle_pitch=10, ankle_roll=11

        # Ankle pitch correction (both feet, same direction)
        final[4] += ankle_pitch_delta    # ankle_pitch_r
        final[10] += ankle_pitch_delta   # ankle_pitch_l

        # Ankle roll correction (opposite sign for L vs R)
        final[5] += ankle_roll_delta     # ankle_roll_r
        final[11] -= ankle_roll_delta    # ankle_roll_l (mirrored)

        # Hip roll correction (opposite sign for L vs R)
        final[1] += hip_roll_delta       # hip_roll_r
        final[7] -= hip_roll_delta       # hip_roll_l (mirrored)

        # ═══════════════════════════════════════════════════════════════
        # 4. PUBLISH full 12-joint trajectory
        # ═══════════════════════════════════════════════════════════════
        traj = JointTrajectory()
        traj.joint_names = LOWER_BODY_JOINTS

        point = JointTrajectoryPoint()
        point.positions = final.tolist()
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms
        traj.points = [point]

        self.traj_pub.publish(traj)

        # ═══════════════════════════════════════════════════════════════
        # 5. LOG (throttled)
        # ═══════════════════════════════════════════════════════════════
        at_target = self.tick >= self.ramp_ticks
        status = '✅ CROUCHED' if at_target else f'⏳ RAMP {self.tick}/{self.ramp_ticks}'

        self.get_logger().info(
            f'{status}  pitch={math.degrees(pitch):+5.1f}°  '
            f'roll={math.degrees(roll):+5.1f}°  '
            f'Δankle={ankle_pitch_delta:+.4f}  '
            f'knee={final[3]:.4f}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = StableCrouchController()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
