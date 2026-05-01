#!/usr/bin/env python3
"""
Standing Calibration — Direct Setpoint

Sends the pre-calibrated ankle offsets that align ZMP ≈ COM.
Values were found experimentally:
  ankle_pitch = -0.0183 rad  →  X_err ≈ -3.8mm (negligible)
  ankle_roll  = -0.0800 rad  →  Y_err ≈ -0.2mm (converged)

Usage:
    ros2 run full_body_mujoco pd_control.py
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class StandingCalibration(Node):
    def __init__(self):
        super().__init__('standing_calibration',
                         allow_undeclared_parameters=True,
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # ── Pre-calibrated ankle offsets ──────────────────────────────────────
        self.ankle_pitch_offset = -0.0183   # fixes X error
        self.ankle_roll_offset  = -0.0800   # fixes Y error

        # ── Publisher ────────────────────────────────────────────────────────
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/lower_body_controller/joint_trajectory', 10)

        # Send once after a short delay to let the controller come up
        self.timer = self.create_timer(2.0, self.send_calibration)
        self.sent = False

        self.get_logger().info('Standing Calibration ready. Will send offsets in 2s...')

    def send_calibration(self):
        if self.sent:
            return

        traj = JointTrajectory()
        traj.joint_names = [
            'ankle_pitch_r', 'ankle_pitch_l',
            'ankle_roll_r', 'ankle_roll_l',
        ]

        point = JointTrajectoryPoint()
        point.positions = [
            self.ankle_pitch_offset, self.ankle_pitch_offset,
            self.ankle_roll_offset, self.ankle_roll_offset,
        ]
        point.time_from_start = Duration(sec=1, nanosec=0)  # smooth 1s transition
        traj.points = [point]

        self.traj_pub.publish(traj)
        self.sent = True

        self.get_logger().info(
            f'✅ Calibration sent!  pitch={self.ankle_pitch_offset:+.4f}rad  '
            f'roll={self.ankle_roll_offset:+.4f}rad')
        self.get_logger().info('Robot should now have ZMP ≈ COM. You can Ctrl+C this node.')


def main(args=None):
    rclpy.init(args=args)
    node = StandingCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
