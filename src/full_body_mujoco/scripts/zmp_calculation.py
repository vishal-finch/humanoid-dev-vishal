#!/usr/bin/env python3
"""
Zero Moment Point (ZMP) Calculator Node

Subscribes to the filtered FootSensorArray topics for both feet,
uses tf2 to transform local forces and torques into the global 'odom' frame,
calculates the overall ZMP, and publishes it as a PointStamped message.
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped, Point, Vector3, Vector3Stamped, TransformStamped
from visualization_msgs.msg import Marker
from full_body_mujoco.msg import FootSensorArray
import tf2_ros
from tf2_geometry_msgs import do_transform_point, do_transform_vector3


class ZMPCalculator(Node):
    def __init__(self):
        super().__init__('zmp_calculator', 
                         allow_undeclared_parameters=True, 
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Subscriptions to the pre-filtered force data arrays
        self.left_sub = self.create_subscription(
            FootSensorArray, 'foot_force_left', self.left_cb, 10)
        self.right_sub = self.create_subscription(
            FootSensorArray, 'foot_force_right', self.right_cb, 10)

        # Publisher for the ZMP position
        self.zmp_pub = self.create_publisher(PointStamped, 'zmp', 10)
        self.marker_pub = self.create_publisher(Marker, 'zmp_marker', 10)

        # Setup TF buffer and listener to get transforms to global frame (odom)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Frame configuration
        self.global_frame = 'odom'
        self.left_frame = 'foot_l'
        self.right_frame = 'foot_r'

        # Store latest data
        self.latest_left = None
        self.latest_right = None

        # Timer to calculate and publish ZMP at 50 Hz
        self.timer = self.create_timer(0.02, self.calculate_zmp)

    def left_cb(self, msg):
        self.latest_left = msg

    def right_cb(self, msg):
        self.latest_right = msg

    def process_sensors(self, msg, frame_id, transform):
        """Transform all sensor forces, torques, and positions into global frame"""
        processed = []
        for sensor in msg.sensors:
            # 1. Transform Position
            pt_local = PointStamped()
            pt_local.header.frame_id = frame_id
            pt_local.point.x = sensor.x
            pt_local.point.y = sensor.y
            pt_local.point.z = sensor.z
            pt_global = do_transform_point(pt_local, transform)

            # 2. Transform Force 
            force_local = Vector3Stamped()
            force_local.header.frame_id = frame_id
            force_local.vector.x = 0.0
            force_local.vector.y = 0.0
            force_local.vector.z = sensor.force
            force_global = do_transform_vector3(force_local, transform)

            # 3. Transform Torque
            torque_local = Vector3Stamped()
            torque_local.header.frame_id = frame_id
            torque_local.vector.x = sensor.torque_x
            torque_local.vector.y = sensor.torque_y
            torque_local.vector.z = sensor.torque_z
            torque_global = do_transform_vector3(torque_local, transform)
            
            processed.append({
                'pos': pt_global.point,
                'force': force_global.vector,
                'torque': torque_global.vector
            })
        return processed

    def calculate_zmp(self):
        left_msg = self.latest_left
        right_msg = self.latest_right

        if left_msg is None and right_msg is None:
            self.get_logger().warn(
                "Waiting for /foot_force_left data... Is foot_force_array_publisher.py running?",
                throttle_duration_sec=5.0)
            return  # No data yet
            
        try:
            # Look up foot poses relative to the robot's root (base_link)
            left_tf = None
            right_tf = None
            now = rclpy.time.Time()
            
            if left_msg is not None:
                left_tf = self.tf_buffer.lookup_transform('base_link', self.left_frame, now)
            if right_msg is not None:
                right_tf = self.tf_buffer.lookup_transform('base_link', self.right_frame, now)
                
            # Leg Odometry: Dynamically attach base_link to an absolute global ground plane (odom)
            if left_tf is not None:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'odom'
                t.child_frame_id = 'base_link'
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                # By defining ground at Z=0, the pelvis is exactly at positive Z = foot_depth
                t.transform.translation.z = -left_tf.transform.translation.z
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().warn(f'Leg Odometry Kinematics Failed: {e}')
            return

        sensors_global = []
        if left_msg is not None and left_tf is not None:
            sensors_global.extend(self.process_sensors(left_msg, self.left_frame, left_tf))
        
        if right_msg is not None and right_tf is not None:
            sensors_global.extend(self.process_sensors(right_msg, self.right_frame, right_tf))

        # ZMP Equation
        sum_fz = 0.0
        sum_moment_y = 0.0  # Used for ZMP X
        sum_moment_x = 0.0  # Used for ZMP Y

        for s in sensors_global:
            fz = s['force'].z
            sum_fz += fz
            
            # Since our Leg Odometry has base_link X/Y perfectly aligned with odom X/Y,
            # we simply compute the base_link geometric ZMP.
            sum_moment_y += (s['pos'].x * fz) - s['torque'].y
            sum_moment_x += (s['pos'].y * fz) + s['torque'].x

        # Deadband threshold to avoid div by zero noise when airborne
        if abs(sum_fz) < 10.0:  
            return

        zmp_x = sum_moment_y / sum_fz
        zmp_y = sum_moment_x / sum_fz

        # Publish the ZMP
        zmp_msg = PointStamped()
        zmp_msg.header.stamp = self.get_clock().now().to_msg()
        zmp_msg.header.frame_id = 'odom'
        zmp_msg.point.x = zmp_x
        zmp_msg.point.y = zmp_y
        zmp_msg.point.z = 0.0  # ZMP is calculated on the level ground surface

        self.zmp_pub.publish(zmp_msg)

        # Publish a Visualization Marker (Red Sphere)
        marker = Marker()
        marker.header = zmp_msg.header
        marker.ns = "zmp"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = zmp_msg.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = ZMPCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
