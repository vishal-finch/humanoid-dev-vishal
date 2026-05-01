#!/usr/bin/env python3
"""
Center of Mass (COM) Calculator Node

Parses the URDF robot_description for link masses and local COM offsets,
uses tf2 to transform each link's COM into the global 'odom' frame,
and publishes the whole-body COM as a PointStamped on /com.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker
import tf2_ros

from urdf_parser_py.urdf import URDF


class COMCalculator(Node):
    def __init__(self):
        super().__init__('com_calculator',
                         allow_undeclared_parameters=True,
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Publishers
        self.com_pub = self.create_publisher(PointStamped, 'com', 10)
        self.marker_pub = self.create_publisher(Marker, 'com_marker', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Parse URDF from robot_description parameter (published by robot_state_publisher)
        self.links_data = []  # list of (link_name, mass, local_com_xyz)
        self.urdf_parsed = False

        # Subscribe with TRANSIENT_LOCAL QoS to receive the latched robot_description
        from std_msgs.msg import String
        qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        self.desc_sub = self.create_subscription(
            String, 'robot_description', self.urdf_cb, qos)

        # Main loop at 50 Hz
        self.timer = self.create_timer(0.02, self.calculate_com)

    def urdf_cb(self, msg):
        """Parse the URDF when robot_description arrives."""
        if self.urdf_parsed:
            return
        try:
            robot = URDF.from_xml_string(msg.data)

            for link in robot.links:
                if link.inertial is not None and link.inertial.mass > 0.001:
                    com_xyz = link.inertial.origin.xyz if link.inertial.origin else [0, 0, 0]
                    self.links_data.append((link.name, link.inertial.mass, com_xyz))

            total_mass = sum(m for _, m, _ in self.links_data)
            self.get_logger().info(
                f'Parsed URDF: {len(self.links_data)} links, total mass = {total_mass:.3f} kg')
            self.urdf_parsed = True

        except Exception as e:
            self.get_logger().warn(f'URDF parse failed: {e}')

    def calculate_com(self):
        if not self.urdf_parsed or len(self.links_data) == 0:
            return

        # Broadcast odom -> base_link (Leg Odometry) if needed
        try:
            self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
        except Exception:
            try:
                tf_base_foot = self.tf_buffer.lookup_transform('base_link', 'foot_l', rclpy.time.Time())
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'odom'
                t.child_frame_id = 'base_link'
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = -tf_base_foot.transform.translation.z
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)
                return  # Wait for next tick
            except Exception:
                return

        # Weighted sum: COM = Sum(m_i * p_i) / Sum(m_i)
        total_mass = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0

        for link_name, mass, local_com in self.links_data:
            try:
                # Get transform from link frame to odom
                tf = self.tf_buffer.lookup_transform('odom', link_name, rclpy.time.Time())
                
                # The link origin in odom
                lx = tf.transform.translation.x
                ly = tf.transform.translation.y
                lz = tf.transform.translation.z

                # Rotate the local COM offset into odom frame
                # For simplicity using quaternion rotation
                qx = tf.transform.rotation.x
                qy = tf.transform.rotation.y
                qz = tf.transform.rotation.z
                qw = tf.transform.rotation.w

                # Quaternion rotation of the local COM vector
                cx, cy, cz = local_com
                # v' = q * v * q_inv  (Hamilton product shortcut)
                t_2 = 2.0
                ix = qw * cx + qy * cz - qz * cy
                iy = qw * cy + qz * cx - qx * cz
                iz = qw * cz + qx * cy - qy * cx
                iw = -qx * cx - qy * cy - qz * cz

                rx = ix * qw + iw * (-qx) + iy * (-qz) - iz * (-qy)
                ry = iy * qw + iw * (-qy) + iz * (-qx) - ix * (-qz)
                rz = iz * qw + iw * (-qz) + ix * (-qy) - iy * (-qx)

                # Global COM of this link = link_origin_in_odom + rotated_local_com
                gx = lx + rx
                gy = ly + ry
                gz = lz + rz

                weighted_x += mass * gx
                weighted_y += mass * gy
                weighted_z += mass * gz
                total_mass += mass

            except Exception:
                continue  # Skip links with missing transforms

        if total_mass < 0.1:
            return

        com_x = weighted_x / total_mass
        com_y = weighted_y / total_mass
        com_z = weighted_z / total_mass

        # Publish COM PointStamped
        com_msg = PointStamped()
        com_msg.header.stamp = self.get_clock().now().to_msg()
        com_msg.header.frame_id = 'odom'
        com_msg.point.x = com_x
        com_msg.point.y = com_y
        com_msg.point.z = com_z
        self.com_pub.publish(com_msg)

        # Publish Visualization Marker (Green Sphere)
        marker = Marker()
        marker.header = com_msg.header
        marker.ns = "com"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = com_msg.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = COMCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
