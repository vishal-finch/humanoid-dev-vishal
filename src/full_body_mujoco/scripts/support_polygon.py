#!/usr/bin/env python3
"""
Support Polygon Visualizer Node

Subscribes to FootSensorArray topics, filters sensors that are in contact
with the ground (|Fz| > threshold), transforms their positions into the
global 'odom' frame, computes the 2D convex hull, and publishes it as a
LINE_STRIP marker for RViz visualization.

Stability Rule: ZMP (red) must stay inside this polygon (blue) to remain balanced.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from full_body_mujoco.msg import FootSensorArray
import tf2_ros


# Minimum |Fz| in Newtons for a sensor to be considered "in contact"
CONTACT_FORCE_THRESHOLD = 5.0


def convex_hull_2d(points):
    """Compute 2D convex hull using Andrew's monotone chain algorithm.
    Input: list of (x, y) tuples.
    Output: list of (x, y) in CCW order forming the hull."""
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def cross(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class SupportPolygon(Node):
    def __init__(self):
        super().__init__('support_polygon',
                         allow_undeclared_parameters=True,
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Subscriptions
        self.left_sub = self.create_subscription(
            FootSensorArray, 'foot_force_left', self.left_cb, 10)
        self.right_sub = self.create_subscription(
            FootSensorArray, 'foot_force_right', self.right_cb, 10)

        # Publishers
        self.marker_pub = self.create_publisher(Marker, 'support_polygon_marker', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Data
        self.latest_left = None
        self.latest_right = None

        # Timer at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_polygon)

    def left_cb(self, msg):
        self.latest_left = msg

    def right_cb(self, msg):
        self.latest_right = msg

    def get_grounded_points(self, msg, foot_frame):
        """Extract global (x, y) positions of sensors with significant ground contact."""
        points = []
        try:
            tf = self.tf_buffer.lookup_transform('odom', foot_frame, rclpy.time.Time())
        except Exception:
            return points

        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y

        for sensor in msg.sensors:
            if abs(sensor.force) < CONTACT_FORCE_THRESHOLD:
                continue  # Sensor is airborne, skip

            # Rotate local sensor position by the foot's quaternion
            cx, cy, cz = sensor.x, sensor.y, sensor.z
            ix = qw * cx + qy * cz - qz * cy
            iy = qw * cy + qz * cx - qx * cz
            iz = qw * cz + qx * cy - qy * cx
            iw = -qx * cx - qy * cy - qz * cz
            rx = ix * qw + iw * (-qx) + iy * (-qz) - iz * (-qy)
            ry = iy * qw + iw * (-qy) + iz * (-qx) - ix * (-qz)

            # Global 2D position on the floor
            gx = tx + rx
            gy = ty + ry
            points.append((gx, gy))

        return points

    def publish_polygon(self):
        if self.latest_left is None and self.latest_right is None:
            return

        # Ensure odom -> base_link exists
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
                return
            except Exception:
                return

        # Collect all grounded contact points from both feet
        all_points = []
        if self.latest_left is not None:
            all_points.extend(self.get_grounded_points(self.latest_left, 'foot_l'))
        if self.latest_right is not None:
            all_points.extend(self.get_grounded_points(self.latest_right, 'foot_r'))

        if len(all_points) < 3:
            # Need at least 3 points to form a polygon
            return

        # Compute convex hull
        hull = convex_hull_2d(all_points)
        if len(hull) < 3:
            return

        # Create LINE_STRIP marker (closed polygon on the floor)
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'odom'
        marker.ns = "support_polygon"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # Line width 5mm
        marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.9)  # Bright blue
        marker.pose.orientation.w = 1.0

        # Add hull vertices + close the loop
        for (x, y) in hull:
            marker.points.append(Point(x=x, y=y, z=0.0))
        # Close the polygon by repeating the first point
        marker.points.append(Point(x=hull[0][0], y=hull[0][1], z=0.0))

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = SupportPolygon()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
