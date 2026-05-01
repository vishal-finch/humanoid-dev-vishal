#!/usr/bin/env python3
"""
Subscribes to the 8 raw /wrench topics from the force_torque_sensor_broadcasters,
applies a first-order exponential low-pass filter, and publishes aggregated
FootSensorArray messages on /foot_force_left and /foot_force_right.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from full_body_mujoco.msg import FootSensor, FootSensorArray

# ── Filter tuning ───────────────────────────────────────────────────────────
# Alpha = dt / (RC + dt), where RC = 1/(2*pi*fc).
# With fc ~15 Hz and dt ~2 ms (500 Hz), alpha ≈ 0.16.
# Lower alpha = smoother but more lag.  Higher = sharper but noisier.
FILTER_ALPHA = 0.16

# Sensor site positions (from MuJoCo XML, in foot frame)
SENSOR_POSITIONS = {
    'foot_l_heel_1': (0.0, -0.048, -0.009),
    'foot_l_heel_2': (0.015, -0.048, -0.009),
    'foot_l_toe_1': (0.0, 0.075, -0.009),
    'foot_l_toe_2': (-0.015, 0.075, -0.009),
    'foot_r_heel_1': (0.0, -0.048, -0.009),
    'foot_r_heel_2': (-0.015, -0.048, -0.009),
    'foot_r_toe_1': (0.0, 0.075, -0.009),
    'foot_r_toe_2': (0.015, 0.075, -0.009),
}


class LowPassFilter:
    """Simple first-order exponential moving average filter."""
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, raw):
        if self.value is None:
            self.value = raw
        else:
            self.value += self.alpha * (raw - self.value)
        return self.value


class FootForceArrayPublisher(Node):
    def __init__(self):
        super().__init__('foot_force_array_publisher')

        # Subscribe to the raw /wrench topics (NOT wrench_filtered)
        self.sensor_topics = [
            ('foot_l_heel_1', '/contact_force_l_heel_1_broadcaster/wrench'),
            ('foot_l_heel_2', '/contact_force_l_heel_2_broadcaster/wrench'),
            ('foot_l_toe_1', '/contact_force_l_toe_1_broadcaster/wrench'),
            ('foot_l_toe_2', '/contact_force_l_toe_2_broadcaster/wrench'),
            ('foot_r_heel_1', '/contact_force_r_heel_1_broadcaster/wrench'),
            ('foot_r_heel_2', '/contact_force_r_heel_2_broadcaster/wrench'),
            ('foot_r_toe_1', '/contact_force_r_toe_1_broadcaster/wrench'),
            ('foot_r_toe_2', '/contact_force_r_toe_2_broadcaster/wrench'),
        ]

        # Per-sensor, per-axis filters (6 axes: fx, fy, fz, tx, ty, tz)
        self.filters = {}
        self.latest = {}
        for label, _ in self.sensor_topics:
            self.filters[label] = {
                'fx': LowPassFilter(FILTER_ALPHA),
                'fy': LowPassFilter(FILTER_ALPHA),
                'fz': LowPassFilter(FILTER_ALPHA),
                'tx': LowPassFilter(FILTER_ALPHA),
                'ty': LowPassFilter(FILTER_ALPHA),
                'tz': LowPassFilter(FILTER_ALPHA),
            }
            self.latest[label] = {'fx': 0.0, 'fy': 0.0, 'fz': 0.0,
                                   'tx': 0.0, 'ty': 0.0, 'tz': 0.0}

        for label, topic in self.sensor_topics:
            self.create_subscription(WrenchStamped, topic, self.make_cb(label), 10)

        self.left_pub = self.create_publisher(FootSensorArray, 'foot_force_left', 10)
        self.right_pub = self.create_publisher(FootSensorArray, 'foot_force_right', 10)
        self.timer = self.create_timer(0.02, self.publish_arrays)  # 50 Hz output

    def make_cb(self, label):
        def cb(msg):
            f = self.filters[label]
            self.latest[label] = {
                'fx': f['fx'].update(msg.wrench.force.x),
                'fy': f['fy'].update(msg.wrench.force.y),
                'fz': f['fz'].update(msg.wrench.force.z),
                'tx': f['tx'].update(msg.wrench.torque.x),
                'ty': f['ty'].update(msg.wrench.torque.y),
                'tz': f['tz'].update(msg.wrench.torque.z),
            }
        return cb

    def publish_arrays(self):
        left_labels = ['foot_l_heel_1', 'foot_l_heel_2', 'foot_l_toe_1', 'foot_l_toe_2']
        right_labels = ['foot_r_heel_1', 'foot_r_heel_2', 'foot_r_toe_1', 'foot_r_toe_2']

        left_array = FootSensorArray()
        right_array = FootSensorArray()
        for label in left_labels:
            d = self.latest[label]
            x, y, z = SENSOR_POSITIONS[label]
            sensor = FootSensor(label=label, force=d['fz'], x=x, y=y, z=z,
                                torque_x=d['tx'], torque_y=d['ty'], torque_z=d['tz'])
            left_array.sensors.append(sensor)
        for label in right_labels:
            d = self.latest[label]
            x, y, z = SENSOR_POSITIONS[label]
            sensor = FootSensor(label=label, force=d['fz'], x=x, y=y, z=z,
                                torque_x=d['tx'], torque_y=d['ty'], torque_z=d['tz'])
            right_array.sensors.append(sensor)
        self.left_pub.publish(left_array)
        self.right_pub.publish(right_array)


def main(args=None):
    rclpy.init(args=args)
    node = FootForceArrayPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
