#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = (
            self.get_parameter("scan_topic").get_parameter_value().string_value
        )
        self.DRIVE_TOPIC = (
            self.get_parameter("drive_topic").get_parameter_value().string_value
        )
        self.SIDE = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY = (
            self.get_parameter("velocity").get_parameter_value().double_value
        )
        self.DESIRED_DISTANCE = (
            self.get_parameter("desired_distance").get_parameter_value().double_value
        )

        self.lidar_subscription = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self.timer_callback, 10
        )
        self.lidar_subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped, self.DRIVE_TOPIC, 10
        )

        self.wall_pub = self.create_publisher(Marker, "wall_marker", 10)

        # Setting the PID
        self.controller = PID(-30, -1.4, 0.00001, 0.02)

        self.get_logger().info("Wall Follower has been started")

    def timer_callback(self, msg: LaserScan):
        self.SIDE = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY = (
            self.get_parameter("velocity").get_parameter_value().double_value
        )
        self.DESIRED_DISTANCE = (
            self.get_parameter("desired_distance").get_parameter_value().double_value
        )

        if self.SIDE == 1:
            self.left_control(msg)
        elif self.SIDE == -1:
            self.right_control(msg)

    def approx_wall(self, xs, ys):
        """
        Given the rangess scan and a side, run a least squares regression to find the wall and find closest distance to the wall.
        Returns: a list of predicted wall positions based on angle and lidar scan
        """
        slope, intercept = np.polyfit(xs, ys, 1)

        return slope, intercept

    def left_control(self, msg: LaserScan):

        ranges = msg.ranges
        length = len(ranges)
        front = ranges[2 * length // 5 : 3 * length // 5]
        left = ranges[3 * length // 5 : 4 * length // 5]
        far_left = ranges[4 * length // 5 :]
        angles = np.linspace(msg.angle_min, msg.angle_max, length)

        xs = np.array(ranges) * np.cos(angles)
        ys = np.array(ranges) * np.sin(angles)

        slope, intercept = self.approx_wall(
            xs[(3 * length) // 5 : int(4 * length) // 5],
            ys[(3 * length) // 5 : int(4 * length) // 5],
        )

        x = np.linspace(0, 2, 200)
        y = slope * x + intercept

        self.get_logger().info("slope " + str(slope))

        VisualizationTools.plot_line(x, y, self.wall_pub, frame="/laser")

        closest_dist = np.min(y)

        self.controller.update_control(self.DESIRED_DISTANCE - closest_dist)
        control = self.controller.get_control()
        self.get_logger().info("Control: " + str(control))

        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = self.VELOCITY

        ack_msg.drive.steering_angle = control
        self.publisher_.publish(ack_msg)

    def right_control(self, msg: LaserScan):
        ranges = msg.ranges
        length = len(ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, length)

        xs = np.array(ranges) * np.cos(angles)
        ys = np.array(ranges) * np.sin(angles)

        slope, intercept = self.approx_wall(
            xs[(1 * length) // 5 : int(2 * length) // 5],
            ys[(1 * length) // 5 : int(2 * length) // 5],
        )

        x = np.linspace(0, 2, 200)
        y = slope * x + intercept

        VisualizationTools.plot_line(x, y, self.wall_pub, frame="/laser")

        closest_dist = np.min(abs(y))

        self.controller.update_control(-(self.DESIRED_DISTANCE - closest_dist))
        control = self.controller.get_control()
        self.get_logger().info("Control: " + str(control))

        ack_msg = AckermannDriveStamped()
        ack_msg.drive.speed = self.VELOCITY
        self.get_logger().info("current error:" + str(self.controller.curr_error))
        ack_msg.drive.steering_angle = control
        self.publisher_.publish(ack_msg)


class PID:
    def __init__(self, Kp, Td, Ti, dt):
        self.Kp = Kp
        self.Td = Td
        self.Ti = Ti
        self.curr_error = 0
        self.prev_error = 0
        self.sum_error = 0
        self.prev_error_deriv = 0
        self.curr_error_deriv = 0
        self.control = 0
        self.dt = dt

    def update_control(self, current_error, reset_prev=False):
        self.prev_error = self.curr_error
        self.curr_error = current_error

        # Calculating the integral error
        self.sum_error = self.sum_error + self.curr_error * self.dt

        # Calculating the derivative error
        self.curr_error_deriv = (self.curr_error - self.prev_error) / self.dt

        # Calculating the PID Control
        self.control = (
            self.Kp * self.curr_error
            + self.Ti * self.sum_error
            + self.Td * self.curr_error_deriv
        )

    def get_control(self):
        return self.control


def main():

    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
