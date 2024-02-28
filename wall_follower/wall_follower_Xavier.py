
#!/usr/bin/env python3
import math
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
        self.max_speed = 4 # meters/s
        self.max_steering_angle = 0.34 #radians
        
        
        self.last_steer = 0
        self.dist_Ks = (24, 0, 1.5)
        self.steer_Ks = (.8, 0, .5)
        self.previousErrorD = 0
        self.previousErrorS = 0
        self.integralD = 0
        self.integralS = 0
        self.line_params = 0,0

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        
    # TODO: Initialize your publishers and subscribers here
        self.publisher1 = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.laser_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.laser_callback, 10)
        self.line_pub = self.create_publisher(Marker, "/wall", 1)
        self.laser_sub  # prevent unused variable warning
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def laser_callback(self, msg):  
        self.get_logger().info("calling laser_callback")
        #Listening to Lasers
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        range_min = 1/8 * math.pi
        range_max = range_min + math.pi/2
        # print(range_min, range_max)
        # print(self.SIDE, self.SIDE == -1)
        if self.SIDE == 1:
            range_min, range_max = -range_max, -range_min
        # print(range_min, range_max)
        # print(f'{msg.angle_min=}, {msg.angle_max=}')
        sliced_scan = self.slice_laser_scan(msg, range_min, range_max)
        self.line_params = self.least_squares_regression(sliced_scan[1], sliced_scan[0])

    def timer_callback(self):
        self.get_logger().info("calling timer_callback")
        dt = .05
        try:
            m, b = self.line_params
        except AttributeError:
            self.get_logger().info("m or b is bad")
            return
        
        self.PID_dist = self.PID_func(b, self.SIDE*self.DESIRED_DISTANCE, self.last_steer, self.dist_Ks, self.previousErrorD, self.integralD, dt)
        self.PID_slope = self.PID_func(m, 0, self.last_steer, self.steer_Ks, self.previousErrorS, self.integralS, dt)
        # print(f'{self.PID_slope=}')
        # print(f'{self.PID_dist=}')
        distance, self.previousErrorD, self.integralD = self.PID_dist
        slope, self.previousErrorS, self.integralS = self.PID_slope
        steer_command = float(distance + slope)
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.header.frame_id = 'map'
        ack_msg.drive.steering_angle = steer_command
        ack_msg.drive.speed = self.VELOCITY
        self.get_logger().info("haah")
        self.publisher1.publish(ack_msg)
        self.last_steer = ack_msg.drive.steering_angle

    def slice_laser_scan(self, msg, range_min, range_max):
        '''
        Takes LaserScan type, specified range, and returns
        cartesian coords values of LaserScan within specified range
        '''
        range_min = max(msg.angle_min, min(range_min, msg.angle_max))
        range_max = max(range_min, min(range_max, msg.angle_max))
        # print(f" AFTER CLAMP: {range_min=}, {range_max=}")
        # Calculate lower and upper indices for slicing msg.ranges
        lower = int((range_min - msg.angle_min) / msg.angle_increment)
        upper = int((range_max- msg.angle_min) / msg.angle_increment)
        
        # Slice msg.ranges to get wallScan
        wallScan = msg.ranges[lower:upper]
        # Convert wallScan to cartesian coordinates
        angles = np.linspace(range_min, range_max, len(wallScan))
        x = wallScan * np.cos(angles)
        y = wallScan * np.sin(angles)
        dist = np.sqrt(x ** 2 + y ** 2) <= self.DESIRED_DISTANCE * 4
        x = x[dist]
        y = y[dist]

        VisualizationTools.plot_line(x, y, self.line_pub, frame="/laser")
        return x, y
    
    def least_squares_regression(self, x, y):
        '''
        Perform least squares regression to fit a line to the given data points.
        Returns tuple containing fit parameters (m, b) for y = mx + b
        '''
        # Convert input data to NumPy arrays
        x = np.array(x)
        y = np.array(y)
        # Calculate necessary sums
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x_squared = np.sum(x**2)
        sum_xy = np.sum(x * y)
        # Calculate slope (m) and y-intercept (b) using least squares formula
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
        b = (sum_y - m * sum_x) / n
        print(f' {m=}, {b=}')
        return m, b
    def PID_func(self, current_val, setpoint, steering, constants, previousError, Integral, dt):
        # compute the error
        error = setpoint - current_val
        KP, KI, KD = constants
        # PID computation
        Integral = Integral + error * dt
        Derivative = (error - previousError) / dt
        output = steering + (KP * error + KI * Integral + KD * Derivative)
        # constrain the wheel speed to lie between -max_steering_angle and max_steering_angle
        if output > self.max_steering_angle:
            output = self.max_steering_angle
        elif output < -self.max_steering_angle:
            output = -self.max_steering_angle
        previousError = error
        return (output, previousError, Integral)
def main():
    
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
