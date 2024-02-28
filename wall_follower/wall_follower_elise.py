#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan # message detials: http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html
from ackermann_msgs.msg import AckermannDriveStamped # message details: http://docs.ros.org/en/jade/api/ackermann_msgs/html/msg/AckermannDrive.html
from visualization_msgs.msg import Marker # message details: http://docs.ros.org/en/noetic/api/visualization_msgs/html/msg/Marker.html

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        ## Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")
        self.declare_parameter('max_velocity', 4) # meters per second
        self.declare_parameter('max_steering_angle', 0.34) # radians

        ## Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle')
            

        ## Set control variables
        self.Kp = 5.8 # 5.8
        self.Kd = 0.3 # 0.5
        self.prev_error = 0
		
	    ## Initialize your publishers and subscribers here
        self.drive_publisher_ = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.lst_sqrs_publisher = self.create_publisher(Marker, '/wall', 10)
        self.subscription = self.create_subscription(
            LaserScan, 
            self.SCAN_TOPIC, 
            self.listener_callback, 
            10)
        self.subscription # avoids a not used error
    
   
    def listener_callback(self, scan_msg): # scan_msg is the laser scan
        ## Added updates so tests run correctly
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        ## Find wall and create marker
        x, y_fitted = self.recognize_wall(scan_msg) # create wall data, read which wall to follow, and return x and y_fitted to plot
        VisualizationTools.plot_line(x, y_fitted, self.lst_sqrs_publisher, frame = "/laser")

        ## Use the data and a control system (PD)
        drive_msg = self.create_drive_command(y_fitted) # use PD control sequence to give steering commands

        ## Log and publish
        self.drive_publisher_.publish(drive_msg)
        self.get_logger().info(f'LaserScan read and drive commands were published.')


    def recognize_wall(self, scan_msg):
        ## Read info from the LaserScan
        angle_min = scan_msg.angle_min  
        angle_max = scan_msg.angle_max  
        angle_increment = scan_msg.angle_increment 
        ranges = np.array(scan_msg.ranges)
        angles = np.arange(angle_min, angle_max, angle_increment)
        data = np.vstack([ranges * np.cos(angles), ranges * np.sin(angles)]) # data has two rows, first is x and second is y and should be shape (2, n)
        
        ## Filter data to read based on what wall we want
        if self.SIDE == 1: # left --> y > 0
            indices = [i for i in range(data.shape[1]) if data[0, i] >= 0 and data[0, i] <= 3.5 and data[1, i] > -1] 
            wall = data[:, indices]
        else: # right --> y < 0
            indices = [i for i in range(data.shape[1]) if data[0, i] >= 0 and data[0, i] <= 3.5 and data[1, i] < 1] 
            wall = data[:, indices]

        ## Create least squares regression
        x = wall[0, :]
        y = wall[1, :]
        m, b = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
        y_fitted = m * x + b

        ## Return x and fitted y data
        return x, y_fitted
    

    def create_drive_command(self, y_fitted):
        ## Use a PD controller to calculate the control value 
        current_error =  np.mean(y_fitted) - self.SIDE * self.DESIRED_DISTANCE 
        k_angle = 0.1 if abs(np.mean(y_fitted)) >= 1.1 else 1 # a proportional steering controller so if far it doesn't turn too quick
        k_speed = 0.5 if abs(np.mean(y_fitted)) <= 0.5 else 1 # a proportional speed controller so slows down if near wall
        control_value = self.Kp*current_error + self.Kd*(current_error - self.prev_error)
        self.prev_error = current_error # for next round, error updates

        ## Create message to publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = k_angle*control_value 
        drive_msg.drive.speed = k_speed*self.VELOCITY
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"  
        
        ## Return message
        return drive_msg


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
