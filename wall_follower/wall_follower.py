#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
import math

from wall_follower.visualization_tools import VisualizationTools

class WallFollower(Node):

   def __init__(self):
       super().__init__("wall_follower")
       # Declare parameters to make them available for use
       self.declare_parameter("scan_topic", "/scan")
       self.declare_parameter("drive_topic", "/drive")
       self.declare_parameter("side", -1)
       self.declare_parameter("velocity", 1.0)
       self.declare_parameter("desired_distance", 1.)
       


       # Fetch constants from the ROS parameter server
       self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
       self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
       self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
       self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
       self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

       self.Kp = 1.5
       self.Kd = 0.13
       self.previous_error = 0.0
       

       # TODO: Initialize your publishers and subscribers here
       self.laser_subscription = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.laser_callback, 10)
       self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

   def find_min_distance(self, msg):
       ranges = np.array(msg.ranges)
       angles = np.linspace(msg.angle_min, msg.angle_max, 100)

       if self.SIDE == -1:
           start_index = 5
           end_index = 30
       else:
           start_index = 70
           end_index = 94
      
       ranges = ranges[start_index:end_index]
       angles = angles[start_index:end_index]

       x_coords = ranges*np.cos(angles)
       y_coords = ranges*np.sin(angles)


       # Fit a line to these coordinates using polyfit (1st degree polynomial)
       m, b = np.polyfit(x_coords, y_coords, 1)
  
       # Define the line function
       # line_func = lambda x: coefficients[0] * x + coefficients[1]
  
       # Calculate perpendicular distances to the line from the robot position
       # For line ax + by + c = 0 and point (x0, y0), the distance d = |ax0 + by0 + c| / sqrt(a^2 + b^2)
       # Our line is y = mx + b, which can be rewritten as mx - y + b = 0, giving a = m, b = -1, c = b
       
       min_distance = np.abs(b) / np.sqrt(m**2 + 1)
  
       # Since we're interested in the wall, we consider only distances for points on the line within our ranges

       front_ranges = (np.array(msg.ranges))[45:55]
       
       front_distance = np.mean(front_ranges)
       if front_distance < 2.5:
           wall_soon = True
       else:
            wall_soon = False

       return min_distance, wall_soon

   def calculate_steering_angle(self, min_distance):
       
       min_steer = -0.34
       max_steer = 0.34
       

       # Calculate the error
       error = self.DESIRED_DISTANCE - min_distance
  
       # Calculate the derivative of the error
       derivative = error - self.previous_error
  
       # Calculate the steering angle using PD control
       control = self.Kp * error + self.Kd * derivative


       # Update previous error
       self.previous_error = error
  
       # Optionally adjust PD gains based on velocity for more dynamic control
       # For example, reduce Kp and Kd at higher speeds to make the control less aggressive
       # if velocity > some_threshold:
       #     adjusted_Kp = Kp * some_factor
       #     adjusted_Kd = Kd * some_factor
       #     steering_angle = adjusted_Kp * error + adjusted_Kd * derivative

       control = np.clip(control, min_steer, max_steer)
       steering_angle = -self.SIDE*control
       return steering_angle


   def laser_callback(self, msg):
       # velocity
       # velocity = self.velocity

       self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
       self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
       self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

       min_distance = self.find_min_distance(msg)[0]

       steering_angle = self.calculate_steering_angle(min_distance)
       # steering_angle = 0.0
       
       # Publish a constant drive command
       drive_msg = AckermannDriveStamped()
       drive_msg.drive.speed = self.VELOCITY  # Set your desired constant speed


       wall_soon = self.find_min_distance(msg)[1]
       if wall_soon == True:
           steering_angle = -self.SIDE*0.34

       drive_msg.drive.steering_angle = steering_angle  # positive angle is CCW
      
       self.drive_pub.publish(drive_msg)
      


   # TODO: Write your callback functions here  
#    def scan_callback(self, msg):
#        min_distance = self.find_min_distance(msg)
#        print(min_distance)
      
#        # Publish command
#        self.publish_constant_command(msg)
      
      
   




def main():
  
   rclpy.init()
   wall_follower = WallFollower()
   rclpy.spin(wall_follower)
   wall_follower.destroy_node()
   rclpy.shutdown()


if __name__ == '__main__':
   main()
  



