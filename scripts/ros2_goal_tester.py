import rclpy
import numpy as np
import sys
from geometry_msgs.msg import Point
from rclpy.node import Node

def pull_to_centre(x, y, r_pull=1):
    theta = np.arctan2(y, x)
    x_pull = r_pull * np.cos(theta)
    y_pull = r_pull * np.sin(theta)
    x_new = x - x_pull
    y_new = y - y_pull
    return x_new, y_new

class PointPublisher(Node):
    def __init__(self):
        super().__init__(self, pub_topic='/goal', ped_topic='/tracked/pedestrians', rate=1) 
        self.pub_topic = pub_topic
        self.ped_topic = ped_topic
        self.goal_publisher: rclpy.Publisher = None
        self.ped_publisher: rclpy.Publisher = None
        self.rate_value = rate
        
    def setup(self):
        node = rclpy.create_node('GoalTester')
        self.goal_publisher: rclpy.Publisher = node.create_publisher(Point , self.pub_topic , queue_size=1)
        self.ped_publisher: rclpy.Publisher = node.create_publisher(Point, self.ped_topic, queue_size=1)
        self.rate = rclpy.Rate(self.rate_value)

    def publish_circle(self, radius=1):
        for theta in np.linspace(np.pi, 2 * np.pi, 10):
            point = Point(radius * np.sin(theta), radius * np.cos(theta), 0)
            print(point)
            print('-'*60)
            self.goal_publisher.publish(point)
            self.ped_publisher.publish(point)
            self.rate.sleep()
            
    def main():
        pub = PointPublisher()
        pub.setup()
        while not rclpy.is_shutdown():
            for i in np.linspace(1, 4, 8):
                pub.publish_circle(radius=i) 
            rclpy.spin(pub)   
        
        
if __name__ == '__main__':
    main()