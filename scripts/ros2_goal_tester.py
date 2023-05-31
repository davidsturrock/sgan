import rclpy
import numpy as np
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
    def __init__(self, pub_topic='/goal', ped_topic='/tracked/pedestrians', rate=1):
        super().__init__('goal_tester')
        self.pub_topic = pub_topic
        self.ped_topic = ped_topic
        self.goal_publisher = None
        self.ped_publisher = None
        self.rate_value = rate

    def setup(self):
        self.goal_publisher = self.create_publisher(Point, self.pub_topic, 1)
        self.ped_publisher = self.create_publisher(Point, self.ped_topic, 1)
        self.timer = self.create_timer(1.0 / self.rate_value, self.publish_circle)

    def publish_circle(self):
        radius_values = np.linspace(1, 4, 8)
        theta_values = np.linspace(np.pi, 2 * np.pi, 10)
        
        for radius in radius_values:
            for theta in theta_values:
                point = Point()
                point.x = radius * np.sin(theta)
                point.y = radius * np.cos(theta)
                point.z = 0
                self.goal_publisher.publish(point)
                self.ped_publisher.publish(point)

def main(args=None):
    rclpy.init(args=args)
    pub = PointPublisher()
    pub.setup()
    rclpy.spin(pub)
    pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
