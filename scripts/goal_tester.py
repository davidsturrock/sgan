import rospy
import numpy as np
import sys
from geometry_msgs.msg import Point

def pull_to_centre(x, y, r_pull=1):
    theta = np.arctan2(y, x)
    # radius = np.linalg.norm(x,y)
    x_pull = r_pull * np.cos(theta)
    y_pull = r_pull * np.sin(theta)
    x_new = x - x_pull
    y_new = y - y_pull
    return x_new, y_new


class PointPublisher:
    def __init__(self, pub_topic='/goal', ped_topic='/tracked/pedestrians', rate=1):
        self.pub_topic = pub_topic
        self.ped_topic = ped_topic
        self.goal_publisher: rospy.Publisher = None
        self.ped_publisher: rospy.Publisher = None
        self.rate_value = rate

    def setup(self):
        rospy.init_node('GoalTester')
        self.goal_publisher: rospy.Publisher = rospy.Publisher(self.pub_topic, Point, queue_size=1)
        self.ped_publisher: rospy.Publisher = rospy.Publisher(self.ped_topic, Point, queue_size=1)
        self.rate = rospy.Rate(self.rate_value)

    def publish_circle(self, radius=1):
        for theta in np.linspace(np.pi, 2 * np.pi, 10):
            point = Point(radius * np.sin(theta), radius * np.cos(theta), 0)
            print(point)
            print('-'*60)
            self.goal_publisher.publish(point)
            self.ped_publisher.publish(point)
            self.rate.sleep()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pub = PointPublisher()
    pub.setup()
    while not rospy.is_shutdown():
        for i in np.linspace(1, 4, 8):
            pub.publish_circle(radius=i)
