
import time

import rospy
import numpy as np
import cv2
import sys
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


sys.path.insert(0, '/home/administrator/code/aru-core/ped/lib/')
# import aru_py_logger
import aru_py_pedestrian


class Tracker:
    def __init__(self, pub_topic='/tracked/pedestrians', sub_topic='/velodyne_points', rate=5):
        self.sub_topic = sub_topic
        self.pub_topic = pub_topic
        self.publisher = None
        self.goal_sub = None
        self.rate_value = rate
        self.init = False
        self.last_callback = time.perf_counter()
        self.pedestrian = None

    def setup(self):
        self.pedestrian = aru_py_pedestrian.PyPedestrian("name")
        rospy.init_node('PedestrianTracker')
        self.odom_sub = rospy.Subscriber(self.sub_topic, PointCloud2, self.callback)
        self.publisher: rospy.Publisher = rospy.Publisher(self.pub_topic, Point, queue_size=1)
        self.rate = rospy.Rate(self.rate_value)

    def callback(self, point_cloud: PointCloud2):
        """Callback for tracker"""
        # Limit callback rate
        if 1 / (time.perf_counter() - self.last_callback) > self.rate_value:
            return

        cloud_points = np.array(
            list(point_cloud2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z")))).astype(np.float64)
        time_out = int(time.perf_counter())
        # Init tracker if not done
        if not self.init:
            self.pedestrian.init_tracker(cloud_points, time_out)
            self.init = True

        else:
            tracks = self.pedestrian.static_tracker(cloud_points, time_out)
            # print(tracks)
            for agent_id in range(10):
                # point( x coord, y coord, agent id no.)
                point = Point(tracks[agent_id, 0], tracks[agent_id, 1], agent_id)
                self.publisher.publish(point)
                time.sleep(0.1)

        self.last_callback = time.perf_counter()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tracker = Tracker()
    tracker.setup()

    while True:
        pass

