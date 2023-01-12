
import time

import rospy
import numpy as np
import cv2
import sys
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/home/administrator/code/aru-core/ped/lib/')
# import aru_py_logger
import aru_py_pedestrian


class Tracker:
    def __init__(self, pub_topic='/tracked/pedestrians', odom_topic='/odometry/filtered',
                 lidar_topic='/velodyne_points', rate=10, agents=9):
        self.agents = agents
        self.odom = None
        self.lidar_topic = lidar_topic
        self.odom_topic = odom_topic
        self.pub_topic = pub_topic
        self.publisher = None
        self.goal_sub = None
        self.rate_value = rate
        self.init = False
        self.odom_last_callback = time.perf_counter()
        self.lidar_last_callback = time.perf_counter()
        self.pedestrian = None
        self.odom_callback_status = False
    def setup(self):
        self.pedestrian = aru_py_pedestrian.PyPedestrian("name")
        rospy.init_node('PedestrianTracker')
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback)
        self.publisher: rospy.Publisher = rospy.Publisher(self.pub_topic, Point, queue_size=1)
        self.rate = rospy.Rate(self.rate_value)

    def odom_callback(self, odom: Odometry):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker"""
        # limit update rate to self.rate_value
        # if 1 / (time.perf_counter() - self.odom_last_callback) > self.rate_value:
        #     return
        self.odom = odom.pose
        quat = self.odom.pose.orientation
        pose = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        tf = np.eye(4)
        tf[0:3, 0:3] = pose.as_matrix()
        tf[:3, 3] = np.array([self.odom.pose.position.x, self.odom.pose.position.y, 0]).T
        self.tf = tf
        # print(tf)
        # print('*'*60)
        # print(f'Callback rate {1 / (time.perf_counter() - self.last_callback):.2f}Hz')
        self.odom_callback_status = True

        self.odom_last_callback = time.perf_counter()

    def lidar_callback(self, point_cloud: PointCloud2):
        """Callback for tracker"""
        # Limit callback rate
        # if not self.odom_callback_status or 1 / (time.perf_counter() - self.lidar_last_callback) > self.rate_value:
        #     return

        cloud_points = np.array(
            list(point_cloud2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z")))).astype(np.float64)
        time_out = int(time.perf_counter())
        # Init tracker if not done
        if not self.init:
            self.pedestrian.init_tracker(cloud_points, time_out)
            self.init = True

        else:
            # tracks = self.pedestrian.static_tracker(cloud_points, time_out)
            tracks = self.pedestrian.dynamic_tracker(cloud_points, time_out, self.tf)
            # print(tracks)
            for agent_id in range(min(self.agents, tracks.shape[0])):
                # point( x coord, y coord, agent id no.)
                point = Point(tracks[agent_id, 0], tracks[agent_id, 1], agent_id)
                self.publisher.publish(point)
                time.sleep(0.1)

        self.lidar_last_callback = time.perf_counter()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tracker = Tracker()
    tracker.setup()

    while True:
        pass

