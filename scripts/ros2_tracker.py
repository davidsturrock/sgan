import time
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import sys
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
import sys

sys.path.insert(0, '/home/administrator/code/aru-core/ped/lib/')
import aru_py_pedestrian


def pull_to_centre(x, y, r_pull=1.5, min_radius=0.2):
    theta = np.arctan2(y, x)
    # radius = np.linalg.norm(x,y)
    x_pull = r_pull * np.cos(theta)
    y_pull = r_pull * np.sin(theta)
    x_new = abs(x) / x * abs(x - x_pull) if x != 0 else min_radius * np.cos(theta)
    y_new = abs(y) / y * abs(y - y_pull) if y != 0 else min_radius * np.sin(theta)
    if x_new ** 2 + y_new ** 2 < min_radius ** 2:
        x_new = min_radius * np.cos(theta)
        y_new = min_radius * np.sin(theta)
    return x_new, y_new


class Tracker(Node):
    def __init__(self, pub_topic='/tracked/pedestrians', odom_topic='/odometry/filtered',
                 lidar_topic='/velodyne_points', rate=10, agents=9):
        super().__init__('pedestrian_tracker')
        self.lidar_callback_status = False
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
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10
        )
        self.publisher = self.create_publisher(
            Point,
            self.pub_topic,
            1
        )
        self.rate = self.create_rate(self.rate_value)

    def odom_callback(self, odom: Odometry):
        if 1 / (time.perf_counter() - self.odom_last_callback) > 2.5:
            return
        self.odom = odom.pose
         # quat = self.odom.pose.orientation
        # pose = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        # tf = np.eye(4)
        # tf[0:3, 0:3] = pose.as_matrix()
        # tf[:3, 3] = np.array([self.odom.pose.position.x, self.odom.pose.position.y, 0]).T
        # self.tf = tf
        # print(tf)
        # print('*'*60)
        # print(f'Callback rate {1 / (time.perf_counter() - self.last_callback):.2f}Hz')
        self.odom_callback_status = True
        self.odom_last_callback = time.perf_counter()

    def lidar_callback(self, point_cloud: PointCloud2):
        if not self.odom_callback_status:
            return
        self.cloud_points = np.array(
            list(point_cloud2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z")))
        ).astype(np.float64)
        self.time_out = int(time.perf_counter())
        self.lidar_callback_status = True

    def publish_tracked_pts(self, r_pull=1.5, min_radius=0.2):
        if not self.init:
            self.pedestrian.init_tracker(self.cloud_points, self.time_out)
            self.init = True
        else:
            tracks = self.pedestrian.static_tracker(self.cloud_points, self.time_out)
            quat = self.odom.pose.orientation
            pose = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            tf = np.eye(4)
            tf[0:3, 0:3] = pose.as_matrix()
            tf[:3, 3] = np.array([self.odom.pose.position.x, self.odom.pose.position.y, 0]).T
            self.tf = tf
            print(tracks)
            for agent_id in range(min(self.agents, tracks.shape[0])):
                point_mat = np.eye(4)
                x = tracks[agent_id, 1]
                y = -tracks[agent_id, 0]
                x, y = pull_to_centre(x=x, y=y, r_pull=r_pull, min_radius=min_radius)
                point_mat[:3, 3] = np.array([x, y, 0]).T
                transformed_pts = self.tf @ point_mat
                point = Point(transformed_pts[0, 3], transformed_pts[1, 3], agent_id)
                print(point)
                self.publisher.publish(point)
                time.sleep(0.05)

    def main(self):
        print('Waiting for lidar')
        while not self.lidar_callback_status:
            pass
        print('Starting tracker')
        while rclpy.ok():
            self.publish_tracked_pts()
            self.rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    tracker = Tracker()
    tracker.setup()
    tracker.main()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

