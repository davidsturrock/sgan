import pathlib
import time
import sys
import torch

from scripts.model_loaders import get_combined_generator
from sgan.utils import relative_to_abs, save_plot_trajectory, abs_to_relative, plot_trajectories

sys.path.insert(0, '/home/administrator/code/aru-core/build/lib')
# sys.path.insert(0, '/home/david/code/aru-core/build/lib')
sys.path.insert(0, '/usr/local/lib/python3.6/dist-packages/cv2/python-3.6')
import numpy as np
import rospy
import std_msgs
from geometry_msgs.msg import Twist, Point
import aru_py_logger
from utilities.Transform import distance_and_yaw_from_transform
from nav_msgs.msg import Odometry

_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ControlParameters:
    __slots__ = "dthresh", "ythresh", "pz", "max_x", "max_z"

    def __init__(self, args):
        """
        self.p_z: float = args.angular_gain
        self.dist_thresh: float = 1
        self.yaw_thresh: float = 30
        self.max_linear_vel: float = args.max_x
        self.max_angular_vel: float = args.max_z
        """
        self.dthresh: float = args.distance_threshold
        self.ythresh: float = args.yaw_threshold
        self.pz: float = 1 / self.ythresh
        self.max_x: float = args.max_x
        self.max_z: float = args.max_z


def create_tf_logger(logfolder=pathlib.Path('logs'), logname=None,
                     overwrite: bool = True) -> aru_py_logger.TransformLogger:
    if logname is None:
        logname = f'live_tfs_log_{time.strftime("%d-%m-%y-%H:%M:%S")}.monolithic'
    pathlib.Path(logfolder).mkdir(parents=True, exist_ok=True)
    logfile = logfolder / logname
    return aru_py_logger.TransformLogger(str(logfile), overwrite)


class Navigator:

    def __init__(self, args, model_path, agents=10, rate=1):
        if pathlib.Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            print('Invalid model path.')
            sys.exit(0)
        self.generator = get_combined_generator(checkpoint)
        self.obs_len = self.generator.goal.obs_len
        self.pred_len = self.generator.goal.pred_len
        self.agents = agents
        self.obs_traj = torch.zeros((self.obs_len, agents, 2), device=_DEVICE_)

        self.published_points = []
        self.tfs = []
        self.tf_to_last_loc = np.eye(4)
        self.logger = create_tf_logger() if args.log else None
        self.verbose: bool = args.verbose
        self.control_params: ControlParameters = ControlParameters(args)
        rospy.init_node('naviganNavigator', anonymous=True)

        self.last_callback = time.perf_counter()
        self.rate_value = rate
        self.rate = rospy.Rate(self.rate_value)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.callback)
        self.publisher: rospy.Publisher = rospy.Publisher(args.pub_topic, Twist, queue_size=1, latch=True)

    def sleep(self):
        self.rate.sleep()

    def goal_step(self, tf):
        distance, yaw = distance_and_yaw_from_transform(tf)
        if distance < self.control_params.dthresh and abs(yaw) < self.control_params.ythresh:
            msg = self.controller(tf)
            self.publisher.publish(msg)

    def controller(self, tf):
        """Controller now assumes TF is within bounds. Check must be performed at higher level"""
        cmd_vel = Twist()
        if np.array_equal(tf, np.eye(4)):
            return cmd_vel
        # TODO Check below comment
        #  Invert for ORB to give correct left and right action (could also just flip sign of yaw)
        # tf = np.linalg.inv(tf)
        distance, yaw = distance_and_yaw_from_transform(tf)
        yaw = -yaw
        # print(f'Yaw: {yaw} deg.')
        # if logging, log the tf for path tracking purposes
        if self.logger is not None:
            # TODO add source and dest timestamps to logging
            self.logger.write_to_file(tf.astype(dtype=np.float64), 0, 1)

        cmd_vel.angular.z = -self.control_params.pz * (0 - yaw)
        # Bound the output of controller to within max values
        if cmd_vel.angular.z > self.control_params.max_z:
            cmd_vel.angular.z = self.control_params.max_z
        elif cmd_vel.angular.z < - self.control_params.max_z:
            cmd_vel.angular.z = -self.control_params.max_z

        # Use constant fwd speed if transform is within thresholds
        cmd_vel.linear.x = self.control_params.max_x

        return cmd_vel

    def callback(self, tracked_pts: Odometry):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker"""
        # limit update rate to self.rate_value
        if 1 / (time.perf_counter() - self.last_callback) > self.rate_value:
            return
        print(f'Callback rate {1 / (time.perf_counter() - self.last_callback):.2f}Hz')
            # Slide all points fwd a timestep
        self.obs_traj[:-1] = self.obs_traj.clone()[1:]
        # From agent 0 (Husky) to agent 9 update
        # x and y coordinate from object tracker
        self.obs_traj[-1, 0, 0] = tracked_pts.pose.pose.position.x
        self.obs_traj[-1, 0, 1] = tracked_pts.pose.pose.position.y
        # for i, pt in enumerate(tracked_pts):
        #     self.obs_traj[-1, i, 0] = pt[0]
        #     self.obs_traj[-1, i, 1] = pt[1]
        self.last_callback = time.perf_counter()

    def seek_live_goal(self, x, y, agent_id=0, title='live_exp'):
        with torch.no_grad():
            pred_traj_gt = torch.zeros(self.obs_traj.shape, device=_DEVICE_)
            obs_traj_rel = abs_to_relative(self.obs_traj)
            seq_start_end = torch.tensor([0, self.obs_traj.shape[1]], device=_DEVICE_).unsqueeze(0)
            # seq_start_end = torch.tensor([0, 1], device=_DEVICE_).unsqueeze(0)
            with np.printoptions(precision=3, suppress=True):
                goal_state = torch.zeros((1, self.obs_traj.shape[1], 2), device=_DEVICE_)
                # goal_state[0, agent_id] = pred_traj_gt[-1, agent_id]
                goal_state[0, agent_id, 0] = x
                goal_state[0, agent_id, 1] = y
                # print(goal_state.shape)

            ota = self.obs_traj.numpy()
            ptga = pred_traj_gt.numpy()
            pred_traj_fake_rel = self.generator(self.obs_traj, obs_traj_rel, seq_start_end, goal_state, 1)
            start_pos = self.obs_traj[-1]
            ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=start_pos)
            # ptfa = relative_to_abs(pred_traj_fake_rel)

            # plot_trajectories(ota, ptga, ptfa, seq_start_end)

            save_plot_trajectory(title, ota, ptga, ptfa, seq_start_end)
        # for i, ped in enumerate(obs_traj.permute(1, 0, 2)):
        #     if i == agent_id:
        #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
        # for i, ped in enumerate(pred_traj_gt.permute(1, 0, 2)):
        #     if i == agent_id:
        #         print(f'Ped {i} predicted gt\tX\n\t\t\t\t\tY\n{ped.T}')
        return ptfa, pred_traj_fake_rel


def get_slice(seq_images=None, seq_tfs=None):
    """Slice list of images to skip calculating existing tfs for images we already have."""

    return slice(1 if seq_tfs.shape == (4, 4) else len(seq_tfs), len(seq_images))


def shape_check(tf):
    if tf.shape == (4, 4):
        tf = tf.reshape(1, 4, 4)
    return tf
