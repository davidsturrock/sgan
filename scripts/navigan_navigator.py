import math
import pathlib
import time
import sys
import torch

from scripts.model_loaders import get_combined_generator
from sgan.utils import relative_to_abs, abs_to_relative, Plotter

sys.path.insert(0, '/home/administrator/code/aru-core/build/lib')
# sys.path.insert(0, '/home/david/code/aru-core/build/lib')
sys.path.insert(0, '/usr/local/lib/python3.6/dist-packages/cv2/python-3.6')
import numpy as np
import rospy
import std_msgs
from geometry_msgs.msg import Twist, Point
# import aru_py_logger
from utilities.Transform import distance_and_yaw_from_transform
# from aru_sil_py.utilities.Transform import distance_and_yaw_from_transform
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

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
        self.pz: float = 1
        self.max_x: float = args.max_x
        self.max_z: float = args.max_z


# def create_tf_logger(logfolder=pathlib.Path('logs'), logname=None,
#                      overwrite: bool = True) -> aru_py_logger.TransformLogger:
#     if logname is None:
#         logname = f'live_tfs_log_{time.strftime("%d-%m-%y-%H:%M:%S")}.monolithic'
#     pathlib.Path(logfolder).mkdir(parents=True, exist_ok=True)
#     logfile = logfolder / logname
#     return aru_py_logger.TransformLogger(str(logfile), overwrite)


class Navigator:

    def __init__(self, args, model_path, agents=9, rate=1, odom_topic='odometry/filtered'):
        self.last_times = np.ones(agents) * time.perf_counter()
        self.odom_callback_status = False
        self.tracked_agents = []
        self.husky_odom = []
        self.odom_topic = odom_topic
        self.plotter = Plotter()
        self.goal = Point(0, 0, 0)
        self.abs_goal = [0, 0]
        self.goal_status = False
        self.callback_status = False
        self.odom = None
        if pathlib.Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            print('Invalid model path.')
            sys.exit(0)
        self.generator = get_combined_generator(checkpoint)
        self.obs_len = self.generator.goal.obs_len
        self.pred_len = self.generator.goal.pred_len
        self.agents = agents
        # self.obs_traj = torch.zeros((self.obs_len, 1 + agents, 2), device=_DEVICE_)
        # TODO consider best solution to init agent values
        # Initialise agents far awa to avoid influencing Husky planned path
        self.obs_traj = torch.ones((self.obs_len, 1 + agents, 2), device=_DEVICE_) * 100
        self.obs_traj[::, 0] = 0
        self.published_points = []
        self.tfs = []
        self.tf_to_last_loc = np.eye(4)
        # self.logger = create_tf_logger() if args.log else None
        self.verbose: bool = args.verbose
        self.control_params: ControlParameters = ControlParameters(args)
        rospy.init_node('naviganNavigator', anonymous=True)

        self.last_callback = time.perf_counter()
        self.odom_last_callback = time.perf_counter()
        self.rate_value = rate
        self.rate = rospy.Rate(self.rate_value)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.ped_sub = rospy.Subscriber('/tracked/pedestrians', Point, self.callback)
        self.goal_sub = rospy.Subscriber('/goal', Point, self.goal_callback)
        # self.publisher: rospy.Publisher = rospy.Publisher(args.pub_topic, Twist, queue_size=1, latch=True)
        self.publisher: rospy.Publisher = rospy.Publisher('/relay', Twist, queue_size=1, latch=True)

    def sleep(self):
        self.rate.sleep()

    def goal_step(self, tf):
        with np.printoptions(precision=2, suppress=True):
            # print(f'Initial TF:\n{tf}')
            x = tf[2, 3]
            # curr_x = self.odom.pose.position.x
            # curr_y = self.odom.pose.position.y
            rot_mat = R.from_matrix(tf[0:3, 0:3])
            # turn_angle = math.atan2(curr_y, curr_x) * 180 / math.pi
            tf_turn_angle = rot_mat.as_euler('zyx', degrees=True)[0]
            # print(f"TF yaw {tf_turn_angle:.2f}")

            quat = self.odom.pose.orientation
            pose = R.from_quat([quat.x, quat.y, quat.z, quat.w])
            curr_yaw = pose.as_euler('zyx', degrees=True)[0]
            # print(f'Final turn angle: = {tf_turn_angle:.2f} - {curr_yaw:.2f} = {tf_turn_angle - curr_yaw:.2f}')
            # # print(f'Turn angle for ({y - curr_y:.2f}, {x - curr_x:.2f}): {turn_angle:.2f} deg.')
            turn_angle = curr_yaw - tf_turn_angle
            r = R.from_euler('z', turn_angle, degrees=True)
            rot = np.array(r.as_matrix())
            tf = np.eye(4)
            tf[0:3, 0:3] = rot
            tf[2, 3] = x
            # print(f'New TF:\n{tf}')
            if x < self.control_params.dthresh and abs(turn_angle) < self.control_params.ythresh:
                msg = self.controller(tf)
                self.publisher.publish(msg)
            if x > self.control_params.dthresh:
                print(f'{x:.2f} > {self.control_params.dthresh}')
            if abs(turn_angle) > self.control_params.ythresh:
                  print(f'{abs(turn_angle):.2f} > {self.control_params.ythresh}')

    def controller(self, tf):
        """Controller now assumes TF is within bounds. Check must be performed at higher level"""
        cmd_vel = Twist()
        if np.array_equal(tf, np.eye(4)):
            return cmd_vel
        # TODO Check below comment
        #  Invert for ORB to give correct left and right action (could also just flip sign of yaw)
        # tf = np.linalg.inv(tf)
        distance, yaw = distance_and_yaw_from_transform(tf)
        pose = R.from_matrix(tf[0:3, 0:3])
        yaw = pose.as_euler('zyx', degrees=True)[0]
        yaw = -yaw
        # print(f'Controller yaw: {yaw:.2f}')
        # print(f'Yaw: {yaw} deg.')
        # if logging, log the tf for path tracking purposes
        # if self.logger is not None:
        # TODO add source and dest timestamps to logging
        # self.logger.write_to_file(tf.astype(dtype=np.float64), 0, 1)
        cmd_vel.angular.z = -self.control_params.pz * (0 - yaw)
        # Bound the output of controller to within max values
        if cmd_vel.angular.z > self.control_params.max_z:
            cmd_vel.angular.z = self.control_params.max_z
        elif cmd_vel.angular.z < - self.control_params.max_z:
            cmd_vel.angular.z = -self.control_params.max_z

        # Use constant fwd speed if transform is within thresholds
        cmd_vel.linear.x = self.control_params.max_x
        print(f'Yaw error {0 - yaw:.2f} deg')
        print(f'Max/ Curr Angular Vel {self.control_params.max_z:.2} / {cmd_vel.angular.z:.2}')
        return cmd_vel

    def callback(self, tracked_pts: Point):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker
        NOTE: 1 is added to agent index because obs_traj row zero is for husky and is updated via wheel odom"""
        # check tracked agent is within limit and limit update rate to self.rate_value
        # if int(tracked_pts.z) > self.agents or 1 / (time.perf_counter() - self.last_callback) > self.rate_value:
        #     return
        # print(f'Agent ID: {tracked_pts.z} x: {tracked_pts.x:.2f}m y: {tracked_pts.y:.2f}m')
        # print(f'X = y + obs_traj[-1, 0 ,0] = {tracked_pts.y:.2f} + {self.obs_traj[-1, 0, 0].item():.2f} = {tracked_pts.y + self.obs_traj[-1, 0, 0].item():.2f}m')
        # print(f'Y = -x + obs_traj[-1, 0 ,1] = {-tracked_pts.x:.2f} + {self.obs_traj[-1, 0, 1].item():.2f} = {-tracked_pts.x + self.obs_traj[-1, 0, 1].item():.2f}m')
        # # print(f'Callback rate {1 / (time.perf_counter() - self.last_callback):.2f}Hz')
        print(
            f'Agent {int(tracked_pts.z)} tracked {time.perf_counter() - self.last_times[int(tracked_pts.z)]:.2f}s ago.')
        if time.perf_counter() - self.last_times[int(tracked_pts.z)] < 0.4:

            return

        # Slide selected agent's points fwd a timestep
        self.obs_traj[:-1, 1 + int(tracked_pts.z)] = self.obs_traj.clone()[1:, 1 + int(tracked_pts.z)]
        # # From agent 0 (Husky) to agent 9 update
        # # x and y coordinate from object tracker
        # # point.z value contains agent id no.
        # TODO flipping x and -y for now because of wrong axes from tracker
        self.obs_traj[-1, 1 + int(tracked_pts.z), 0] = -tracked_pts.x
                                                       # + self.obs_traj[0, 0, 0]
        # + self.obs_traj[-1, 0, 0].item()
        self.obs_traj[-1, 1 + int(tracked_pts.z), 1] = tracked_pts.y
                                                       # + self.obs_traj[0, 0, 1]
        # Check all values are same, if so most likely dead point from out of scene. Reset to 100
        # if np.all(self.obs_traj[::, int(tracked_pts.z), 0].numpy() ==
        #           self.obs_traj[0, int(tracked_pts.z), 0].item()) \
        #         and np.all(self.obs_traj[::, int(tracked_pts.z), 1].numpy() ==
        #                    self.obs_traj[0, int(tracked_pts.z), 1].item()):
        #     self.obs_traj[::, int(tracked_pts.z), 0] = 100
        #     self.obs_traj[::, int(tracked_pts.z), 1] = 100
        # + self.obs_traj[-1, 0, 1].item()
        # # for i, pt in enumerate(tracked_pts):
        # #     self.obs_traj[-1, i, 0] = pt[0]
        # #     self.obs_traj[-1, i, 1] = pt[1]
        self.last_times[int(tracked_pts.z)] = time.perf_counter()
        self.last_callback = time.perf_counter()
        self.callback_status = True

    def odom_callback(self, odom: Odometry):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker"""
        # limit update rate to self.rate_value
        # if 1 / (time.perf_counter() - self.odom_last_callback) > self.rate_value:
        if 1 / (time.perf_counter() - self.odom_last_callback) > 2.5:
            return
        self.odom = odom.pose
        # self.husky_odom.append([self.odom.pose.position.x, self.odom.pose.position.y])
        # Slide husky points along by one
        self.obs_traj[:-1, 0] = self.obs_traj.clone()[1:, 0]
        # Update latest observed pts with new odom x and y
        #TODO adding x flip for network prediction issue
        self.obs_traj[-1, 0, 0] = -self.odom.pose.position.x
        self.obs_traj[-1, 0, 1] = self.odom.pose.position.y
        # print(f'Callback rate {1 / (time.perf_counter() - self.last_callback):.2f}Hz')
        self.odom_callback_status = True

        self.odom_last_callback = time.perf_counter()

    def goal_callback(self, goal: Point):
        """update_obs_traj when a new list msg of tracked pts are received from the object tracker"""
        # limit update rate to self.rate_value
        self.abs_goal = [goal.x, goal.y]
        self.goal = goal
        self.goal.x -= self.obs_traj[-1, 0, 0].item()
        self.goal.y -= self.obs_traj[-1, 0, 1].item()
        self.goal_status = True

    def seek_live_goal(self, agent_id=0, x=40, y=10, title='live_exp', filename=None):
        # self.goal = Point(x - self.obs_traj[-1, 0, 0].item() , y - self.obs_traj[-1, 0, 1].item(), 0)
        with torch.no_grad():
            pred_traj_gt = torch.zeros(self.obs_traj.shape, device=_DEVICE_)
            # obs_traj_rel = abs_to_relative(self.obs_traj)
            # obs_traj_rel[::, 1] = 0
            # self.obs_traj[::, 1] = 0
            # obs_traj_rel = torch.ones((self.obs_len, 1, 2), device=_DEVICE_) * 0.8
            seq_start_end = torch.tensor([0, self.obs_traj.shape[1]], device=_DEVICE_).unsqueeze(0)
            # seq_start_end = torch.tensor([0, 1], device=_DEVICE_).unsqueeze(0)
            with np.printoptions(precision=3, suppress=True):
                goal_state = torch.zeros((1, 1, 2), device=_DEVICE_)

                # goal_state[0, agent_id] = pred_traj_gt[-1, agent_id]
                # goal_state[0, agent_id, 0] = x

                goal_state[0, agent_id, 0] = self.goal.x - self.obs_traj[-1, 0, 0].item()
                goal_state[0, agent_id, 1] = self.goal.y - self.obs_traj[-1, 0, 1].item()
                # print(f'X {self.goal.x:.2f}, Y {self.goal.y:.2f}')
                # print(goal_state.shape)
            # ota = self.obs_traj.numpy()
            ptga = pred_traj_gt.numpy()
            # self.obs_traj[::, 0] = self.husky_odom[-8::]
            # self.obs_traj[::, 1::] = self.tracked_agents[-8::]
            # with np.printoptions(precision=2, suppress=True):
            #     print(self.obs_traj[::, 0:2, 0].T)
            #     print(self.obs_traj[::, 0:2, 1].T)
            # print(self.obs_traj[, 1].T)
            # TODO Resolve pred imbalance for now flip x - axis
            # obs_traj = self.obs_traj
            # obs_traj[::, ::, 0] = -obs_traj[::, ::, 0]
            obs_traj_rel = abs_to_relative(self.obs_traj)
            pred_traj_fake_rel = self.generator(self.obs_traj, obs_traj_rel, seq_start_end, goal_state, goal_aggro=0.5)
            # self.obs_traj[:-1, 0] = self.obs_traj.clone()[1:, 0]

            # with np.printoptions(precision=2, suppress=True):
            #     print(f'Rel Pred Ped 0:\n{pred_traj_fake_rel[::, 1].T}')
            #     print(f'Abs Observed Ped 0:\n{self.obs_traj[::, 1].T}')
            #     print(f'Rel Observed Ped 0:\n{obs_traj_rel[::, 1].T}')
            start_pos = self.obs_traj[-1]
            ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=start_pos)
            pred_traj_to_plot = ptfa.clone()
            pred_traj_to_plot[::, ::, 0] = -pred_traj_to_plot[::, ::, 0]


            # self.obs_traj[-1, 0] = ptfa[0, 0]
            obs_traj_to_plot = self.obs_traj.clone()
            obs_traj_to_plot[::, ::, 0] = -obs_traj_to_plot[::, ::, 0]
            with open(filename, 'a') as f:
                p = ptfa.numpy()
                f.write(f'{p[0, 0, 0]:.3f}\t{p[0, 0, 1]:.3f}\n')

            # print(ptfa[::,0].T)
            # ptfa = relative_to_abs(pred_traj_fake_rel)

            # plot_trajectories(ota, ptga, ptfa, seq_start_end)
            self.plotter.xlim = [obs_traj_to_plot[-1, 0, 0] + -5, obs_traj_to_plot[-1, 0, 0] + 5]
            self.plotter.ylim = [obs_traj_to_plot[-1, 0, 1] + -5, obs_traj_to_plot[-1, 0, 1] + 5]
            # abs_goal = [goal_state[0, 0, 0] + self.obs_traj[-1, 0, 0], goal_state[0, 0, 1] + self.obs_traj[-1, 0, 1]]
            self.plotter.display(
                title=f'\nRel Goal {goal_state[0, 0, 0].item():.2f}m {goal_state[0, 0, 1].item():.2f}m\n' \
                      f'Abs Goal {self.abs_goal[0]:.2f}m {self.abs_goal[1]:.2f}m', ota=obs_traj_to_plot,
                ptfa=pred_traj_to_plot,
                goal_centre=self.abs_goal, sse=seq_start_end)

        # for i, ped in enumerate(self.obs_traj.permute(1, 0, 2)):
        #     if i == agent_id:
        #         print(f'Ped {i} observed traj\tX\n\t\t\t\t\tY\n{ped.T}')
        # for i, ped in enumerate(ptfa.permute(1, 0, 2)):
        #     if i == agent_id:
        #         print(f'Ped {i} predicted\tX\n\t\t\t\t\tY\n{ped.T}')
        return ptfa, pred_traj_fake_rel


def get_slice(seq_images=None, seq_tfs=None):
    """Slice list of images to skip calculating existing tfs for images we already have."""

    return slice(1 if seq_tfs.shape == (4, 4) else len(seq_tfs), len(seq_images))


def shape_check(tf):
    if tf.shape == (4, 4):
        tf = tf.reshape(1, 4, 4)
    return tf
