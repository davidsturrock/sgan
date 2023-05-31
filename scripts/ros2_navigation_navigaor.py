import pathlib
import time
import sys
import torch

from scripts.model_loaders import get_combined_generator
from sgan.utils import relative_to_abs, abs_to_relative, Plotter

sys.path.insert(0, '/home/administrator/code/aru-core/build/lib')
sys.path.insert(0, '/usr/local/lib/python3.6/dist-packages/cv2/python-3.6')

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Navigator:

    def __init__(self, model_path, agents=9, rate=1, odom_topic='odometry/filtered'):
        self.last_times = np.ones(agents) * time.perf_counter()

        self.odom_callback_status = False

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
        self.obs_traj = torch.ones((self.obs_len, 1 + agents, 2), device=_DEVICE_) * 100
        self.obs_traj[::, 0] = 0

        rclpy.init(args=None)

        self.last_callback = time.perf_counter()
        self.odom_last_callback = time.perf_counter()
        self.rate_value = rate
        self.node = rclpy.create_node('SocialNav')
        self.odom_sub = self.node.create_subscription(Odometry, self.odom_topic, self.odom_callback)
        self.ped_sub = self.node.create_subscription(Point, '/tracked/pedestrians', self.callback)
        self.goal_sub = self.node.create_subscription(Point, '/goal', self.goal_callback)
        self.publisher = self.node.create_publisher(Pose, '/pose', 1)

        self.timer = self.node.create_timer(1.0 / self.rate_value, self.timer_callback)
        self.node.spin()

    def timer_callback(self):
        self.goal_step()

    def goal_step(self, step_horizon=11):
        with torch.no_grad():
            if self.callback_status and self.goal_status and self.odom_callback_status:
                seq_start_end = torch.tensor([0, self.obs_traj.shape[1]], device=_DEVICE_).unsqueeze(0)

                goal_state = torch.zeros((1, 1, 2), device=_DEVICE_)
                goal_state[0, 0, 0] = self.goal.x - self.obs_traj[-1, 0, 0].item()
                goal_state[0, 0, 1] = self.goal.y - self.obs_traj[-1, 0, 1].item()

                obs_traj_rel = abs_to_relative(self.obs_traj)
                pred_traj_fake_rel = self.generator(
                    self.obs_traj, obs_traj_rel, seq_start_end, goal_state, goal_aggro=0.5
                )

                start_pos = self.obs_traj[-1]
                ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=start_pos)
                pred_traj_to_plot = ptfa.clone()
                pred_traj_to_plot[::, ::, 0] = -pred_traj_to_plot[::, ::, 0]

                obs_traj_to_plot = self.obs_traj.clone()
                obs_traj_to_plot[::, ::, 0] = -obs_traj_to_plot[::, ::, 0]

                self.plotter.xlim = [obs_traj_to_plot[-1, 0, 0] + -5, obs_traj_to_plot[-1, 0, 0] + 5]
                self.plotter.ylim = [obs_traj_to_plot[-1, 0, 1] + -5, obs_traj_to_plot[-1, 0, 1] + 5]

                self.plotter.display(
                    title=f'\nRel Goal {goal_state[0, 0, 0].item():.2f}m {goal_state[0, 0, 1].item():.2f}m\n' \
                          f'Abs Goal {self.abs_goal[0]:.2f}m {self.abs_goal[1]:.2f}m', ota=obs_traj_to_plot,
                    ptfa=pred_traj_to_plot,
                    goal_centre=self.abs_goal, sse=seq_start_end
                )

                pose = Pose()
                pose.orientation.w = 1
                pose.position.x = -pred_traj_fake_rel[step_horizon, 0, 0]
                pose.position.y = pred_traj_fake_rel[step_horizon, 0, 1]
                self.publisher.publish(pose)

                self.callback_status = False
                self.goal_status = False
                self.odom_callback_status = False

    def callback(self, tracked_pts: Point):
        if time.perf_counter() - self.last_times[int(tracked_pts.z)] < 0.4:
            return

        self.obs_traj[:-1, 1 + int(tracked_pts.z)] = self.obs_traj.clone()[1:, 1 + int(tracked_pts.z)]
        self.obs_traj[-1, 1 + int(tracked_pts.z), 0] = -tracked_pts.x
        self.obs_traj[-1, 1 + int(tracked_pts.z), 1] = tracked_pts.y

        self.last_times[int(tracked_pts.z)] = time.perf_counter()
        self.last_callback = time.perf_counter()
        self.callback_status = True

    def odom_callback(self, odom: Odometry):
        if 1 / (time.perf_counter() - self.odom_last_callback) > 2.5:
            return
        self.odom = odom.pose
          # self.husky_odom.append([self.odom.pose.position.x, self.odom.pose.position.y])
        # Slide husky points along by one
        self.obs_traj[:-1, 0] = self.obs_traj.clone()[1:, 0]
        # Update latest observed pts with new odom x and y
        # TODO adding x flip for network prediction issue
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

    def seek_live_goal(self, agent_id=0, x=40, y=10):
        with torch.no_grad():

            seq_start_end = torch.tensor([0, self.obs_traj.shape[1]], device=_DEVICE_).unsqueeze(0)

            goal_state = torch.zeros((1, 1, 2), device=_DEVICE_)
            goal_state[0, agent_id, 0] = self.goal.x - self.obs_traj[-1, 0, 0].item()
            goal_state[0, agent_id, 1] = self.goal.y - self.obs_traj[-1, 0, 1].item()

            # TODO Resolve pred imbalance for now flip x - axis
            obs_traj_rel = abs_to_relative(self.obs_traj)
            pred_traj_fake_rel = self.generator(self.obs_traj, obs_traj_rel, seq_start_end, goal_state, goal_aggro=0.5)

            start_pos = self.obs_traj[-1]
            ptfa = relative_to_abs(pred_traj_fake_rel, start_pos=start_pos)
            # Make separate variable to plot. Flipped along x-axis to look visually correct.
            pred_traj_to_plot = ptfa.clone()
            pred_traj_to_plot[::, ::, 0] = -pred_traj_to_plot[::, ::, 0]

            obs_traj_to_plot = self.obs_traj.clone()
            obs_traj_to_plot[::, ::, 0] = -obs_traj_to_plot[::, ::, 0]

            self.plotter.xlim = [obs_traj_to_plot[-1, 0, 0] + -5, obs_traj_to_plot[-1, 0, 0] + 5]
            self.plotter.ylim = [obs_traj_to_plot[-1, 0, 1] + -5, obs_traj_to_plot[-1, 0, 1] + 5]

            self.plotter.display(
                title=f'\nRel Goal {goal_state[0, 0, 0].item():.2f}m {goal_state[0, 0, 1].item():.2f}m\n' \
                      f'Abs Goal {self.abs_goal[0]:.2f}m {self.abs_goal[1]:.2f}m', ota=obs_traj_to_plot,
                ptfa=pred_traj_to_plot,
                goal_centre=self.abs_goal, sse=seq_start_end)

        return ptfa, pred_traj_fake_rel
        
