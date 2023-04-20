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
import rclpy #ros2
from rclpy.node import Node #ros2

from geometry_msgs.msg import Twist, Point, Pose

from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Navigator(Node):
    def __init__(self):
        super().__init__(self, model_path, agents=9, rate=1, odom_topic='odometry/filtered')
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
        # self.obs_traj = torch.zeros((self.obs_len, 1 + agents, 2), device=_DEVICE_)
        # TODO consider best solution to init agent values
        # Initialise agents far away to avoid influencing Husky planned path
        self.obs_traj = torch.ones((self.obs_len, 1 + agents, 2), device=_DEVICE_) * 100
        self.obs_traj[::, 0] = 0   
        
    def sleep(self):
        self.rate.sleep()

    def goal_step(self, predictions, step_horizon=11):
        """predictions: absolute coordinate predictions from sgan network of shape(12, no.agents, 2)
            Row 0 is predictions for husky.
            step_horizon: index of predictions to send as goal to move_base_mapless_demo controller"""

        pose = Pose()
        pose.orientation.w = 1
        # Inver x value to deal with network bias issue. To be fixed
        pose.position.x = -predictions[step_horizon, 0, 0]
        pose.position.y = predictions[step_horizon, 0, 1]

        self.publisher_.publish(pose)

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
        # print(
        # f'Agent {int(tracked_pts.z)} tracked {time.perf_counter() - self.last_times[int(tracked_pts.z)]:.2f}s ago.')
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
        
def main(args=None):
    rclpy.init(args=args)
    node = MyNode('SocialNav', anonymous=True)
    
    self.last_callback = time.perf_counter()
    self.odom_last_callback = time.perf_counter()
    self.rate_value = rate
    self.rate = rclpy.Rate(self.rate_value)
    
    #subscriber
    self.odom_sub = node.create_subscription(Odometry, self.odom_topic, self.odom_callback)
    
    self.ped_sub = node.create_subscription(Point, '/tracked/pedestrians', self.callback)
    
    self.goal_sub = node.create_subscription(Point, '/goal', self.goal_callback)
    
    self.publisher: rclpy.Publisher = node.create_publisher(Pose, '/pose', queue_size=1, latch=True)
    
    rclpy.spin(node)
    rclpy.shutdown()
            
if __name__ == '__main__':
    main()