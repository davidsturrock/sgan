import time

import numpy as np
import rospy
import sys
sys.path.insert(0, '/home/administrator/code/aru_sil_py/navigation/__init__.py')
import argparse
import os
import torch
from navigation.nav_options import NavOptions
from navigan_navigator import Navigator
from utilities.Transform import distance_and_yaw_from_transform
from attrdict import AttrDict
from pathlib import Path

from scripts.goal import seek_goal, seek_live_goal, create_obs_traj, pts_to_tfs
from scripts.model_loaders import get_combined_generator
from sgan.data.loader import data_loader
from sgan.utils import get_dset_path, save_plot_trajectory

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def make_scene(count, tf, tf1, tfs):
    obs_traj = create_obs_traj(tfs)

    other_ped = obs_traj.clone()
    other_ped[::, 0, 1] = other_ped[::, 0, 1] + 5
    obs_traj = torch.cat([obs_traj, other_ped], dim=1)
    # print(obs_traj.shape)
    # print(obs_traj[::,0].T)
    # print(obs_traj[::,1].T)
    return obs_traj


def update_scene(obs_traj, count, tf, tf1, tfs, pred):
    tfs.append(tf if count % 2 == 0 else tf1)
    # Update other ped
    other_ped = create_obs_traj(tfs)
    other_ped[::, 0, 1] = other_ped[::, 0, 1] + 5
    # obs_traj[::, 1] = other_ped[::]
    # Update goal agent
    obs_traj[:-1, 0] = obs_traj.clone()[1:, 0]
    obs_traj[-1] = pred[0, 0]
    new_agent_traj = obs_traj[::, 0].clone()
    new_agent_traj = new_agent_traj.unsqueeze(1)
    obs_traj = torch.cat([new_agent_traj, other_ped], dim=1)
    return obs_traj


def main(args):
    # ----------------------------------------------
    # Navigator setup
    nav_args = NavOptions().parse()
    nav = Navigator(nav_args, args.model_path, agents=1, rate=1)
    # -----------------------------------------------
    count = 0
    tf = np.eye(4)
    tf[2, 3] = 0.5
    tf[0, 3] = 0
    tf1 = np.eye(4)
    tf1[2, 3] = 0.5
    tf1[0, 3] = 0.2
    tfs = [tf for _ in range(8)]
    # obs_traj = make_scene(count, tf, tf1, tfs)
    # nav.obs_traj = obs_traj
    # nav.obs_traj = torch.zeros((8, 1, 2))
    while not nav.callback_status:
        pass
    print('Wheel Odom received')
    while False in np.all([nav.obs_traj[::, 0].numpy()], axis=0):
        pass
    print('8 points observed.')
    while not nav.goal_status:
        pass
    print('Goal received')
    # print(nav.obs_traj[::, 0].T)
    # print()
    # sys.exit(0)

    while not rospy.is_shutdown():
        start = time.perf_counter()
        # x, y = (t.item() for t in obs_traj[-1, 0])
        pred, pred_rel = nav.seek_live_goal(x=40, y=20, title=f'Loop iteration {count}')
        goal_tfs = pts_to_tfs(pred_rel)
        nav.goal_step(goal_tfs[0])
        # print(goal_tfs[0])
        # print(f"Goal pt: {x + 20:.2f}, {y + 20:.2f}")
        # nav.obs_traj = update_scene(obs_traj, count, tf, tf1, tfs, pred)
        count += 1
        nav.sleep()
        print(f"Loop rate {1 / (time.perf_counter() - start):.2f}Hz")
        # if count >= 1:
        #     sys.exit(0)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
