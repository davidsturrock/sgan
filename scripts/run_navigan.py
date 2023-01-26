import time

import numpy as np
import rospy
import sys

sys.path.insert(0, '/home/administrator/code/aru_sil_py/navigation/__init__.py')
sys.path.insert(0, '/home/administrator/code/sgan')
import argparse
import os
import torch
from navigation.nav_options import NavOptions
# from aru_sil_py.navigation.nav_options import NavOptions
from navigan_navigator import Navigator

from pathlib import Path

from scripts.goal import pts_to_tfs

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # ----------------------------------------------
    # Navigator setup
    nav_args = NavOptions().parse()
    nav_args.max_x = 0.1
    nav_args.max_z = 0.35
    nav = Navigator(nav_args, args.model_path, rate=5)

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
    print('Waiting for wheel odometry')
    while not nav.odom_callback_status:
        pass
    print('Wheel odometry received')
    # while False in np.all([nav.obs_traj[::, 0, 0].numpy()], axis=0):
    #     pass
    # print('Husky observed for 8 timesteps.')
    #while not nav.goal_status:
    #    time.sleep(0.1)
    #print('Goal received')
    # print(nav.obs_traj[::, 0].T)
    # print()
    # sys.exit(0)
    goal_tfs = []
    accum_time = 0
    filename = f'{time.time():.0f}_traj.txt'
    # with open(filename, 'w') as f:
    #     f.write('-'*60 + '\n')
    while not rospy.is_shutdown():
        start = time.perf_counter()
        # x, y = (t.item() for t in obs_traj[-1, 0])
        # if len(goal_tfs) < 8:
        pred, pred_rel = nav.seek_live_goal(title=f'Jan_{count}', filename=filename)
        # goal_tfs = list(pts_to_tfs(pred_rel))
        nav.goal_step(pred[11])
        # nav.sleep()
        # with np.printoptions(precision=2, suppress=True):
        #     print(nav.obs_traj.T)
        print(f"Loop rate {1 / (time.perf_counter() - start):.2f}Hz")
        accum_time += time.perf_counter() - start

        if count == 5:
            # print(f"5 Loop Avg. Loop rate {1 / (accum_time/count):.2f}Hz")
            count = 0
            accum_time = 0
        count += 1
        # if count > 10:

        if count >= 60:
        #     nav.plotter.save_video()
            sys.exit(0)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
