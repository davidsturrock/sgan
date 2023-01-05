import numpy as np
import rospy
import sys

import argparse
import torch
from aru_sil_py.navigation.nav_options import NavOptions

from navigan_navigator import Navigator

from scripts.goal import pts_to_tfs
from scripts.navigan_navigator import make_scene, update_scene

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # ----------------------------------------------
    # Navigator setup
    nav_args = NavOptions().parse()
    nav = Navigator(nav_args, args.model_path)
    # -----------------------------------------------
    count = 0
    tf = np.eye(4)
    tf[2, 3] = 0.5
    tf[0, 3] = 0
    tf1 = np.eye(4)
    tf1[2, 3] = 0.5
    tf1[0, 3] = 0.2
    tfs = [tf for _ in range(8)]
    obs_traj = make_scene(count, tf, tf1, tfs)
    print(obs_traj.shape)
    # sys.exit(0)
    r = rospy.Rate(5)

    while not rospy.is_shutdown():
        # start = time.perf_counter()
        x, y = (t.item() for t in obs_traj[-1, 0])
        pred, pred_rel = nav.seek_live_goal(x=x + 20, y=y + 20, title=f'Loop iteration {count}')
        goal_tfs = pts_to_tfs(pred_rel)
        nav.goal_step(goal_tfs[0])
        print(f"Goal pt: {x + 20:.2f}, {y + 20:.2f}")
        nav.obs_traj = update_scene(obs_traj, count, tf, tf1, tfs, pred)
        count += 1
        r.sleep()
        # print(f"Loop rate {1 / (time.perf_counter() - start):.2f}Hz")
        if count >= 25:
            sys.exit(0)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


