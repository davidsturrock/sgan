import time
import rospy
import sys

sys.path.insert(0, '/home/administrator/code/aru_sil_py/navigation/__init__.py')
sys.path.insert(0, '/home/administrator/code/sgan')
import argparse

import torch
# from aru_sil_py.navigation.nav_options import NavOptions
from navigan_navigator import Navigator

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # ----------------------------------------------
    # Navigator setup
    nav = Navigator(args.model_path, rate=5)

    # -----------------------------------------------

    print('Waiting for wheel odometry')
    while not nav.odom_callback_status:
        pass
    print('Wheel odometry received')
    # while False in np.all([nav.obs_traj[::, 0, 0].numpy()], axis=0):
    #     pass
    # print('Husky observed for 8 timesteps.')
    # while not nav.goal_status:
    #    time.sleep(0.1)
    # print('Goal received')

    while not rospy.is_shutdown():
        start = time.perf_counter()
        pred, pred_rel = nav.seek_live_goal()
        nav.goal_step(pred)
        print(f"Loop rate {1 / (time.perf_counter() - start):.2f}Hz")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
