import time
import rclpy
from rclpy.node import Node
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

def main(args=None):
    rclpy.init(args=args)
    nav = Navigator(args.model_path, rate=5)
    print('Waiting for wheel odometry')
    while not nav.odom_callback_status:
        pass
    print('Wheel odometry received')
    
    while not rclpy.is_shutdown():
        start = time.perf_counter()
        pred, pred_rel = nav.seek_live_goal()
        nav.goal_step(pred)
        print(f"Loop rate {1 / (time.perf_counter() - start):.2f}Hz")
        rclpy.spin(node)
    rclpy.shutdown()
                    
if __name__ == '__main__':
    main()

