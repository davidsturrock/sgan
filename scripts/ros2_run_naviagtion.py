import time
import argparse
import torch

import rclpy
from rclpy.node import Node

from navigan_navigator import Navigator

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NavigatorNode(Node):
    def __init__(self, args):
        super().__init__('navigator_node')
        self.nav = Navigator(args.model_path, rate=5)

        self.subscription = self.create_subscription(
            WheelOdometry,
            'wheel_odom',
            self.odom_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def odom_callback(self, msg):
        # Wheel odometry received
        self.nav.odom_callback_status = True

    def main_loop(self):
        while rclpy.ok():
            start = time.perf_counter()
            pred, pred_rel = self.nav.seek_live_goal()
            self.nav.goal_step(pred)
            self.get_logger().info(f"Loop rate {1 / (time.perf_counter() - start):.2f}Hz")


def main(args):
    rclpy.init(args=args)

    navigator_node = NavigatorNode(args)
    navigator_node.main_loop()

    rclpy.shutdown()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

