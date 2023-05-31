# """NOTE Python 2.7 used for now. Receives Pose msg from Husky and publishes move_base_msgs/MoveBaseActionGoal msg
# to move_base_mapless_demo session on laptop"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import time


class GoalPublisher(Node):
    def __init__(self, pub_topic='/goal_pose', sub_topic='/pose', rate=1):
        super().__init__('pose_to_goal_relay')
        self.move_goal = None
        self.rate = rate
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.goal_pub = None
        self.pose_sub = None
        self.pose = None
        self.create_subscription(
            PoseStamped,
            self.sub_topic,
            self.callback,
            10
        )
        self.create_timer(1.0 / self.rate, self.publish_goal)
        self.client = self.create_client(NavigateToPose, 'navigate_to_pose')

    def callback(self, pose):
        self.move_goal = pose

    def publish_goal(self):
        if self.move_goal is not None:
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = self.move_goal
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.header.frame_id = 'odom'
            self.client.wait_for_service()
            self.send_goal(goal_msg)
            self.move_goal = None

    def send_goal(self, goal_msg):
        future = self.client.call_async(goal_msg)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    response = future.result()
                    if response.result == response.RESULT_ACCEPTED:
                        self.get_logger().info('Goal accepted')
                    else:
                        self.get_logger().info('Goal rejected')
                except Exception as e:
                    self.get_logger().info('Goal execution failed: {}'.format(e))
                break

    def main_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self)


def main(args=None):
    rclpy.init(args=args)
    pub = GoalPublisher()
    pub.main_loop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

                