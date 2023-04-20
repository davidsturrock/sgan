# """NOTE Python 2.7 used for now. Receives Pose msg from Husky and publishes move_base_msgs/MoveBaseActionGoal msg
# to move_base_mapless_demo session on laptop"""

import rclpy
import numpy as np
import sys
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from rclpy.node import Node

class GoalPublisher(Node):
    def __init__(self):  #initialisation
        super().__init__(self, pub_topic='/move_base/goal', sub_topic='/pose', rate=1)
        self.move_goal = None
        self.rate = None
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.goal_pub = None
        self.pose_sub = None
        self.rate_value = rate
        self.pose = None
        
    def setup(self):
        node = rclpy.create_node('PoseToGoalRelay') #creates node
        self.pose_sub = node.create_subscription(Pose, self.sub_topic,self.callback) #subscriber
        self.rate = rclpy.Rate(self.rate_value)
        #client - service
        self.client = ActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        print('Setup complete')

    def callback(self, pose):
        self.move_goal = MoveBaseGoal()
        self.move_goal.target_pose.header.frame_id = "odom"
        self.move_goal.target_pose.header.stamp = rclpy.Time.now()
        self.move_goal.target_pose.pose.position = pose.position
        self.move_goal.target_pose.pose.orientation = pose.orientation

    def move_base_action(self):
        self.client.send_goal(self.move_goal)
        # wait = self.client.wait_for_result()
        # if not wait:
        #     rclpy.logerr("Action server not available!")
        #     rclpy.signal_shutdown("Action server not available!")
        # else:
            # return self.client.get_result()    
        
    def main(args=None):
        #rclpy.init(args=args)
        #node = MyNode('SocialNav', anonymous=True) 
        pub = GoalPublisher()
        pub.setup()
        while not rclpy.is_shutdown():
            if pub.move_goal is not None:
                pub.move_base_action()
                print(pub.move_goal)
                pub.move_goal = None
            else:
                print('Waiting for goal.')
    
            pub.rate.sleep()
            
            rclpy.spin(node)
        rclpy.shutdown()
                                
if __name__ == '__main__':
    main()
                