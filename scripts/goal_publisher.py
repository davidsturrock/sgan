# """NOTE Python 2.7 used for now. Receives Pose msg from Husky and publishes move_base_msgs/MoveBaseActionGoal msg
# to move_base_mapless_demo session on laptop"""
import rospy
import numpy as np
import sys
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib

# import move_base_msgs
# print(move_base_msgs.__file__)
# sys.exit(0)


class GoalPublisher:
    def __init__(self, pub_topic='/move_base/goal', sub_topic='/pose', rate=1):
        self.move_goal = None
        self.rate = None
        self.pub_topic = pub_topic
        self.sub_topic = sub_topic
        self.goal_pub = None
        self.pose_sub = None
        self.rate_value = rate
        self.pose = None

    def setup(self):
        rospy.init_node('PoseToGoalRelay')
        self.pose_sub = rospy.Subscriber(self.sub_topic, Pose, self.callback)
        self.rate = rospy.Rate(self.rate_value)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        print('Setup complete')

    def callback(self, pose):
        self.move_goal = MoveBaseGoal()
        self.move_goal.target_pose.header.frame_id = "odom"
        self.move_goal.target_pose.header.stamp = rospy.Time.now()
        self.move_goal.target_pose.pose.position = pose.position
        self.move_goal.target_pose.pose.orientation = pose.orientation

    def move_base_action(self):
        self.client.send_goal(self.move_goal)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            return self.client.get_result()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pub = GoalPublisher()
    pub.setup()
    while not rospy.is_shutdown():
        if pub.move_goal is not None:
            result = pub.move_base_action()
            if result:
                rospy.loginfo("Goal execution done!")
            print(pub.move_goal)
            pub.move_goal = None
        else:
            print('Waiting for goal.')

        pub.rate.sleep()

# if __name__ == '__main__':
#     try:
#         rospy.init_node('movebase_client_py')
#         result = movebase_client()
#         if result:
#             rospy.loginfo("Goal execution done!")
#     except rospy.ROSInterruptException:
#         rospy.loginfo("Navigation test finished.")
