## Experiment Setup 

SSH into Husky
```bash
ssh -X administrator@CPR-A200-0656
````
If required, reset wheel odom by typing:
```bash
rosservice call /set_pose 
```
then hit double tab, ROS message will auto-complete. Hit Enter.
```bash
conda activate silpy
python ~/code/sgan/scripts/run_navigan.py --model_path /home/administrator/code/sgan/scripts/SocialNavRelativeGoal.pt
````
To set a goal from the laptop
```bash
export ROS_MASTER_URI=http://CPR-A200-0656:11311 
````
Then type
```bash
rostopic pub /goal geometry_msgs/Point 
```
hit double tab, ROS message will auto-complete. Edit x and y values, then hit Enter.

In a new terminal, SSH into Husky
```bash
ssh -X administrator@CPR-A200-0656
conda activate silpy
```
Start the tracker.
```bash
python ~/code/sgan/scripts/tracker.py
```
Run the following on laptop. Ensure you are connected to Husky Wi-Fi
```bash
export ROS_MASTER_URI=http://CPR-A200-0656:11311
roslaunch husky_navigation move_base_mapless_demo.launch
```
CAUTION: once the goal_publisher script begins the Husky will begin to move. Ensure there is a supervisor with controller at hand.
Also note the use of python2.
```bash
python2 ~/code/sgan/scripts goal_publisher.py          
```
NOTE: Python scripts should be killable with Ctrl-C. failing that, use Ctrl-Z

### Behaviour of move_base controller can be dynamically reconfigured.
The most pertinent parameters to us are:
``` markdown
/move_base/DWAPlannerROS/max_vel_x
/move_base/DWAPlannerROS/max_rot_vel
/move_base/DWAPlannerROS/xy_goal_tolerance
/move_base/DWAPlannerROS/yaw_goal_tolerance
```
Useful commands here are:
```bash
rosparam get /move_base/DWAPlannerROS/max_vel_x
rosparam set /move_base/DWAPlannerROS/max_vel_x
rosparam list
```
