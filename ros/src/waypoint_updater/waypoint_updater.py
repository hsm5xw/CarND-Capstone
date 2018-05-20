#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Float64
from scipy.spatial import KDTree
from tf.transformations import euler_from_quaternion

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


# calculate Euclidean distance
def distance( x1, y1, x2, y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # incoming topics
        rospy.Subscriber('/current_pose',   PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # outgoing topic
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.cte_pub = rospy.Publisher('cteFromWayPointUpdater', Float64, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose           = None
        self.base_waypoints = None
        self.waypoints_2d   = None
        self.waypoint_tree  = None
        self.cte            = None
        self.loop()
    
    # Control publishing frequency
    def loop(self):
        rate = rospy.Rate(50)  # 50 Hz

        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree: 
                # Get closest waypoint  
                closest_waypoint_idx, cte = self.get_closest_waypoint_idx()
                self.publish_waypoints( closest_waypoint_idx, cte)

            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        #rospy.logwarn("Got Pose: {a:f}:{b:f}".format(a=x, b=y))
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]  # kd tree (1st closest, idx)
        
        # Check if closest waypoint is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord    = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coors
        cl_vect       = np.array( closest_coord)
        prev_vect     = np.array( prev_coord)
        pos_vect      = np.array( [x,y])

        val = np.dot( cl_vect - prev_vect, pos_vect - cl_vect)
        # Car is ahead of the closest waypoint
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)


        # For computing CTE
        x0 = x
        y0 = y

        x1 = self.waypoints_2d[closest_idx - 1][0]
        y1 = self.waypoints_2d[closest_idx - 1][1]

        x2 = self.waypoints_2d[closest_idx][0]
        y2 = self.waypoints_2d[closest_idx][1]

        cte = math.fabs((y2 - y1)*x0  -  (x2 - x1)*y0  + x2*y1  - y2*x1) / math.sqrt( math.pow(y2 - y1, 2.0) +  math.pow(x2 - x1, 2.0))

        # For computing CTE - #2
        orientation = self.pose.pose.orientation   # convert quaternion to yaw
        _, _,yaw    = euler_from_quaternion( [orientation.x, orientation.y, orientation.z, orientation.w])
        yaw         = yaw * (-1)                   # sign in the simulator is flipped

        N = 10
        start_idx = max( closest_idx - 5, 0)
        end_idx   = min( closest_idx + 5, len(self.waypoints_2d)-1 )

        ptsx_transformed = []
        ptsy_transformed = []
 
        # Convert Waypoints from Map's coordinates to Car's coordinates
        for i in range(start_idx, end_idx):
            # shift car reference
            pts = self.waypoints_2d[i]
            dx  = pts[0] - x0
            dy  = pts[1] - y0
 
            # Now the car is at (0,0) with angle 0
            ptsx_transformed.append( dx * math.cos(yaw) - dy * math.sin(yaw) )
            ptsy_transformed.append( dx * math.sin(yaw) + dy * math.cos(yaw) )

        ptsx_transformed_arr = np.array( ptsx_transformed)
        ptsx_transformed_arr = np.array( ptsy_transformed)

        # Fit polynomial to Waypoints (3rd order) 
        coeffs = np.polyfit( ptsx_transformed, ptsy_transformed, 3) # coeffs of 3rd order polynomial
        cte2  = -np.polyval( coeffs, 0);  

        rospy.logwarn("cte : {}".format(cte) )
        rospy.logwarn("cte2: {}".format( abs(cte2) ) )

        #rospy.logwarn("current_yaw: {}".format(current_yaw) )
        #rospy.logwarn("current_yaw (degrees): {}".format(current_yaw * 180 / math.pi) )

        return closest_idx, cte

    def publish_waypoints(self, closest_idx, cte):
        lane = Lane()

        end_pt = min( closest_idx + LOOKAHEAD_WPS, len(self.base_waypoints.waypoints) )
        lane.header    = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[ closest_idx: end_pt]
        #rospy.logwarn("publishing waypoints: {a:d}:{b:d}".format(a=closest_idx, b=end_pt))
        self.final_waypoints_pub.publish( lane )
        self.cte_pub.publish( cte )

    # Incoming topic #1 callback 
    def pose_cb(self, msg):
        self.pose = msg     # Store the car's pose
        #rospy.logwarn("Got Pose in Callback: {a:f}:{b:f}".format(a=self.pose.pose.position.x, b=self.pose.pose.position.y))

    # Incoming topic #2 callback 
    ''' NOTE: This is a latched subscriber. 
              So, once the callback is called, it doesn't send the base waypoints (which don't change) anymore.
              It is only called once. 
    '''
    def waypoints_cb(self, waypoints):

        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
