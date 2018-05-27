#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Float64, Int32
from scipy.spatial import cKDTree  # use cKDTree instead
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
#LOOKAHEAD_WPS  = 100 # Number of waypoints we will publish. You can change this number
#PUBLISHER_RATE = 4

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # global debug counter
        self.debug_counter = 0

        rospy.Subscriber('/current_pose',   PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.cte_pub             = rospy.Publisher('cteFromWayPointUpdater', Float64, queue_size=1) # extra (*)

        # TODO: Add other member variables you need below
        self.pose           = None
        self.base_waypoints = None
        self.waypoints_2d   = None
        self.waypoint_tree  = None
        self.cte            = None
        self.traffic_waypoint = None
        
        # the loop waiting for messsages
        self.loop()
    
    def loop(self):
        rate = rospy.Rate(4)  # (PUBLISHER_RATE) Hz

        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree: 
                closest_waypoint_idx, cte = self.get_closest_waypoint_idx()
                if self.traffic_waypoint and self.traffic_waypoint > -1:
                    # there is a Red light ahead
                    # this is very basic, I just want to see that the car can stop when a Red light signal is received
                    lane = Lane()
                    lane.header = self.base_waypoints.header
                    lane.waypoints = self.base_waypoints.waypoints[closest_waypoint_idx: self.traffic_waypoint]
                    self.set_waypoint_velocity(lane.waypoints, len(lane.waypoints)-2, 0)
                    #publish the waypoints
                    self.final_waypoints_pub.publish(lane)
                    self.cte_pub.publish(cte)
                else:              
                    self.publish_waypoints( closest_waypoint_idx, cte)
                    self.debug_counter += 1

            rate.sleep()

    '''
    find sign of cte, by determining whether the car is left/right of the lane
    @ closest_idx: closest idx of waypoint ahead of the car

    returns (+) if car is on the left  (positive angle)
            (-) if car is on the right (negative angle)

    (source: https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane)
    '''
    def find_cte_sign( self, closest_idx):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        end_idx = min( closest_idx + 1, len(self.base_waypoints.waypoints)-1 )

        next_coord  = self.waypoints_2d[end_idx]
        prev_coord  = self.waypoints_2d[closest_idx - 1]

        next_wp     = np.array( next_coord)
        prev_wp     = np.array( prev_coord)
        pos_vect    = np.array( [x,y])

        v1 = next_wp   - prev_wp
        v2 = pos_vect  - prev_wp
  
        cross = np.cross( np.append(v1, 0), np.append(v2, 0) )
        sign  = np.dot( [0,0,1], cross)    

        return sign

    # get closest waypoint index ahead of the car
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y])[1]  # ckd tree (1st closest, idx)
           
        # Check if closest waypoint is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord    = self.waypoints_2d[closest_idx - 1]
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
                   
        # Find the sign of CTE
        sign = self.find_cte_sign( closest_idx)
        cte  = cte * sign

        #if self.debug_counter % 4 == 0:  # modulo by publisher rate
        #    rospy.logwarn("cte: {a:f}, closest_idx: {b:f}".format( a=cte, b=closest_idx) )
  
        return closest_idx, cte

    def publish_waypoints(self, closest_idx, cte):
        LOOKAHEAD_WPS  = 100 # Number of waypoints we will publish. You can change this number
        closest_idx += 2     # see Slack for discussion

        end_pt = min( closest_idx + LOOKAHEAD_WPS, len(self.base_waypoints.waypoints) )

        lane = Lane()
        lane.header    = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[ closest_idx: end_pt]
        
        self.final_waypoints_pub.publish( lane )
        self.cte_pub.publish( cte )  # extra (*)

    # callback routines
    # ---------------------------------------------------------------------------------------
    def pose_cb(self, msg):
        self.pose = msg     # Store the car's pose
        
    def waypoints_cb(self, waypoints):

        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = cKDTree( self.waypoints_2d )

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_waypoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
    # ----------------------------------------------------------------------------------------
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
