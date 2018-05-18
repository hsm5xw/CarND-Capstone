#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
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
MAX_DECEL = 0.5 # From the walkthrough.

# calculate Euclidean distance
def distance( x1, y1, x2, y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # incoming topics
        rospy.Subscriber('/current_pose',   PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # outgoing topic
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=400)

        # TODO: Add other member variables you need below
        self.pose           = None
        self.base_waypoints = None
        self.waypoints_2d   = None
        self.waypoint_tree  = None
        self.stopline_wp_idx = None

        self.loop()
    
    # Control publishing frequency
    def loop(self):
        rate = rospy.Rate(50)  # 50 Hz

        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree: 
                # Get closest waypoint  
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints( closest_waypoint_idx)
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

        return closest_idx

    def publish_waypoints(self, closest_idx):

        # With traffic light updates
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish( final_lane )



        '''
        # --------Without traffic light updates ----------
        lane = Lane()

        end_pt = min( closest_idx + LOOKAHEAD_WPS, len(self.base_waypoints.waypoints) )
        lane.header    = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[ closest_idx: end_pt]
        #rospy.logwarn("publishing waypoints: {a:d}:{b:d}".format(a=closest_idx, b=end_pt))
        self.final_waypoints_pub.publish( lane)
        
        # --------Without traffic light updates ----------
        '''


    # Generate lane incorporating information from traffic lights
    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx : farthest_idx]

        # No Red traffic light, or it is further than the end of the planned trajectory
        # Also: at init when no stopline has yet been returned -- just pass the computed trajectory without traffic info.
        if ((self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx) or (not self.stopline_wp_idx)):
            lane.waypoints = base_waypoints
        else:
            # Traffic Light is Red. Tune planned ego velocity so that the vehicle stop at the stop line.
            rospy.logwarn("Waypoint updater: BRAKE.")
            lane.waypoints = self.decelerate_waypoints( base_waypoints, closest_idx)

        return lane


    # Attach required ego velocity information with each waypoint so that the vehicle stop before the stop line
    # at a traffic light.
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []

        # For each waypoint in the planned trajectory
        for i, wp in enumerate(waypoints):

            # A new waypoint
            p = Waypoint()

            # copy the pose. Only ego velocity is modulated.
            p.pose = wp.pose

            # The slack in terms of the number of waypoints we have before we have to come to a full stop.
            # We do not want negative slack (i.e., ego overan the stop line, and is required to reverse). Therefore, the
            # quickest stop would be immediate halt (= 0 slack)
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)

            # Compute the physical distance between the current waypoint (index i) and the waypoint where the ego must
            # come to a complete halt.
            dist = self.distance(waypoints, i, stop_idx)

            # Compute the required velocity for this waypoint subject to deceleration limit so that the ego can come
            # to a full stop at the stop line
            # Equation of motion: v^2 = u^2 + 2*a*S, with v = 0, u = waypoint velocity we would like to set.
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if(vel < 1.):
                # Too low. Just stop
                vel = 0.

            # In case the vehicle is going slower than compute velocity, do not accelerate.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)

            # Add to the waypoints
            temp.append(p)

        # Return the waypoint annotated with velocity information
        return temp






    # Incoming topic #1 callback 
    def pose_cb(self, msg):
        self.pose = msg     # Store the car's pose


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
        # The traffic light detector sends in the index of the waypoint closest the traffic light we should bother about.
        self.stopline_wp_idx = msg.data

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
