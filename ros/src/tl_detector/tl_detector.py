#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
from scipy.spatial import cKDTree # use cKDTree instead

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        # global debug counter
        self.debug_counter = 0

        self.use_classifier  = rospy.get_param('use_classifier', False)
        rospy.logwarn("use_classifier: {}".format(self.use_classifier) ) # see if it works ...
        
        self.pose            = None
        self.base_waypoints  = None
        self.waypoints_2d    = None
        self.camera_image    = None
        self.waypoint_tree   = None
        self.lights          = []
        self.has_image       = False

        # put all variables above publisher/subscribers.. 
        # -----------------------------------------------------------------------------------
        config_string = rospy.get_param("/traffic_light_config")
        self.config   = yaml.load(config_string)

        # output
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge           = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener         = tf.TransformListener()

        self.state       = TrafficLight.UNKNOWN
        self.last_state  = TrafficLight.UNKNOWN
        self.last_wp     = -1
        self.state_count = 0
        
        # keep count of the image so far, so only process the nth image
        # This is to help the lag problem
        self.image_count = 0

        # input ---------------------------------------------------------------------------
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # TODO: replace '/image_color' with '/image_raw' (*)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb) # ground-truth from simulator
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        # ---------------------------------------------------------------------------------

        # Let the tl_detector spin, all the work is done by the _cb functions.
        # as the image_cb is called and received a new image, it will determine the color of the stop light
        rospy.spin()
        
    # callback routines
    # ---------------------------------------------------------------------------------------
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):

        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = cKDTree(self.waypoints_2d)

    def traffic_cb(self, msg):   # ground-truth from simulator
        self.lights = msg.lights

    def image_cb(self, msg):     # incoming image
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        
        if (self.image_count > 0):
            # don't process anything, just publish and return
            self.image_count -= 1
            self.upcoming_red_light_pub.publish( Int32(-100) ) 
            return 1
        else:
            # this is the nth image, let's start processing
            self.image_count = rospy.get_param('image_count', 5)    # set to default 5, only every 5th image is processed
            
            self.has_image    = True
            self.camera_image = msg

            # Get (waypoint index, color) of the closest traffic light ahead
            light_wp, state   = self.process_traffic_lights()

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            # color transition  ex) green -> yellow
            if self.state != state:
                self.state_count = 0
                self.state       = state

            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp        = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))

            self.state_count += 1
    # ----------------------------------------------------------------------------------------

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        closest_idx = self.waypoint_tree.query([x, y])[1]  # ckd tree (1st closest, idx)

        # Check if closest waypoint is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord    = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coors
        cl_vect   = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect  = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        # Car is ahead of the closest waypoint
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if self.use_classifier:       
            if(not self.has_image):
                return None

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            #Get classification
            return self.light_classifier.get_classification(cv_image) 
        
        else:
            # For testing, just return the ground-truth light state, directly coming from simulator
            return light.state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None
        line_wp_index = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if(self.pose):

            car_wp_index = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            #TODO find the closest visible traffic light (if one exists)

            # We want to find the smallest "diff". Initialize to the maximum possible value: total number of waypoints
            diff = len(self.base_waypoints.waypoints)

            # Now go over each light position in the known map (total of 8) and find the closest one to the ego
            for i, light in enumerate(self.lights):

                # Extract the position of the stop line associated with the light (index = i)
                line = stop_line_positions[i]  # position of the stop line

                # Extract the closest waypoint to x=line[0] and y=line[1] associated to traffic light (index = i) and in front of the ego
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                # Find the number of waypoints between the ego and the temp_wp_idx
                d = temp_wp_idx - car_wp_index

                # If this the smallest so far?
                if (d >= 0) and (d < diff):
                    # update diff to the smallest d (corresponding to traffic light (index = i) found so far
                    diff = d

                    # The closest light to the ego so far
                    closest_light = light
                    line_wp_index = temp_wp_idx

        # If we did find a closest light:
        if (closest_light):
            # Read the state of the closest light and update the state (i.e., the color of the light)
            # Red: 0, Yellow: 1, Green: 2, Unknown: 4
            state = self.get_light_state(closest_light)

            if state is None:
                 rospy.logwarn("Closest Traffic Light: no state ...")
                 return -1, TrafficLight.UNKNOWN
                
            color_dict = { TrafficLight.RED:     'Red',    \
                           TrafficLight.YELLOW:  'Yellow', \
                           TrafficLight.GREEN:   'Green',  \
                           TrafficLight.UNKNOWN: 'Unknown'}

            rospy.logwarn("Closest Traffic Light: ({0}, {1})".format( color_dict[state], line_wp_index) )
            
            return line_wp_index, state
        else:
            # rospy.logwarn("No traffic light found.")
            return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
