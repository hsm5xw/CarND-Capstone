import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg  import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport  # Drive-by-wire messages
from geometry_msgs.msg import TwistStamped

from yaw_controller import YawController  # yaw_controller.py
from pid import PID                       # pid.py
from lowpass import LowPassFilter         # lowpass.py

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

MIN_SPEED = 0.1

def clamp(val, min_val, max_val):
    val = max( val, min_val)
    val = min( val, max_val)

    return val

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                       accel_limit, wheel_radius, wheel_base, steer_ratio, 
                       max_lat_accel, max_steer_angle):

        # global debug counter
        self.debug_counter = 0

        # parameters
        self.vehicle_mass   = vehicle_mass
        self.fuel_capacity  = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit    = decel_limit
        self.accel_limit    = accel_limit
        self.wheel_radius   = wheel_radius
        self.last_time      = rospy.get_time()

        # TODO: Implement
        # (init) yaw controller
        self.yaw_controller = YawController( wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
           
        # (init) throttle controller
        kp = rospy.get_param('Kp', 1.3)  # set parameters from 'launch/styx.launch', without recompiling every time 
        ki = rospy.get_param('Ki', 0.0)
        kd = rospy.get_param('Kd', 0.5)

        throttle_min = 0.      # minimum throttle value
        throttle_max = 0.2     # maximum throttle value
        self.throttle_controller = PID( kp, ki, kd, throttle_min, throttle_max)

        # (init) low pass filter
        tau = 0.5              # 1/(2pi*tau) = cutoff frequency
        ts  = 0.02             # sample time
        self.vel_lpf = LowPassFilter(tau, ts)

    """
    :@ linear_vel : target linear  velocity
    :@ angular_vel: target angular velocity
    """
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.        

        # steering control
        current_vel = self.vel_lpf.filt( current_vel)  # remove noise
        steering    = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
      

        # Begin to calculate Throttle Value
        vel_error     = linear_vel - current_vel        
        self.last_vel = current_vel

        current_time    = rospy.get_time()
        sample_time     = current_time - self.last_time
        self.last_time  = current_time

        # throttle control
        throttle = self.throttle_controller.step( vel_error, sample_time)
        brake    = 0.

        # If target linear velocity = 0, then go very slow        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0.
            brake    = 400   # N*m - to hold the car in place if we are stopped at a light. Acceleration - 1m/s^2
        
        # If throttle is really small and velocity error < 0 (i.e. we're going faster than we want to be)
        elif throttle < 0.1 and vel_error < 0.:
            throttle = 0.
            decel    = max( vel_error, self.decel_limit)  # a negative number
            brake    = abs(decel) * self.vehicle_mass * self.wheel_radius   # Torque N*m

        #throttle = clamp(throttle, 0, 0.05)

        rospy.logwarn( "########### debug_count: {0} ###############".format(self.debug_counter) )
        #rospy.logwarn( "Target  linear  vel: {0}".format(linear_vel) )
        #rospy.logwarn( "Target  angular vel: {0}".format(angular_vel) )
        #rospy.logwarn( "Current vel: \t {0}".format(current_vel) )
        
        #rospy.logwarn( "\n")
        #rospy.logwarn( "Filtered vel: \t {0}".format(self.vel_lpf.get()) )

        rospy.logwarn( "steering: {0}".format(steering) )
        rospy.logwarn( "throttle: \t {0}".format(throttle) )
        rospy.logwarn( "\n")
        self.debug_counter = self.debug_counter + 1
 
        return throttle, brake, steering

