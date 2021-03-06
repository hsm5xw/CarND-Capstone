import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg  import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport  # Drive-by-wire messages
from geometry_msgs.msg import TwistStamped

from yaw_controller import YawController  # yaw_controller.py
from pid import PID                       # pid.py
from lowpass import LowPassFilter         # lowpass.py

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

# ------------ PARAMETERS  --------------------------
MIN_SPEED = 0.1
kp = 1.0
ki = 0.1 
kd = 0.1

mn = 0.      # minimum throttle value
mx = 0.15     # maximum throttle value

tau = 0.5    # 1/(2pi*tau) = cutoff frequency
ts  = 0.02   # sample time
# -------------------------------------------------

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                       accel_limit, wheel_radius, wheel_base, steer_ratio, 
                       max_lat_accel, max_steer_angle):

        self.vehicle_mass    = vehicle_mass
        self.fuel_capacity   = fuel_capacity
        self.brake_deadband  = brake_deadband
        self.decel_limit     = decel_limit
        self.accel_limit     = accel_limit
        self.wheel_radius    = wheel_radius
        self.wheel_base      = wheel_base
        self.steer_ratio     = steer_ratio
        self.max_lat_accel   = max_lat_accel
        self.max_steer_angle = max_steer_angle

        # global debug counter
        self.debug_counter = 0

        # yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        
        # throttle controller
        self.throttle_controller = PID( kp, ki, kd, mn, mx)
        self.vel_lpf             = LowPassFilter(tau, ts)

        self.last_time = rospy.get_time()
        
    """
    :@ linear_vel : target linear velocity
    :@ angular_vel: target angular velocity
    """
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel, cte):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            self.vel_lpf.reset()
            return 0., 0., 0.

        # Begin to calculate Throttle Value
        current_vel   = self.vel_lpf.filt( current_vel)
        vel_error     = linear_vel - current_vel        

        current_time    = rospy.get_time()
        sample_time     = current_time - self.last_time
        self.last_time  = current_time

        # throttle control
        throttle = self.throttle_controller.step( vel_error, sample_time)
        brake    = 0.

        # steering control
        angular_vel = (1.5)* angular_vel  # (*) adjust (decrease) radius of pure pursuit
        
        steering   = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        steering   = max(-self.max_steer_angle, min(self.max_steer_angle, steering) )
        
        # If target linear velocity = 0, then go very slow        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0.
            brake    = 700   # N*m - to hold the car in place if we are stopped at a light. Acceleration - 1m/s^2
        
        # If throttle is really small and velocity error < 0 (i.e. we're going faster than we want to be)
        elif throttle < 0.1 and vel_error < 0.:
            throttle = 0.
            decel    = max( vel_error, self.decel_limit)  # a negative number
            brake    = abs(decel) * self.vehicle_mass * self.wheel_radius   # Torque N*m

        #rospy.logwarn( "########### debug_count: {0} ###############".format(self.debug_counter) )
        #rospy.logwarn( "steering: {0}".format(steering) )
        #rospy.logwarn( "throttle: \t {0}".format(throttle) )
        #rospy.logwarn( "\n")
        self.debug_counter = self.debug_counter + 1
 
        return throttle, brake, steering

