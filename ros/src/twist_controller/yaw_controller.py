# For steering commands ...
#      A controller that can be used to convert target linear and angular velocities to steering commands

import math
from math import atan
import rospy
class YawController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle, pSteering, iSteering, dSteering):
        self.wheel_base    = wheel_base
        self.steer_ratio   = steer_ratio
        self.min_speed     = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle     = -max_steer_angle
        self.max_angle     = max_steer_angle

        self.integralError = 0.
        self.proportionalError = 0.
        self.differentialError = 0.
        self.oldCTE = 0.

        self.pSteering = pSteering
        self.iSteering = iSteering
        self.dSteering = dSteering

        #self.last_steering_angle = 0.0

    def get_angle(self, radius):
        if math.fabs(radius) < 0.00001:
            return 0.0
        else:
            angle = atan(self.wheel_base / radius) * self.steer_ratio
        return max(self.min_angle, min(self.max_angle, angle))

    def get_steering(self, linear_velocity, angular_velocity, current_velocity, lastSteeringWheelAngle):

        angular_velocity = current_velocity * angular_velocity / linear_velocity if abs(linear_velocity) > 0. else 0.
        ##AUTOMATEDCOMMENTREMOVAL:rospy.logwarn("angular_vel(28): {a:f}, linear_velocity: {b:f}, current_velocity: {c:f}".format(a=angular_velocity, b=linear_velocity, c=current_velocity))

        if abs(current_velocity) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / current_velocity);
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))



        steering_angle_new = self.get_angle(max(current_velocity, self.min_speed) / angular_velocity) if abs(angular_velocity) > 0. else 0.0;

        #steering_angle = steering_angle_new - self.last_steering_angle

        #self.last_steering_angle = steering_angle_new


        ##AUTOMATEDCOMMENTREMOVAL:rospy.logwarn(
        #    "angular_vel(39): {a:f}, linear_velocity: {b:f}, current_velocity: {c:f}, steering: {d:f}".format(a=angular_velocity,
        #                                                                                     b=linear_velocity,
        #                                                                                     c=current_velocity,
        #                                                                                     d=steering_angle_new))


        return steering_angle_new


    def get_steeringFromCTE(self, linear_vel, angular_vel, current_vel, lastSteeringWheelAngle, cte):
        self.integralError = self.integralError + cte
        self.proportionalError  = cte
        self.differentialError = cte - self.oldCTE
        self.oldCTE = cte

        steering = - (self.pSteering * self.proportionalError + self.iSteering * self.integralError + self.dSteering * self.differentialError)

        #steering = steering - (lastSteeringWheelAngle * math.pi / 180.)

        return max(self.min_angle, min(self.max_angle, steering))




