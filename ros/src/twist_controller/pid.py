
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn, mx):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = self.last_error = 0.

    def reset(self):
        self.int_val = 0.0

    def step(self, error, sample_time):

        integral   = self.int_val + error * sample_time;
        if(sample_time == 0.):
            derivative = 0
        else:
            derivative = (error - self.last_error) / sample_time;

        val = self.kp * error + self.ki * self.int_val + self.kd * derivative

        # Bound the result.
        val = max(self.min, min(val, self.max))

        #rospy.logwarn("error: {a:f}, integral: {b:f}, derivative: {c:f}, pid: {d:f}".format(a=error,b=self.int_val,c=derivative, d=val))
        #rospy.logwarn("error: {a:f}".format(a=error))

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        self.last_error = error

        return val
