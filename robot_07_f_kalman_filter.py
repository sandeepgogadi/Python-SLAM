# The full Kalman filter, consisting of prediction and correction step.
#

from robot import *
from math import sin, cos, pi, atan2, sqrt
from numpy import *
from robot_library import get_observations, write_cylinders


class ExtendedKalmanFilter:
    def __init__(self, state, covariance,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The state. This is the core data of the Kalman filter.
        self.state = state
        self.covariance = covariance

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

    @staticmethod
    def g(state, control, w):
        x, y, theta = state
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + l * cos(theta)
            g2 = y + l * sin(theta)
            g3 = theta

        return array([g1, g2, g3])

    @staticmethod
    def dg_dstate(state, control, w):
        theta = state[2]
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            a = (rad + w*.5) * (cos(theta + alpha) - cos(theta))
            b = (rad + w*.5) * (sin(theta + alpha) - sin(theta))
            # --->>> Put your code here.
            # This is for the case r != l.
            # g has 3 components and the state has 3 components, so the
            # derivative of g with respect to all state variables is a
            # 3x3 matrix. To construct such a matrix in Python/Numpy,
            # use: m = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            # where 1, 2, 3 are the values of the first row of the matrix.
            # Don't forget to return this matrix.
            m = array([[1, 0, a], [0, 1, b], [0, 0, 1]])

        else:
            a = -l * sin(theta)
            b = l * cos(theta)
            # --->>> Put your code here.
            # This is for the special case r == l.
            m = array([[1, 0, a], [0, 1, b], [0, 0, 1]])

        return m

    @staticmethod
    def dg_dcontrol(state, control, w):
        theta = state[2]
        l, r = tuple(control)
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            # This is for the case l != r.
            # Note g has 3 components and control has 2, so the result
            # will be a 3x2 (rows x columns) matrix.
            m11 = (w*r)*( sin(theta+alpha) - sin(theta))/(r-l)**2 \
                   -(r+l)*cos(theta+alpha)/(2*(r-l))
            m21 = (w*r)*(-cos(theta+alpha) + cos(theta))/(r-l)**2 \
                   -(r+l)*sin(theta+alpha)/(2*(r-l))


            m12 = (-w*l)*( sin(theta+alpha) - sin(theta))/(r-l)**2 \
                   +(r+l)*cos(theta+alpha)/(2*(r-l))
            m22 = (-w*l)*(-cos(theta+alpha) + cos(theta))/(r-l)**2 \
                   +(r+l)*sin(theta+alpha)/(2*(r-l))
        else:
            # This is for the special case l == r.
            m11 = 0.5*(cos(theta) + (l/w)*sin(theta))
            m21 = 0.5*(sin(theta) - (l/w)*cos(theta))
            m12 = 0.5*(-(l/w)*sin(theta) + cos(theta))
            m22 = 0.5*( (l/w)*cos(theta) + sin(theta))

        m31 = -1/w
        m32 =  1/w

        m = array([[m11, m12], [m21, m22], [m31, m32]])

        return m

    @staticmethod
    def get_error_ellipse(covariance):
        """Return the position covariance (which is the upper 2x2 submatrix)
           as a triple: (main_axis_angle, stddev_1, stddev_2), where
           main_axis_angle is the angle (pointing direction) of the main axis,
           along which the standard deviation is stddev_1, and stddev_2 is the
           standard deviation along the other (orthogonal) axis."""
        eigenvals, eigenvects = linalg.eig(covariance[0:2,0:2])
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        return (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1]))

    def predict(self, control):
        """The prediction step of the Kalman filter."""
        # covariance' = G * covariance * GT + R
        # where R = V * (covariance in control space) * VT.
        # Covariance in control space depends on move distance.
        left, right = control

        # --->>> Put your code to compute the new self.covariance here.
        # First, construct the control_covariance, which is a diagonal matrix.
        # In Python/Numpy, you may use diag([a, b]) to get
        # [[ a, 0 ],
        #  [ 0, b ]].
        # Then, compute G using dg_dstate and V using dg_dcontrol.
        # Then, compute the new self.covariance.
        # Note that the transpose of a Numpy array G is expressed as G.T,
        # and the matrix product of A and B is written as dot(A, B).
        # Writing A*B instead will give you the element-wise product, which
        # is not intended here.

        a = (self.control_motion_factor * left)**2 + (self.control_turn_factor * (left - right))**2
        b = (self.control_motion_factor * right)**2 + (self.control_turn_factor * (left - right))**2

        E_control = diag([a, b]) # control_covariance

        G = self.dg_dstate(self.state, control, self.robot_width)
        V = self.dg_dcontrol(self.state, control, self.robot_width)

        self.covariance = dot(G, dot(self.covariance, G.T)) + dot(V, dot(E_control, V.T))


        # state' = g(state, control)

        self.state = self.g(self.state, control, self.robot_width)

    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (x, y) landmark, and returns the
           measurement (range, bearing)."""
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi

        return array([r, alpha])

    @staticmethod
    def dh_dstate(state, landmark, scanner_displacement):

        # Note that:
        # x y theta is state[0] state[1] state[2]
        # x_m y_m is landmark[0] landmark[1]
        # The Jacobian of h is a 2x3 matrix.

        x, y, theta = state
        x_m, y_m = landmark
        d = scanner_displacement
        x_l = x + d*cos(theta)
        y_l = y + d*sin(theta)

        delta_x = x_m - x_l
        delta_y = y_m - y_l

        q = delta_x**2 + delta_y**2

        m11 = -delta_x/sqrt(q)
        m12 = -delta_y/sqrt(q)
        m13 = (d/sqrt(q)) * (delta_x*sin(theta) - delta_y*cos(theta))
        m21 = delta_y/q
        m22 = -delta_x/q
        m23 = -(d/q) * (delta_x*cos(theta) + delta_y*sin(theta)) - 1


        return array([[m11, m12, m13], [m21, m22, m23]])

    def correct(self, measurement, landmark):
        """The correction step of the Kalman filter."""

        # --->>> Put your new code here.
        #
        # You will have to compute:
        # H, using dh_dstate(...).
        # Q, a diagonal matrix, from self.measurement_distance_stddev and
        #  self.measurement_angle_stddev (remember: Q contains variances).
        # K, from self.covariance, H, and Q.
        #  Use linalg.inv(...) to compute the inverse of a matrix.
        # The innovation: it is easy to make an error here, because the
        #  predicted measurement and the actual measurement of theta may have
        #  an offset of +/- 2 pi. So here is a suggestion:
        #   innovation = array(measurement) -\
        #                self.h(self.state, landmark, self.scanner_displacement)
        #   innovation[1] = (innovation[1] + pi) % (2*pi) - pi
        # Then, you'll have to compute the new self.state.
        # And finally, compute the new self.covariance. Use eye(3) to get a 3x3
        #  identity matrix.
        #
        # Hints:
        # dot(A, B) is the 'normal' matrix product (do not use: A*B).
        # A.T is the transposed of a matrix A (A itself is not modified).
        # linalg.inv(A) returns the inverse of A (A itself is not modified).
        # eye(3) returns a 3x3 identity matrix.

        H = self.dh_dstate(self.state, landmark, self.scanner_displacement)
        Q = diag([self.measurement_distance_stddev, self.measurement_angle_stddev])
        K = dot(self.covariance, dot(H.T, linalg.inv(dot(H, dot(self.covariance, H.T)) + Q)))

        innovation = array(measurement) - self.h(self.state, landmark, self.scanner_displacement)

        innovation[0] = (innovation[0] + pi) % (2*pi) - pi
        innovation[1] = (innovation[1] + pi) % (2*pi) - pi

        self.state = self.state + dot(K, innovation)

        self.covariance = dot((eye(3) - dot(K, H)), self.covariance)



if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0
    max_cylinder_distance = 300.0

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.
    measurement_distance_stddev = 200.0  # Distance measurement error of cylinders.
    measurement_angle_stddev = 15.0 / 180.0 * pi  # Angle measurement error.

    # Measured start position.
    initial_state = array([1850.0, 1897.0, 213.0 / 180.0 * pi])
    # Covariance at start position.
    initial_covariance = diag([100.0**2, 100.0**2, (10.0 / 180.0 * pi) ** 2])
    # Setup filter.
    kf = ExtendedKalmanFilter(initial_state, initial_covariance,
                              robot_width, scanner_displacement,
                              control_motion_factor, control_turn_factor,
                              measurement_distance_stddev,
                              measurement_angle_stddev)

    # Read data.
    logfile = RobotLogfile()
    logfile.read("robot4_motors.txt")
    logfile.read("robot4_scan.txt")
    logfile.read("robot_arena_landmarks.txt")
    reference_cylinders = [l[1:3] for l in logfile.landmarks]

    # Loop over all motor tick records and all measurements and generate
    # filtered positions and covariances.
    # This is the Kalman filter loop, with prediction and correction.
    states = []
    covariances = []
    matched_ref_cylinders = []
    for i in range(len(logfile.motor_ticks)):
        # Prediction.
        control = array(logfile.motor_ticks[i]) * ticks_to_mm
        kf.predict(control)

        # Correction.
        observations = get_observations(
            logfile.scan_data[i],
            depth_jump, minimum_valid_distance, cylinder_offset,
            kf.state, scanner_displacement,
            reference_cylinders, max_cylinder_distance)
        for j in range(len(observations)):
            kf.correct(*observations[j])

        # Log state, covariance, and matched cylinders for later output.
        states.append(kf.state)
        covariances.append(kf.covariance)
        matched_ref_cylinders.append([m[1] for m in observations])

    # Write all states, all state covariances, and matched cylinders to file.
    f = open("kalman_prediction_and_correction.txt", "w")
    for i in range(len(states)):
        # Output the center of the scanner, not the center of the robot.
        print ("F %f %f %f" % \
            tuple(states[i] + [scanner_displacement * cos(states[i][2]),
                               scanner_displacement * sin(states[i][2]),
                               0.0]), file=f)
        # Convert covariance matrix to angle stddev1 stddev2 stddev-heading form
        e = ExtendedKalmanFilter.get_error_ellipse(covariances[i])
        print ("E %f %f %f %f" % (e + (sqrt(covariances[i][2,2]),)), file=f)
        # Also, write matched cylinders.
        write_cylinders(f, "W C", matched_ref_cylinders[i])

    f.close()
