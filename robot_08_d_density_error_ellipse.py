# The particle filter, prediciton and correction.
# In addition to the previous code:
# 1.
# the second moments are computed and are output as an error ellipse and
# heading variance.
# 2.
# the particles are initialized uniformly distributed in the arena, and a
# larger number of particles is used.
# 3.
# predict and correct are only called when control is nonzero.
#

from robot import *
from robot_library import get_cylinders_from_scan, assign_cylinders
from math import sin, cos, pi, atan2, sqrt
import random
import numpy as np
from scipy.stats import norm as normal_dist


class ParticleFilter:
    def __init__(self, initial_particles,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The particles.
        self.particles = initial_particles

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

    # State transition. This is exactly the same method as in the Kalman filter.
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

        return (g1, g2, g3)

    def predict(self, control):
        """The prediction step of the particle filter."""
        l, r = control
        alpha1, alpha2 = self.control_motion_factor, self.control_turn_factor
        sigmal = sqrt((alpha1*l)**2 + alpha2*(l - r)**2)
        sigmar = sqrt((alpha1*r)**2 + alpha2*(l - r)**2)

        controls = [(random.gauss(l, sigmal), random.gauss(r, sigmar)) for p in self.particles]

        self.particles = [self.g(p, u, self.robot_width) for p, u in zip(self.particles, controls)]


    # Measurement. This is exactly the same method as in the Kalman filter.
    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (x, y) landmark, and returns the
           corresponding (range, bearing)."""
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
        return (r, alpha)

    def probability_of_measurement(self, measurement, predicted_measurement):
        """Given a measurement and a predicted measurement, computes
           probability."""
        # Compute differences to real measurements.

        sigma_d = self.measurement_distance_stddev
        sigma_alpha = self.measurement_angle_stddev

        z = np.array(measurement)
        pred_z = np.array(predicted_measurement)

        z[1] = atan2(sin(z[1]), cos(z[1]))
        pred_z[1] = atan2(sin(pred_z[1]), cos(pred_z[1]))

        delta_z = z - pred_z
        delta_z[1] = atan2(sin(delta_z[1]), cos(delta_z[1]))

        p_d = normal_dist.pdf(delta_z[0], 0, sigma_d)
        p_alpha = normal_dist.pdf(delta_z[1], 0, sigma_alpha)


        # --->>> Compute difference in distance and bearing angle.
        # Important: make sure the angle difference works correctly and does
        # not return values offset by 2 pi or - 2 pi.
        # You may use the following Gaussian PDF function:
        # scipy.stats.norm.pdf(x, mu, sigma). With the import in the header,
        # this is normal_dist.pdf(x, mu, sigma).
        # Note that the two parameters sigma_d and sigma_alpha discussed
        # in the lecture are self.measurement_distance_stddev and
        # self.measurement_angle_stddev.
        return p_d*p_alpha

    def compute_weights(self, cylinders, landmarks):
        """Computes one weight for each particle, returns list of weights."""
        weights = []
        for p in self.particles:
            # Get list of tuples:
            # [ ((range_0, bearing_0), (landmark_x, landmark_y)), ... ]
            assignment = assign_cylinders(cylinders, p,
                self.scanner_displacement, landmarks)

            wt = 1.
            for z, landmark in assignment:
                z_pred = self.h(p, landmark, self.scanner_displacement)
                wt *= self.probability_of_measurement(z, z_pred)
            weights.append(wt)
        total = sum(weights)
        weights = [w/total for w in weights]
            # This will require a loop over all (measurement, landmark)
            # in assignment. Append weight to the list of weights.
        return weights

    def resample(self, weights):
        """Return a list of particles which have been resampled, proportional
           to the given weights."""

        # You may implement the 'resampling wheel' algorithm
        # described in the lecture.
        old_particles = self.particles
        N = len(old_particles)
        new_particles = []

        # Resampling Wheel
        index = int(random.random() * N)
        beta = 0.0
        mw = max(weights)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % N
            new_particles.append(old_particles[index])

        '''
        new_particles = [self.particles[np.random.choice(N, p=weights)]
                for i in range(N)]
        '''

        return new_particles

    def correct(self, cylinders, landmarks):
        """The correction step of the particle filter."""
        # First compute all weights.
        weights = self.compute_weights(cylinders, landmarks)
        # Then resample, based on the weight array.
        self.particles = self.resample(weights)

    def print_particles(self, file_desc):
        """Prints particles to given file_desc output."""
        if not self.particles:
            return
        print("PA", file=file_desc, end=' ')
        for p in self.particles:
            print("%.0f %.0f %.3f" % p, file=file_desc, end=' ')
        print(file=file_desc)

    def get_mean(self):
        """Compute mean position and heading from all particles."""

        # --->>> This is the new code you'll have to implement.
        # Return a tuple: (mean_x, mean_y, mean_heading).

        mean_x, mean_y, mean_heading, vx, vy = 0., 0., 0., 0, 0
        m = len(self.particles)

        for p in self.particles:
            x, y, theta = p
            mean_x += x
            mean_y += y
            vx += cos(theta)
            vy += sin(theta)

        mean_x /= m
        mean_y /= m
        mean_heading = atan2(vy, vx)
        return (mean_x, mean_y, mean_heading)


    # *** Modification 1: Extension: This computes the error ellipse.
    def get_error_ellipse_and_heading_variance(self, mean):
        """Returns a tuple: (angle, stddev1, stddev2, heading-stddev) which is
           the orientation of the xy error ellipse, the half axis 1, half axis 2,
           and the standard deviation of the heading."""
        center_x, center_y, center_heading = mean
        n = len(self.particles)
        if n < 2:
            return (0.0, 0.0, 0.0, 0.0)

        # Compute covariance matrix in xy.
        sxx, sxy, syy = 0.0, 0.0, 0.0
        for p in self.particles:
            dx = p[0] - center_x
            dy = p[1] - center_y
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        cov_xy = np.array([[sxx, sxy], [sxy, syy]]) / (n-1)

        # Get variance of heading.
        var_heading = 0.0
        for p in self.particles:
            dh = (p[2] - center_heading + pi) % (2*pi) - pi
            var_heading += dh * dh
        var_heading = var_heading / (n-1)

        # Convert xy to error ellipse.
        eigenvals, eigenvects = np.linalg.eig(cov_xy)
        ellipse_angle = atan2(eigenvects[1,0], eigenvects[0,0])

        return (ellipse_angle, sqrt(abs(eigenvals[0])),
                sqrt(abs(eigenvals[1])),
                sqrt(var_heading))


if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.
    measurement_distance_stddev = 200.0  # Distance measurement error of cylinders.
    measurement_angle_stddev = 15.0 / 180.0 * pi  # Angle measurement error.

    # Generate initial particles. Each particle is (x, y, theta).
    # *** Modification 2: Generate the particles uniformly distributed.
    # *** Also, use a large number of particles.
    number_of_particles = 500
    # Alternative: uniform init.
    initial_particles = []
    for i in range(number_of_particles):
        initial_particles.append((
            random.uniform(0.0, 2000.0), random.uniform(0.0, 2000.0),
            random.uniform(-pi, pi)))

    # Setup filter.
    pf = ParticleFilter(initial_particles,
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

    # Loop over all motor tick records.
    # This is the particle filter loop, with prediction and correction.
    f = open("particle_filter_ellipse.txt", "w")
    for i in range(len(logfile.motor_ticks)):
        control = map(lambda x: x * ticks_to_mm, logfile.motor_ticks[i])
        # *** Modification 3: Call the predict/correct step only if there
        # *** is nonzero control.
        if control != [0.0, 0.0]:
            # Prediction.
            pf.predict(control)

            # Correction.
            cylinders = get_cylinders_from_scan(logfile.scan_data[i], depth_jump,
                minimum_valid_distance, cylinder_offset)
            pf.correct(cylinders, reference_cylinders)

        # Output particles.
        pf.print_particles(f)

        # Output state estimated from all particles.
        mean = pf.get_mean()
        print("F %.0f %.0f %.3f" %\
              (mean[0] + scanner_displacement * cos(mean[2]),
               mean[1] + scanner_displacement * sin(mean[2]),
               mean[2]), file=f)

        # Output error ellipse and standard deviation of heading.
        errors = pf.get_error_ellipse_and_heading_variance(mean)
        print("E %.3f %.0f %.0f %.3f" % errors, file=f)

    f.close()
