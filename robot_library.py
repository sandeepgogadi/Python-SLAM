from math import sin, cos, pi, atan2, sqrt
from robot import *
import numpy as np

# This function takes the old (x, y, heading) pose and the motor ticks
# (ticks_left, ticks_right) and returns the new (x, y, heading).
def filter_step(old_pose, motor_ticks, ticks_to_mm, robot_width,
                scanner_displacement):

    # Find out if there is a turn at all.
    if motor_ticks[0] == motor_ticks[1]:
        # No turn. Just drive straight.
        dist = motor_ticks[0] * ticks_to_mm
        theta = old_pose[2]
        x = old_pose[0] + dist * cos(theta)
        y = old_pose[1] + dist * sin(theta)
        return (x, y, theta)

    else:
        # Turn. Compute alpha, R, etc.
        # Get old center
        old_theta = old_pose[2]
        old_x = old_pose[0]
        old_y = old_pose[1]

        # Modification: subtract offset to compute center.
        old_x -= cos(old_theta) * scanner_displacement
        old_y -= sin(old_theta) * scanner_displacement

        l = motor_ticks[0] * ticks_to_mm
        r = motor_ticks[1] * ticks_to_mm
        alpha = (r - l) / robot_width
        R = l / alpha
        new_theta = (old_theta + alpha) % (2*pi)
        new_x = old_x + (R + robot_width/2.0) * (sin(new_theta) - sin(old_theta))
        new_y = old_y + (R + robot_width/2.0) * (-cos(new_theta) + cos(old_theta))

        # Modification: add offset to compute location of scanner.
        new_x += cos(new_theta) * scanner_displacement
        new_y += sin(new_theta) * scanner_displacement

        return (new_x, new_y, new_theta)

# Find the derivative in scan data, ignoring invalid measurements.
def compute_derivative(scan, min_dist):
    jumps = [ 0 ]
    for i in range(1, len(scan) - 1):
        l = scan[i-1]
        r = scan[i+1]
        if l > min_dist and r > min_dist:
            derivative = (r - l) / 2.0
            jumps.append(derivative)
        else:
            jumps.append(0)
    jumps.append(0)
    return jumps

# For each area between a left falling edge and a right rising edge,
# determine the average ray number and the average depth.
def find_cylinders(scan, scan_derivative, jump, min_dist):
    cylinder_list = []
    on_cylinder = False
    sum_ray, sum_depth, rays = 0.0, 0.0, 0

    for i in range(len(scan_derivative)):
        if scan_derivative[i] < -jump:
            # Start a new cylinder, independent of on_cylinder.
            on_cylinder = True
            sum_ray, sum_depth, rays = 0.0, 0.0, 0
        elif scan_derivative[i] > jump:
            # Save cylinder if there was one.
            if on_cylinder and rays:
                cylinder_list.append((sum_ray/rays, sum_depth/rays))
            on_cylinder = False
        # Always add point, if it is a valid measurement.
        elif scan[i] > min_dist:
            sum_ray += i
            sum_depth += scan[i]
            rays += 1
    return cylinder_list

# Given detected cylinder coordinates: (beam_id, distance), return
# cartesian coordinates (x, y). This is a polar to cartesian conversion
# with an added offset.
def compute_cartesian_coordinates(cylinders, cylinder_offset):
    result = []
    for c in cylinders:
        angle = RobotLogfile.beam_index_to_angle(c[0])
        r = c[1] + cylinder_offset
        result.append( (r*cos(angle), r*sin(angle)) )
    return result

# Returns a new similarity transform, which is the concatenation of
# transform a and b, "a after b".
# The transform is described in the form of:
# (scale, cos(angle), sin(angle), translate_x, translate_y)
# i.e., the angle is described by a direction vector.
def concatenate_transform(a, b):
    laa, ca, sa, txa, tya = a
    lab, cb, sb, txb, tyb = b

    # New lambda is multiplication.
    la = laa * lab

    # New rotation matrix uses trigonometric angle sum theorems.
    c = ca*cb - sa*sb
    s = sa*cb + ca*sb

    # New translation is a translation plus rotated b translation.
    tx = txa + laa * ca * txb - laa * sa * tyb
    ty = tya + laa * sa * txb + laa * ca * tyb

    return (la, c, s, tx, ty)

# Utility to write a list of cylinders to (one line of) a given file.
# Line header defines the start of each line, e.g. "D C" for a detected
# cylinder or "W C" for a world cylinder.
def write_cylinders(file_desc, line_header, cylinder_list):
    output = line_header+' '+' '.join("%.1f %.1f" % c for c in cylinder_list)
    file_desc.write(output + '\n')

# Utility to write a list of cylinders to (one line of) a given file.
# Line header defines the start of each line, e.g. "D C" for a detected
# cylinder or "W C" for a world cylinder.
def write_cylinders_fastslam(file_desc, line_header, cylinder_list):
    print (line_header, end = ' ', file = file_desc)
    for c in cylinder_list:
        print ("%.1f %.1f" % tuple(c), end = ' ', file = file_desc)
    print (end = '\n', file = file_desc)

# This function does all processing needed to obtain the cylinder observations.
# It matches the cylinders and returns distance and angle observations together
# with the corresponding cylinder in the reference dataset.
# In detail:
# - It takes scan data and detects cylinders.
# - For every cylinder, it computes its world coordinate using
#   the polar coordinates from the cylinder detection and the robot's pose,
#   taking into account the scanner's displacement.
# - Using the world coordinate, it finds the closest cylinder in the
#   reference_cylinder list, which has a distance of at most
#   max_reference_distance.
# - If there is such a closest cylinder, the (distance, angle) pair from the
#   scan measurement (these are the two original observations) and the matched
#   cylinder from reference_cylinders are added to the output list.
# - This is repeated for every cylinder detected in the scan.
def get_observations(scan, jump, min_dist, cylinder_offset,
                     robot_pose, scanner_displacement,
                     reference_cylinders, max_reference_distance):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    # Compute scanner pose from robot pose.
    scanner_pose = (robot_pose[0] + cos(robot_pose[2]) * scanner_displacement,
                    robot_pose[1] + sin(robot_pose[2]) * scanner_displacement,
                    robot_pose[2])

    # For every detected cylinder which has a closest matching pole in the
    # reference cylinders set, put the measurement (distance, angle) and the
    # corresponding reference cylinder into the result list.
    result = []
    for c in cylinders:
        # Compute the angle and distance measurements.
        angle = RobotLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        # Compute x, y of cylinder in world coordinates.
        x, y = distance*cos(angle), distance*sin(angle)
        x, y = RobotLogfile.scanner_to_world(scanner_pose, (x, y))
        # Find closest cylinder in reference cylinder set.
        best_dist_2 = max_reference_distance * max_reference_distance
        best_ref = None
        for ref in reference_cylinders:
            dx, dy = ref[0] - x, ref[1] - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_ref = ref
        # If found, add to both lists.
        if best_ref:
            result.append(((distance, angle), best_ref))

    return result

# Detects cylinders and computes bearing, distance and cartesian coordinates (in
# the scanner's coordinate system).
# Result is a list of tuples: (range, bearing, x, y).
def get_cylinders_from_scan(scan, jump, min_dist, cylinder_offset):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    result = []
    for c in cylinders:
        # Compute the angle and distance measurements.
        bearing = RobotLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        # Compute x, y of cylinder in the scanner system.
        x, y = distance*cos(bearing), distance*sin(bearing)
        result.append( (distance, bearing, x, y) )
    return result

# Detects cylinders and computes range, bearing and cartesian coordinates
# (in the scanner's coordinate system).
# The result is modified from previous versions: it returns a list of
# tuples of two numpy arrays, the first being (distance, bearing), the second
# being (x, y) in the scanner's coordinate system.
def get_cylinders_from_scan_fastslam(scan, jump, min_dist, cylinder_offset):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    result = []
    for c in cylinders:
        # Compute the angle and distance measurements.
        bearing = RobotLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        # Compute x, y of cylinder in the scanner system.
        x, y = distance*cos(bearing), distance*sin(bearing)
        result.append( (np.array([distance, bearing]), np.array([x, y])) )
    return result

# For a given pose, assign cylinders.
# cylinders is a list of cylinder measurements
#  (range, bearing, x, y)
#  where x, y are cartesian coordinates in the scanner's system.
# Returns a list of matches, where each element is a tuple of 2 tuples:
#  [ ((range_0, bearing_0), (landmark_x, landmark_y)), ... ]
def assign_cylinders(cylinders, robot_pose, scanner_displacement,
                     reference_cylinders):
    # Compute scanner pose from robot pose.
    scanner_pose = (robot_pose[0] + cos(robot_pose[2]) * scanner_displacement,
                    robot_pose[1] + sin(robot_pose[2]) * scanner_displacement,
                    robot_pose[2])

    # Find closest cylinders.
    result = []
    for c in cylinders:
        # Get world coordinate of cylinder.
        x, y = RobotLogfile.scanner_to_world(scanner_pose, c[2:4])
        # Find closest cylinder in reference cylinder set.
        best_dist_2 = 1e300
        best_ref = None
        for ref in reference_cylinders:
            dx, dy = ref[0] - x, ref[1] - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_ref = ref
        # If found, add to both lists.
        if best_ref:
            result.append((c[0:2], best_ref))

    return result


# Utility to write a list of error ellipses to (one line of) a given file.
# Line header defines the start of each line.
def write_error_ellipses(file_desc, line_header, error_ellipse_list):
    print(line_header, end = ' ', file=file_desc)
    for e in error_ellipse_list:
        print("%.3f %.1f %.1f" % e, end=' ', file=file_desc)
    print('', file=file_desc)

# Utility to write a list of error ellipses to (one line of) a given file.
# Line header defines the start of each line.
# Note that in contrast to previous versions, this takes a list of covariance
# matrices instead of list of ellipses.
def write_error_ellipses_fastslam(file_desc, line_header, covariance_matrix_list):
    print (line_header, end = ' ', file = file_desc)
    for m in covariance_matrix_list:
        eigenvals, eigenvects = np.linalg.eig(m)
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        print ("%.3f %.1f %.1f" % \
                 (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1])), end = ' ', file = file_desc)
    print ('', file = file_desc)

# This function does all processing needed to obtain the cylinder observations.
# It matches the cylinders and returns distance and angle observations together
# with the cylinder coordinates in the world system, the scanner
# system, and the corresponding cylinder index (in the list of estimated parameters).
# In detail:
# - It takes scan data and detects cylinders.
# - For every detected cylinder, it computes its world coordinate using
#   the polar coordinates from the cylinder detection and the robot's pose,
#   taking into account the scanner's displacement.
# - Using the world coordinate, it finds the closest cylinder in the
#   list of current (estimated) landmarks, which are part of the current state.
#
# - If there is such a closest cylinder, the (distance, angle) pair from the
#   scan measurement (these are the two observations), the (x, y) world
#   coordinates of the cylinder as determined by the measurement, the (x, y)
#   coordinates of the same cylinder in the scanner's coordinate system,
#   and the index of the matched cylinder are added to the output list.
#   The index is the cylinder number in the robot's current state.
# - If there is no matching cylinder, the returned index will be -1.
def get_observations_ekfslam(scan, jump, min_dist, cylinder_offset,
                     robot,
                     max_cylinder_distance):
    der = compute_derivative(scan, min_dist)
    cylinders = find_cylinders(scan, der, jump, min_dist)
    # Compute scanner pose from robot pose.
    scanner_pose = (
        robot.state[0] + cos(robot.state[2]) * robot.scanner_displacement,
        robot.state[1] + sin(robot.state[2]) * robot.scanner_displacement,
        robot.state[2])

    # For every detected cylinder which has a closest matching pole in the
    # cylinders that are part of the current state, put the measurement
    # (distance, angle) and the corresponding cylinder index into the result list.
    result = []
    for c in cylinders:
        # Compute the angle and distance measurements.
        angle = RobotLogfile.beam_index_to_angle(c[0])
        distance = c[1] + cylinder_offset
        # Compute x, y of cylinder in world coordinates.
        xs, ys = distance*cos(angle), distance*sin(angle)
        x, y = RobotLogfile.scanner_to_world(scanner_pose, (xs, ys))
        # Find closest cylinder in the state.
        best_dist_2 = max_cylinder_distance * max_cylinder_distance
        best_index = -1
        for index in range(robot.number_of_landmarks):
            pole_x, pole_y = robot.state[3+2*index : 3+2*index+2]
            dx, dy = pole_x - x, pole_y - y
            dist_2 = dx * dx + dy * dy
            if dist_2 < best_dist_2:
                best_dist_2 = dist_2
                best_index = index
        # Always add result to list. Note best_index may be -1.
        result.append(((distance, angle), (x, y), (xs, ys), best_index))

    return result

def get_mean(particles):
    """Compute mean position and heading from a given set of particles."""
    # Note this function would more likely be a part of FastSLAM or a base class
    # of FastSLAM. It has been moved here for the purpose of keeping the
    # FastSLAM class short in this tutorial.
    mean_x, mean_y = 0.0, 0.0
    head_x, head_y = 0.0, 0.0
    for p in particles:
        x, y, theta = p.pose
        mean_x += x
        mean_y += y
        head_x += cos(theta)
        head_y += sin(theta)
    n = max(1, len(particles))
    return np.array([mean_x / n, mean_y / n, atan2(head_y, head_x)])

def get_error_ellipse_and_heading_variance(particles, mean):
    """Given a set of particles and their mean (computed by get_mean()),
       returns a tuple: (angle, stddev1, stddev2, heading-stddev) which is
       the orientation of the xy error ellipse, the half axis 1, half axis 2,
       and the standard deviation of the heading."""
    # Note this function would more likely be a part of FastSLAM or a base class
    # of FastSLAM. It has been moved here for the purpose of keeping the
    # FastSLAM class short in this tutorial.
    center_x, center_y, center_heading = mean
    n = len(particles)
    if n < 2:
        return (0.0, 0.0, 0.0, 0.0)

    # Compute covariance matrix in xy.
    sxx, sxy, syy = 0.0, 0.0, 0.0
    for p in particles:
        x, y, theta = p.pose
        dx = x - center_x
        dy = y - center_y
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    cov_xy = np.array([[sxx, sxy], [sxy, syy]]) / (n-1)

    # Get variance of heading.
    var_heading = 0.0
    for p in particles:
        dh = (p.pose[2] - center_heading + pi) % (2*pi) - pi
        var_heading += dh * dh
    var_heading = var_heading / (n-1)

    # Convert xy to error ellipse.
    eigenvals, eigenvects = np.linalg.eig(cov_xy)
    ellipse_angle = atan2(eigenvects[1,0], eigenvects[0,0])

    return (ellipse_angle, sqrt(abs(eigenvals[0])),
            sqrt(abs(eigenvals[1])),
            sqrt(var_heading))

def print_particles(particles, file_desc):
    # Note this function would more likely be a part of FastSLAM or a base class
    # of FastSLAM. It has been moved here for the purpose of keeping the
    # FastSLAM class short in this tutorial.
    """Prints particles to given file_desc output."""
    if not particles:
        return
    print ("PA", end=' ', file=file_desc)
    for p in particles:
        print ("%.0f %.0f %.3f" % tuple(p.pose), end=' ', file=file_desc)
    print (end='\n', file=file_desc)
