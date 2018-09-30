# Compute the derivative of a scan.

import matplotlib.pyplot as plt
from robot import *

# Find the derivative in scan data, ignoring invalid measurements.
def compute_derivative(scan, min_dist):
    jumps = [0]
    for i in range(1, len(scan) - 1):
        # Compute derivative using formula "(f(i+1) - f(i-1)) / 2".
        # Do not use erroneous scan values, which are below min_dist.
        l, r = scan[i-1], scan[i+1]
        if l > min_dist and r > min_dist:
            der = (r - l)/2
        else:
            der = 0
        jumps.append(der)

    jumps.append(0)
    return jumps


if __name__ == '__main__':

    minimum_valid_distance = 20.0

    # Read the logfile which contains all scans.
    logfile = RobotLogfile()
    logfile.read("robot4_scan.txt")

    # Pick one scan.
    scan_no = 235
    scan = logfile.scan_data[scan_no]

    # Compute derivative, (-1, 0, 1) mask.
    der = compute_derivative(scan, minimum_valid_distance)

    # Plot scan and derivative.
    plt.title("Plot of scan %d" % scan_no)
    plt.plot(scan)
    plt.plot(der)
    plt.show()
