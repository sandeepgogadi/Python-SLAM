# Plot a scan of the robot using matplotlib.

import matplotlib.pyplot as plt
from robot import *

# Read the logfile which contains all scans.
logfile = RobotLogfile()
logfile.read("robot4_scan.txt")

# Plot one scan.
plt.plot(logfile.scan_data[8])
plt.show()
