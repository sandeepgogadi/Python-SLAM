# Plot the increments of the left and right motor.

import matplotlib.pyplot as plt
from robot import RobotLogfile

if __name__ == '__main__':

    logfile = RobotLogfile()
    logfile.read("robot4_motors.txt")

    plt.plot(logfile.motor_ticks)
    plt.show()
