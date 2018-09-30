# Plot the ticks from the left and right motor.

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read all ticks of left and right motor.
    # Format is:
    # M timestamp[in ms] pos-left[in ticks] * * * pos-right[in ticks] ...
    # we are interested in field 2 (left) and 6 (right).

    f = open("robot4_motors.txt")
    left_list = []
    right_list = []
    for l in f:
        sp = l.split()
        left_list.append(int(sp[2]))
        right_list.append(int(sp[6]))

    plt.plot(left_list, label='Left Wheel')
    plt.plot(right_list, label='Right Wheel')
    plt.xlabel('Increments')
    plt.ylabel('Starting Values')
    plt.legend()
    plt.show()
