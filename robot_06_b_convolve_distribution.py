# Instead of moving a distribution, move (and modify) it using a convolution.

import matplotlib.pyplot as plt
from distribution import *

def move(distribution, delta):
    """Returns a Distribution that has been moved (x-axis) by the amount of
       delta."""
    return Distribution(distribution.offset + delta, distribution.values)

def convolve(a, b):
    """Convolve distribution a and b and return the resulting new distribution."""
    a = move(a, b.offset)
    posterior = []
    for idx, a_val in enumerate(a.values):
        new_val = [a_val*b_val for b_val in b.values]
        d = Distribution(a.offset + idx, new_val)
        posterior.append(d)

    return Distribution.sum(posterior)


if __name__ == '__main__':
    arena = (0,100)

    # Move 3 times by 20.
    moves = [20] * 3

    # Start with a known position: probability 1.0 at position 10.
    position = Distribution.unit_pulse(10)
    plt.plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
         linestyle='steps')

    # Now move and plot.
    for m in moves:
        move_distribution = Distribution.triangle(m, 2)
        position = convolve(position, move_distribution)
        plt.plot(position.plotlists(*arena)[0], position.plotlists(*arena)[1],
             linestyle='steps')

    plt.ylim(0.0, 1.1)
    plt.show()
