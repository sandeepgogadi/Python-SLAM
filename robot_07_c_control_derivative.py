# This adds the derivative of g, this time with respect to the control
# (left and right motor movement).
#

from robot import *
from math import sin, cos, pi
from numpy import *

class ExtendedKalmanFilter:

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


if __name__ == '__main__':
    # If the partial derivative with respect to l and r (the control)
    # are correct, then the numerical derivative and the analytical
    # derivative should be the same.

    # Set some variables. Try other variables as well.
    # In particular, you should check cases with l == r and l != r.
    x = 10.0
    y = 20.0
    theta = 35. / 180. * pi
    state = array([x, y, theta])
    l = 50.0
    r = 54.32
    control = array([l, r])
    w = 150.0

    # Compute derivative numerically.
    print ("Numeric differentiation dl, dr")
    delta = 1e-7
    control_l = array([l + delta, r])
    control_r = array([l, r + delta])
    dg_dl = (ExtendedKalmanFilter.g(state, control_l, w) -\
             ExtendedKalmanFilter.g(state, control, w)) / delta
    dg_dr = (ExtendedKalmanFilter.g(state, control_r, w) -\
             ExtendedKalmanFilter.g(state, control, w)) / delta
    dg_dcontrol_numeric = column_stack([dg_dl, dg_dr])
    print (dg_dcontrol_numeric)

    # Use the above code to compute the derivative analytically.
    print ("Analytic differentiation dl, dr:")
    dg_dcontrol_analytic = ExtendedKalmanFilter.dg_dcontrol(state, control, w)
    print (dg_dcontrol_analytic)

    # The difference should be close to zero (depending on the setting of
    # delta, above).
    print ("Difference:")
    print (dg_dcontrol_numeric - dg_dcontrol_analytic)
    print ("Seems correct:", allclose(dg_dcontrol_numeric, dg_dcontrol_analytic))
