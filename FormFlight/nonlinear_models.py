
""" @file nonlinear_models.py
"""

from scipy.linalg import solve_continuous_are as care
import scipy.integrate as scint

import numpy as np
import math

def NonlinearModel2D():

    dx = 3

    # State components
    pos_components = np.array([0, 1])
    att_components = np.array([2])

    statespace = {
            'position': pos_components,
            'velocity': None, # vel_components,
            'attitude': att_components,
            'angular_velocity': None
            }

    du = 1

    def f(t, x, u):
        """
        Point mass equations of turning aircraft in horizontal plane
        states: (x, y, heading)
        control: u = tan(bank angle)
        """
        V = 150 # m/s
        g = 9.81 # m/s^2
        # enter dynamics here
        # print('t', t)
        # print('x', x)
        # print('u', u)
        # print("here!", x.shape)
        out = np.zeros((3))
        out[0] = V * np.cos(x[2])
        out[1] = V * np.sin(x[2])
        out[2] = (g / V) * u[0]
        return out        

    return f, dx, du, statespace


def NonlinearModel3D():

    dx = 6

    # State components
    pos_components = np.array([0, 1, 2])
    vel_components = np.array([3, 4, 5])

    statespace = {
            'position': pos_components,
            'velocity': vel_components,
            'attitude': None,
            'angular_velocity': None
            }

    du = 3

    def f(t, x, u):
        # enter dynamics here
        # print('t', t)
        # print('x', x)
        # print('u', u)

        dummy_result = np.ones(x.size)
        for i in range(x.size):
            dummy_result[i] = math.sin(x[i])

        # print(dummy_result)
        return dummy_result

    return f, dx, du, statespace


def Quadrotor():

    # Differentially flat formulation
    dx = 4

    # State components
    pos_components = np.array([0, 1, 2])
    attitude_components = np.array([3])

    statespace = {
            'position': pos_components,
            'velocity': None,
            'attitude': attitude_components,
            'angular_velocity': None
            }

    du = 4

    def f(t, x, u):

        new_state = np.zeros(3)

        g = -9.81

        # test
        accel_desired = np.array([10, 10, 10])

        x = x[0]
        y = x[1]
        z = x[2]
        psi = x[3]

        T = np.array([accel_desired[0], accel_desired[1], accel_desired[2] + g])

        # define world frame
        xw = np.array([1, 0, 0])
        yw = np.array([0, 1, 0])
        zw = np.array([0, 0, 1])

        # intermediate frame serving to translate between world frame and body frame
        xc = np.array([np.cos(psi), np.sin(psi), 0])

        # define body-fixed frame
        zb = T / np.linalg.norm(T)
        yb = np.cross(zb, xc) / np.linalg.norm(np.cross(zb, xc))
        xb = np.cross(yb, zb)

        return new_state


    return f, dx, du, statespace

class NonlinearController():

    """ Class representing generic nonlinear tracker for tracking non-constant states
    """

    def __init__(self):

        """ Nonelinear constructor

        Input:

        """

    def evaluate(self, time, state1, state2):

        """ Computes control input
        """
        h = 0.5
        Q1 = 1.0
        Q2 = 1.0
        Q3 = 1.0
        g = 9.81
        R = 1000
        omega = 0.15
        V = 150

        dx = state1[0] - state2[0]
        dy = state1[1] - state2[1]
        dpsi = state1[2] - state2[2]

        ddotx = V * np.cos(state1[2]) - V * np.cos(state2[2])
        ddoty = V * np.sin(state1[2]) - V * np.sin(state2[2])

        psi = state1[2]

        P = 0.25 * g * (Q1 * np.sin(psi)**2 + Q2 * np.cos(psi)**2) + \
            g * Q3 / V**2

        term1 = 1/2/h**2 *(-Q1 * (dx + h * ddotx) * np.sin(psi) + \
                           Q2 * (dy + h * ddoty) * np.cos(psi))
        term2 = -0.25 * R * omega**2 * (Q1 * np.sin(psi) * np.cos(omega * time) - \
                                        Q2 * np.cos(psi) * np.sin(omega * time))
        term3 = Q3 / (h*V) * (dpsi - h * omega)

        control = (term1 + term2 + term3) / -P

        # print("control = ", control)
        return np.array([control])


    def sim_combined_dynamics(self, time, state1, state2):

        R = 1000
        g = 9.81
        V = 150

        tan_bank_angle_ref = V**2 / (R*g)


        def dynamics(t, x):

            out = np.zeros((6))
            out[0] = V * np.cos(x[2])
            out[1] = V * np.sin(x[2])
            out[2] = self.evaluate(t, x[:3], x[3:])
            out[3] = V * np.cos(x[5])
            out[4] = V * np.cos(x[5])
            out[5] = (g / V) * tan_bank_angle_ref
            return out
        
        h = 0.5
        final_time = 100
        all_time = np.arange(0, final_time, h)
        integrate_time = all_time[all_time > time-1e-10]
        print("integrate time", integrate_time[:4], state1, state2)
        states = np.concatenate((state1, state2))
        print("states = ", states)
        res = scint.solve_ivp(dynamics, (integrate_time[0], integrate_time[-1]), states,
                              t_eval=integrate_time, method='RK45')
        return res, integrate_time
        
    def cost_to_go(self, time, state1, state2):

        """ Computes cost_to_go for a agent/target to track a single agent/target
        """
        # cost = np.linalg.norm(state1 - state2);
        
        # Brute force cost-to-go computer

        # these should be consistent with the main function, this is a hack for now!

        # print("time = ", time)


        # simulate the system into the future
        res, integrate_time = self.sim_combined_dynamics(time, state1, state2)

        # evaluate the controls at all the times
        controls = np.array([self.evaluate(integrate_time[ii], res.y[:3, ii], res1[3:, ii]) \
                             for ii in range(integrate_time.shape[0])])
        r = 1.0 # this is confusing is it 1 or zero?
        cost_u = 0.5*r*np.cumsum(controls * controls)[:-1]
        
        # evaluate all of the residuals
        Q = 1.0    
        cost = 0.5*np.cumsum([Q*np.linalg.norm(res.y[:3, ii+1] - res.y[3:, ii+1])**2
                              for ii in range(integrate_time.shape[0]-1)])
        
        # sum up teh costs
        total_cost = cost + cost_u
        # return time[:-1], total_cost
        return total_cost[-1]

        # return cost

class NonlinearControllerTarget():

    """ Class representing generic nonlinear tracker for tracking non-constant states
    """

    def __init__(self):

        """ Nonelinear constructor

        Input:

        """

    def evaluate(self, time, state1):

        """ Computes control input
        """
        V = 150
        R = 1000
        g = 9.81
        tan_bank_angle_ref = V**2 / (R*g)
        # print("control = ", control)
        return np.array([tan_bank_angle_ref])

    def cost_to_go(self, time, state1, state2):
        assert 1 == 0
        """ Computes cost_to_go for a agent/target to track a single agent/target
        """


