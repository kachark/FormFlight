
""" @file linear_models_2D.py
"""

import numpy as np

def double_integrator_2D():

    dx = 4

    # State components
    pos_components = np.array([0, 1])
    vel_components = np.array([2, 3])

    statespace = {
            'position': pos_components,
            'velocity': vel_components,
            'attitude': None,
            'angular_velocity': None
            }

    du = 2
    A = np.array([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]], dtype=np.float_)

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float_)

    C = np.eye(dx)

    D = 0

    return A, B, C, D, dx, du, statespace

def quadcopter_2D():
    # linearized quadcopter dynamics
    m = 0.1         #kg
    Ixx = 0.00062   #kg-m^2
    Iyy = 0.00113   #kg-m^2
    Izz = 0.9*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
    # dx = 0.114      #m
    # dy = 0.0825     #m
    g = 9.81  #m/s/s
    # DTR = 1/57.3; RTD = 57.3

    dx = 8

    # State components
    pos_components = np.array([0, 1])
    vel_components = np.array([4, 5])

    # TODO update statespace with attitude and ang vel components

    statespace = {
            'position': pos_components,
            'velocity': vel_components,
            'attitude': None,
            'angular_velocity': None
            }

    du = 3
    A = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, -g, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, g, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    B = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 1.0/Ixx, 0.0],
                  [0.0, 0.0, 1.0/Iyy]])

    C = np.eye(A.shape[0])
    D = 0

    return A, B, C, D, dx, du, statespace

