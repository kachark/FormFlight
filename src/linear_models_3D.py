
""" @file linear_models_3D.py
"""

import numpy as np

# TODO update to match how it's defined in our paper
def quadcopter_3D():
    # linearized quadcopter dynamics
    m = 0.1         #kg
    Ixx = 0.00062   #kg-m^2
    Iyy = 0.00113   #kg-m^2
    Izz = 0.9*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
    # dx = 0.114      #m
    # dy = 0.0825     #m
    g = 9.81  #m/s/s
    # DTR = 1/57.3; RTD = 57.3

    dx = 12

    # State components
    pos_components = np.array([9, 10, 11])
    vel_components = np.array([6, 7, 8])

    # TODO update statespace with attitude and ang vel components

    statespace = {
            'position': pos_components,
            'velocity': vel_components,
            'attitude': None,
            'angular_velocity': None
            }

    du = 4
    A = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, -g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

    B = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0/Ixx, 0.0, 0.0],
                  [0.0, 0.0, 1.0/Iyy, 0.0],
                  [0.0, 0.0, 0.0, 1.0/Izz],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [1.0/m, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])

    C = np.eye(A.shape[0])
    D = 0

    return A, B, C, D, dx, du, statespace

def double_integrator_3D():
    # 3D double integrator
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
    A = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    B = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    C = np.eye(dx)
    D = 0

    return A, B, C, D, dx, du, statespace

