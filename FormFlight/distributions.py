
""" @file distributions.py
"""

import numpy as np
import copy


def circle(dim, radius, nsamples, sample):
    """ Computes the x,y,z position on a circle for a given number of points
    r: radius of circle
    ntargets: total number of points on circle
    target: nth point along the circle
    """

    if dim == 2:
        angle = sample*(2*np.pi)/nsamples
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return x, y
    elif dim == 3:
        angle = sample*(2*np.pi)/nsamples
        x = radius * np.cos(angle)
        y = 0
        z = radius * np.sin(angle)
        return x, y, z


def fibonacci_sphere(r, nsamples, sample):
    """ Computes the x,y,z positions on a sphere for a given number of points
    http://blog.marmakoide.org/?p=1
    r: radius of sphere / scaling factor
    nsamples: total number of points on sphere
    sample: nth point along the sphere
    """

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * sample
    z_i = (1 - 1/nsamples) * (1 - (2*sample)/(nsamples-1))
    radius = np.sqrt(1 - z_i * z_i)

    x = r * radius * np.cos(theta)
    y = r * radius * np.sin(theta)
    z = r * z_i
    return x,y,z

