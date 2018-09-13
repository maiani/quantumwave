#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions

Authors:
- Andrea Maiani <andrea.maiani@mail.polimi.it>
- Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>
"""

import numpy as np


# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, k0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))


def gauss_k(k, a, x0, k0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi)) ** 0.5
            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))


# Utility functions for building potentials
def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y


def square_barrier(x, width, height):
    width = width / 2
    return height * (theta(x + width) - theta(x - width))


def double_square_barrier(x, width, height, space):
    width = width / 2
    space = space / 2
    return height * (theta(x + space + width)
                     - theta(x + space - width)
                     + theta(x - space + width)
                     - theta(x - space - width))


def well(x, width, height):
    width = width / 2
    return height * (-theta(x + width) + theta(x - width))


def reflectionless(x, l=1, V_0=1):
    """
    Poschl-Teller potential
    """
    x = np.asarray(x)
    return -V_0 * l * (l + 1) / np.cosh(np.sqrt(V_0) * x) ** 2


def invisible_susy(x, a=1j, c=1):
    x = np.asarray(x)
    return 2 * c ** 2 / (c * x + a) ** 2


def invisible_kk(x, n=5, a=-3j):
    x = np.asarray(x)
    return np.exp(0.5j * np.pi * (n + 1)) / (x + a) ** (n + 1)


def kk(x, n, c, A, a):
    x = np.asarray(x)
    return A / (c * x + a) ** n
