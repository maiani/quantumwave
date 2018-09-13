#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collection of predefined simulations.

Authors:
   - Andrea Maiani <andrea.maiani@mail.polimi.it>
   - Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>"""

import numpy as np

import quickfunctions as qf
from schrodingersim import SchrodingerSimulation


def square_barrier():
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "squar_barrier"
    p0 = 1
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = qf.square_barrier(x, 2, 2)
    ss.set_static_potential(V_x)

    return ss, x0, p0


def invisible_susy(a=1j, c=1):
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "invisible_susy_a" + str(a) + "c" + str(c)
    p0 = 1
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = qf.invisible_susy(x, a=a, c=c)
    ss.set_static_potential(V_x)

    return ss, x0, p0


def reflectionless(l=1):
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "reflectionless-" + str(l)
    p0 = 0.8
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = qf.reflectionless(x, l=l)
    ss.set_static_potential(V_x)

    return ss, x0, p0


def kk(n, a, c, A):
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "kk_n" + str(n) + "a" + str(a) + "c" + str(c) + "A" + str(A)
    p0 = 1
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = qf.kk(x=x, n=n, a=a, A=A, c=c)
    ss.set_static_potential(V_x)

    return ss, x0, p0


def kk_wrong(n, a, c, A):
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "kk-wrong_n" + str(n) + "a" + str(a) + "c" + str(c) + "A" + str(A)
    p0 = -1
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = 50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = qf.kk(x=x, n=n, a=a, A=A, c=c)
    ss.set_static_potential(V_x)

    return ss, x0, p0


def tentativo():
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "tentativo"
    p0 = 0.8
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -40
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = 1j * np.imag(qf.kk(x, n=1, a=1j, c=3, A=20))
    ss.set_static_potential(V_x)

    return ss, x0, p0


def dancing():
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "dancing"
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    p0 = 0
    d = 3
    x0 = 0
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    V_x = qf.theta(x + 1) - qf.theta(x - 1)
    V_x[x < -20] = 1E6
    V_x[x > 20] = 1E6
    V_x = np.array([V_x])
    ss.set_static_potential(V_x)

    return ss, x0, p0


# Time dependent potentials

def osc_barrier():
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "oscillating_barrier"
    p0 = 1
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    freq = 0.1
    amp = 10
    V_x = np.zeros((Nt, Nx))
    for i in range(Nt - 1):
        V = np.zeros((Nx))
        V[x < -99] = 1E6
        V[x > 99] = 1E6
        V += qf.square_barrier(x + amp * np.cos(2 * np.pi * freq * t[i]), 4, 2)
        V_x[i] = V

    ss.set_potential(V_x)

    return ss, x0, p0


def osc_invisible():
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "oscillating_invisible"
    p0 = 0.75
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    freq = 0.3
    amp = 10
    V_x = np.zeros((Nt, Nx), dtype=np.complex128)
    for i in range(Nt - 1):
        V = np.zeros((Nx), dtype=np.complex128)
        V[x < -99] = 1E6
        V[x > 99] = 1E6
        V += qf.invisible_susy(x + amp * np.cos(2 * np.pi * freq * t[i]))
        V_x[i] = V

    ss.set_potential(V_x)

    return ss, x0, p0


def osc_reflectionless(l=1):
    Nx = 2 ** 13
    dx = 0.03
    t0 = 0.0
    dt = 0.02
    Nt = int(105 / dt)
    x = dx * (np.arange(Nx) - 0.5 * Nx)
    t = t0 + dt * np.arange(Nt)

    simname = "oscillating_reflectionless"
    p0 = 0.75
    dp2 = p0 * p0 * 1. / 80
    d = 1 / np.sqrt(2 * dp2)
    x0 = -50
    psi_x0 = qf.gauss_x(x, d, x0, p0)
    norm = np.sqrt((np.abs(psi_x0) ** 2).sum() * 2 * np.pi / dx)
    psi_x0 = psi_x0 / norm

    ss = SchrodingerSimulation(Nx=Nx, dx=dx, t0=t0, Nt=Nt, dt=dt,
                               psi_x0=psi_x0, normalize=False,
                               simname=simname)

    freq = 0.05
    amp = 10
    V_x = np.zeros((Nt, Nx), dtype=np.complex128)
    for i in range(Nt - 1):
        V = np.zeros((Nx), dtype=np.complex128)
        V += qf.reflectionless(x + amp * np.cos(2 * np.pi * freq * t[i]), l)
        V_x[i] = V

    ss.set_potential(V_x)

    return ss, x0, p0
