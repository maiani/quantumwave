#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transmission and reflection coefficients with transfer matrix method.

Authors:
- Andrea Maiani <andrea.maiani@mail.polimi.it>
- Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import quickfunctions as qf


def coefficients_calc(x, p, V):
    Nx = x.size
    Np = p.size

    dx = x[1] - x[0]

    rl = np.zeros((Np), dtype=np.complex128)
    rr = np.zeros((Np), dtype=np.complex128)
    t = np.zeros((Np), dtype=np.complex128)

    TT = np.matrix(np.eye(2), dtype=np.complex128)

    for i in range(0, Np):
        TT = np.eye(2, dtype=np.complex128)
        for j in range(0, Nx):
            lam = np.sqrt(p[i] ** 2 - V[j])
            H = np.zeros((2, 2), dtype=np.complex128)
            H[0, 0] = np.cos(lam * dx)
            H[0, 1] = np.sin(lam * dx) / lam
            H[1, 0] = -np.sin(lam * dx) * lam
            H[1, 1] = np.cos(lam * dx)
            TT = H @ TT

        S = np.matrix(((1, 1), (1j * p[i], -1j * p[i])))
        M = S.I * TT * S

        rl[i] = -M[1, 0] / M[1, 1]
        rr[i] = M[0, 1] / M[1, 1]
        t[i] = 1 / M[1, 1]

    return ((rl, rr, t))


def coeff_plot_large(p, rl, rr, t, L):
    xlim = (0, p.max())
    ylim = (-0.05, 1.05)

    fig = plt.figure(figsize=(14, 12), dpi=300)

    plot1 = fig.add_subplot(221, xlim=xlim, ylim=ylim, title="Reflectance (left incidence)")
    plot1.set_xlabel("Momentum")
    plot1.plot(p, np.abs(rl) ** 2)

    plot2 = fig.add_subplot(222, xlim=xlim, ylim=ylim, title="Reflectance (right incidence)")
    plot2.set_xlabel("Momentum")
    plot2.plot(p, np.abs(rr) ** 2)

    plot3 = fig.add_subplot(223, xlim=xlim, ylim=ylim, title="Transmittance")
    plot3.set_xlabel("Momentum")
    plot3.plot(p, np.abs(t) ** 2)

    plot4 = fig.add_subplot(224, xlim=xlim, title="Phase difference")
    plot4.set_xlabel("Momentum")
    plot4.plot(p, np.unwrap(np.angle(t * np.exp(-2j * p * L))))

    return fig


def coeff_plot_small(p, rl, rr, t, L):
    xlim = (0, p.max())
    ylim = (-0.05, 1.05)

    fs = 20

    fig = plt.figure(figsize=(13, 6), dpi=300)

    plot1 = fig.add_subplot(121, xlim=xlim, ylim=ylim)
    plot1.set_title("Left incidence", fontsize=fs)
    plot1.set_xlabel("Momentum", fontsize=fs)
    plot1.set_ylabel("Reflectance and Transmittance", fontsize=fs)
    rl_line, = plot1.plot([], [], c='b', label=r'$|R^{(l)}|^2$')
    t_line, = plot1.plot([], [], c='r', label=r'$|T|^2$')
    plot1.tick_params(axis='both', labelsize=fs)
    plot1.xaxis.set_ticks([0, 1, 2, 3])
    plot1.yaxis.set_ticks([0, 0.5, 1])

    plot2 = fig.add_subplot(122, xlim=xlim, ylim=ylim)
    plot2.set_title("Right incidence", fontsize=fs)
    plot2.set_xlabel("Momentum", fontsize=fs)
    plot2.set_ylabel("Reflectance", fontsize=fs)
    rr_line, = plot2.plot([], [], c='b', label=r'$|R^{(r)}|^2$')
    plot2.tick_params(axis='both', labelsize=fs)
    plot2.xaxis.set_ticks([0, 1, 2, 3])
    plot2.yaxis.set_ticks([0, 0.5, 1])

    plot1.legend(prop=dict(size=fs))
    plot2.legend(prop=dict(size=fs))
    rl_line.set_data(p, np.abs(rl) ** 2)
    t_line.set_data(p, np.abs(t) ** 2)
    rr_line.set_data(p, np.abs(rr) ** 2)

    return fig


# Some predefined potentials

def reflectionless(l=1):
    global V
    global simname

    # Potential
    V = qf.reflectionless(x, V_0=1, l=l)
    V = np.array(V, dtype=np.complex128)

    simname = "reflectionless-" + str(l)


def square_barrier():
    global V
    global simname

    # Potential
    V = qf.square_barrier(x, 2, 2)
    V = np.array(V, dtype=np.complex128)

    simname = "square_barrier"


def invisible_susy(a=1j, c=1):
    global V
    global simname

    simname = "invisible_susy_a" + str(a) + "c" + str(c)

    # Potential
    V = qf.invisible_susy(x, a=a, c=c)
    V = np.array(V, dtype=np.complex128)


def kk(n, a, c, A):
    global V
    global simname

    simname = "kk_n" + str(n) + "a" + str(a) + "c" + str(c) + "A" + str(A)

    # Potential
    V = qf.kk(x=x, n=n, a=a, A=A, c=c)
    V = np.array(V, dtype=np.complex128)


if __name__ == "__main__":

    sim_type = "normal"

    if (sim_type == "normal"):
        Nx = 2.5E4
        dx = 0.007
        L = dx * Nx / 2
        x = dx * (np.arange(Nx) - 0.5 * Nx)

        dp = 0.006
        Np = 2 ** 9
        p = 0.001 + dp * np.arange(Np)

        #####################
        kk(n=1, a=1j, c=1, A=4)
        #####################

        print("COEFFICIENTS CALCULATOR")
        print("SIMULATION NAME: " + simname)
        print("[*] Running simulation...   ", end='')
        start = time.time()
        rl, rr, t = coefficients_calc(x, p, V)
        end = time.time()
        print("completed in %.2fs" % (end - start))

        print("[*] Building output...   ", end='')
        os.makedirs("./output/" + simname, exist_ok=True)
        start = time.time()
        fig = coeff_plot_large(p, rl, rr, t, L)
        fig.savefig("./output/" + simname + "/" + simname + "-coefficients-large.pdf")
        fig = coeff_plot_small(p, rl, rr, t, L)
        fig.savefig("./output/" + simname + "/" + simname + "-coefficients-small.png")
        end = time.time()
        print("completed in %.2fs" % (end - start))

        print("[*] Execution completed")

    if (sim_type == "tolerance"):
        dp = 0.006
        Np = 2 ** 9
        p = 0.001 + dp * np.arange(Np)

        print("Building tolerance plot...   ")
        start = time.time()

        dx = 0.005

        Nx1 = 4E4
        x1 = dx * (np.arange(Nx1) - 0.5 * Nx1)
        V1 = qf.kk(x1, n=1, a=1j, c=5, A=10)
        rl1, rr1, t1 = coefficients_calc(x1, p, V1)
        print("First done")

        Nx2 = 2E4
        x2 = dx * (np.arange(Nx2) - 0.5 * Nx2)
        V2 = qf.kk(x2, n=1, a=1j, c=5, A=10)
        rl2, rr2, t2 = coefficients_calc(x2, p, V2)
        print("Second done")

        Nx3 = 5E3
        x3 = dx * (np.arange(Nx3) - 0.5 * Nx3)
        V3 = qf.kk(x3, n=1, a=1j, c=5, A=10)
        rl3, rr3, t3 = coefficients_calc(x3, p, V3)
        print("Third done")

        Nx4 = 2E3
        x4 = dx * (np.arange(Nx4) - 0.5 * Nx4)
        V4 = qf.kk(x4, n=1, a=1j, c=5, A=10)
        rl4, rr4, t4 = coefficients_calc(x4, p, V4)
        print("Fourth done")

        Nx5 = 1E3
        x5 = dx * (np.arange(Nx5) - 0.5 * Nx5)
        V5 = qf.kk(x5, n=1, a=1j, c=5, A=10)
        rl5, rr5, t5 = coefficients_calc(x5, p, V5)
        print("Fifth done")

        end = time.time()
        print("Completed in %.2fs" % (end - start))

        xlim = (0, p.max())
        ylim = (-0.05, 1.05)

        fig = plt.figure(figsize=(12, 12), dpi=300)

        plot1 = fig.add_subplot(111, xlim=xlim, ylim=ylim, title="Left side reflectance of $V(x)=10/(5x+i)$")
        plot1.set_xlabel("Momentum", fontsize=14)
        rl_line1, = plot1.plot(p, np.abs(rl1) ** 2, c='b', label="Cutted at $x=\pm100$")
        rl_line2, = plot1.plot(p, np.abs(rl2) ** 2, c='g', label="Cutted at $x=\pm50$")
        rl_line3, = plot1.plot(p, np.abs(rl3) ** 2, c='y', label="Cutted at $x=\pm12.5$")
        rl_line4, = plot1.plot(p, np.abs(rl4) ** 2, c='orange', label="Cutted at $x=\pm5$")
        rl_line5, = plot1.plot(p, np.abs(rl5) ** 2, c='r', label="Cutted at $x=\pm2.5$")
        plot1.tick_params(axis='both', labelsize=14)
        plot1.xaxis.set_ticks([0, 1, 2, 3])
        plot1.yaxis.set_ticks([0, 0.5, 1])
        plot1.legend(prop=dict(size=12))

        fig.savefig("./output/tolerance.png")
