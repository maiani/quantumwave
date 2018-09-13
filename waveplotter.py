#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:51:21 2017

Wavefunction Plotting and Animation

Authors:
    - Andrea Maiani <andrea.maiani@mail.polimi.it>
    - Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class WavePlotter:
    """
    This class provide a set of function in order to plot some features of a 
    solution of the Schrodinger equation
    """

    def __init__(self, SS, x0, p0):
        self.SS = SS
        self.x0 = x0
        self.p0 = p0

    def plot_x(self, t):
        """
        Plot the wavefunction module at time t
        """
        i = self.SS.t2i(t)
        fig = plt.figure()
        xlim = (-100, 100)
        fig1 = fig.add_subplot(111, xlim=xlim, ylim=(0, 1))
        psi_x_line, = fig1.plot([], [], c='r', label=r'$|\psi(x)|$')
        psi_x_line.set_data(self.SS.x, np.abs(self.SS.psi_x[i]))
        return fig

    def plot_xk(self, t, ymin=0, ymax=2, center_line=True, max_line=True):
        """
        Plot the wavefunction module squared and power spectrum  at time t
        """
        # xc=1/np.sqrt(2)  #Schrodinger
        xc = 1  # Helmholtz
        i = self.SS.t2i(t)
        x = self.SS.x * xc

        fig = plt.figure(dpi=300)

        # plotting limits
        xlim = (-100, 100)
        klim = (-4, 4)

        ax1 = fig.add_subplot(211, xlim=xlim,
                              ylim=(ymin - 0.2 * (ymax - ymin),
                                    ymax + 0.2 * (ymax - ymin)))

        psi_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
        V_x_line_re, = ax1.plot([], [], c='b', label=r'$\mathrm{Re}\{V(x)\}$')
        V_x_line_im, = ax1.plot([], [], c='g', ls=':', label=r'$\mathrm{Im}\{V(x)\}$')

        title = ax1.set_title("")
        ax1.legend(prop=dict(size=12), loc='center left', bbox_to_anchor=(1, 0.5))

        ax1.set_xlabel('$x$')

        ymin = np.abs(self.SS.psi_k[0]).min()
        ymax = np.abs(self.SS.psi_k[0]).max()

        ax2 = fig.add_subplot(212, xlim=klim,
                              ylim=(ymin - 0.2 * (ymax - ymin),
                                    ymax + 0.2 * (ymax - ymin)))
        psi_k_line, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')

        ax2.axvline(-self.p0, c='k', ls=':', label=r'$\pm p_0$')
        ax2.axvline(self.p0, c='k', ls=':')

        ax2.legend(prop=dict(size=12), loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel('$k$')
        title.set_text("t = %.2f , Norm = %.3f" % (self.SS.t[i], self.SS.psi_norm[i]))

        V_x_line_re.set_data(x, np.real(self.SS.V_x[i]))
        V_x_line_im.set_data(x, np.imag(self.SS.V_x[i]))
        psi_x_line.set_data(x, 5 * np.abs(self.SS.psi_x[i]))
        psi_k_line.set_data(self.SS.k, np.abs(self.SS.psi_k[i]))

        if center_line:
            center_line = ax1.axvline(0, c='k', ls=':', label=r"$x_0 + v_0t$")
            center_line.set_data([self.x0 * xc + self.SS.t[i] * self.p0 * xc], [0, 1])

        if max_line:
            max_line = ax1.axvline(0, c='r', ls=':', label=r"max")
            max_line.set_data(x[np.argmax(np.abs(self.SS.psi_x[i]))], [0, 1])

        return fig

    def plot_Vx(self, t, ymin=-1.5, ymax=1.5):
        """
        Plot the wavefunction module squared and power spectrum  at time t
        """
        # xc=1/np.sqrt(2)  #Schrodinger
        xc = 1  # Helmholtz
        i = self.SS.t2i(t)
        x = self.SS.x * xc

        fig = plt.figure(dpi=300)

        # plotting limits
        xlim = (-40, 40)

        ax1 = fig.add_subplot(211, xlim=xlim,
                              ylim=(ymin - 0.2 * (ymax - ymin),
                                    ymax + 0.2 * (ymax - ymin)))

        V_x_line_re, = ax1.plot([], [], c='b', label=r'$\mathrm{Re}\{V(x)\}$')
        V_x_line_im, = ax1.plot([], [], c='g', ls=':', label=r'$\mathrm{Im}\{V(x)\}$')

        ax1.legend(prop=dict(size=12))

        ax1.set_xlabel('$x$')

        ymin = np.abs(self.SS.psi_k[0]).min()
        ymax = np.abs(self.SS.psi_k[0]).max()

        V_x_line_re.set_data(x, np.real(self.SS.V_x[i]))
        V_x_line_im.set_data(x, np.imag(self.SS.V_x[i]))

        return fig

    def createmp4(self, filename, ymin=-1, ymax=3, center_line=True, max_line=True):
        """
        Create the mp4 simulation
        """

        # Functions to animate the plot
        def init():
            psi_x_line.set_data([], [])
            V_x_line_re.set_data([], [])
            V_x_line_im.set_data([], [])
            psi_k_line.set_data([], [])
            center_line.set_data([], [])
            max_line.set_data([], [])

            title.set_text("")
            return (psi_x_line, V_x_line_re, V_x_line_im, center_line, psi_k_line, title)

        def animate(i):
            i *= multiplier

            psi_x_line.set_data(self.SS.x, 5 * np.abs(self.SS.psi_x[i]))
            V_x_line_re.set_data(self.SS.x, np.real(self.SS.V_x[i]))
            V_x_line_im.set_data(self.SS.x, np.imag(self.SS.V_x[i]))
            psi_k_line.set_data(self.SS.k, np.abs(self.SS.psi_k[i]))

            center_line.set_data(self.x0 + self.SS.t[i] * self.p0, [0, 1])
            max_line.set_data(self.SS.x[np.argmax(np.abs(self.SS.psi_x[i]))], [0, 1])

            title.set_text("t = %.2f , Norm = %.3f" % (self.SS.t[i], self.SS.psi_norm[i]))
            return (psi_x_line, V_x_line_re, V_x_line_im, center_line, psi_k_line, title)

        fig = plt.figure(figsize=(15, 7))

        xlim = (-100, 100)
        klim = (-4, 4)

        ylim = (ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))

        ax1 = fig.add_subplot(211, xlim=xlim, ylim=ylim)

        psi_x_line, = ax1.plot([], [], c='r', label=r'$|\psi(x)|$')
        V_x_line_re, = ax1.plot([], [], c='b', label=r'$\mathrm{Re}\{V(x)\}$')
        V_x_line_im, = ax1.plot([], [], c='g', ls=':', label=r'$\mathrm{Im}\{V(x)\}$')
        center_line = ax1.axvline(0, c='k', ls=':', label=r"$x_0 + v_0t$")
        max_line = ax1.axvline(0, c='r', ls=':', label=r"max")

        title = ax1.set_title("")
        ax1.legend(prop=dict(size=12), loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('$x$')

        ymin = np.abs(self.SS.psi_k[0]).min()
        ymax = np.abs(self.SS.psi_k[0]).max()
        ylim = (ymin - 0.2 * (ymax - ymin), ymax + 0.2 * (ymax - ymin))

        ax2 = fig.add_subplot(212, xlim=klim, ylim=ylim)

        psi_k_line, = ax2.plot([], [], c='r', label=r'$|\psi(k)|$')

        ax2.axvline(-self.p0, c='k', ls=':', label=r'$\pm p_0$')
        ax2.axvline(self.p0, c='k', ls=':')

        ax2.legend(prop=dict(size=12), loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel('$k$')

        fps = 15
        interval = 10
        frames = fps * interval

        multiplier = max(1, int(self.SS.Nt / frames))

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frames, interval=interval, blit=True)
        anim.save(filename, fps=fps,
                  extra_args=['-vcodec', 'libx264'], dpi=200)

    def plot_optics(self, t0=0, levels=64):
        """
        Contour plot of a solution of the Schrodinger equation interpreted as a
        solution of a 2D-Helmoltz equation
        """

        fs = 20

        Delta = 10000
        t0 = self.SS.t2i(t0)
        xmin = int((-50.0 - self.SS.x[0]) / self.SS.dx)
        xmax = int((50.0 - self.SS.x[0]) / self.SS.dx)
        x = self.SS.x[xmin:xmax]
        y = self.SS.t[t0:(t0 + Delta)]
        X, Y = np.meshgrid(x, y)

        f = self.SS.psi_x[t0:(t0 + Delta), xmin:xmax]

        f = np.abs(f)

        fig = plt.figure(figsize=(8, 6), dpi=1000)

        CS = plt.contourf(Y, X, f, levels, cmap="magma")
        for c in CS.collections:
            c.set_edgecolor("face")

        plt.colorbar(CS)
        plt.title('Wavepacket propagation', fontsize=fs)
        plt.xlabel('Time t', fontsize=fs)
        plt.ylabel('Space x', fontsize=fs)
        return fig

    def plot_norm(self):
        """
        Plot of the norm of the solution as a function of time
        """
        xlim = (0, 90)
        fig = plt.figure(figsize=(8, 8), dpi=180)
        plt.plot(self.SS.t, self.SS.psi_norm)
        plt.xlim(xlim)
        plt.xlabel("Time")
        plt.ylabel("Norm")
        return fig
