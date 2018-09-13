#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a Schrodinger simulation and collects the results

Authors:
    - Andrea Maiani <andrea.maiani@mail.polimi.it>
    - Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>
"""

import numpy as np

from schrodinger import Schrodinger


class SchrodingerSimulation:
    """
    Class which run the spit-step engine and collects the results
    """

    def __init__(self, Nx, dx, t0, Nt, dt, psi_x0, normalize=False, simname=""):
        """
        Parameters
        ----------
        Nx : integer
            Number of cells of the spatial lattice
        dx : float
            Length of the spatial cell
        t0 : float
            Initial time
        Nt : float
            Number of time steps
        dt : float
            Width of time step
        psi_x0: numpy array
            Initial wavefuncion
        normalize: bool
            Normalization at each step
        simname: string
            Simulation name (for saving results and plotting)
       """

        self.Nx = Nx
        self.dx = dx
        self.x = self.dx * (np.arange(self.Nx) - 0.5 * self.Nx)

        self.Nt = Nt
        self.dt = dt
        self.t = t0 + self.dt * np.arange(self.Nt)

        self.k = None

        self.psi_x = np.zeros((Nt, Nx), dtype=np.complex128)
        self.psi_k = np.zeros((Nt, Nx), dtype=np.complex128)
        self.V_x = np.zeros((Nt, Nx), dtype=np.complex128)
        self.psi_norm = np.zeros(Nt)

        psi_x0 = np.asarray(psi_x0)
        assert psi_x0.shape == (self.Nx,)
        self.psi_x[0] = psi_x0

        self.wf = None

        self.normalize = normalize
        self.simname = simname

    def t2i(self, t):
        """
        Converts the time t into an index i for accessing the arrays
        """
        return np.int((t - self.t[0]) / self.dt)

    def set_potential(self, V_x):
        """
        Set the potential 
        """
        V_x = np.asarray(V_x)
        assert V_x.shape == (self.Nt, self.Nx)
        self.V_x = V_x

    def set_static_potential(self, V_x):
        """
        Set a static potential 
        """
        assert V_x.shape == (1, self.Nx)
        self.V_x = np.repeat(V_x, self.Nt, axis=0)

    def run(self):
        """
        Run the simulation
        """
        self.wf = Schrodinger(self.x, self.psi_x[0], self.V_x[0], k0=-25, t0=self.t[0])

        self.k = self.wf.k
        for i in range(int(self.Nt)):
            self.wf.V_x = self.V_x[i]
            self.wf.time_step(self.dt, Nsteps=1, normalize=self.normalize)
            self.psi_x[i] = self.wf.psi_x
            self.psi_k[i] = self.wf.psi_k
            self.psi_norm[i] = self.wf.norm
