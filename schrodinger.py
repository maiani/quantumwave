#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical Solver for the 1D Time-Dependent Schrodinger Equation.

Authors:
   - Andrea Maiani <andrea.maiani@mail.polimi.it>
   - Ciro Pentangelo <ciro.pentangelo@mail.polimi.it>
"""

import numpy as np
from scipy import fftpack


class Schrodinger():
    """
    Class which implements a numerical solution of the time-dependent
    Schrodinger equation for an arbitrary potential
    """

    def __init__(self, x, psi_x, V_x, k0=None, t0=0.0):
        """
        Parameters
        ----------
        x : array_like, float
            Length-N array of evenly spaced spatial coordinates
        psi_xi : array_like, complex
            Length-N array of the initial wave function at time t0
        V_x : array_like, float
            Length-N array giving the potential at each x
        k0 : float
            The minimum value of k.  Note that, because of the workings of the
            Fast Fourier Transform, the momentum wave-number will be defined
            in the range
              k0 < k < 2*pi / dx ,
            where dx = x[1]-x[0].  If you expect nonzero momentum outside this
            range, you must modify the inputs accordingly.  If not specified,
            k0 will be calculated such that the range is [-k0,k0]
        t0 : float
            Initial time (default = 0)
        """

        # Validation of inputs
        self.x = np.asarray(x)
        psi_x = np.asarray(psi_x)
        self._V_x = np.asarray(V_x)

        self.Nx = self.x.size
        assert self.x.shape == (self.Nx,)
        assert psi_x.shape == (self.Nx,)
        assert self._V_x.shape == (self.Nx,)

        # Set internal parameters
        self.t = t0
        self._dt = 0
        self.dx = self.x[1] - self.x[0]
        self.dk = 2 * np.pi / (self.Nx * self.dx)

        # Set momentum scale
        if k0 == None:
            self.k0 = -0.5 * self.Nx * self.dk
        else:
            assert k0 < 0
            self.k0 = k0

        self.k = self.k0 + self.dk * np.arange(self.Nx)

        self.psi_x = psi_x
        self.compute_k_from_x()

        # Variables which hold steps in evolution
        self.x_evolve_half = None
        self.x_evolve = None
        self.k_evolve = None

    # psi_x getter and setter
    def _set_psi_x(self, psi_x):
        assert psi_x.shape == self.x.shape
        self.psi_mod_x = (psi_x * np.exp(-1j * self.k[0] * self.x)
                          * self.dx / np.sqrt(2 * np.pi))
        self.psi_mod_x /= self.norm
        self.compute_k_from_x()

    def _get_psi_x(self):
        return (self.psi_mod_x * np.exp(1j * self.k[0] * self.x)
                * np.sqrt(2 * np.pi) / self.dx)

    # psi_k getter and setter
    def _set_psi_k(self, psi_k):
        assert psi_k.shape == self.x.shape
        self.psi_mod_k = psi_k * np.exp(1j * self.x[0] * self.dk
                                        * np.arange(self.Nx))
        self.compute_x_from_k()
        self.compute_k_from_x()

    def _get_psi_k(self):
        return self.psi_mod_k * np.exp(-1j * self.x[0] * self.dk
                                       * np.arange(self.Nx))

    # V_x getter and setter
    def _get_V_x(self):
        return self.V_x

    def _set_V_x(self, V_x):
        assert V_x.shape == (self.Nx,)
        self._V_x = V_x
        self.x_evolve_half = np.exp(-0.5 * 0.5 * 1j * self._V_x * self.dt)
        self.x_evolve = self.x_evolve_half * self.x_evolve_half

    # dt getter and setter
    def _get_dt(self):
        return self._dt

    def _set_dt(self, dt):
        assert dt != 0
        if dt != self._dt:
            self._dt = dt
            self.x_evolve_half = np.exp(- 0.5 * 0.5 * 1j * self._V_x * self.dt)
            self.x_evolve = self.x_evolve_half * self.x_evolve_half
            self.k_evolve = np.exp(- 0.5 * 1j * (self.k * self.k) * self.dt)

    # Norm getter
    def _get_norm(self):
        return self.wf_norm(self.psi_mod_x)

    # Properties definitions
    psi_x = property(_get_psi_x, _set_psi_x)
    psi_k = property(_get_psi_k, _set_psi_k)
    norm = property(_get_norm)
    dt = property(_get_dt, _set_dt)
    V_x = property(_get_V_x, _set_V_x)

    def compute_k_from_x(self):
        self.psi_mod_k = fftpack.fft(self.psi_mod_x)

    def compute_x_from_k(self):
        self.psi_mod_x = fftpack.ifft(self.psi_mod_k)

    def wf_norm(self, wave_fn):
        """
        Returns the norm of a wave function.

        Parameters
        ----------
        wave_fn : array
            Length-N array of the wavefunction in the position representation
        """
        assert wave_fn.shape == self.x.shape

        return np.sqrt((np.abs(wave_fn) ** 2).sum() * 2 * np.pi / self.dx)

    def solve(self, dt, Nsteps=1, eps=1e-3, max_iter=1000):
        """
        Propagate the Schrodinger equation forward in imaginary
        time to find the ground state.

        Parameters
        ----------
        dt : float
            The small time interval over which to integrate
        Nsteps : float, optional
            The number of intervals to compute (default = 1)
        eps : float
            The criterion for convergence applied to the norm (default = 1e-3)
        max_iter : float
            Maximum number of iterations (default = 1000)
        """
        eps = np.abs(eps)
        assert eps > 0
        t0 = self.t
        old_psi = self.psi_x
        d_psi = 2 * eps
        num_iter = 0
        while (d_psi > eps) and (num_iter <= max_iter):
            num_iter += 1
            self.time_step(-1j * dt, Nsteps)
            d_psi = self.wf_norm(self.psi_x - old_psi)
            old_psi = 1. * self.psi_x
        self.t = t0

    def time_step(self, dt, Nsteps=1, normalize=False):
        """
        Perform a series of time-steps via the time-dependent Schrodinger
        Equation.

        Parameters
        ----------
        dt : complex
            The small time interval over which to integrate
        Nsteps : float, optional
            The number of intervals to compute.  The total change in time at
            the end of this method will be dt * Nsteps (default = 1)
        normalize: boolean, optional
            Option to normalize the wave function 
        """
        assert Nsteps >= 0
        self.dt = dt
        if Nsteps > 0:
            self.psi_mod_x *= self.x_evolve_half

            for num_iter in range(Nsteps - 1):
                self.compute_k_from_x()
                self.psi_mod_k *= self.k_evolve
                self.compute_x_from_k()
                self.psi_mod_x *= self.x_evolve

            self.compute_k_from_x()
            self.psi_mod_k *= self.k_evolve
            self.compute_x_from_k()
            self.psi_mod_x *= self.x_evolve_half

            if (normalize):
                self.psi_mod_x /= self.norm

            self.compute_k_from_x()
            self.t += dt * Nsteps
