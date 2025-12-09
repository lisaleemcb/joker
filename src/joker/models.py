import os

import h5py
import healpy as hp
import matplotlib.pyplot as plt

# from astropy.io import fits
import numpy as np

from joker import __version__
from joker.cosmology import *

# Physical constants
c = 2.99792458e10  # Speed of light, cm/s
h = 6.62606957e-27  # Planck constant, erg s
k = 1.3806488e-16  # Boltzmann constant, erg/K
Tcmb = 2.7255  # CMB temperature, K


def LM_powerlaw(masses, L0, alpha, scatter_fraction=0.01):
    M0 = np.mean(masses)
    luminosities = L0 * (masses / M0) ** alpha
    luminosities += luminosities * np.random.normal(
        scale=scatter_fraction, size=luminosities.shape
    )

    return luminosities


def tSZ_per_nu(nu):
    x = (h * nu) / (k * Tcmb)

    scaling = x * (np.exp(x) + 1) / np.expm1(x) - 4

    return scaling


def synchrotron_per_nu(nu):
    # Reference frequency
    nu_ref = 30.0 * 1.0e9  # Hz
    sync_beta = -1.2

    scaling = (nu / nu_ref) ** sync_beta  # * G_nu(nu_ref, Tcmb) / G_nu(nu, Tcmb)

    return scaling


def B_nu(nu, T):
    """
    Planck blackbody function.
    """
    return 2.0 * h * nu**3.0 / (c**2.0 * np.expm1(h * nu / (k * T)))


def G_nu(nu, T=Tcmb):
    """
    Conversion factor from intensity to T_CMB, i.e. I_nu = G_nu * deltaT_CMB.
    """
    x = h * nu / (k * T)
    return B_nu(nu, T) * x * np.exp(x) / (np.expm1(x) * T)


def rj2cmb(nu, T):
    """
    Convert a Rayleigh-Jeans temperature to a CMB temperature.
    """
    return 2.0 * k * (nu / c) ** 2.0 * T / G_nu(nu, Tcmb)


def cmb2rj(nu, dT):
    """
    Convert a CMB temperature to a Rayleigh-Jeans temperature.
    """
    return dT * G_nu(nu, Tcmb) / (2.0 * k * (nu / c) ** 2.0)
