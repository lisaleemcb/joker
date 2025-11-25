"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = joker.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import os
import sys

# from astropy.io import fits
import h5py
import healpy as hp
import matplotlib.pyplot as plt

from joker import __version__
from joker.cosmology import *

__author__ = "Lisa McBride"
__copyright__ = "Lisa McBride"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from joker.skeleton import fib`,
# when using this Python module as a library.

resolution = 4096


def make_halo_catalogue(filename, verbose=False):
    if verbose:
        print(f"Loading halo catalogue from {filename}")
    halos = {}

    basename, ext = os.path.splitext(filename)
    if verbose:
        print(f"Parsing file type {ext}...")

    allowed_exts = [".h5", ".npz"]
    if ext not in allowed_exts:
        raise ValueError(
            f"File extension '{ext}' not allowed. Allowed extensions are: {allowed_exts}"
        )

    if ext == ".h5":
        with h5py.File(filename, "r") as f:
            # List all groups (like folders in the file)
            for key, item in f.items():
                print(f"adding {key}...")
                halos[key] = item[()]

    elif ext == ".npz":
        f = np.load(filename, allow_pickle=True)
        for key, item in f.items():
            print(f"adding {key}...")
            halos[key] = item[()]

    chi = np.sqrt(halos["x"] ** 2 + halos["y"] ** 2 + halos["z"] ** 2)  # Mpc
    redshift = zofchi(chi)

    halos["redshift"] = redshift

    return halos


def make_sky(
    coordinates, nside=resolution, mask=None, weights=None, fwhm=None, verbose=False
):
    """Make sky from data

    Args:
      n (int): integer
      mask (arr): mask in order to pick out certain redshifts

    Returns:
      int: healpix map
    """

    sky = np.zeros((hp.nside2npix(nside)))

    if mask is None:
        mask = np.ones_like(coordinates["x"], dtype=bool)

    pix = hp.vec2pix(
        nside, coordinates["x"][mask], coordinates["y"][mask], coordinates["z"][mask]
    )
    # pix = hp.ang2pix(nside, coordinates["theta"], coordinates["phi"])  # does the same

    if weights is None:
        weights = np.ones_like(coordinates["x"])

    np.add.at(sky, pix, weights[mask])

    if fwhm is not None:
        if verbose:
            print(f"Smoothing map assuming fwhm={fwhm}...")

        sky = hp.sphtfunc.smoothing(sky, fwhm=fwhm)

    return sky


def zoom_in(sky, nside=resolution, coordinates=(0.0, -30.7), width=20, height=10):
    ra, dec = coordinates

    # Convert to radians
    theta_center = np.radians(90 - dec)
    phi_center = np.radians(ra)

    # Create 2D grid of theta/phi
    npix_x = 400
    npix_y = 500
    theta_offsets = np.radians(np.linspace(-height / 2, height / 2, npix_y))
    phi_offsets = np.radians(np.linspace(-width / 2, width / 2, npix_x))
    theta_grid, phi_grid = np.meshgrid(theta_offsets, phi_offsets, indexing="ij")
    theta = theta_center + theta_grid
    phi = phi_center + phi_grid

    # Wrap phi
    phi = np.mod(phi, 2 * np.pi)

    # Use full-resolution map and interpolate values
    patch = hp.get_interp_val(sky, theta.flatten(), phi.flatten()).reshape(
        npix_y, npix_x
    )

    return patch


# # Plot
# plt.imshow(patch, origin='lower',
#            extent=[-width_deg/2, width_deg/2, -height_deg/2, height_deg/2],
#            aspect='auto', cmap='viridis')
# plt.axis('off')
# plt.show()


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"joker {__version__}",
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print(f"The {args.n}-th Fibonacci number is {fib(args.n)}")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m joker.skeleton 42
    #
    run()
