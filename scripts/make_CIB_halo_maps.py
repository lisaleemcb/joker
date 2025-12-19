# from astropy.io import fits
import os

import h5py
import healpy as hp
import numpy as np

# import matplotlib.pyplot as plt
import joker.cosmology as cosmo
import joker.maps as maps

print("hello, world!")

dir_home = "/home/emc-brid"
dir_Websky = f"/data/cluster/emc-brid/Datasets/Websky"
dir_websky = f"{dir_Websky}/websky/v0.0"
dir_rewrite = f"{dir_Websky}/websky_halos_rewrite"
dir_co = f"{dir_Websky}/cib_co_sources"
dir_data = "/data/cluster/emc-brid/Datasets/Websky/maps/correlations_to_halofield"

resolution = 4096

redshifts_bands = [(0.5 * i, 0.5 * i + 0.5) for i in range(10)]
redshifts_bands = redshifts_bands[:-1]
print(f"SAVING ALL FILES TO {dir_data}")

print("=====================================")

fwhm = 5.0  # arcmin
fwhm = np.deg2rad(fwhm / 60)  # to convert arcmin to degrees

coordinates = (20, -30)
width = 10
height = 10

print()
print("Now loading CO halos...")

halo_fn_co = f"{dir_co}/websky_halos-light.hdf5"
with h5py.File(halo_fn_co, "r") as f:
    for key, item in f.items():
        data = item[:]

halos_co = dict(zip(["x", "y", "z", "M200m"], data.T))
chi = np.sqrt(halos_co["x"] ** 2 + halos_co["y"] ** 2 + halos_co["z"] ** 2)  # Mpc
redshift = cosmo.zofchi(chi)

halos_co["redshift"] = redshift

cib_freqs = ["100", "143", "217", "353", "545", "857"]
fluxes_halos = []


def load_fluxes(nu_obs, dir=dir_co):
    print("Now loading cib fluxes...")
    fluxes = []
    for chunk in [1, 2, 3, 4]:
        fn_halos = f"{dir}/sources/cen_chunk{chunk}_flux_{nu}.h5"

        print(f"\t {fn_halos}")
        with h5py.File(fn_halos, "r") as f:
            data = f["flux"][:]  # Load it into a NumPy array
            fluxes.append(data)

    fluxes = np.concatenate(fluxes)

    return fluxes


def make_maps(fluxes, dz, nu_obs):
    redshift_mask = (halos_co["redshift"] < max(dz)) & (halos_co["redshift"] >= min(dz))

    print(
        f"{(np.sum(redshift_mask) / len(halos_co['redshift'])) * 100:.2f}% of halos are within this redshift slice..."
    )
    print()
    print("making CO maps...")

    maps_nu = []
    for j, nu in enumerate(cib_freqs):
        print(f"\t Now on nu={nu}...")
        m = maps.make_sky(
            halos_co,
            weights=fluxes,
            fwhm=fwhm,
            mask=redshift_mask,
            verbose=True,
        )
        p = maps.zoom_in(
            m, coordinates=coordinates, height=height, width=width, verbose=True
        )

        maps_nu.append(p)

    return np.asarray(maps_nu)


for nu in cib_freqs:
    print("========================================================")
    print(f"Now making maps for nu_obs={nu}...")
    print("========================================================")
    fluxes = load_fluxes(nu)
    for i, dz in enumerate(redshifts_bands):
        fn = f"{dir_data}/cib_maps_nu{nu}_z{min(dz)}_to_z{max(dz)}"
        if os.path.exists(f"{fn}.npy"):
            print(f"file {fn} already written...")
            continue
        print("========================================================")
        print(f"Now probing a redshift range of delta_z = {dz}...")
        print("========================================================")
        maps_nu = make_maps(fluxes, dz, nu)

        print(f"saving cib map to {fn}...")
        np.save(fn, maps_nu)

print("finished!")

print("Et voila !")
