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

CO_obs_freqs = ["090", "150", "220"]
CO_rest_freqs = [115.271, 230.538, 345.796, 461.041, 576.268, 691.473, 806.652]
CO_fluxes_halos = []

print("Now loading CO fluxes...")

for freq in CO_obs_freqs:
    CO_halos = []
    for chunk in [1, 2, 3, 4]:
        fn_halos = f"{dir_co}/sources/cen_chunk{chunk}_fluxCO_{freq}.h5"

        print(fn_halos)
        with h5py.File(fn_halos, "r") as f:
            data = f["flux"][:]  # Load it into a NumPy array
            CO_halos.append(data)

    CO_fluxes_halos.append(np.concatenate(CO_halos, axis=1))

for i, dz in enumerate(redshifts_bands):
    print("========================================================")
    print(f"Now probing a redshift range of delta_z = {dz}...")
    print("========================================================")
    redshift_mask = (halos_co["redshift"] < max(dz)) & (halos_co["redshift"] >= min(dz))

    print(
        f"{(np.sum(redshift_mask) / len(halos_co['redshift'])) * 100:.2f}% of halos are within this redshift slice..."
    )
    print()
    print("making CO maps...")

    fn = f"{dir_data}/correlations_to_halofield/CO_maps_z{min(dz)}_to_z{max(dz)}"
    if os.path.exists(f"{fn}.npy"):
        print(f"file {fn} already written...")
        continue

    maps_CO_090_halos = []
    maps_CO_150_halos = []
    maps_CO_220_halos = []

    for j in range(len(CO_rest_freqs)):
        print(f"\t Now on nu_rest={CO_rest_freqs[i]}...")
        m = maps.make_sky(
            halos_co,
            weights=CO_fluxes_halos[0][j],
            fwhm=fwhm,
            mask=redshift_mask,
            verbose=True,
        )
        p = maps.zoom_in(m, coordinates=coordinates, height=height, width=width)

        maps_CO_090_halos.append(p)

        m = maps.make_sky(
            halos_co,
            weights=CO_fluxes_halos[1][j],
            fwhm=fwhm,
            mask=redshift_mask,
            verbose=True,
        )
        p = maps.zoom_in(m, coordinates=coordinates, height=height, width=width)

        maps_CO_150_halos.append(p)

        m = maps.make_sky(
            halos_co,
            weights=CO_fluxes_halos[2][j],
            fwhm=fwhm,
            mask=redshift_mask,
            verbose=True,
        )
        p = maps.zoom_in(m, coordinates=coordinates, height=height, width=width)

        maps_CO_220_halos.append(p)

    print("saving CO halo maps...")
    np.save(f"{dir_data}/maps_CO_090_z{min(dz)}_to_z{max(dz)}", maps_CO_090_halos)
    np.save(f"{dir_data}/maps_CO_150_z{min(dz)}_to_z{max(dz)}", maps_CO_150_halos)
    np.save(f"{dir_data}/maps_CO_220_z{min(dz)}_to_z{max(dz)}", maps_CO_220_halos)

    LIM_maps = [maps_CO_090_halos, maps_CO_150_halos, maps_CO_220_halos]
    LIM_maps = np.asarray(LIM_maps)

    print(f"saving CO maps to {fn}...")
    np.save(fn, LIM_maps)
    print()


print("finished!")

print("Et voila !")
