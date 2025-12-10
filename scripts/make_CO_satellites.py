# from astropy.io import fits
import h5py
import healpy as hp
import numpy as np

# import matplotlib.pyplot as plt
import joker.cosmology as cosmo
import joker.maps as maps

print("hello, world!")

dir_data = "/data/cluster/emc-brid/Datasets/Websky"
dir_save = f"{dir_data}/patches_z2p4_to_3p4"
dir_co = f"{dir_data}/cib_co_sources"

print(f"SAVING ALL FILES TO {dir_save}!")

print("=====================================")

fwhm = np.deg2rad(20.0 / 60)  # to convert arcmin to degrees
redshift_range = [2.4, 3.4]

print()
print("Now loading CO signal...")

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
CO_fluxes_sats = []


for freq in CO_obs_freqs:
    CO_sats = []
    for chunk in [1, 2, 3, 4]:
        fn_sats = f"{dir_co}/sources/sat_chunk{chunk}_fluxCO_{freq}.h5"

        print(fn_sats)
        with h5py.File(fn_sats, "r") as f:
            # List all groups (like folders in the file)
            print("Keys:", list(f.keys()))

            # Access a dataset by key\n",
            data = f["flux"][:]  # Load it into a NumPy array
            print("Data shape:", data.shape)
            # print("Data type:", data.dtype)

            CO_sats.append(data)

    CO_fluxes_sats.append(np.concatenate(CO_sats, axis=1))

redshift_mask = (halos_co["redshift"] < max(redshift_range)) & (
    halos_co["redshift"] > min(redshift_range)
)

maps_CO_090 = []
maps_CO_150 = []
maps_CO_220 = []

for i in range(len(CO_rest_freqs)):
    print(f"Now on {i}...")
    m = maps.make_sky(
        halos_co,
        weights=CO_fluxes_sats[0][i],
        fwhm=fwhm,
        mask=redshift_mask,
        verbose=True,
    )
    p = maps.zoom_in(m)

    maps_CO_090.append(p)

    m = maps.make_sky(
        halos_co,
        weights=CO_fluxes_sats[1][i],
        fwhm=fwhm,
        mask=redshift_mask,
        verbose=True,
    )
    p = maps.zoom_in(m)

    maps_CO_150.append(p)

    m = maps.make_sky(
        halos_co,
        weights=CO_fluxes_sats[2][i],
        fwhm=fwhm,
        mask=redshift_mask,
        verbose=True,
    )
    p = maps.zoom_in(m)

    maps_CO_220.append(p)

print("saving CO patches...")
np.save(f"{dir_save}/maps_CO_090_sats", maps_CO_090)
np.save(f"{dir_save}/maps_CO_150_sats", maps_CO_150)
np.save(f"{dir_save}/maps_CO_220_sats", maps_CO_220)

print("finished!")

print("Et voila !")
