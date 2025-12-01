# from astropy.io import fits
import h5py
import healpy as hp
import numpy as np

# import matplotlib.pyplot as plt
import joker.cosmology as cosmo
import joker.maps as maps

print("hello, world!")

dir_data = f"/data/cluster/emc-brid/Datasets/Websky"
dir_websky = f"{dir_data}/websky/v0.0"
dir_rewrite = f"{dir_data}/websky_halos_rewrite"
dir_co = f"{dir_data}/cib_co_sources/cib_co_sources"

fwhm = np.deg2rad(20.0 / 60)  # to convert arcmin to degrees
redshift_range = [2.4, 3.4]

halo_filename = f"{dir_rewrite}/websky_halos-lesslight_20230612.h5"
halos = maps.make_halo_catalogue(halo_filename, verbose=True)

print(f"making full halo patch...")
map_halo = maps.make_sky(
    halos, weights=halos["M200m"] ** (5.0 / 3.0), fwhm=fwhm, verbose=True
)

patch_halo = maps.zoom_in(map_halo)

np.save("patch_halo", patch_halo)
print("saved patch_halo!")

print(f"making redshift-masked halo patch in range {redshift_range}...")

redshift_mask = (halos["redshift"] < max(redshift_range)) & (
    halos["redshift"] > min(redshift_range)
)
print(
    f"{np.sum(redshift_mask) / len(halos['redshift']) * 100:.3f}% of halos within specified redshift range"
)
map_halo_masked = maps.make_sky(
    halos,
    weights=halos["M200m"] ** (5.0 / 3.0),
    fwhm=fwhm,
    mask=redshift_mask,
    verbose=True,
)
patch_halo_masked = maps.zoom_in(map_halo_masked)
np.save("patch_halo_masked", patch_halo_masked)
print("saved patch_halo_masked!")

print()

print("Now loading on tSZ signal...")
tsz = hp.read_map(f"{dir_websky}/websky/0.4/tsz_8192_hp.fits", h=False)
tsz = hp.smoothing(tsz, fwhm=fwhm)
patch_tsz = maps.zoom_in(tsz)

np.save("patch_tsz", patch_tsz)

print("Now loading CIB signal...")
path_cib = "/data/cluster/emc-brid/Datasets/Websky/websky/v0.0"
suffix = "cib_nu"
# cib_freqs = ["0093", "0143", "0217"]

cib_freqs = ["0100", "0143", "0145", "0217", "0278", "0353", "0545", "0857"]

data_cib = []

for freq in cib_freqs:
    print(f"making cib data for freq {freq}...")
    m = hp.read_map(f"{path_cib}/{suffix}{freq}.fits")
    p = maps.zoom_in(hp.smoothing(m, fwhm=fwhm))

    data_cib.append(p)

data_cib = np.asarray(data_cib)

print("Saving data_cib...")
np.save("data_cib", data_cib)

print("Now loading CO signal...")
CO_freqs = ["090", "150", "220"]

CO_fluxes = []

for freq in CO_freqs:
    CO_chunks = []
    for chunk in [1, 2, 3, 4]:
        filename = f"{dir_co}/cen_chunk{chunk}_fluxCO_{freq}.h5"
        print(filename)
        with h5py.File(filename, "r") as f:
            # List all groups (like folders in the file)
            # print(f"Keys:", list(f.keys()))

            # Access a dataset by key\n",
            data = f["flux"][:]  # Load it into a NumPy array
            # print("Data shape:", data.shape)
            # print("Data type:", data.dtype)

            CO_chunks.append(data)

    CO_fluxes.append(np.concatenate(CO_chunks, axis=1))

maps_CO_090 = []
maps_CO_150 = []
maps_CO_220 = []

for i in range(7):
    print(f"Now on {i}...")
    m = maps.make_sky(
        halos, weights=CO_fluxes[0][i], fwhm=fwhm, mask=redshift_mask, verbose=True
    )
    p = maps.zoom_in(m)

    maps_CO_090.append(p)

    m = maps.make_sky(
        halos, weights=CO_fluxes[1][i], fwhm=fwhm, mask=redshift_mask, verbose=True
    )
    p = maps.zoom_in(m)

    maps_CO_150.append(p)

    m = maps.make_sky(
        halos, weights=CO_fluxes[2][i], fwhm=fwhm, mask=redshift_mask, verbose=True
    )
    p = maps.zoom_in(m)

    maps_CO_220.append(p)

print("saving CO patches...")
np.save("maps_CO_090", maps_CO_090)
np.save("maps_CO_150", maps_CO_150)
np.save("maps_CO_220", maps_CO_220)

print("finished!")

print("Et voila !")
