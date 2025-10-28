import numpy as np
import healpy as hp

# from astropy.io import fits
import h5py
# import matplotlib.pyplot as plt

from joker.cosmology import *
from joker.maps import *

print("hello, world!")

dir_websky = "/home/emc-brid/Websky_maps"
dir_rewrite = "/home/emc-brid/websky_halos_rewrite"
dir_co = "/home/emc-brid/cib_co_data"


filename = f"{dir_rewrite}/websky_halos-lesslight_20230612.h5"
print(f"Now loading in {filename}...")
halos = {}
with h5py.File(filename, "r") as f:
    # List all groups (like folders in the file)
    for key, item in f.items():
        print(f"adding {key}...")
        halos[key] = item[()]

chi = np.sqrt(halos["x"] ** 2 + halos["y"] ** 2 + halos["z"] ** 2)  # Mpc
redshift = zofchi(chi)

halos["redshift"] = redshift

print(f"Now loading CO fluxes...")
CO_90 = []
nCO = 0
index = 0
for i in range(1, 5):
    filename = f"{dir_co}/cen_chunk{i}_fluxCO_090.h5"
    print(filename)
    with h5py.File(filename, "r") as f:
        # List all groups (like folders in the file)
        print(f"Keys:", list(f.keys()))

        # Access a dataset by key\n",
        data = f["flux"][:]  # Load it into a NumPy array
        print("Data shape:", data.shape)
        print("Data type:", data.dtype)

        CO_90.append(data)

CO_90 = np.concatenate(CO_90, axis=1)


mask = (redshift <= 2.05) & (redshift >= 1.95)
print(mask.sum() / len(redshift))

print("Making maps...")

map_halo = make_sky(halos, weights=halos["M200m"])
map_CO = make_sky(halos, weights=np.sum(CO_90, axis=0))

print("Zooming in...")

patch_halos = zoom_in(map_halo)
patch_CO = zoom_in(map_CO)

print("Saving files...")

np.save("patch_halos", patch_halos)
np.save("patch_CO", patch_CO)

print("Et voila !")
