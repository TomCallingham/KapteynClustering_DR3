# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:py39]
#     language: python
#     name: conda-env-py39-py
# ---

# %%
# Setup
import numpy as np
from multiprocessing import Pool, RawArray
import time
import KapteynClustering.significance_funcs as sigf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.dic_funcs as dicf
import sys
sys.path.append('../../Notebooks/')
from params import gaia2

if gaia2:
    Folder = "/data/users/callingham/data/clustering/KapteynClustering_DR3/Notebooks/"
else:
    Folder = "/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapteynClustering_DR3/Notebooks/"

sig_params = dicf.pickle_load(Folder + "Significance_params")

sig_files = sig_params["files"]
min_members = sig_params["min_members"]
max_members = sig_params["max_members"]
features = sig_params["features"]
N_art = sig_params["N_art"]
N_sigma_ellipse_axis = sig_params["N_sigma_ellipse_axis"]
N_max = sig_params["N_max"]
N_process = sig_params["N_process"]


sample_file = sig_files["sample_file"]
art_file = sig_files["art_file"]


cluser_data = dicf.h5py_load(sig_files["file_base"] + sig_files["cluster_file"])
Z = cluser_data["Z"]
N_clusters = len(Z[:, 0])

from KapteynClustering.legacy import load_sofie_data as lsd
print("USING SOF ART")
stars = lsd.load_sof_df()
X = clusterf.find_X(features, stars)
del stars
sof_art_stars = lsd.load_sof_art()
art_X = clusterf.art_find_X(features, sof_art_stars)
del sof_art_stars

tree_members = clusterf.find_tree(Z, prune=True)
list_tree_members = [tree_members[i + N_clusters+1] for i in range(N_clusters)]
print("LOADED all data")


# %%
def worker_func(members):
    # print(members)
    my_results = sigf.cut_expected_density_members(members, N_std=N_sigma_ellipse_axis, X=X, art_X=art_X, N_art=N_art, min_members=min_members)
    return my_results



# %%
print("Test")

# %% tags=[]
T = time.time()
region_count = np.zeros((N_clusters))
art_region_count = np.zeros((N_clusters))
art_region_count_std = np.zeros((N_clusters))

for i, m in enumerate(list_tree_members):
    region_count[i], art_region_count[i], art_region_count_std[i] = worker_func(m)
    if i%500==0:
        print(i)
print("Finished")
dt = time.time() - T
print(f" Time taken: {dt/60} mins")


# %%
significance = (region_count - art_region_count) / \
    np.sqrt(region_count + (art_region_count_std**2))
pca_data = {"region_count": region_count,
            "art_region_count": art_region_count,
            "art_region_count_std": art_region_count_std,
            "significance": significance}

print("Using cut. SLOWER")
file_base = sig_files["file_base"]
save_name = sig_files["save_name"] + "_SOF_ART_cut_serial"

dicf.h5py_save(file_base + save_name, pca_data, verbose=True, overwrite=True)

# %%
