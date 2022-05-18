# Setup
from multiprocessing import Pool, RawArray
import time
import KapteynClustering.significance_funcs as sigf
import numpy as np
import KapteynClustering.cluster_funcs as clusterf
import dic_funcs as dicf
import data_funcs as dataf
# if gaia2:
#     Folder = "/data/users/callingham/data/clustering/KapteynClustering_DR3/Notebooks/"
# else:
#     Folder = "/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapteynClustering_DR3/Notebooks/"
import sys
param_file = sys.argv[1]

# LOADING Params
params = dataf.read_param_file(param_file)
data_params = params["data"]
result_folder = data_params["result_folder"]
cluster_file, sample_file, art_file = data_params["cluster"] ,data_params["sample"] , data_params["art"]
save_name = data_params["sig"]

cluster_params = params["cluster"]
min_members, max_members = cluster_params["min_members"], cluster_params["max_members"]
features = cluster_params["features"]
N_art, N_std, N_process = cluster_params["N_art"] ,cluster_params["N_sigma_ellipse_axis"],cluster_params["N_process"]


# LOADING all necessary Data
sample_data = dicf.h5py_load(result_folder + sample_file)
stars = sample_data["stars"]
X = clusterf.find_X(features, stars)
del stars

art_data = dicf.h5py_load(result_folder + art_file)
art_stars = art_data["stars"]
art_X = clusterf.art_find_X(features, art_stars)
del art_stars, art_data

cluster_data = dicf.h5py_load(result_folder + cluster_file)
Z = cluster_data["Z"]
N_clusters = len(Z[:, 0])
tree_members = clusterf.find_tree(Z, prune=True)
list_tree_members = [tree_members[i + N_clusters+1] for i in range(N_clusters)]

print("LOADED all data")
print("Starting finding significance")
print(f"N_clusters = {N_clusters}")
print(f"N_art = {N_art}")

# ALL DATA LOADED


def worker_func(members):
    return sigf.cut_expected_density_members(members, N_std=N_std, X=X, art_X=art_X, N_art=N_art,
                                         min_members=min_members)

# Start the process pool and do the computation.

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

## Save Result
significance = (region_count - art_region_count) / \
    np.sqrt(region_count + (art_region_count_std**2))
pca_data = {"region_count": region_count,
            "art_region_count": art_region_count,
            "art_region_count_std": art_region_count_std,
            "significance": significance}


dicf.h5py_save(result_folder + save_name+ "_serial_cut", pca_data, verbose=True, overwrite=True)
