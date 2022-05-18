import numpy as np
import time
import KapteynClustering.significance_funcs as sigf
import KapteynClustering.dic_funcs as dicf
import sys

# LOADING Params
param_file = sys.argv[1]
save_name, result_folder, max_members, min_members, X, art_X, list_tree_members, N_clusters, N_process, N_art, N_std = sigf.sig_load_data(param_file)

print("LOADED all data")
print("Running in Serial")
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
