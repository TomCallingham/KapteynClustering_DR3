# Setup
import numpy as np
import cluster_funcs as clusterf
import dic_funcs as dicf
from ..params import gaia2

if gaia2:
    Folder = "/data/users/callingham/data/clustering/KapteynClustering/"
else:
    Folder = "/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapteynClustering/"

sig_params = dicf.pickle_load(Folder + "Significance_params")

sig_files = sig_params["files"]
min_members =  sig_params["min_members"]
max_members =  sig_params["max_members"]
features =  sig_params["features"]
N_art =  sig_params["N_art"]
N_sigma_ellipse_axis =  sig_params["N_sigma_ellipse_axis"]
N_max = sig_params["N_max"]
N_process = sig_params["N_process"]


file_base = sig_files["file_base"]
cluster_file = sig_files["cluster_file"]
sample_file = sig_files["sample_file"]
art_file = sig_files["art_file"]
save_name = sig_files["save_name"]



cluser_data = dicf.h5py_load(file_base + cluster_file)
Z = cluser_data["Z"]
sample_data = dicf.h5py_load(file_base + sample_file)
stars = sample_data["stars"]
art_data = dicf.h5py_load(file_base + art_file)
art_stars = art_data["stars"]
N_clusters = len(Z[:,0])


X = clusterf.find_X(features, stars)
art_stars = art_data["stars"]
art_X = clusterf.art_find_X(features, art_stars)
del art_stars
del art_data


tree_members = clusterf.find_tree(Z, prune=True)

list_tree_members = [tree_members[i + N_clusters+1] for i in range(N_clusters)]
print("LOADED all data")

#### ALL DATA LOADED

# PCA Fit

import significance_funcs as sigf

import time
from multiprocessing import Pool, RawArray

if N_max is  None:
    N_max = N_clusters
else:
    N_max = int(N_max)
    save_name += f"_N_max_{N_max}"

print("Starting finding significance")
print(f"N_clusters = {N_clusters}")
print(f"N_max = {N_max}")
print(f"N_art = {N_art}")
# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(raw_X, X_shape, art_dic):
    print("Initialised")
    var_dict["share_X"] = np.frombuffer(raw_X).reshape(X_shape)
    share_art_X = {}
    for n in list(art_dic.keys()):
        share_art_X[n] =  np.frombuffer(art_dic[n]["art_X"]).reshape(art_dic[n]["art_X_shape"])
    var_dict["share_art_X"] = share_art_X

def worker_func(members):
    return sigf.expected_density_members(members, N_std = N_sigma_ellipse_axis, X=var_dict["share_X"], art_X=var_dict["share_art_X"], N_art=N_art,
            min_members=min_members)




X_shape = np.shape(X)
raw_X = RawArray('d', X_shape[0] * X_shape[1])
# Wrap X as an numpy array so we can easily manipulates its data.
share_X = np.frombuffer(raw_X).reshape(X_shape)
# Copy data to our shared array.
np.copyto(share_X, X)

art_dic = {}
share_dic = {}
raw_dic = {}
for n in list(art_X.keys()):
    art_dic[n] = {}

    art_dic[n]["art_X_shape"] =np.shape(art_X[n])
    raw_dic[n] = RawArray('d', art_dic[n]["art_X_shape"][0]*art_dic[n]["art_X_shape"][1])
    # Wrap art_X_array as an numpy array so we can easily manipulates its data.
    share_dic[n] = np.frombuffer(raw_dic[n]).reshape(art_dic[n]["art_X_shape"])
    # Copy data to our shared array.
    np.copyto(share_dic[n], art_X[n])
    art_dic[n]["art_X"] = raw_dic[n]

# Start the process pool and do the computation.
# Here we pass X and X_shape to the initializer of each worker.
# (Because X_shape is not a shared variable, it will be copied to each child process.)
T = time.time()

with Pool(processes=N_process, initializer=init_worker, initargs=(raw_X, X_shape,art_dic)) as pool:
    result = pool.map(worker_func, list_tree_members[:N_max] )
print("Finished")
dt = time.time() - T
print(f" Time taken: {dt/60} mins")

result = np.array(result)
region_count = result[:,0]
art_region_count = result[:,1]
art_region_count_std = result[:,2]
significance = (region_count - art_region_count)/ np.sqrt( region_count + (art_region_count_std**2))
pca_data = {"region_count":region_count,
            "art_region_count": art_region_count,
            "art_region_count_std": art_region_count_std,
            "significance": significance}

import dic_funcs as dicf

dicf.h5py_save(file_base + save_name, pca_data, verbose=True, overwrite=True)
