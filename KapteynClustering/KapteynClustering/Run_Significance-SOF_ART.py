# Setup
from KapteynClustering.legacy import load_sofie_data as lsd
import numpy as np
from multiprocessing import Pool, RawArray
import time
import KapteynClustering.significance_funcs as sigf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.dic_funcs as dicf
# import sys
# sys.path.append('../../Notebooks/')
# from params import gaia2
# if gaia2:
#     Folder = "/data/users/callingham/data/clustering/KapteynClustering_DR3/Notebooks/"
# else:
#     Folder = "/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapteynClustering_DR3/Notebooks/"
Folder = "/data/users/callingham/data/clustering/KapteynClustering_DR3/Notebooks/"

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

print("Using cut. SLOWER")
save_name = sig_files["save_name"] + "_SOF_ART_cut"

cluser_data = dicf.h5py_load(
    sig_files["file_base"] + sig_files["cluster_file"])
Z = cluser_data["Z"]
N_clusters = len(Z[:, 0])

print("USING SOF ART")
stars = lsd.load_sof_df()
X = clusterf.find_X(features, stars)
del stars
sof_art_stars = lsd.load_sof_art()
art_X = clusterf.art_find_X(features, sof_art_stars)
del sof_art_stars

tree_members = clusterf.find_tree(Z, prune=True)
list_tree_members = [tree_members[i + N_clusters+1] for i in range(N_clusters)]
print("LOADED all data")  # PCA Fit

print("Starting finding significance")
print(f"N_clusters = {N_clusters}")
print(f"N_art = {N_art}")
# A global dictionary storing the variables passed from the initializer.
var_dict = {}


def init_worker(share_dic):
    raw_X = share_dic["raw_X"]
    X_shape = share_dic["X_shape"]
    art_dic = share_dic["art_dic"]

    print("Initialised")
    var_dict["share_X"] = np.frombuffer(raw_X).reshape(X_shape)
    share_art_X = {}
    for n in list(art_dic.keys()):
        share_art_X[n] = np.frombuffer(
            art_dic[n]["art_X"]).reshape(art_dic[n]["art_X_shape"])
    var_dict["share_art_X"] = share_art_X


def worker_func(members):
    return sigf.cut_expected_density_members(members, N_std=N_sigma_ellipse_axis, X=var_dict["share_X"], art_X=var_dict["share_art_X"], N_art=N_art,
                                             min_members=min_members)


def init_memory(X, art_X):
    print("Initialising Memory")
    X_shape = np.shape(X)
    raw_X = RawArray('d', X_shape[0] * X_shape[1])
    # Wrap X as an numpy array so we can easily manipulates its data.
    share_X = np.frombuffer(raw_X).reshape(X_shape)
    # Copy data to our shared array.
    np.copyto(share_X, X)

    art_dic = {}
    raw_dic = {}
    for n in list(art_X.keys()):
        art_dic[n] = {}
        art_dic[n]["art_X_shape"] = np.shape(art_X[n])
        raw_dic[n] = RawArray('d', art_dic[n]["art_X_shape"]
                              [0]*art_dic[n]["art_X_shape"][1])
        # Wrap art_X_array as an numpy array so we can easily manipulates its data.
        temp_share = np.frombuffer(raw_dic[n]).reshape(
            art_dic[n]["art_X_shape"])
        # Copy data to our shared array.
        np.copyto(temp_share, art_X[n])
        art_dic[n]["art_X"] = raw_dic[n]
    share_dic = {}
    share_dic["raw_X"] = raw_X
    share_dic["X_shape"] = X_shape
    share_dic["art_dic"] = art_dic
    print("Initialised")
    return share_dic  # Start the process pool and do the computation.


# Here we pass X and X_shape to the initializer of each worker.
# (Because X_shape is not a shared variable, it will be copied to each child process.)
share_dic = init_memory(X, art_X)
T = time.time()

with Pool(processes=N_process, initializer=init_worker, initargs=(share_dic,)) as pool:
    result = pool.map(worker_func, list_tree_members)
print("Finished")
dt = time.time() - T
print(f" Time taken: {dt/60} mins")

result = np.array(result)
region_count = result[:, 0]
art_region_count = result[:, 1]
art_region_count_std = result[:, 2]
significance = (region_count - art_region_count) / \
    np.sqrt(region_count + (art_region_count_std**2))
pca_data = {"region_count": region_count,
            "art_region_count": art_region_count,
            "art_region_count_std": art_region_count_std,
            "significance": significance}


dicf.h5py_save(sig_files["file_base"] + save_name,
               pca_data, verbose=True, overwrite=True)
