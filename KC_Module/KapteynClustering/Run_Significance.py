from threadpoolctl import threadpool_limits
import numpy as np
import time
import KapteynClustering.significance_funcs as sigf
import KapteynClustering.data_funcs as dataf
from multiprocessing import Pool, RawArray
import sys

# LOADING Params
param_file = sys.argv[1]
save_name, max_members, min_members, X, art_X, list_tree_members, N_clusters, N_process, N_art, N_std = sigf.sig_load_data( param_file)

print("LOADED all data")
print("Starting finding significance")
print(f"N_process = {N_process}")
print(f"N_clusters = {N_clusters}")
print(f"N_art = {N_art}")


# Initialise MPI
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


# print("No MAX")
# save_name += "_cut"
# print("Using cut function")
# def worker_func(members):
#     return sigf.cut_expected_density_members(members, N_std=N_std, X=var_dict["share_X"], art_X=var_dict["share_art_X"], N_art=N_art,
#                                              min_members=min_members, max_members=max_members)
def worker_func(members):
    return sigf.expected_density_members(members, N_std=N_std, X=var_dict["share_X"], art_X=var_dict["share_art_X"], N_art=N_art,
                                         min_members=min_members, max_members=max_members)


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
    return share_dic


share_dic = init_memory(X, art_X)

# Start the process pool and do the computation.
T = time.perf_counter()
with threadpool_limits(limits=1):
    with Pool(processes=N_process, initializer=init_worker, initargs=(share_dic,)) as pool:
        result = pool.map(worker_func, list_tree_members)
print("Finished")
dt = time.perf_counter() - T
print(f" Time taken: {dt/60} mins")


# Save Result
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


dataf.write_data(save_name, pca_data,
                 verbose=True, overwrite=True)
