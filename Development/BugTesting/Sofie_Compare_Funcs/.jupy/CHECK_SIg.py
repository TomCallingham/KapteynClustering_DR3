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
import numpy as np
import time
import KapteynClustering.significance_funcs as sigf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.cluster_funcs as clusterf
import sys

# LOADING Params
param_file = "sof_check_params.yaml"
save_name, result_folder, max_members, min_members, scaled_X, scaled_art_X, list_tree_members, N_clusters, N_process, N_art, N_std = sigf.sig_load_data(
    param_file, scaled_force=False)
save_name, result_folder, max_members, min_members, X, art_X, list_tree_members, N_clusters, N_process, N_art, N_std = sigf.sig_load_data(
    param_file, scaled_force=False)

print("LOADED all data")
print("Running in Serial")
print("Starting finding significance")
print(f"N_clusters = {N_clusters}")
print(f"N_art = {N_art}")

# ALL DATA LOADED

# %%

# %% tags=[]
# s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
# features  = ["scaled_En", "scaled_Lperp", "scaled_Lz"]
# # features  = ["En", "Lperp", "Lz"]
# features_to_be_scaled  = ["En", "Lperp", "Lz"]
# sof_art_stars = dataf.read_data(f'{s_result_path}df_artificial_3D.hdf5')
# sof_art_stars = dicf.groups(sof_art_stars, group="index", verbose=True)
# art_X = clusterf.art_find_X(features, sof_art_stars)

# stars = dataf.read_data(f'{s_result_path}df.hdf5')
# stars = clusterf.scale_features(stars, features = features_to_be_scaled, plot=False)[0]

# X = clusterf.find_X(features, stars)

# %%
params = dataf.read_param_file(param_file)
data_params = params["data"]
result_folder = data_params["result_folder"]
cluster_file, sample_file, art_file = data_params[
    "cluster"], data_params["sample"], data_params["art"]

cluster_data = dicf.h5py_load(result_folder + cluster_file)
Z = cluster_data["Z"]


# %%
import vaex
import cluster_utils
# Path to folder where intermediate and final results are stored
s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/'
df = vaex.open(f'{s_result_path}df.hdf5')
minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300],
                                 Lz=[-4500, 4600])  # Fix origin for scaling of features
df = cluster_utils.scaleData(df, ["En", "Lperp", "Lz"], minmax_values)
df_artificial = vaex.open(f'{s_result_path}df_artificial_3D.hdf5')
print("HACKY SCALE WORKAROUND")
for x in ["En", "Lperp", "Lz"]:
    df[x] = df["scaled_" + x]
    df_artificial[x] = df_artificial["scaled_" + x]


# %% [markdown]
# # Funcs

# %%
def my_func_cut(members):
    return sigf.cut_expected_density_members(members, N_std=N_std, X=X, art_X=art_X, N_art=N_art,
                                             min_members=min_members)


def my_func(members):
    return sigf.expected_density_members(members, N_std=N_std, X=X, art_X=art_X, N_art=N_art,
                                         min_members=min_members)


def my_scaled_func(members):
    return sigf.expected_density_members(members, N_std=N_std, X=scaled_X, art_X=art_scaled_X, N_art=N_art,
                                         min_members=min_members)


# %%
from Sof_Sig_funcs import original_s_expected_density_parallel
s_dic = {}
s_dic["Z"] = Z
s_dic["min_members"] = min_members
s_dic["max_members"] = max_members
s_dic["df"] = df
s_dic["df_artificial"] = df_artificial
s_dic["N_sigma_ellipse_axis"] = N_std
print(N_std)
s_dic["N_datasets"] = 100


def s_func(idx):
    return original_s_expected_density_parallel(idx, s_dic)


# %%

# %%
# Path to folder where intermediate and final results are stored
s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/'
s_stats = dataf.read_data(f"{s_result_path}stats")
s_sav_region_count = s_stats["within_ellipse_count"]
s_sav_art_region_count = s_stats["region_counts"]
s_sav_art_region_count_std = s_stats["region_count_stds"]

# %% tags=[]
# Start the process pool and do the computation.


region_count = np.zeros((N_clusters))
art_region_count = np.zeros((N_clusters))
art_region_count_std = np.zeros((N_clusters))

scal_region_count = np.zeros((N_clusters))
scal_art_region_count = np.zeros((N_clusters))
scal_art_region_count_std = np.zeros((N_clusters))

s_region_count = np.zeros((N_clusters))
s_art_region_count = np.zeros((N_clusters))
s_art_region_count_std = np.zeros((N_clusters))

for i, m in enumerate(list_tree_members):
    region_count[i], art_region_count[i], art_region_count_std[i] = my_func(m)
    sav_diff = np.abs(region_count[i] - s_sav_region_count[i])\
        + np.abs(art_region_count[i] - s_sav_art_region_count[i])\
        + np.abs(art_region_count_std[i] - s_sav_art_region_count_std[i])

    scal_region_count[i], scal_art_region_count[i], scal_art_region_count_std[i] = my_scaled_func(m)
    sav_diff = np.abs(region_count[i] - s_sav_region_count[i])\
        + np.abs(art_region_count[i] - s_sav_art_region_count[i])\
        + np.abs(art_region_count_std[i] - s_sav_art_region_count_std[i])

    s_region_count[i], s_art_region_count[i], s_art_region_count_std[i] = s_func(i)
    s_sav_diff = np.abs(s_region_count[i] - s_sav_region_count[i])\
        + np.abs(s_art_region_count[i] - s_sav_art_region_count[i])\
        + np.abs(s_art_region_count_std[i] - s_sav_art_region_count_std[i])

    live_diff = np.abs(region_count[i] - s_region_count[i])\
        + np.abs(art_region_count[i] - s_art_region_count[i])\
        + np.abs(art_region_count_std[i] - s_art_region_count_std[i])
    if region_count[i] > 0:
        print(i)
        print(region_count[i], art_region_count[i], art_region_count_std[i])
        print(scal_region_count[i], scal_art_region_count[i], scal_art_region_count_std[i])
        print(s_region_count[i], s_art_region_count[i], s_art_region_count_std[i])
        print(s_sav_region_count[i], s_sav_art_region_count[i], s_sav_art_region_count_std[i])
        # break

    if (sav_diff + s_sav_diff + live_diff) > 1e-3:
        print("BAD")
        print(i)
        print(region_count[i], art_region_count[i], art_region_count_std[i])
        print(s_region_count[i], s_art_region_count[i], s_art_region_count_std[i])
        print(s_sav_region_count[i], s_sav_art_region_count[i], s_sav_art_region_count_std[i])
        break
    else:
        if region_count[i] + s_region_count[i] > 0:
            print("GOOD")
            print(i)
            print(sav_diff, s_sav_diff, live_diff)

# %%
# Save Result
significance = (region_count - art_region_count) / \
    np.sqrt(region_count + (art_region_count_std**2))
pca_data = {"region_count": region_count,
            "art_region_count": art_region_count,
            "art_region_count_std": art_region_count_std,
            "significance": significance}

# %%
# dicf.h5py_save(result_folder + save_name + "_serial_cut",
#                pca_data, verbose=True, overwrite=True)
