# %% [markdown]
# This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
# Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.
#
#
# The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf

# from params import data_params, gaia2, auriga, cosma

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %% [markdown]
# # LOAD

# %%
params = dataf.read_param_file("../../../Params/sof_check_params.yaml")
data_params = params["data"]

cut_sig_data = dataf.read_data(fname=data_params["sig"], extra="cut" )
cut_sig = cut_sig_data["significance"]
cut_label_data = dataf.read_data( data_params["label"], extra="cut")
cut_my_labels = cut_label_data["labels"]

sig_data = dataf.read_data(fname=data_params["sig"] )
label_data = dataf.read_data( data_params["label"])
my_labels = label_data["labels"]

# %% [markdown]
# ## Load Sof

# %%
# sof_stats = dataf.load_vaex_to_dic("/net/gaia2/data/users/dodd/Clustering_EDR3/cluster_in_3D_test/results/df_labels_3D_3sigma")
s_result_path = "/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/"
sof_stats = dataf.read_data(fname=s_result_path+ "stats" )
sof_Z = np.stack((sof_stats["index1"], sof_stats["index2"])).T
s_sig_data = {}
s_sig_data["significance"] = sof_stats["significance"]
s_sig_data["region_count"] = sof_stats["within_ellipse_count"]
s_sig_data["art_region_count"] = sof_stats["region_counts"]
s_sig_data["art_region_count_std"] = sof_stats["region_count_stds"]

# %% [raw]
# import vaex
# result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
# stats = vaex.open(f'{result_path}stats.hdf5')
# s_sig_data = {}
# s_sig_data["region_count"] =  stats["within_ellipse_count"].values
# s_sig_data["art_region_count"]  = stats["region_counts"].values
# s_sig_data["art_region_count_std"] = stats["region_count_stds"].values
#
# s_sig_data["significance"]  = stats["significance"].values

# %% [markdown]
# # Sig ANALYSIS

# %%
for k in s_sig_data.keys():
    print(k)
    # filt = np.ones((len(Z)),dtype=bool)
    # s_filt=filt
    # filt = (sig_data["region_count"]!=0)
    # s_filt = (s_sig_data["region_count"]!=0)
    x = sig_data[k]
    y = s_sig_data[k]
    # x = x[np.isfinite(x)]
    # y = y[np.isfinite(y)]
    diff = np.abs(x-y)
    diff[(~np.isfinite(x))*(~np.isfinite(y))] = 0
    n_diff = (np.abs(x-y)>1e-6).sum()
    print(n_diff/len(x))
    fin_filt =  np.isfinite(x)*np.isfinite(y)
    x = x[fin_filt]
    y = y[fin_filt]
    plt.figure()
    plt.hist(x, bins="auto", density=True, label="me")
    plt.hist(y, bins="auto", density=True, histtype="step", label="sof")
    plt.xlabel(k)
    plt.legend()
    plt.show()
    
    Fig, axs = plt.subplots(ncols=3, figsize=(15,5))
    axs[0].scatter(x, y)
    axs[0].set_ylabel("sof")
    axs[1].scatter(x, y - x)
    axs[1].set_ylabel("sof - me")
    axs[2].scatter(x, y/x)
    axs[2].set_ylabel("sof / me")
    for ax in axs:
        ax.set_xlabel("me")
        # if k != "significance":
        #     ax.set_xscale("log")
    plt.title(k)
    plt.show()

# %% [markdown]
# # Label Analysis

# %%
# sof_stats = dataf.load_vaex_to_dic("/net/gaia2/data/users/dodd/Clustering_EDR3/cluster_in_3D_test/results/df_labels_3D_3sigma")
s_result_path = "/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/"
sof_label_data = dataf.read_data(fname=s_result_path+ "df_labels_3D_3sigma" )
sof_labels = sof_label_data["labels"]

import KapteynClustering.label_funcs as labelf
o_sof_labels = labelf.order_labels(sof_labels,fluff_label=0)[0]

f_match = (o_sof_labels==my_labels).sum()/len(sof_labels)
print(f_match)

# %%
print(my_labels)
print(o_sof_labels)

# %%
