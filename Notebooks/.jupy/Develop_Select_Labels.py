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
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

from params import data_params

# %%
import time

# %%
N_sigma_significance = 3

# %%
%load_ext line_profiler

# %% [markdown]
# # Params

# %%
data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = data["stars"]

# %%
cluster_data = dataf.read_data(fname=data_params["cluster"], data_params=data_params)
Z = cluster_data["Z"]

# %%
sig_data = dataf.read_data(fname=data_params["sig"], data_params=data_params)
significance = sig_data["significance"]

# %%
tree_members = clusterf.find_tree(Z, prune=True)

# %%
N_clusters = len(Z[:, 0])
N_clusters1 = N_clusters + 1

# %%
import vaex
import cluster_utils
stats_dic = {"significance": significance,
             "i": np.arange(len(Z)),
             "index1": Z[:, 0],
             "index2": Z[:, 1],
             }
stats = vaex.from_dict(stats_dic)

# %% [markdown]
# # Select MaxSig Clusters

# %% [markdown]
# ## My MaxSig Cluster

# %% tags=[]
selected, statistical_significance = clusterf.select_maxsig_clusters_from_tree(
    significance, Z, minimum_significance=N_sigma_significance)

# %% tags=[]
print(selected)
print(len(selected))
print(statistical_significance)

# %% [markdown]
# ## Get Cluster Labels
# ### My Cluster Labels

# %%
labels, significance_list = clusterf.get_cluster_labels_with_significance(
    selected, statistical_significance, tree_members, N_clusters)

# %% tags=[]
%lprun -f  clusterf.get_cluster_labels_with_significance clusterf.get_cluster_labels_with_significance( selected, statistical_significance, tree_members, N_clusters)

# %% [markdown]
# ### Sofie Cluster Labels

# %%
maxsigclusters = stats.take(selected)
s_labels, s_significance_list = cluster_utils.get_cluster_labels_with_significance(maxsigclusters, Z)

# %% tags=[]
%lprun -f  cluster_utils.get_cluster_labels_with_significance  cluster_utils.get_cluster_labels_with_significance(maxsigclusters, Z)

# %%
print((labels==s_labels).sum()/len(labels))

# %%
Groups, Pops = np.unique(labels[labels!=-1], return_counts=True)
G_sig = np.array([significance_list[labels==g][0] for g in Groups])
print(len(Groups))

# %%
s_Groups, s_Pops = np.unique(s_labels, return_counts=True)
s_G_sig = np.array([significance_list[s_labels==g][0] for g in s_Groups])
print(len(s_Groups))

# %%
plt.figure(figsize=(8,8))
plt.scatter(Pops[G_sig>3], G_sig[G_sig>3])
for g,p,s in zip(Groups, Pops, G_sig):
    plt.text(p,s,g, size=15)
plt.scatter(s_Pops[s_G_sig>3], s_G_sig[s_G_sig>3], marker="+", zorder=10)
# for g,p,s in zip(s_Groups, s_Pops, s_G_sig):
#     plt.text(p+0.5,s+0.5,g, size=15)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
plt.show()

# %%
