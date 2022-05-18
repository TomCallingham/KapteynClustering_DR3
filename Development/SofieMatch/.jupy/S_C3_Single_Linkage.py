# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

from params import data_params

# %% [markdown]
# # Linkage Clustering
#
# ### SOFIE
# Perform clustering on the original data.
# This should not take too long, around 2-3 minutes with an input size of 5e10^4.
# Z is the linkage matrix outputted by the single linkage algorithm, where each row i indicates one step of the algorithm:
# At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster (N+i),
# where N is the number of original data points. Z[i, 3] indicates the number of members in the group formed at step i.
# ### Me
# Takes 2 mins on 5e5 of Lv4 Au
# 15s on 5e5 Gaia sample

# %%
fname = data_params["cluster"]
folder = data_params["result_path"] + data_params["data_folder"]

cluster_data = dicf.h5py_load(folder+fname)

# %% [markdown]
# # Plots
# %%
Z = cluster_data["Z"]
print(np.shape(Z))

# %%
sof_stats = dataf.load_vaex_to_dic("/net/gaia2/data/users/dodd/Clustering_EDR3/cluster_in_3D_test/results/stats")
print(sof_stats)
sof_Z = np.stack((sof_stats["index1"], sof_stats["index2"])).T

# %%
print(np.shape(Z))
print(np.shape(sof_Z))

print((Z[:,0] == sof_Z[:,0]).sum()/len(Z))
print((Z[:,1] == sof_Z[:,1]).sum()/len(Z))

# %%
# Dendagram, Single linkage
