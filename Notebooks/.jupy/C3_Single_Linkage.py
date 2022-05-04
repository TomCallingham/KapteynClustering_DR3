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
np.random.seed(0)

# %%
import sys
# sys.path.append("/cosma/home/dp004/dc-call1/scripts/Python/AuClustering/KapetynClustering/")
sys.path.append('../KapetynClustering/')
import dic_funcs as dicf
import data_funcs as dataf
import plot_funcs as plotf
import dynamics_funcs as dynf

# %%
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
import cluster_funcs as clusterf

# %%
data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = data["stars"]

# %%
print(len(stars["x"]))

# %%
from default_params import cluster_params0
features = cluster_params0["features"]

# %%
cluster_data = clusterf.clusterData(stars, features=features)

# %%
cluster_data["selection"] = data["selection"]

# %%
fname = data_params["cluster"]
folder = data_params["result_path"] + data_params["data_folder"]

dicf.h5py_save(folder+fname, cluster_data)

# %%
Z = cluster_data["Z"]
print(np.shape(Z))

# %% [markdown]
# # Plots

# %%
# Dendagram, Single linkage
# %%
print("Added in py")
# %%
print("Added in py")
# %%
print("New C3, made from check notebook scripts")

# %%
print("Now added in py")
# %%
