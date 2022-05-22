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
# # Setup

# %% tags=[]
import numpy as np
import vaex
import matplotlib.pyplot as plt
# import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.cluster_funcs as clusterf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_p = params["data"]
cluster_p = params["cluster"]
scales, features = cluster_p["scales"], cluster_p["features"]

# %%
print(data_p)

# %% [markdown]
# # Load

# %%
stars = vaex.open( data_p["sample"])

# %%
print(stars.count())

# %% [markdown]
# # Linkage Clustering
# Takes ~10s on 5e4 sample
# Applies single linkage clustering.

# %%
print("Using the following features and scales:")
print(scales, "\n", features)

# %%
cluster_data = clusterf.clusterData(stars, features=features, scales = scales)

# %%
print(cluster_data.keys())

# %% [markdown]
# # Save

# %%
dataf.write_data(data_p["cluster"], cluster_data)

# %% [markdown]
# # Plots
# %%
Z = cluster_data["Z"]
print(np.shape(Z))

# %%
# Dendagram, Single linkage
