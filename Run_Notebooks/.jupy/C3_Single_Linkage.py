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
# # Setup

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_params = params["data"]
cluster_params = params["cluster"]
scales, features = cluster_params["scales"], cluster_params["features"]

# %%
print(data_params)

# %% [markdown]
# # Load

# %%
stars = dataf.read_data(fname=data_params["result_folder"] + data_params["sample"])

# %%
print(len(stars["x"]))

# %% [markdown]
# # Linkage Clustering
# Takes ~30s on 5e4 sample
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
dicf.h5py_save(data_params["result_folder"]+data_params["cluster"], cluster_data)

# %% [markdown]
# # Plots
# %%
Z = cluster_data["Z"]
print(np.shape(Z))

# %%
# Dendagram, Single linkage
