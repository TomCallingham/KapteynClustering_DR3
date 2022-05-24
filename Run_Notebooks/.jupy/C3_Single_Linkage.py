# %% [markdown]
# # Setup

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
# import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.cluster_funcs as clusterf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
cluster_p = params["cluster"]
scales, features = cluster_p["scales"], cluster_p["features"]

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
