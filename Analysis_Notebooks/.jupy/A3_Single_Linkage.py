# %% [markdown]
# # Setup

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
# import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.linkage_funcs as linkf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
linkage_p = params["linkage"]
scales, features = linkage_p["scales"], linkage_p["features"]

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

# %% [markdown]
# # Save

# %%
link_data = dataf.read_data(data_p["linkage"])

# %% [markdown]
# # Plots
