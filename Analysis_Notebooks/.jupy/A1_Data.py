# %% [markdown]
# # Data Notebook
# Notebook to setup the basic dataset.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import vaex
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.param_funcs as paramf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]

# %%
print(params["solar"])

# %%
print(data_p["sample"])
stars = vaex.open(data_p["sample"])

# %%
print(stars.count())

# %%
print(np.median(stars["vy"].values))

# %%
print(np.median(stars["vY"].values))

# %%
print(np.median(stars["vT"].values))

# %%
print(np.median(stars["X"].values))

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy import plotting_utils
plotting_utils.plot_original_data(stars)

# %%
print(stars.column_names)

# %%
