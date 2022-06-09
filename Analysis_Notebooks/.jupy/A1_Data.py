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

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]

# %%
print(solar_p)

# %%
print(data_p["sample"])
stars = vaex.open(data_p["sample"])

# %%
print(stars.count())

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy import plotting_utils
plotting_utils.plot_original_data(stars)

# %%
print(stars.column_names)

# %%
plt.figure()
plt.hist(stars["x"].values, bins=100)
plt.show()

plt.figure()
plt.hist(stars["X"].values, bins=100)
plt.show()

# %%
