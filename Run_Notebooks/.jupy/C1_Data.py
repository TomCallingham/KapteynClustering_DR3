# %% [markdown]
# # Data Notebook
# Notebook to setup the basic dataset.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import vaex
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.plot_funcs as plotf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
solar_p = params["solar"]
link_p = params["linkage"]
features = link_p["features"]
scales = link_p["scales"]

# %%
print(solar_p)

# %% [markdown]
# # Basic Dataset
# The basic dataset needs to contain
# - (x,y,z,vx,vy,vz). These are helio centric (following sofie)

# %%
print(data_p["base_data"])
stars = vaex.open(data_p["base_data"])


# %% [markdown]
# # Sample
# - (pos,Vel) in N x 3 form. These are GALACTIC centric
# Given solar params, we can create the galactic pos,vel (along with XYZ, vXvYvZ)
# Apply the vtoomre>210

# %%
stars = dataf.create_galactic_posvel(stars, solar_params=solar_p)
stars = dynf.add_cylindrical(stars)
stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210, solar_params=solar_p)

# %% [markdown]
# ## Dynamics
# Read in in the potential name and calculates the dynamics based on that. The pot_name can be "H99", or any Agama ini file in packaged with Agama, or the path to a specific Agama ini file.

# %%
stars = dynf.add_dynamics(stars, data_p["pot_name"], additional_dynamics=["circ"])#, "JR"])
stars = stars[stars["En"] < 0]

# %% [markdown]
# ## Save Sample

# %%
print(data_p["sample"])
stars.export(data_p["sample"])

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
