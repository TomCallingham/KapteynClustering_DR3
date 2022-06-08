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
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.plot_funcs as plotf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
solar_p = params["solar"]
cluster_p = params["cluster"]
features = cluster_p["features"]
scales = cluster_p["scales"]

# %%
print(solar_p)

# %% [markdown]
# # Basic Dataset
# The basic dataset needs to contain
# - (pos,Vel) in N x 3 form. These are GALACTIC centric
# - (x,y,z,vx,vy,vz). These are helio centric (following sofie)

# %%
print(data_p["base_data"])
gaia_catalogue = vaex.open(data_p["base_data"])


# %% [markdown]
# # Create Dynamics
# Given solar params, we can create the galactic pos,vel (along with _x, _vx)

# %%
stars = dataf.create_galactic_posvel(gaia_catalogue, solar_params=solar_p)

# %% [markdown]
# ## Galactic PosVel
# Read in in the potential name and calculates the dynamics based on that. The pot_name can be "H99", or any Agama ini file in packaged with Agama, or the path to a specific Agama ini file.

# %%
print(data_p["pot_name"])
stars = dynf.add_dynamics(stars, data_p["pot_name"], additional_dynamics=["circ"])
stars = stars[stars["En"] < 0]

# %% [markdown]
# ## Save Dyn

# %%
print(data_p["base_dyn"])
stars.export(data_p["base_dyn"])

# %% [markdown]
# # Toomre Sample
# Apply the vtoomre>210

# %%
toomre_stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210, solar_params=solar_p)
toomre_stars = toomre_stars[toomre_stars["En"] < 0]

# %% [markdown]
# ## Save Sample

# %%
print(data_p["sample"])
toomre_stars.export(data_p["sample"])

# %%
print(toomre_stars.count())

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy import plotting_utils
plotting_utils.plot_original_data(toomre_stars)

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
