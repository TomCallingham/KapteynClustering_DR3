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
# # Data Notebook
# Notebook to setup the basic

# %% [markdown]
# ## Setup

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import vaex
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.plot_funcs as plotf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_p = params["data"]
solar_p = params["solar"]
cluster_p = params["cluster"]
features = cluster_p["features"]
scales = cluster_p["scales"]
# solar_p={'vslr': 232.8, 'R0': 8.2, '_U': 11.1, '_V': 12.24, '_W': 7.25}

# %%
print(params.keys())

# %%
print(params["data"].keys())

# %%
print(params["data"]["base_data"])

# %%
print(solar_p)

# %% [markdown] tags=[]
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

# %%
print(stars.count())

# %% [markdown]
# ## Galactic PosVel
# Read in in the potential name and calculates the dynamics based on that. The pot_name can be "H99", or any Agama ini file in packaged with Agama, or the path to a specific Agama ini file.

# %%
print(data_p["pot_name"])
stars = dynf.add_dynamics(stars, data_p["pot_name"], circ=True)
stars = stars[stars["En"]<0]
stars = clusterf.scale_features(stars, features=features,scales=scales, plot=False)[0]

# %% [markdown]
# ## Save Dyn

# %%
print(data_p["base_dyn"])
stars.export( data_p["base_dyn"])

# %% [markdown] tags=[]
# # Toomre Sample
# Apply the vtoomre>210

# %%
toomre_stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210,solar_params=solar_p)
toomre_stars = toomre_stars[toomre_stars["En"]<0]

# %% [markdown]
# ## Save Sample

# %%
print(data_p["sample"])
toomre_stars.export( data_p["sample"])

# %%
print(toomre_stars.count())

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy import plotting_utils
plotting_utils.plot_original_data(toomre_stars)
