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

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_params = params["data"]
folder = data_params["result_folder"]

# %% [markdown] tags=[]
# # Basic Dataset
# The following data is needed to start.
# - stars.hdf5 : (Pos,Vel) in N x 3 form
# - selection

# %% [markdown] tags=[]
# # Load basic data
# Load Gaia, make toomre selection

# %%
# data = dataf.read_data(fname=data_params["base_data"], data_params=data_params)
stars = dataf.read_data(fname=data_params["base_data"])

# %%
print(len(stars["x"]))

# %% [markdown]
# # Create Dynamics

# %%
pot_name = data_params["pot_name"]
stars = dynf.add_dynamics(stars, pot_name, circ=True)
stars = dynf.add_cylindrical(stars)

# %% [markdown]
# ## Save Dyn

# %%
dataf.write_data(fname=folder + data_params["base_dyn"], dic=stars, verbose=True, overwrite=True)

# %% [markdown] tags=[]
# # Toomre Sample
# Apply the vtoomre>210

# %%
toomre_stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210)
toomre_stars = dicf.filt(toomre_stars, stars["En"]<0)

# %% [markdown]
# ## Save Sample

# %%
fname = data_params["sample"]
dataf.write_data(fname=folder + fname, dic=toomre_stars, verbose=True, overwrite=True)

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy.vaex_funcs import vaex_from_dict

df = vaex_from_dict(stars)

# %%
from KapteynClustering.legacy import plotting_utils
plotting_utils.plot_original_data(df)

# %%

# %%
