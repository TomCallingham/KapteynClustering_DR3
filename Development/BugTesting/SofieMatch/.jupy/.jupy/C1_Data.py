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
# This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
# Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.
#
#
# The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

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
from params import data_params

# %% [markdown] tags=[]
# # Basic Dataset
# The following data is needed to start. By default, the data is named  as the  'f"{data}_{data_name}"'
# - stars.hdf5 : (Pos,Vel) in N x 3 form
# - selection

# %% [markdown]
# # Added
# - Add Dynamics
# - Make Selection (Toomre)

# %% [markdown] tags=[]
# # Load basic data
# Load Gaia, make toomre selection

# %%
data = dataf.read_data(fname=data_params["base_data"], data_params=data_params)

# %%
selection = data["selection"]
stars = data["stars"]
print(len(stars["x"]))

# %% [markdown]
# # Create Dynamics

# %%
pot_name = data_params["pot_name"]
stars = dynf.add_dynamics(stars, pot_name, circ=True)

# %%
data["selection"] = selection 
data["stars"] = stars

# %% [markdown]
# '''
# '''

# %% [markdown]
# # Save Dyn

# %%
fname = data_params["base_dyn"]
folder = data_params["result_path"] + data_params["data_folder"]
dicf.h5py_save(fname=folder + fname, dic=data, verbose=True, overwrite=True)

# %% [markdown] tags=[]
# # Toomre Sample
# Apply the vtoomre>210

# %%
data = dataf.read_data(fname=data_params["base_dyn"], data_params=data_params)

# %%
data = dataf.apply_toomre_filt_dataset(data, v_toomre_cut=210)
stars = data["stars"]
print(len(stars["x"]))
print(stars.keys())
stars = dicf.filt(stars, stars["E"]<0)
data["stars"] = stars

fname = data_params["sample"]
folder = data_params["result_path"] + data_params["data_folder"]
dicf.h5py_save(fname=folder + fname, dic=data, verbose=True, overwrite=True)

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy.vaex_funcs import vaex_from_dict
df = vaex_from_dict(stars)

# %%
from KapteynClustering.legacy import plotting_utils
plotting_utils.plot_original_data(df)

# %% [markdown]
# #  COMPARE

# %%
for x in ["En", "Lz", "Lperp", "circ"]:
    plt.figure()
    plt.scatter(stars[x], stars["original_" + x])
    plt.xlabel(x)
    plt.ylabel("original" + x)
    plt.show()

    plt.figure()
    plt.scatter(stars[x], stars["original_" + x]/stars[x])
    plt.xlabel(x)
    plt.ylabel("original / me" + x)
    plt.show()

    plt.figure()
    plt.scatter(stars[x], stars["original_" +x]-stars[x])
    plt.xlabel(x)
    plt.ylabel("original - me" + x)
    plt.show()

# %%
plt.figure()
plt.scatter(stars["En"], stars["original_En"])
plt.show()

plt.figure()
plt.scatter(stars["En"], stars["original_En"]/stars["En"])
plt.show()

plt.figure()
plt.scatter(stars["En"], stars["original_En"]-stars["En"])
plt.show()
