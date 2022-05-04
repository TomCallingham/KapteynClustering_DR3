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
# This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
# Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.
#
#
# The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

# %%
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../KapetynClustering/')
import dic_funcs as dicf

import data_funcs as dataf
import plot_funcs as plotf
import dynamics_funcs as dynf

# %%
from jupytext.config import global_jupytext_configuration_directories
list(global_jupytext_configuration_directories())

# %%
print("Recreating Notebooks. Creating blanks, then copying the jupytext files. Run on cosma")

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
from params import data_params, gaia2, auriga

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
print(auriga)

# %%
if not auriga:
    print("make dynamics")
    a_pot = dynf.load_a_potential(data_params["pot_file"])
    # agama_pot = load_a_potential(data_params["pot_file"])


    stars = dynf.add_dynamics(stars, a_pot, circ=True)
    stars["RPhi"] =stars["phi"]
# import TomScripts.auriga_AGAMA_pot as aap
# agama_pot = aap.Auriga_AGAMA_AxiPot_Load()

# %% [markdown]
# # Scale
# Plots to show scaling. This is actually an important paramater - must scale consistently

# %%
stars, scale_data = dataf.scale_features(stars)

# %%
data["scale"] = scale_data
data["stars"] = stars

# %%
print(stars.keys())

# %% [markdown]
# # Save Base

# %%
fname = data_params["base_dyn"]
folder = data_params["result_path"] + data_params["data_folder"]
dicf.h5py_save(fname=folder + fname, dic=data, verbose=True, overwrite=True)

# %% [markdown] tags=[]
# # Toomre Sample
# Apply the vtoomre>210

# %%
data = dataf.apply_toomre_filt_dataset(data, v_toomre_cut=210)
stars = data["stars"]
print(len(stars["x"]))
stars = dicf.filt(stars, stars["En"]<0)
data["stars"] = stars

fname = data_params["sample"]
folder = data_params["result_path"] + data_params["data_folder"]
dicf.h5py_save(fname=folder + fname, dic=data, verbose=True, overwrite=True)

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
print(stars.keys())

# %%
import vaex
stars = data["stars"]
del stars["N_Part"]
stars["Lperp"] = stars["Lp"]
stars["Circ"] = stars["circ"]

# %%
df = vaex.from_dict(stars)

# %%
import plotting_utils
plotting_utils.plot_original_data(df)

# %%
print("Added in py")
# %%
print("Added in py on gaia")
# %%
