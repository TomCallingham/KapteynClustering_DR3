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

# %% [markdown]
# # Setup

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
from params import data_params, gaia2, auriga

# %% [markdown]
# ## Load Data
# Load data before TOOMRE selection. We apply that after

# %%
data = dataf.read_data(fname=data_params["base_dyn"], data_params=data_params)
stars = data["stars"]

# %%
scale = data["scale"]
selection = data["selection"]

# %%
if gaia2:
    a_pot = dynf.load_a_potential(data_params["pot_file"])
if auriga:
    from TomScripts import auriga_AGAMA_pot as aap
    a_pot = aap.Auriga_AGAMA_AxiPot_Load(Halo_n=5, Level=4)

# %% [markdown]
# # Artificial DataSet
#
# Shuffles vx, vz.
# Keeps pos + vy
# Recalculates dynamics

# %%
import artificial_data_funcs as adf

# %%

# %%
from default_params import art_params0
N_art = art_params0["N_art"]
N_art =10

# %% tags=[]
if auriga:
    additional_props = ["R", "group", "Fe_H", "Age"]
elif gaia2:
    additional_props = []
art_data = adf.get_shuffled_artificial_set(N_art, data, a_pot, additional_props)

# %%
fname = data_params["art"]
folder = data_params["result_path"] + data_params["data_folder"]
dicf.h5py_save(fname=folder + fname, dic=art_data, verbose=True, overwrite=True)

# %% [markdown]
# # Plots comparing Shuffled Smooth dataset

# %%
import plotting_utils
import vaex
print(art_data.keys())

# %%
art_stars = art_data["stars"][0]
del art_stars["N_Part"]

# %%
art_stars["Lperp"] = art_stars["Lp"]
art_stars["circ"] = art_stars["Circ"]

# %%
df_artificial = vaex.from_dict(art_stars)

# %%
plotting_utils.plot_original_data(df_artificial)