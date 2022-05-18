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

# %% [markdown] tags=[]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.artificial_data_funcs as adf
# import KapteynClustering.plot_funcs as plotf
# from params import data_params, gaia2, auriga

# %% [markdown]
# ## Params

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_params = params["data"]
art_params = params["art"]
N_art = art_params["N_art"]
pot_name = data_params["pot_name"]

# %% [markdown]
# ## Load Data
# Load data before TOOMRE selection. We apply that after

# %%
data = dataf.read_data(fname=data_params["result_folder"]+data_params["base_dyn"])

# %% [markdown]
# # Artificial DataSet
#
# Shuffles vx, vz.
# Keeps pos + vy
# Recalculates dynamics

# %% [markdown] tags=[]
# if auriga:
#     additional_props = ["R", "group", "Fe_H", "Age"]
# elif gaia2:
#     additional_props = []

# %%
additional_props = []

# %% tags=[]
art_data = adf.get_shuffled_artificial_set(N_art, data, pot_name, additional_props)

# %%
fname = data_params["art"]
folder = data_params["result_folder"]
dicf.h5py_save(fname=folder + fname, dic=art_data, verbose=True, overwrite=True)

# %% [markdown]
# # Plots comparing Shuffled Smooth dataset

# %%
print(art_data["stars"].keys())

# %%
from KapteynClustering.legacy import plotting_utils, vaex_funcs
df_artificial = vaex_funcs.vaex_from_dict(art_data["stars"][0])
df= vaex_funcs.vaex_from_dict(data["stars"])

# %%
plotting_utils.plot_original_data(df)

# %% tags=[]
plotting_utils.plot_original_data(df_artificial)

# %%
art_stars = art_data["stars"][0]
sample_data = dataf.read_data(fname=data_params["result_folder"] + data_params["sample"])
sample_stars = sample_data["stars"]
stars = data["stars"]

for x in ["En", "Lz", "Lperp", "circ"]:
    plt.figure()
    plt.hist(stars[x], label="Original", bins="auto",histtype="step")
    plt.hist(sample_stars[x], label="sample", bins="auto",histtype="step")
    plt.hist(art_stars[x], label="art", bins="auto", histtype="step")
    plt.legend()
    plt.xlabel(x)
    plt.show()

# %%
