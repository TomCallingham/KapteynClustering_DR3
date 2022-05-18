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
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf

from params import data_params, gaia2, auriga, cosma

# %% [markdown]
# # Paramsgaia2

# %%
from KapteynClustering.default_params import  cluster_params0
min_members = cluster_params0["min_members"]
max_members = cluster_params0["max_members"]
features = cluster_params0["features"]

# %% tags=[]
N_sigma_ellipse_axis = 2.83  # Length (in standard deviations) of axis of ellipsoidal cluster boundary
N_art = 10

# %% [markdown]
# # Load Data

# %%
sig_params = {
    "min_members": min_members,
    "max_members": max_members,
    "features": features,
    "N_art": N_art,
    "N_sigma_ellipse_axis": N_sigma_ellipse_axis,
    "N_max": None, #2e4, #NONE # TESTING purpose. Adds N_max_{} to back of save_name
    "N_process": 8}

sig_files = {
    "file_base" : data_params["result_path"] + data_params["data_folder"],
    "cluster_file": data_params["cluster"],
    "sample_file": data_params["sample"],
    "art_file": data_params["art"],
    "save_name": data_params["sig"]
}
sig_params["files"] = sig_files


# %%
# CAREFUL! DO NOT overwrite
if True:
    dicf.pickle_save(sig_params, "Significance_params")

# %% [markdown] tags=[]
# # PCA Fit
# multiprocessing struggles in notebook. Below launches the scripts with given parameters.
# ## Original
# For N_art = 100 took 20-30 min on 20 cores for 5e4 clusters
# ## New
# for N_art = 10
# New takes 5 mins on 8 cores for 5e4 clusters.
# around 90 mins on 8 cores for 5e5 clusters.

# %%
# If on cosma, sbatch this!

# %%
if True:
    if cosma:
        !sbatch "Batch/"
    elif gaia2:
        !python "../KapteynClustering/KapteynClustering/Run_Significance.py"

# %% [markdown]
# ## ANALYSIS

# %%
sig_data = dataf.read_data(fname=data_params["sig"], data_params=data_params, extra = "_N_max_20000")

# %%

# %%

# %% [markdown]
# import time
# from multiprocessing import Pool, RawArray

# %%
plt.figure()
plt.hist(region_count, bins="auto",density=True, label="fit", histtype="step")
plt.hist(art_region_count, bins="auto",density=True, label= "art")
plt.legend()
plt.show()
