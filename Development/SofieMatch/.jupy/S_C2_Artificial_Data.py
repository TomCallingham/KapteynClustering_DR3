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

# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf

from params import data_params, gaia2, auriga

# %% [markdown]
# ## Load Data
# Load data before TOOMRE selection. We apply that after

# %%
data = dataf.read_data(fname=data_params["base_dyn"], data_params=data_params)
stars = data["stars"]

# %% [markdown]
# # Artificial DataSet
#
# Shuffles vx, vz.
# Keeps pos + vy
# Recalculates dynamics

# %%
fname = data_params["art"]
folder = data_params["result_path"] + data_params["data_folder"]
art_data = dicf.h5py_load(fname=folder + fname, verbose=True)

# %% [markdown]
# # Plots comparing Shuffled Smooth dataset

# %%
from KapteynClustering.legacy import plotting_utils, vaex_funcs
df_artificial_0 = vaex_funcs.vaex_from_dict(art_data["stars"][0])
df= vaex_funcs.vaex_from_dict(stars)

# %%
plotting_utils.plot_original_data(df)

# %%

# %% tags=[]
plotting_utils.plot_original_data(df_artificial_0)

# %%
art_stars = art_data["stars"][0]
sample_data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
sample_stars = sample_data["stars"]

for x in ["E", "Lz", "Lp", "circ"]:
    plt.figure()
    plt.hist(stars[x], label="Original", bins="auto",histtype="step")
    plt.hist(sample_stars[x], label="sample", bins="auto",histtype="step")
    plt.hist(art_stars[x], label="art", bins="auto", histtype="step")
    plt.legend()
    plt.xlabel(x)
    plt.show()

# %% [markdown]
# # Sofie Compare

# %%
import vaex
s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
df_artificial = vaex.open(f'{s_result_path}df_artificial_3D.hdf5')
plotting_utils.plot_original_data(df_artificial[df_artificial.index==2])

# %%

sof_art_stars = dataf.load_vaex_to_dic(f'{s_result_path}df_artificial_3D.hdf5')
trans_pops = {"En":"E", "Lperp":"Lp"}
for k in trans_pops.keys():
    sof_art_stars[trans_pops[k]] = sof_art_stars[k]
    del sof_art_stars[k]
sof_art_stars = dicf.groups(sof_art_stars, group="index", verbose=True)

# %%
import KapteynClustering.cluster_funcs as clusterf
features = ["E", "Lz", "Lp"]
sof_art_X = clusterf.art_find_X(features,sof_art_stars)

# %%
print(np.shape(sof_art_X))

# %%
my_E = sof_art_stars[0]["E"]

# %%
sof_E = df_artificial[df_artificial.index==0].En.values

# %%
print( (my_E == sof_E).sum()/len(my_E))

# %%

# %%
