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
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.artificial_data_funcs as adf
import vaex

# %% [markdown]
# ## Params

# %%
params = dataf.read_param_file("gaia_params.yaml")

data_p = params["data"]
pot_name =  data_p["pot_name"]

art_p = params["art"]
N_art, additional_props = art_p["N_art"], art_p["additional"]

cluster_p = params["cluster"]
scales, features = cluster_p["scales"], cluster_p["features"]

# %% [markdown]
# ## Load Data
# Load the initial dynamics dataset, not the toomre cut setection.

# %%
stars = vaex.open(data_p["base_dyn"])

# %% [markdown]
# # Artificial DataSet
# Calculates N_art number of artificial halos by shuffling galactic vx, vz (keeping pos + vy). Recalculates dynamics. Can also propogate forward additional properties of the initial dataset.

# %%
print(additional_props)

# %% tags=[]
art_stars = adf.get_shuffled_artificial_set(N_art, stars, pot_name, features_to_scale=features, scales=scales)

# %% [markdown]
# ## Save Artifical Dataset
# Note that we save and read this dataset with dataf.write_data and dataf.read_data. The artificial datasets can be different lengths, which vaex does not like. These functions write as hdf5 similar to vaex, but load all of the data into a nested dictionary and numpy array format.

# %%
dataf.write_data(fname=data_p["art"], dic=art_stars, verbose=True, overwrite=True)

# %% [markdown]
# # Plots comparing Shuffled Smooth dataset

# %%
print(art_stars.keys())
print(art_stars[0].keys())
print(art_stars[0]["En"])

# %%
from KapteynClustering.legacy import plotting_utils #, vaex_funcs
df_artificial = vaex.from_dict(art_stars[0])
df= stars

# %%
plotting_utils.plot_original_data(df)

# %% tags=[]
plotting_utils.plot_original_data(df_artificial)

# %%
art_stars0 = art_stars[0]
sample_stars = dataf.read_data(fname=data_p["sample"])

for x in ["En", "Lz", "Lperp", "circ"]:
    plt.figure()
    plt.hist(stars[x], label="Original", bins="auto",histtype="step")
    plt.hist(sample_stars[x], label="sample", bins="auto",histtype="step")
    plt.hist(art_stars0[x], label="art", bins="auto", histtype="step")
    plt.legend()
    plt.xlabel(x)
    plt.show()

# %%
