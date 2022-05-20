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
# # TODO
#
# - [ ] Resample or scale number density?
# - [ ] Compare Artifical Halos

# %% [markdown]
# # Paramaters
# params are given in .yaml file

# %% [markdown]
# # NOTEBOOKS

# %% [markdown]
# ## Initial Dataset and selections
# ### Auriga Put into format
# [Data_Create](./C1_Auriga_Data_Vaex.ipynb)

# %% [markdown]
# ## Create Artifical DataSet
# [Artificial Data ](./C2_Artificial_Data.ipynb)

# %% [markdown]
# ## Single Linkage Cluster
# [Peform Single_Linkage](./C3_Single_Linkage.ipynb)

# %% [markdown]
# ## Calculate Significance
# [Find Significance with PCA ](./C4_PCA_Analysis.ipynb)

# %% [markdown]
# ## Significance Analysis
# [Find how Signficant Clusters Are](./C5_Significance_Analysis.ipynb)

# %% [markdown]
# ## Analysis

# %% [markdown]
# #  INTRO
# This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
# Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.
#
#
# The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.
# # Running Kapetyn Clustering
# Implentation of Lövdal et al. (2021).
# Can just import KapetynCluster module
# Notebooks below take through the entire process
