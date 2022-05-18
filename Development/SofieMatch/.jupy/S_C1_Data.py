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

# %% [markdown]
# # Load Dyn

# %%
dyn_data = dataf.read_data(fname=data_params["base_dyn"], data_params=data_params)

# %% [markdown] tags=[]
# # Toomre Sample
# Apply the vtoomre>210

# %%
sample_data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = sample_data["stars"]

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
import vaex
result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
sof_df = vaex.open(f'{result_path}df.hdf5')

# %%
plotting_utils.plot_original_data(sof_df)

# %%
translate = {"E":"En"}
for x in ["E", "Lz", "Lp", "circ"]:
    my_x = stars[x]
    original_x = stars["original_" + translate.get(x,x)]
    plt.figure()
    plt.scatter(my_x, original_x)
    plt.xlabel(x)
    plt.ylabel("original" + x)
    plt.show()

    plt.figure()
    plt.scatter(my_x, original_x/my_x)
    plt.xlabel(x)
    plt.ylabel("original / me" + x)
    plt.show()

    plt.figure()
    plt.scatter(my_x, original_x - my_x)
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

# %%
