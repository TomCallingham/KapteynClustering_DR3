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
params = dataf.read_param_file("../../Run_Notebooks/gaia_params.yaml")
data_params = params["data"]
result_folder = data_params["result_folder"]

# %% [markdown]
# # Load Me
# Dyn and Toomre

# %%
dyn_data = dataf.read_data(fname=result_folder + data_params["base_dyn"])
dyn_stars = dyn_data["stars"]

sample_data = dataf.read_data(fname=result_folder+data_params["sample"])
stars = sample_data["stars"]

# %%
import KapteynClustering.legacy.load_sofie_data as lsd
sof_stars = lsd.load_sof_df()

# %% [markdown]
# # Plots
# Plots of initial Sample

# %%
from KapteynClustering.legacy.vaex_funcs import vaex_from_dict
from KapteynClustering.legacy import plotting_utils
df = vaex_from_dict(stars)

# %%
plotting_utils.plot_original_data(df)

# %%
import vaex
result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
sof_df = vaex.open(f'{result_path}df.hdf5')

# %%
plotting_utils.plot_original_data(sof_df)

# %% [markdown]
# #  COMPARE

# %%
for x in ["En", "Lz", "Lperp", "circ", "x", "y", "z", "vx", "vy", "vz"]:
    plt.figure()
    if x in ["Lz", "circ"]:
        plt.hist(-stars[x], bins=100, density=True, label="Me")
    else:
        plt.hist(stars[x], bins=100, density=True, label="Me")
    plt.hist(sof_stars[x], bins=100, density=True, label="Sof", histtype="step", zorder=10)
    plt.xlabel(x)
    plt.legend()
    plt.show()

# %%
