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
# # Setup

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import os
import KapteynClustering as KC

# %% [markdown]
# ## Paramaters
# Consistent across Notebooks

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_params = params["data"]

# %% [markdown] tags=[]
# # PCA Fit
# multiprocessing struggles in notebook. Below launches the scripts with given parameters.

# %%
# If on cosma, sbatch this!

# if True:
#     if cosma:
#         !sbatch "Batch/"
#     elif gaia2:
#         !python "../KapteynClustering/KapteynClustering/Run_Significance.py"

# %%
if False: # Careful, time consuming!
    module_folder = os.path.dirname(KC.__file__)
    script_file = module_folder + "/Run_Significance.py"
    os.system(f"python {script_file} {param_file}")
    print("Finished")

# %% [markdown]
# # Plots

# %%
sig_data = dataf.read_data(fname=data_params["result_folder"] +data_params["sig"])

sig, region_count, art_region_count, art_region_count_std =\
sig_data["significance"], sig_data["region_count"], sig_data["art_region_count"], sig_data["art_region_count_std"]

# %%
plt.figure()
plt.hist(region_count, bins="auto",density=True, label="fit", histtype="step")
plt.hist(art_region_count, bins="auto",density=True, label= "art")
plt.legend()
plt.show()

# %%
