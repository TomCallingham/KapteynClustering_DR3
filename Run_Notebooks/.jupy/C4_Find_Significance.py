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
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import os
import KapteynClustering as KC

# %% [markdown]
# ## Paramaters
# Consistent across Notebooks

# %%
param_file = "gaia_params.yaml"
params = dataf.read_param_file(param_file)
data_p= params["data"]

# %% [markdown] tags=[]
# # PCA Fit
# multiprocessing struggles in notebook. Below launches the scripts with given parameters.

# %% tags=[]
if False: # Careful, time consuming! Better To Run in cmdline to see output
    module_folder = os.path.dirname(KC.__file__)
    script_file = module_folder + "/Run_Significance.py"
    run_cmd = f"python {script_file} {param_file}"
    print(run_cmd)
    os.system(run_cmd)
    # !python run_cmd
    print("Finished")

# %% [markdown]
# # Plots

# %%
sig_data = dataf.read_data(fname=data_p["sig"])

sig, region_count, art_region_count, art_region_count_std =\
sig_data["significance"], sig_data["region_count"], sig_data["art_region_count"], sig_data["art_region_count_std"]

# %%
plt.figure()
plt.hist(region_count, bins="auto",density=True, label="fit", histtype="step")
plt.hist(art_region_count, bins="auto",density=True, label= "art")
plt.legend()
plt.show()

# %%
