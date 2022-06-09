# %% [markdown]
# # Setup

# %%
import matplotlib.pyplot as plt
import numpy as np
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import os
import KapteynClustering as KC

# %% [markdown]
# ## Paramaters
# Consistent across Notebooks

# %%
param_file = "default_params.yaml"
params = paramf.read_param_file(param_file)
data_p= params["data"]

# %% [markdown]
# # Plots

# %%
sig_data = dataf.read_data(fname=data_p["sig"])

sig, region_count, art_region_count, art_region_count_std =\
sig_data["significance"], sig_data["region_count"], sig_data["art_region_count"], sig_data["art_region_count_std"]

# %%
print((~np.isfinite(sig)).sum())
plt.figure()
plt.hist(sig[np.isfinite(sig)], bins="auto",density=True)
plt.xlabel("Significance")
plt.show()

# %%
