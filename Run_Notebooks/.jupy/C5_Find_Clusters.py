# %% [markdown]
# # Setup

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
# params = dataf.read_param_file("sof_check_params.yaml")
data_p = params["data"]
min_sig = params["label"]["min_sig"]

# %% [markdown]
# ## Load

# %%
stars = vaex.open(data_p["sample"])

# %%
cluster_data = dataf.read_data(fname=data_p["cluster"])
Z = cluster_data["Z"]

# %%
sig_data = dataf.read_data(fname=data_p["sig"])
sig = sig_data["significance"]

# %% [markdown]
# # Find Clusters

# %%
label_data =labelf.find_group_data(sig, Z, minimum_significance=min_sig)

[labels, star_sig, Groups, Pops, G_sig] = [ label_data[p] for p in
                                           ["labels","star_sig", "Groups", "Pops", "G_sig"]]

# %% [markdown]
# ## Save

# %%
dataf.write_data(fname= data_p["label"], dic=label_data, verbose=True, overwrite=True)

# %% [markdown]
# # RESULTS

# %%
N_fluff = Pops[Groups==-1]
N_grouped = len(labels) - N_fluff
f_grouped = N_grouped/len(labels)

print(f"Fraction Grouped: {f_grouped}")

# %%
plt.figure(figsize=(8,8))
plt.scatter(Pops[G_sig>min_sig], G_sig[G_sig>min_sig])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Groups[G_sig>min_sig], Pops[G_sig>min_sig], G_sig[G_sig>min_sig]):
    plt.text(p,s,g, size=15)
plt.show()
