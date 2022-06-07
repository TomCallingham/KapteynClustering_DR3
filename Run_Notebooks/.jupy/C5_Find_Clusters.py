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

# %%
cluster_data = dataf.read_data(fname=data_p["cluster"])
Z = cluster_data["Z"]

# %%
sig_data = dataf.read_data(fname=data_p["sig"])
sig = sig_data["significance"]

# %% [markdown]
# # Find Clusters

# %%
label_data =labelf.find_cluster_data(sig, Z, minimum_significance=min_sig)

[labels, star_sig, Clusters, cPops, C_sig] = [ label_data[p] for p in
                                           ["labels","star_sig", "Clusters", "cPops", "C_sig"]]

# %% [markdown]
# # Save
# ## Save indendant labels

# %%
dataf.write_data(fname= data_p["label"], dic=label_data, verbose=True, overwrite=True)

# %% [markdown]
# ## Save Labled Sample?

# %%
stars = vaex.open(data_p["sample"])
stars["label"] = labels
stars["cluster_sig"] = star_sig

stars.export(data_p["labelled_sample"])

# %% [markdown]
# # RESULTS

# %%
N_fluff = cPops[Clusters==-1]
N_clustered = len(labels) - N_fluff
f_clustered = N_clustered/len(labels)

print(f"Fraction clustered: {f_clustered}")

# %%
plt.figure(figsize=(8,8))
plt.scatter(cPops[C_sig>min_sig], C_sig[C_sig>min_sig])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Clusters[C_sig>min_sig], cPops[C_sig>min_sig], C_sig[C_sig>min_sig]):
    plt.text(p,s,g, size=15)
plt.show()
