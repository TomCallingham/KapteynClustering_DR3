# %% [markdown]
# # Setup

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.cluster_funcs as clusterf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
# params = dataf.read_param_file("sof_check_params.yaml")
data_p = params["data"]
min_sig = params["cluster"]["min_sig"]

# %% [markdown]
# ## Load

# %%
link_data = dataf.read_data(fname=data_p["linkage"])
Z = link_data["Z"]

# %%
print(Z[:,2])

# %%
plt.figure()
plt.plot(Z[:,2])
plt.show()

# %%
sig_data = dataf.read_data(fname=data_p["sig"])
sig = sig_data["significance"]

# %% [markdown]
# # LoadData

# %%
cluster_data = dataf.read_data(fname= data_p["clusters"] )
[labels, star_sig, Clusters, cPops, C_sig] = [ cluster_data[p] for p in
                                           ["labels","star_sig", "Clusters", "cPops", "C_sig"]]

# %%
stars = vaex.open(data_p["labelled_sample"])

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

# %%

# %%
