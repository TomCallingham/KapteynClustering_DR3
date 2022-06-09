# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps

# %% [markdown]
# # Load

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
min_sig=params["label"]["min_sig"]

stars = vaex.open(data_p["labelled_sample"])

clusters_data = dataf.read_data(fname= data_p["clusters"])
[labels, star_sig, Clusters, cPops, C_sig] = [ clusters_data[p] for p in
                                           ["labels","star_sig", "Clusters", "cPops", "C_sig"]]

# %% [markdown]
# # RESULTS

# %%
print(len(Clusters))

# %%
C_Colours = ps.get_G_colours(Clusters)
C_Colours_list = np.array([C_Colours[g] for g in Clusters])

# %%
N_fluff = cPops[Clusters==-1]
N_clustered = len(labels) - N_fluff
f_clustered = N_clustered/len(labels)

print(f"Fraction clustered: {f_clustered}")

# %%
plt.figure(figsize=(8,8))
plt.scatter(cPops[C_sig>min_sig], C_sig[C_sig>min_sig],c=C_Colours_list[C_sig>min_sig])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Clusters[C_sig>min_sig], cPops[C_sig>min_sig], C_sig[C_sig>min_sig]):
    plt.text(p,s,g, size=15)
plt.show()

# %% [markdown]
# ## Plots

# %%
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"],
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
plot_simple_scatter(stars, xy_keys, Clusters, labels)
plt.show()

# %%
xy_keys = (["Lz", "Lperp"], ["En"])

plot_simple_scatter(stars, xy_keys, Clusters, labels)
plt.show()

# %%
xy_keys = (["vx", "vy"], ["En"])

plot_simple_scatter(stars, xy_keys, Clusters, labels)
plt.show()

# %%
xy_keys = [["l", "b"]]
plot_simple_scatter(stars, xy_keys, Clusters, labels)
plt.show()

# %%
print(stars.column_names)

# %%
