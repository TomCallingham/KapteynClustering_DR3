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

label_data = dataf.read_data(fname= data_p["label"])
[labels, star_sig, Clusters, cPops, C_sig] = [ label_data[p] for p in
                                           ["labels","star_sig", "Clusters", "cPops", "C_sig"]]

# %%
fit_data = dataf.read_data(fname= data_p["gaussian_fits"])

# %%
print(fit_data.keys())

# %% [markdown]
# # RESULTS

# %%
C_Colours = ps.get_C_colours(Clusters)
C_Colours_list = np.array([C_Colours[g] for g in Clusters])

# %%
N_fluff = Pops[Groups==-1]
N_grouped = len(labels) - N_fluff
f_grouped = N_grouped/len(labels)

print(f"Fraction Grouped: {f_grouped}")

# %%
plt.figure(figsize=(8,8))
plt.scatter(Pops[G_sig>min_sig], G_sig[G_sig>min_sig],c=G_Colours_list[G_sig>min_sig])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Groups[G_sig>min_sig], Pops[G_sig>min_sig], G_sig[G_sig>min_sig]):
    plt.text(p,s,g, size=15)
plt.show()

# %% [markdown]
# ## Plots

# %%
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"], 
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
plot_simple_scatter(stars, xy_keys, Groups, labels)
plt.show()

# %%
xy_keys = (["Lz", "Lperp"], ["En"])
           
plot_simple_scatter(stars, xy_keys, Groups, labels)
plt.show()

# %%
xy_keys = (["vx", "vy"], ["En"])
           
plot_simple_scatter(stars, xy_keys, Groups, labels)
plt.show()

# %%
xy_keys = [["l", "b"]]
print(np.shape(xy_keys))
           
plot_simple_scatter(stars, xy_keys, Groups, labels)
plt.show()

# %%
print(stars.column_names)

# %%
