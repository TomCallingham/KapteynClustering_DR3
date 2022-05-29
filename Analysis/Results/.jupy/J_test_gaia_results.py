# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %% [markdown]
# # Load

# %%
params = paramf.read_param_file("../Params/J_gaia_params.yaml")
data_p = params["data"]
min_sig=params["label"]["min_sig"]

stars = vaex.open(data_p["sample"])

label_data = dataf.read_data(fname= data_p["label"])
[labels, star_sig, Groups, Pops, G_sig] = [ label_data[p] for p in
                                           ["labels","star_sig", "Groups", "Pops", "G_sig"]]

# %% [markdown]
# # RESULTS

# %%
print(len(Groups))

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

# %% [markdown]
# ## Plots

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter

# %%
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"], 
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
plot_simple_scatter(stars, xy_keys, Groups, labels)
plt.show()

# %%
xy_keys =( ["JR", "Jz", "Lz"], ["En"]) 
         
plot_simple_scatter(stars, xy_keys, Groups, labels)
plt.show()

# %%
