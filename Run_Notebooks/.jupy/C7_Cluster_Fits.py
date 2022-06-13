# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.linkage_funcs as linkf

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps

# %% [markdown]
# ### Params

# %%
params = paramf.read_param_file("default_params.yaml")
print(params.keys())
data_p = params["data"]
min_sig=params["cluster"]["min_sig"]
features = params["linkage"]["features"]

# %%

# %% [markdown]
# # Load

# %%
stars = vaex.open(data_p["labelled_sample"])

cluster_data = dataf.read_data(fname= data_p["clusters"])
[labels, fClusters, fPops, fC_sig] = [ cluster_data[p] for p in
                                           ["labels", "Clusters", "cPops", "C_sig"]]

# %%
Clusters, Pops, C_sig = fClusters[fClusters!=-1], fPops[fClusters!=-1], fC_sig[fClusters!=-1]

# %% [markdown]
# # Make Fits

# %%
print(len(Clusters))

# %%
mean_cluster, cov_cluster = linkf.fit_gaussian_clusters(stars, features, Clusters, labels)

# %% [markdown]
# ## Save Fits

# %%
fit_data = {"Clusters":Clusters, "Pops":Pops, "C_sig":C_sig,
            "mean":mean_cluster, "covariance":cov_cluster,
            "features":features}
dataf.write_data(fname=data_p["gaussian_fits"], dic=fit_data)

# %%
# To plot use Similar code to GC project on cosma!

# %% [markdown]
# # Plot Fits

# %%
import copy
from KapteynClustering.plot_funcs import plot_simple_fit



# %%
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"],
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
plot_simple_fit(fit_data, xy_keys, Clusters, background_stars=stars)
plt.show()
