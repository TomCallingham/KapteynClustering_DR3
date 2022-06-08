# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf
import KapteynClustering.cluster_funcs as clusterf

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps

# %% [markdown]
# ### Params

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
min_sig=params["label"]["min_sig"]
features = params["cluster"]["features"]

# %%

# %% [markdown]
# # Load

# %%
stars = vaex.open(data_p["labelled_sample"])

label_data = dataf.read_data(fname= data_p["label"])
[labels, fClusters, fPops, fC_sig] = [ label_data[p] for p in
                                           ["labels", "Clusters", "cPops", "C_sig"]]

# %%
Clusters, Pops, C_sig = fClusters[fClusters!=-1], fPops[fClusters!=-1], fC_sig[fClusters!=-1]

# %% [markdown]
# # Make Fits

# %%
print(len(Clusters))

# %%
mean_cluster, cov_cluster = clusterf.fit_gaussian_clusters(stars, features, Clusters, labels)

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
# # Adding Stars To Clusters

# %%
from KapteynClustering import mahalanobis_funcs as mahaf
from KapteynClustering import cluster_funcs as clusterf

# %%
test_stars = dataf.read_data(data_p["art"])[5]

# %%
test_labels = mahaf.add_maha_members_to_clusters(test_stars, data_p["gaussian_fits"], max_dis=2.13, plot=True)

# %% [markdown]
# # Plots

# %%
C_Colours = ps.get_G_colours(Clusters)
C_Colours_list = np.array([C_Colours[g] for g in Clusters])

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter

# %%
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"],
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
plot_simple_scatter(test_stars, xy_keys, Clusters, test_labels)
plt.show()

# %%
