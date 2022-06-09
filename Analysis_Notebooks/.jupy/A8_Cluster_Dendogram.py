# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.cluster_funcs as clusterf

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps 

# %% [markdown]
# # Load

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]

# %%
fit_data = dataf.read_data(fname= data_p["gaussian_fits"])
[Clusters, Pops, mean, covar] = [ fit_data[p] for p in
                                           ["Clusters", "Pops", "mean", "covariance"]]

# %%
C_Colours = ps.get_G_colours(Clusters)
C_Colours_list = np.array([C_Colours[g] for g in Clusters])

# %% [markdown]
# # RESULTS

# %%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# %%
from KapteynClustering import grouping_funcs as groupf


# %%
def Dendogram_plot(Z, Clusters, cut=4):
    def llf(id):
        '''For labelling the leaves in the dendrogram according to cluster index'''
        return str(Clusters[id])

    plt.subplots(figsize=(20,10))
    d_result = dendrogram(Z, leaf_label_func=llf, leaf_rotation=90, color_threshold=cut)
    plt.minorticks_off()
    return d_result


# %%
D, dm = groupf.get_cluster_distance_matrix(Clusters, cluster_mean=mean, cluster_cov=covar)

# %%
Z = linkage(dm, 'single')

# %%
cut = 4
n_cut = np.where(Z[:,2]>cut)[0][0]
Fig, axs = plt.subplots(nrows=2, figsize=(8,8),sharey=True)

axs[0].plot(Z[:,2])
axs[0].axhline(cut, c="r")
axs[0].axvline(n_cut, c="purple")
axs[0].set_ylabel("Mahalanobis Dis")

axs[1].plot(np.diff(Z[:,2]))
axs[1].axvline(n_cut, c="purple")
axs[1].set_ylabel("Change in Mahalanobis Dis")
for ax in axs:
    ax.set_xlim([0,len(Z)])

plt.show()

# %%
d_result = Dendogram_plot(Z,Clusters)
plt.title(f'Relationship between 3$\\sigma$ clusters')
plt.show()

# %%
print(d_result.keys())

# %% [markdown]
# # Labelled Stars

# %%
stars = vaex.open(data_p["labelled_sample"])
labels = stars["label"].values

# %%
Grouped_Clusters = fcluster(Z,t=4, criterion="distance")


# %%
def merge_clusters_labels(Clusters, Grouped_Clusters, labels):
    groups = -np.ones_like(labels)
    for c,g in zip(Clusters, Grouped_Clusters):
        c_filt = labels==c
        groups[c_filt] = g
    return groups
groups = merge_clusters_labels(Clusters, Grouped_Clusters, labels)    


# %%
Groups = np.unique(groups)

# %%
import KapteynClustering.plot_funcs as pf
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"],
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
pf.plot_simple_scatter(stars,xy_keys, Groups, groups)
plt.show()

pf.plot_simple_scatter(stars,xy_keys, Clusters, labels)
plt.show()


# %%
