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
[Clusters, Pops, mean, covar] = [ fit_data[p] for p in
                                           ["Clusters", "Pops", "mean", "covariance"]]

# %%
print(fit_data.keys())

# %%
C_Colours = ps.get_G_colours(Clusters)
C_Colours_list = np.array([C_Colours[g] for g in Clusters])

# %% [markdown]
# # RESULTS

# %%
from KapteynClustering import mahalanobis_funcs as mahaf


# %%
def get_cluster_distance_matrix(Clusters, cluster_mean, cluster_covar):
    N_clusters = len(Clusters)
    n_dim = len(cluster_mean[Clusters[0]])
    X = np.array([cluster_mean[c] for c in Clusters])
    dis = mahaf.maha_dis_to_clusters(X, Clusters, cluster_mean, cluster_covar)
    return dis


# %%
D = get_cluster_distance_matrix(Clusters, cluster_mean=mean, cluster_covar=covar)

# %%
print(np.shape(D))

# %%
from scipy.cluster.hierarchy import linkage, dendrogram

# %%

# %%


# Perform single linkage with the custom made distance matrix
Z = linkage(np.sqrt(D), 'single')

def llf(id):
    '''For labelling the leaves in the dendrogram according to cluster index'''
    return str(Clusters[id])

plt.subplots(figsize=(20, 5))
dendrogram(Z, leaf_label_func=llf, leaf_rotation=90, color_threshold=6)
plt.title(f'Relationship between 3$\\sigma$ clusters')
plt.show()



# %%
