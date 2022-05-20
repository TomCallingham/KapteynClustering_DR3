# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:py39]
#     language: python
#     name: conda-env-py39-py
# ---

# %% [markdown]
# This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
# Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.
#
#
# The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

from params import data_params

# %%
import time

# %%
N_sigma_significance = 3

# %%
%load_ext line_profiler

# %% [markdown]
# # Params

# %%
data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = data["stars"]

# %%
cluster_data = dataf.read_data(fname=data_params["cluster"], data_params=data_params)
Z = cluster_data["Z"]

# %%
sig_data = dataf.read_data(fname=data_params["sig"], data_params=data_params)
significance = sig_data["significance"]

# %%
tree_members = clusterf.find_tree(Z, prune=True)

# %%
N_clusters = len(Z[:, 0])
N_clusters1 = N_clusters + 1

# %%
import vaex
import cluster_utils
stats_dic = {"significance": significance,
             "i": np.arange(len(Z)),
             "index1": Z[:, 0],
             "index2": Z[:, 1],
             }
stats = vaex.from_dict(stats_dic)

# %% [markdown]
# # Select MaxSig Clusters

# %% [markdown]
# ## My MaxSig Cluster

# %% tags=[]
selected, statistical_significance = clusterf.select_maxsig_clusters_from_tree(
    significance, Z, minimum_significance=N_sigma_significance)

# %% tags=[]
print(selected)
print(len(selected))
print(statistical_significance)

# %% [markdown]
# ## Get Cluster Labels
# ### My Cluster Labels

# %%
labels, significance_list = clusterf.get_cluster_labels_with_significance(
    selected, statistical_significance, tree_members, N_clusters)

# %% tags=[]
%lprun -f  clusterf.get_cluster_labels_with_significance clusterf.get_cluster_labels_with_significance( selected, statistical_significance, tree_members, N_clusters)

# %% [markdown]
# ### Sofie Cluster Labels

# %%
maxsigclusters = stats.take(selected)
labels, significance_list = cluster_utils.get_cluster_labels_with_significance(maxsigclusters, Z)

# %%
%lprun -f  cluster_utils.get_cluster_labels_with_significance  cluster_utils.get_cluster_labels_with_significance(maxsigclusters, Z)


# %% [markdown]
# ## Sofie MaxSig Cluster

# %% tags=[]
def select_maxsig_clusters_from_tree(stats, minimum_significance=3):
    
    '''
    Takes all (possibly hierarchically overlapping) significant clusters from the
    single linkage merging process and determines where each nested structure reaches its
    maximum statistical significance in the dendrogram (or merging tree). Returns the indices
    of these maximum significance clusters such that we can extract flat cluster labels 
    for the star catalogue.
    
    Parameters:
    stats(vaex.DataFrame): Dataframe containing information from the single linkage merging process.
    minimum_significance: The smallest significance (sigma) that we accept for a cluster
    
    Returns: 
    selected(np.array): The indices (i in stats) of the clusters that have been selected from the tree
    max_sign(np.array): The statistical significance of each selected cluster
    
    '''
    print('Picking out the clusters with maximum significance from the tree...')
    
    significant = stats[stats.significance > minimum_significance]
    max_i = np.max(significant.i.values)
    N = stats.count()+1

    selected_i = 0

    traversed = [] #list of indices i that we have already traversed
    selected = [] #list of merges in the tree where the significance is maximized
    max_sign = [] #the significance at the selected step

    array = significant['significance', 'i'].values
    array = np.flip(array[array[:, 0].argsort()]) #Sort according to significance

    for merge_idx in array[:, 0]: #Traverse over the significant clusters, starting with the most significant ones.

            traversed_currentpath = []
            maxval = 0
            row = significant[significant.i == merge_idx]
            i = row.i.values[0]
            significance = row.significance.values[0]
            visited = 0

            #traverse up the whole tree
            while(i <= max_i):
                #only go up the tree, in case we have not already examined this path and found that the
                #max significance was already found at a lower level.
                if(i not in traversed):

                    traversed_currentpath.append(i) #Track what we have traversed since the max-significance

                    if(significance > maxval):
                        maxval = significance
                        selected_i = i
                        traversed_currentpath = []

                    next_cluster = stats[(stats.index1 == i+N) | (stats.index2 == i+N)]
                    try:
                        i = next_cluster.i.values[0]
                        significance = next_cluster.significance.values[0]
                        visited = 1
                    except Exception:
                        print("no next cluster, at the top!")
                        break

                else: #we already traversed this path, so exit
                    break

            if((maxval > 0) and (visited == 1)): #It means that there was some part of this path which was not yet traversed.
                traversed.extend(traversed_currentpath)
                
                if(selected_i not in selected): #Same cluster might have been marked the most significant one for multiple paths.
                    selected.append(selected_i)
                    max_sign.append(maxval)
                    
    return np.array(selected), np.array(max_sign)               



# %% tags=[]
T=time.time()
s_selected, s_statistical_significance = select_maxsig_clusters_from_tree(
    stats, minimum_significance=N_sigma_significance)
dt = time.time() -T
print(f"time taken: {dt/60} mins")

# %% [markdown]
# ## Compare

# %% tags=[]
print(selected)
print(s_selected)

# %%
print(len(selected))
print(len(s_selected))

# %%
print(len(np.unique(selected)))
print(len(np.unique(s_selected)))

# %% tags=[]
print(statistical_significance)
print(s_statistical_significance)

# %% [markdown]
# ## Time

# %% tags=[]
%lprun -f clusterf.select_maxsig_clusters_from_tree clusterf.select_maxsig_clusters_from_tree( significance, Z, minimum_significance=N_sigma_significance)

# %%
%lprun -f  cluster_utils.select_maxsig_clusters_from_tree cluster_utils.select_maxsig_clusters_from_tree( stats, minimum_significance=N_sigma_significance)

# %%
