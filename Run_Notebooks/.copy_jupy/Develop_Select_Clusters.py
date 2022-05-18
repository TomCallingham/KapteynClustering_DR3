# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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
import time
from data_funcs import load_cluster, result_path, read_auriga_data
from cluster_funcs import find_tree, find_X, art_find_X
from TomScripts import dic_funcs as dicf
np.random.seed(0)

# %% [markdown]
# # Params 

# %%
Halo_n = 5
Level = 4

# smallest cluster size that we are interested in checking (for computational efficiency)
min_members = 10
# smallest cluster size that we are interested in checking (for computational efficiency)
max_members = 25000
N_sigma_ellipse_axis = 3.12  # Length (in standard deviations) of axis of ellipsoidal cluster boundary
features = ['scaled_En', 'scaled_Lperp', 'scaled_Lz']  # , 'circ'] #Clustering features
N_art = 10

fname = "toomre_"

# %%
c_data = load_cluster(Halo_n=Halo_n, Level=Level, fname="toomre_")
stars = c_data["stars"]
Z = c_data["cluster"]["Z"]
N_clusters = len(Z[:, 0])
N_clusters1 = N_clusters + 1


# %%

def load_pca(Halo_n=5, Level=4, fname="toomre_"):
    save_name = "pca_" + fname + f"Lv{Level}Au{Halo_n}Stars"
    pca_data = dicf.h5py_load(result_path + save_name)
    data = read_auriga_data(fname=fname, Halo_n=Halo_n, Level=Level)
    region_count = pca_data["region_count"]
    art_region_count = pca_data["art_region_count"]
    art_region_count_std = pca_data["art_region_count_std"]

    significance = (region_count - art_region_count) / np.sqrt(region_count + (art_region_count_std**2))
    pca_data["significance"] = significance
    data["pca"] = pca_data
    return data


# %% [markdown]
# ## Load 

# %%
data = load_pca(Halo_n=Halo_n, Level=4, fname="toomre_")

# %%
pca_data = data["pca"]
significance = pca_data["significance"]

# %% tags=[]
from tqdm.notebook import tqdm


def select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
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
    T = time.time()
    print('Picking out the clusters with maximum significance from the tree...')
    
    N = len(significance)

    index = np.arange(len(significance))
    sig_filt = (significance > minimum_significance)

    s_significance = significance[sig_filt]
    s_index = index[sig_filt]

    N = len(significance) + 1
    max_i = np.max(s_index)

    selected_i = 0

    traversed = []  # list of indices i that we have already traversed
    selected = []  # list of merges in the tree where the significance is maximized
    max_sign = []  # the significance at the selected step

    argsort_sig = np.argsort(s_significance)[::-1]
    sort_s_significance = s_significance[argsort_sig]
    sort_s_index = s_index[argsort_sig]

    Z_index1 = Z[:, 0]
    Z_index2 = Z[:, 1]

    for m,(merge_i, merge_sig) in enumerate(zip(sort_s_index, sort_s_significance)):
        # print(m, len(sort_s_index), len(traversed))
        if merge_i in traversed:
            # print("Instant Traverse")
            pass
        else:
            # Traverse over the significant clusters, starting with the most significant ones.
            # print(f"merge i: {merge_i}, sig: {merge_sig}")
            # print(len(traversed), N)

            traversed_currentpath = []
            maxval = 0
            sig = merge_sig
            i = merge_i

            visited = 0

            # traverse up the whole tree
            while(i <= max_i):

                # only go up the tree, in case we have not already examined this path and found that the
                # max significance was already found at a lower level.
                if(i not in traversed):

                    # Track what we have traversed since the max-significance
                    traversed_currentpath.append(i)

                    if(sig > maxval):
                        if maxval!=0:
                            print("CHANGED! FOUND HIGHER!")
                        maxval = sig
                        selected_i = i
                        traversed_currentpath = []
                        # traversed_currentpath = [i] #CHANGE

                    next_filt = np.logical_or((Z_index1 == i + N), Z_index2 == i + N)
                    if next_filt.sum() == 0:
                        print("Broken! No next?")
                        print("Current:", i, sig, visited)
                        visited=1
                        break
                    i = index[next_filt]
                    sig = significance[next_filt]
                    visited = 1
                    # print(f"new i: {i}, sig: {sig}")

                else:  # we already traversed this path, so exit
                    # print(f"traversed")
                    break
            # print("While escape")
            # print(i, sig)
            # print(selected_i, maxval)

            # It means that there was some part of this path which was not yet traversed.
            if((maxval > 0) and (visited == 1)):
                # print("NEW, adding!")
                # print(traversed_currentpath)
                traversed.extend(traversed_currentpath)

                # Same cluster might have been marked the most significant one for multiple paths.
                if(selected_i not in selected):
                    selected.append(selected_i)
                    max_sign.append(maxval)
                else:
                    print("DUPLICATE")
            else:
                print("Not added?", maxval, visited)
    print("Finished")
    print((time.time()-T)/60)
    return np.array(selected), np.array(max_sign)

old_selected, old_max_sign = select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
print(len(old_selected))
print(old_selected)
print(old_max_sign)

# %%
print(np.isin(f_selected,old_selected).sum()/len(f_selected))
print(np.isin(old_selected,f_selected).sum()/len(old_selected))

# %%
f_missing_filt = np.isin(f_selected, old_selected,invert=True, assume_unique=True)
print(f_missing_filt.sum())
print(f_max_sign[f_missing_filt])

# %%
old_missing_filt = np.isin(old_selected, f_selected,invert=True, assume_unique=True)
print(old_missing_filt.sum())
print(old_max_sign[old_missing_filt])

# %%
import copy


# %% tags=[]
def slow_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
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
    T = time.time()
    print('Picking out the clusters with maximum significance from the tree...')
    N = len(significance)

    index = np.arange(len(significance))


    N = len(significance) + 1

    selected_i = 0

    traversed = []  # list of indices i that we have already traversed
    selected = []  # list of merges in the tree where the significance is maximized
    max_sign = []  # the significance at the selected step

    argsort_sig = np.argsort(significance)[::-1]
    
    
    sort_significance = significance[argsort_sig]
    sort_index = index[argsort_sig]
    
    sort_sig_filt = (sort_significance > minimum_significance)
    
    max_i = np.max(sort_index[sort_sig_filt])
    
    Z_index1 = Z[:,0]
    Z_index2 = Z[:,1]

    # sort_Z_index1 = Z[:,0][argsort_sig]
    # sort_Z_index2 = Z[:,1][argsort_sig]
    
    traversed = []
    # untraversed_filt = np.ones_like(s_index, dtype=bool)
    untraversed_filt = sort_sig_filt
    N_untraversed = untraversed_filt.sum()
    while N_untraversed>0:
        
        i = sort_index[untraversed_filt][0]
        sig  = sort_significance[untraversed_filt][0]
        # untraversed_filt[np.where(untraversed_filt)[0][0]] = False
        # print(i,sig)

        # Traverse over the significant clusters, starting with the most significant ones.
        # print(f"merge i: {merge_i}, sig: {merge_sig}")
        # print(len(traversed), N)

        traversed_currentpath = []
        maxval = 0
        # traverse up the whole tree
        while(i <= max_i):

            # only go up the tree, in case we have not already examined this path and found that the
            # max significance was already found at a lower level.
            # Track what we have traversed since the max-significance
            traversed_currentpath.append(i)

            if(sig > maxval):
                if maxval>0:
                    print("Better path, dropping old?")
                maxval = sig
                selected_i = i
                # traversed_currentpath = []
                traversed_currentpath = [i] #CHANGED

            # next_filt = np.logical_or((sort_Z_index1 == i + N), sort_Z_index2 == i + N)
            next_filt = np.logical_or((Z_index1 == i + N), Z_index2 == i + N)
            
            if next_filt.sum() == 0:
                print("Broken! No next?")
                print("Current:", i, sig)
                break
            
            if not untraversed_filt[next_filt[argsort_sig]]:
            # if not untraversed_filt[next_filt]:
                # print("Traversed")
                break
                
            i = index[next_filt][0]
            sig = significance[next_filt][0]

        traversed.extend(traversed_currentpath)

        # Same cluster might have been marked the most significant one for multiple paths.
        # Not with ChangEto including on traversed path!
        selected.append(selected_i)
        max_sign.append(maxval)
            
        untraversed_filt[np.where(untraversed_filt)[0]] = np.isin(sort_index[untraversed_filt],traversed_currentpath,invert=True)
        N_untraversed = untraversed_filt.sum()
        # print(f"{len(traversed_currentpath)} searched, {len(traversed)} total searched, {N_untraversed} to explore")
    print("FINISHED!")
            
    dt = time.time() - T            
    print(f"Taken {dt} s")
    return np.array(selected), np.array(max_sign)


# %% tags=[]
def fast_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
    '''
    Takes all (possibly hierarchically overlapping) significant clusters from the
    single linkage merging process and determines where each nested structure reaches its
    maximum statistical significance in the dendrogram (or merging tree). Returns the indices
    of these maximum significance clusters such that we can extract flat cluster labels
    for the star catalogue.

    Parameters:
    significance
    Z
    min_significance that we accept for each cluster

    Returns:
    selected(np.array): The indices (i in stats) of the clusters that have been selected from the tree
    max_sign(np.array): The statistical significance of each selected cluster

    '''
    T = time.time()
    
    print('Picking out the clusters with maximum significance from the tree...')
    N = len(significance)
    index = np.arange(N)

    argsort_sig = np.argsort(significance)[::-1]
    sort_significance = significance[argsort_sig]
    sort_index = index[argsort_sig]

    sort_Z_index1 = Z[:,0][argsort_sig]
    sort_Z_index2 = Z[:,1][argsort_sig]
    index_dic = {}
    index_dic = {sort_Z_index1[n]:n for n in range(N)}|{sort_Z_index2[n]:n for n in range(N)}

    traversed = []  # list of indices i that we have already traversed
    selected = []  # list of merges in the tree where the significance is maximized
    max_sign = []  # the significance at the selected step
    
    untraversed_filt = (sort_significance > minimum_significance)
    N_untraversed = untraversed_filt.sum()
    
    while N_untraversed>0:

        i = sort_index[untraversed_filt][0]
        sig  = sort_significance[untraversed_filt][0]

        selected.append(i)
        max_sign.append(sig)

        traversed_currentpath = []
        while True :# (i <= max_i):

            # only go up the tree, in case we have not already examined this path and found that the
            # max significance was already found at a lower level.
            # Track what we have traversed since the max-significance
            traversed_currentpath.append(i)

            try:
                next_index = index_dic[i+N+1]
            except:
                break
            if not untraversed_filt[next_index]:
                # print("Traversed")
                break

            i = sort_index[next_index]
            # sig = sort_significance[next_index]


        #Add New Path
        traversed.extend(traversed_currentpath)
        untraversed_filt[np.where(untraversed_filt)[0]] = np.isin(sort_index[untraversed_filt],traversed_currentpath,invert=True)
        N_untraversed-= len(traversed_currentpath)  #= untraversed_filt.sum() #FASTER
        
        # print(f"{len(traversed_currentpath)} searched, {len(traversed)} total searched, {N_untraversed} to explore")
    print("FINISHED!")

    dt = time.time() - T
    print(f"Taken {dt} s")

    return np.array(selected), np.array(max_sign)

# %% tags=[]
selected, max_sign = select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
print(len(selected))
print(selected)
print(max_sign)

# %% tags=[]
f_selected, f_max_sign = fast_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
print(len(f_selected))
print(len(f_max_sign))
print(f_selected)
print(f_max_sign)

# %% tags=[]
f_selected, f_max_sign = fast_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
print(len(f_selected))
print(f_selected)
print(f_max_sign)

# %%

# %%
%load_ext line_profiler

# %% tags=[]
%lprun -f   fast_select_maxsig_clusters_from_tree fast_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)


# %% [markdown]
# ## OLD

# %%
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
    N = stats.count() + 1

    selected_i = 0

    traversed = []  # list of indices i that we have already traversed
    selected = []  # list of merges in the tree where the significance is maximized
    max_sign = []  # the significance at the selected step

    array = significant['significance', 'i'].values
    array = np.flip(array[array[:, 0].argsort()])  # Sort according to significance

    # Traverse over the significant clusters, starting with the most significant ones.
    for merge_idx in tqdm(array[:, 0]):

        traversed_currentpath = []
        maxval = 0
        row = significant[significant.i == merge_idx]
        i = row.i.values[0]
        significance = row.significance.values[0]
        visited = 0

        # traverse up the whole tree
        while(i <= max_i):

            # only go up the tree, in case we have not already examined this path and found that the
            # max significance was already found at a lower level.
            if(i not in traversed):

                # Track what we have traversed since the max-significance
                traversed_currentpath.append(i)

                if(significance > maxval):
                    maxval = significance
                    selected_i = i
                    traversed_currentpath = []

                next_cluster = stats[(stats.index1 == i + N) | (stats.index2 == i + N)]
                i = next_cluster.i.values[0]
                significance = next_cluster.significance.values[0]
                visited = 1

            else:  # we already traversed this path, so exit
                break

        # It means that there was some part of this path which was not yet traversed.
        if((maxval > 0) and (visited == 1)):
            traversed.extend(traversed_currentpath)

            # Same cluster might have been marked the most significant one for multiple paths.
            if(selected_i not in selected):
                selected.append(selected_i)
                max_sign.append(maxval)

    return np.array(selected), np.array(max_sign)


# %% [markdown]
# ## Find Clusters

# %%
'''Select all clusters that satisfy your chosen minimum statistical significance level.
To distinguish between hierarchically overlapping significant clusters when extracting the final 
cluster labels, select the cluster which displays the maximum statistical significance.'''

selected, statistical_significance = cluster_utils.select_maxsig_clusters_from_tree(
    stats, minimum_significance=N_sigma_significance)

maxsigclusters = stats.take(selected)
labels, significance_list = cluster_utils.get_cluster_labels_with_significance(maxsigclusters, Z)
df['labels'] = labels
df['maxsig'] = significance_list

plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %%
'''Add labels for subgroup membership in velocity space. Requires having the hdbscan library installed.'''

import velocity_space

if('vR' not in df.get_column_names()):
    # in case the polar coordinate velocities are missing, re-run it trough the get_gaia function
    df = importdata.get_GAIA(df)

df = velocity_space.apply_HDBSCAN_vspace(df)

# Plot an example: The subgroups for cluster nr 1
plotting_utils.plot_subgroup_vspace(df, clusterNr=2, savepath=None)

# %%
'''Compute membership probability for each star belonging to a cluster. Current method does not use XDGMM.'''

import membership_probability

df = cluster_utils.scaleData(df, features_to_be_scaled, minmax_values)
df = membership_probability.get_membership_indication_PCA(df, features=features)

# Plot example: Cluster nr 1 color coded by membership probability
plotting_utils.plot_membership_probability(df, clusterNr=2, savepath=None)

# %%
'''Get a probability matrix of any star belonging to any cluster'''

probability_table = membership_probability.get_probability_table_PCA(df, features=features)
probability_table['source_id'] = df.source_id.values

# %%
'''If desired: Export result catalogue and probability table'''

df.export_hdf5(f'{result_path}df_labels.hdf5')
probability_table.export_hdf5(f'{result_path}probability_table.hdf5')

# %%
