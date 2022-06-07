# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

import sys
sys.path.append("../Notebooks/")
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
stats_dic = {"significance": significance,
             "i": np.arange(len(Z)),
             "index1": Z[:, 0],
             "index2": Z[:, 1],
             }
stats = vaex.from_dict(stats_dic)

# %% [markdown]
# # ORIGNAL

# %%
import cluster_utils
import time

# %%
T = time.time()
s_selected, s_max_sign = cluster_utils.select_maxsig_clusters_from_tree(stats,minimum_significance=3)
print("Finished")
print((time.time()-T)/60)

# %%
print(len(s_selected))

# %%
print(len(s_selected))
print(s_selected)
print(s_max_sign)

# %% [markdown]
# # TEST

# %%
from tqdcluster_utilsebook import tqdm


def old_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
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
        if merge_i not in traversed:
            traversed_currentpath = []
            maxval = 0
            sig = merge_sig
            i = merge_i
            visited = 0
            while(i <= max_i):
                if(i not in traversed):
                    traversed_currentpath.append(i)
                    if(sig > maxval):
                        maxval = sig
                        selected_i = i
                        traversed_currentpath = []
                    next_filt = np.logical_or((Z_index1 == i + N), Z_index2 == i + N)
                    if next_filt.sum() == 0:
                        print("FOUND TOP")
                        print("Current:", i, sig, visited)
                        visited=1
                        break
                    i = index[next_filt]
                    sig = significance[next_filt]
                    visited = 1
                else:  # we already traversed this path, so exit
                    break
            if((maxval > 0) and (visited == 1)):
                traversed.extend(traversed_currentpath)

                # Same cluster might have been marked the most significant one for multiple paths.
                if(selected_i not in selected):
                    selected.append(selected_i)
                    max_sign.append(maxval)
                # else:
                #     print("DUPLICATE")
                #     print(maxval)
            else:
                print("Not added?", maxval, visited)
    print("Finished")
    print((time.time()-T)/60)
    return np.array(selected), np.array(max_sign)

old_selected, old_max_sign = old_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
print(len(old_selected))
print(old_selected)
print(old_max_sign)

# %%
from tqdm.notebook import tqdm


def old2_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
    T = time.time()
    print('Picking out the clusters with maximum significance from the tree...')
    
    N = len(significance)

    index = np.arange(len(significance))
    sig_filt = (significance > minimum_significance)

    s_significance = significance[sig_filt]
    s_index = index[sig_filt]

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
    index_dic = {}
    index_dic = {Z_index1[n]: n for n in range(N)} | {Z_index2[n]: n for n in range(N)}

    dup=0
    for m,(merge_i, merge_sig) in enumerate(zip(sort_s_index, sort_s_significance)):
        print("starting sig:", merge_sig)
        if merge_i not in traversed:
            traversed_currentpath = []
            maxval = 0
            sig = merge_sig
            i = merge_i
            visited = 0
            while(i <= max_i):
                if(i not in traversed):
                    traversed_currentpath.append(i)
                    if(sig > maxval):
                        if maxval>0:
                            print("Higher found?")
                            print(sig, maxval)
                        maxval = sig
                        selected_i = i
                        print("Path cleaned")
                        traversed_currentpath = []
                        # traversed_currentpath = [i]
                        
                    try:
                        next_index = index_dic[i + N+1]
                    except:
                        print("FOUND TOP")
                        print(i)
                        visited=1
                        break
                        
                    i = index[next_index]
                    sig = significance[next_index]
                    visited = 1
                else:  # we already traversed this path, so exit
                    break
            if((maxval > 0) and (visited == 1)):
                traversed.extend(traversed_currentpath)

                # Same cluster might have been marked the most significant one for multiple paths.
                if(selected_i not in selected):
                    selected.append(selected_i)
                    max_sign.append(maxval)
                else:
                    print("DUPLICATE")
                    print(maxval)
                    dup+=1
            else:
                print("Not added?", maxval, visited)
    print("Finished")
    print((time.time()-T)/60)
    print("Dup", dup)
    return np.array(selected), np.array(max_sign)

old2_selected, old2_max_sign = old2_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
%lprun -f old2_select_maxsig_clusters_from_tree old2_select_maxsig_clusters_from_tree( significance, Z, minimum_significance=N_sigma_significance)

# %%
print(len(old2_selected))
print(old2_selected)
print(old2_max_sign)

# %% [markdown]
# # NEW FIXED and Current

# %%
from tqdm.notebook import tqdm


def select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
    T = time.time()
    print('Picking out the clusters with maximum significance from the tree...')
    
    N = len(significance)

    index = np.arange(len(significance))
    

    selected_i = 0

    traversed = []  # list of indices i that we have already traversed
    selected = []  # list of merges in the tree where the significance is maximized
    max_sign = []  # the significance at the selected step

    argsort_sig = np.argsort(significance)[::-1]
    sort_significance = significance[argsort_sig]
    sort_index = index[argsort_sig]
    
    sort_Z_index1 = Z[:, 0][argsort_sig]
    sort_Z_index2 = Z[:, 1][argsort_sig]
    index_dic = {}
    index_dic = {sort_Z_index1[n]: n for n in range(N)} | {sort_Z_index2[n]: n for n in range(N)}
    


    untraversed_filt = np.ones_like(sort_significance,dtype=bool)
    sig_untraversed_filt = (sort_significance>minimum_significance)
    N_sig_untraversed = sig_untraversed_filt.sum()>0
    
    while N_sig_untraversed>0:
        next_index = np.where(sig_untraversed_filt)[0][0]
        i = sort_index[next_index]
        sig = sort_significance[next_index]
        sig_untraversed_filt[next_index] =False
        
        traversed_currentpath = []
        maxval = 0
        while True:
            traversed_currentpath.append(i)
            if(sig > maxval):
                traversed_currentpath = []
                # traversed_currentpath = [i]
                maxval = sig
                selected_i = i

            try:
                next_index = index_dic[i + N+1]
            except:
                print("FOUND TOP")
                break

            if not untraversed_filt[next_index]:
                # print("Traversed")
                break

            i = sort_index[next_index]
            sig = sort_significance[next_index]

        traversed.extend(traversed_currentpath)

        # Same cluster might have been marked the most significant one for multiple paths.
        if(selected_i not in selected):
            selected.append(selected_i)
            max_sign.append(maxval)
        else:
            # faster to just find unique at end?
            print("DUPLICATE")
            print(traversed_currentpath)

        if len(traversed_currentpath)>0:
            untraversed_filt[np.where(untraversed_filt)[0]] = np.isin(
                sort_index[untraversed_filt], traversed_currentpath, invert=True)

            sig_untraversed_filt = untraversed_filt*sig_untraversed_filt
        else:
            print("Len 0")
            
        N_sig_untraversed  = sig_untraversed_filt.sum() #FASTER -= len(traversed_currentpath)  #
        print(f"{N_sig_untraversed} still untraversed")
        
    print("Finished")
    print((time.time()-T)/60)
    return np.array(selected), np.array(max_sign)

selected, max_sign = select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3)

# %%
print(len(selected))
print(selected)
print(max_sign)

# %%
print((s_selected==selected).sum()/len(selected))
print((s_max_sign==max_sign).sum()/len(max_sign))


# %% [markdown]
# # PREVIOUS

# %%
def prev_select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
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
    # Pick max sig clusters, go traverse up the tree to tick off.
    # Go to the next untraversed most significant cluster

    T = time.time()

    print('Picking out the clusters with maximum significance from the tree...')
    N = len(significance)
    index = np.arange(N)

    argsort_sig = np.argsort(significance)[::-1]
    sort_significance = significance[argsort_sig]
    sort_index = index[argsort_sig]

    sort_Z_index1 = Z[:, 0][argsort_sig]
    sort_Z_index2 = Z[:, 1][argsort_sig]
    index_dic = {}
    index_dic = {sort_Z_index1[n]: n for n in range(N)} | {sort_Z_index2[n]: n for n in range(N)}

    traversed = []  # list of indices i that we have already traversed
    selected = []  # list of merges in the tree where the significance is maximized
    max_sign = []  # the significance at the selected step

    untraversed_filt = (sort_significance > minimum_significance)
    N_untraversed = untraversed_filt.sum()

    while N_untraversed > 0:

        i = sort_index[untraversed_filt][0]
        sig = sort_significance[untraversed_filt][0]
        
        selected.append(i)
        max_sign.append(sig)

        traversed_currentpath = []
        while True:
            # Track what we have traversed since the max-significance
            traversed_currentpath.append(i)

            try:
                next_index = index_dic[i + N + 1]
            except:
                # print("Found Top")
                break
            if not untraversed_filt[next_index]:
                # print("Traversed")
                break

            i = sort_index[next_index]

        # Add New Path
        traversed.extend(traversed_currentpath)
        untraversed_filt[np.where(untraversed_filt)[0]] = np.isin(
            sort_index[untraversed_filt], traversed_currentpath, invert=True)
        N_untraversed -= len(traversed_currentpath)  # = untraversed_filt.sum() #FASTER

    dt = time.time() - T
    print(f"Finished. Time {dt} s")

    return np.array(selected), np.array(max_sign)



# %%
print(len(selected))
print(selected)
print(max_sign)

# %%
print(len(Z))

# %%
