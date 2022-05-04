import numpy as np
from fastcluster import linkage_vector
import time


def clusterData(stars, features, linkage_method="single"):
    print("Starting clutering.")
    print("INCLUDE SCALING HERE!")

    T= time.time()
    nstars = len(stars["x"])
    nfeatures = len(features)
    X = np.zeros((nstars, nfeatures))
    for i, p in enumerate(features):
        X[:, i] = stars["scaled_" + p]
    Z = linkage_vector(X, linkage_method)
    dt = time.time() - T
    print(f"Finished! Time taken: {dt/60} minutes")
    cluster_data = {"Z":Z, "features": features}
    return cluster_data

###

from queue import Queue

def fast_get_members(i, Z):
    '''
    SOFIE
    Returns the list of members for a specific cluster included in the encoded linkage matrix

    Parameters:
    i(int): candidate cluster number
    Z(np.ndarray): the linkage matrix

    Returns:
    members(np.array): array of members corresponding to candidate cluster i
    '''
    members = []
    N = len(Z) + 1  # the number of original data points

    # Initializing a queue
    q = Queue(maxsize=N)
    q.put(i)

    while(not q.empty()):

        index = q.get()

        if(Z[index, 0] < N):
            members.append(int(Z[index, 0]))  # append original member
        else:
            q.put(int(Z[index, 0] - N))  # backtract one step, enqueue this branch

        if(Z[index, 1] < N):
            members.append(int(Z[index, 1]))  # append original member
        else:
            q.put(int(Z[index, 1] - N))  # enqueue this branch

    return np.array(members)


def next_members(tree_dic, Z, i, N_cluster):
    tree_dic[i + N_cluster] = tree_dic[Z[i, 0]] + tree_dic[Z[i, 1]]
    return tree_dic


def next_members_del(tree_dic, Z, i, N_cluster):
    tree_dic[i + N_cluster] = tree_dic[Z[i, 0]] + tree_dic[Z[i, 1]]
    del tree_dic[Z[i, 0]], tree_dic[Z[i, 1]]
    return tree_dic


def find_members(ind, Z):
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    tree_dic = {i: [i] for i in range(N_cluster1)}
    for i in range(N_cluster)[:ind + 1]:
        tree_dic = next_members(tree_dic, Z, i, N_cluster1)
    members = tree_dic[ind + N_cluster1]
    return members, tree_dic


def find_tree(Z, i_max=None, prune=False):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    tree_dic = {i: [i] for i in range(N_cluster1)}
    # members= {}
    if i_max is None:
        i_max = N_cluster1
    for i in range(N_cluster)[:i_max]:
        tree_dic = next_members(tree_dic, Z, i, N_cluster1)
    print("Tree found")
    if prune:
        print("del")
        for i in range(N_cluster):
            del tree_dic[i]
    return tree_dic


def find_X(features, stars):
    n_features = len(features)
    N_stars = len(stars["x"])
    X = np.empty((N_stars, n_features))
    for n, p in enumerate(features):
        X[:, n] = stars[p]
    return X


def art_find_X(features, art_stars):
    arts = list(art_stars.keys())
    art_X = {}
    for a in arts:
        art_X[a] = find_X(features, art_stars[a])
    return art_X


import time


def select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
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

def get_cluster_labels_with_significance(selected, significance, tree_members, N_clusters):
    '''
    Extracts flat cluster labels given a list of statistically significant clusters
    in the linkage matrix Z, and also returns a list of their correspondning statistical significance.

    Parameters:
    significant(vaex.DataFrame): Dataframe containing statistics of the significant clusters.
    Z(np.ndarray): The linkage matrix outputted from single linkage

    Returns:
    labels(np.array): Array matching the indices of stars in the data set, where label 0 means noise
                      and other natural numbers encode significant structures.
    '''

    print("Extracting labels...")
    print(f"Initialy {len(significance)} clusters")
    N = len(tree_members.keys())
    print(N)

    labels = -np.ones((N_clusters+1))
    significance_list = np.zeros(N_clusters+1)
    # i_list = np.sort(significant.i.values)
    # Sort increasing, so last are includedIn the best cluster
    sig_sort = np.argsort(significance)
    s_significance = significance[sig_sort]
    s_index = selected[sig_sort]

    for i_cluster, sig_cluster in zip(s_index, s_significance) :
        members = tree_members[i_cluster+N_clusters+1]
        labels[members]=i_cluster
        significance_list[members] = sig_cluster

    Groups, pops = np.unique(labels, return_counts=True)
    p_sort = np.argsort(pops)[::-1] # Decreasing order
    Groups, pops = Groups[p_sort], pops[p_sort]

    for i,l in enumerate(Groups):
        labels[labels==l] = i

    print(f'Number of clusters: {len(Groups)-1}')

    return labels, significance_list
