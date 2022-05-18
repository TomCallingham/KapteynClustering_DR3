import numpy as np
try:
    from . import cluster_funcs as clusterf
except Exception:
    from KapteynClustering import cluster_funcs as clusterf

def find_group_data(significance, Z, minimum_significance=3):
    selected, cluster_sig = select_maxsig_clusters_from_tree(significance, Z, minimum_significance)
    tree_members = clusterf.find_tree(Z, prune=True)
    N_clusters = len(Z)
    labels, star_sig =  get_cluster_labels_with_significance(selected, cluster_sig, tree_members, N_clusters)
    labels, Groups, Pops = order_labels(labels)
    G_sig = np.array([star_sig[labels==g][0] for g in Groups])
    print(f'Number of clusters: {len(Groups)-1}')
    label_data = {"labels":labels, "star_sig": star_sig,
            "Groups":Groups, "Pops":Pops, "G_sig":G_sig}
    return label_data

def select_maxsig_clusters_from_tree(significance, Z, minimum_significance=3):
    # T = time.time()
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
    index_dic = {sort_Z_index1[n]: n for n in range(
        N)} | {sort_Z_index2[n]: n for n in range(N)}

    untraversed_filt = np.ones_like(sort_significance, dtype=bool)
    sig_untraversed_filt = (sort_significance > minimum_significance)
    N_sig_untraversed = sig_untraversed_filt.sum() > 0

    while N_sig_untraversed > 0:
        next_index = np.where(sig_untraversed_filt)[0][0]
        i = sort_index[next_index]
        sig = sort_significance[next_index]
        sig_untraversed_filt[next_index] = False

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
                # print("FOUND TOP")
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
            # faster to just find unique at end?

        if len(traversed_currentpath) > 0:
            untraversed_filt[np.where(untraversed_filt)[0]] = np.isin(
                sort_index[untraversed_filt], traversed_currentpath, invert=True)

            sig_untraversed_filt = untraversed_filt*sig_untraversed_filt

        # FASTER -= len(traversed_currentpath)  #
        N_sig_untraversed = sig_untraversed_filt.sum()
        # print(f"{N_sig_untraversed} still untraversed")

    # print("Finished")
    # print((time.time()-T)/60)
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
    # print(f"Initialy {len(significance)} clusters")
    N = len(tree_members.keys())

    labels = -np.ones((N_clusters+1), dtype=int)
    significance_list = np.zeros(N_clusters+1)
    # i_list = np.sort(significant.i.values)
    # Sort increasing, so last are includedIn the best cluster
    sig_sort = np.argsort(significance)
    s_significance = significance[sig_sort]
    s_index = selected[sig_sort]

    for i_cluster, sig_cluster in zip(s_index, s_significance):
        members = tree_members[i_cluster+N_clusters+1]
        labels[members] = i_cluster
        significance_list[members] = sig_cluster

    NG = len(np.unique(labels))
    return labels, significance_list




def order_labels(old_labels, fluff_label=-1):
    ''' relabel, using 0 as largest group then descending. Fluff on -1'''
    Groups, Pops = np.unique(old_labels[old_labels!=fluff_label], return_counts=True)
    p_sort = np.argsort(Pops)[::-1]  # Decreasing order
    old_Groups, Pops = Groups[p_sort].astype(int), Pops[p_sort]
    N_groups = len(old_Groups)
    Groups = -np.ones((N_groups+1), dtype=int)
    labels = -np.ones_like(old_labels, dtype=int)
    for i, l in enumerate(old_Groups, start=0):
        labels[old_labels == l] = i
        Groups[i+1] = i
    Nfluff = (old_labels==fluff_label).sum()
    Pops = np.concatenate(([[Nfluff], Pops]))
    return labels, Groups, Pops
