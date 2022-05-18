from queue import Queue
import numpy as np
from fastcluster import linkage_vector
import time


def clusterData(stars, features, linkage_method="single", scales={}):
    print(features)
    scale_features(stars, features=features, scales=scales, plot=False)

    T = time.time()
    X = find_X(features, stars, scaled=True)
    print("Starting clustering.")
    Z = linkage_vector(X, linkage_method)
    dt = time.time() - T
    print(f"Finished! Time taken: {dt/60} minutes")
    cluster_data = {"Z": Z, "features": features}
    return cluster_data

###


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
            # backtract one step, enqueue this branch
            q.put(int(Z[index, 0] - N))

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



def find_X(features, stars, scaled=False):
    n_features = len(features)
    N_stars = len(stars["vx"])
    X = np.empty((N_stars, n_features))
    for n, p in enumerate(features):
        if scaled:
            X[:, n] = stars["scaled_"+p]
            print("using scaled ",p)
        else:
            X[:, n] = stars[p]
    return X


def art_find_X(features, art_stars):
    arts = list(art_stars.keys())
    art_X = {}
    for a in arts:
        art_X[a] = find_X(features, art_stars[a])
    return art_X

def find_art_X_array(features, art_stars, scaled=False):
    n_features = len(features)
    arts = list(art_stars.keys())
    N_art = len(arts)
    N_stars = len(art_stars[0]["vx"])
    art_X_array = np.empty((N_art,N_stars, n_features))
    for na,a in enumerate(arts):
        stars = art_stars[a]
        for n, p in enumerate(features):
            if scaled:
                art_X_array[na,:, n] = stars["scaled_"+p]
                print("using scaled ",p)
            else:
                art_X_array[na,:, n] = stars[p]
    return art_X_array

from .default_params import cluster_params0
import copy
###
def scale_features(stars, features=cluster_params0["features"],
                   scales=cluster_params0["scales"], plot=True):
    scale_data = {}
    percent = 5
    for p in features:
        scale = scales.get(p, None)
        if scale is None:
            print(f"No scale found for {p}, creating own using {percent} margin")
            scale = get_scale(x=stars[p])
        stars["scaled_" +
              p] = apply_scale(stars, xkey=p, scale=scale, plot=plot)
        scale_data[p] = scale
    return stars, scale_data


def get_scale(x, percent=5):
    scale = np.array(np.percentile(x, [percent, 100 - percent]))
    return scale


def apply_scale(data, xkey, scale, plot=False):
    x = data[xkey]
    scale_range = scale[1] - scale[0]
    scaled_x = ((x - scale[0]) / (scale_range / 2)) - 1

    if plot:
        import matplotlib.pyplot as plt
        n_x = len(x)
        f_below = (x < scale[0]).sum() / n_x
        f_above = (x > scale[1]).sum() / n_x
        print(xkey)
        print(f"Scale: below {f_below}, above{f_above}")
        x_sort = np.sort(copy.deepcopy(x))

        plt.figure()
        plt.plot(x_sort, np.linspace(0, 1, n_x))
        plt.axvline(scale[0], c='r')
        plt.axvline(scale[1], c='r')
        plt.xlabel(xkey)
        plt.ylabel("cdf")
        plt.show()

        plt.figure()
        plt.plot(x, scaled_x)
        plt.axvline(scale[0], c='r')
        plt.axvline(scale[1], c='r')
        plt.axhline(-1, c='r')
        plt.axhline(1, c='r')
        plt.xlabel(xkey)
        plt.ylabel("scaled " + xkey)
        plt.show()

    return scaled_x
