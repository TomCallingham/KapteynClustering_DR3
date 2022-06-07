from .default_params import cluster_params0
import copy
from queue import Queue
import numpy as np
from fastcluster import linkage_vector
import time


def clusterData(stars, features, linkage_method="single", scales={}):
    print(features)
    stars = scale_features(stars, features=features,
                           scales=scales, plot=False)[0]
    X = find_X(features, stars, scaled=True)
    T = time.perf_counter()
    print("Starting clustering.")
    Z = linkage_vector(X, linkage_method).astype(int)
    dt = time.perf_counter() - T
    print(f"Finished! Time taken: {dt/60} minutes")
    cluster_data = {"Z": Z, "features": features}
    return cluster_data

###


def get_members(i, Z):
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
        for i in range(N_cluster):
            del tree_dic[i]
    return tree_dic


def find_X(features, stars, scaled=False):
    n_features = len(features)
    try:
        N_stars = stars.count()
    except Exception:
        N_stars = len(stars[list(stars.keys())[0]])
    X = np.empty((N_stars, n_features))
    for n, p in enumerate(features):
        if scaled:
            p = "scaled_" + p
            print("using scaled ", p)
        try:
            X[:, n] = stars[p].values
        except Exception:
            X[:, n] = stars[p]
    return X


def art_find_X(features, art_stars, scaled=False):
    arts = list(art_stars.keys())
    art_X = {}
    for a in arts:
        art_X[a] = find_X(features, art_stars[a], scaled=scaled)
    return art_X


def find_art_X_array(features, art_stars, scaled=False):
    n_features = len(features)
    arts = list(art_stars.keys())
    N_art = len(arts)
    N_stars = len(art_stars[0]["vx"])
    art_X_array = np.empty((N_art, N_stars, n_features))
    for na, a in enumerate(arts):
        stars = art_stars[a]
        for n, p in enumerate(features):
            if scaled:
                art_X_array[na, :, n] = stars["scaled_"+p]
                print("using scaled ", p)
            else:
                art_X_array[na, :, n] = stars[p]
    return art_X_array


###


def scale_features(stars, features=cluster_params0["features"],
                   scales=cluster_params0["scales"], plot=False):
    scale_data = {}
    percent = 5
    for p in features:
        given_scale = scales.get(p, None)
        print(f"for {p}, given scale is {given_scale}")
        if given_scale is None or None in given_scale:
            print(f"No scale found for {p}, creating own using {percent} margin")
        try:
            x = stars[p].values
        except Exception:
            x = stars[p]
        scale = get_partial_scale(x=x, percent=percent, given_scale = given_scale)
        print(f"{p}, using scale: {scale}")
        stars["scaled_" + p] = apply_scale(stars, xkey=p, scale=scale, plot=plot)
        scale_data[p] = scale
    return stars, scale_data


def get_scale(x, percent=5):
    scale = np.array(np.percentile(x, [percent, 100 - percent]))
    return scale

def get_partial_scale(x,percent=5, given_scale=None):
    if given_scale is None:
        return get_scale(x, percent)
    scale = get_scale(x, percent=5)
    if given_scale[0] not in [None,"None"]:
        scale[0] = given_scale[0]
    if given_scale[1] not in [None,"None"]:
        print("replacing top scale")
        scale[1] = given_scale[1]
    return scale


def apply_scale(data, xkey, scale, plot=False):
    try:
        x = data[xkey].values
    except Exception:
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

def fit_clusters(stars, features, Groups, labels):
    import KapteynClustering.mahalanobis_funcs as mahaf
    x = find_X(features, stars, scaled=False)
    mean_group = {}
    cov_group = {}
    for g in Groups:
        g_filt = (labels == g)
        mean_group[g] , cov_group[g] = mahaf.fit_gaussian(x[g_filt,:])
    return mean_group, cov_group
