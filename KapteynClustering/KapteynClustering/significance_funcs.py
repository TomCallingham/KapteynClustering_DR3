import numpy as np
try:
    from .mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian
    from . import data_funcs as dataf
    from . import dic_funcs as dicf
    from . import cluster_funcs as clusterf
except Exception:
    from KapteynClustering.mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian
    from KapteynClustering import data_funcs as dataf
    from KapteynClustering import dic_funcs as dicf
    from KapteynClustering import cluster_funcs as clusterf


max_members =  25000
def expected_density_members(members, N_std, X, art_X, N_art, min_members):
    '''
    Returns:
    [members within region, mean_art_region_count, std_art_region_cound](np.array): The number of stars in the artificial halo
                                                in the region of cluster i, and the standard
                                                deviation in counts across the artificial
                                                halo data sets.
    '''

    # get a list of members (indices in our halo set) of the cluster C we are currently investigating

    N_members = len(members)

    # ignore these clusters, return placeholder (done for computational efficiency)
    if((N_members > max_members) or (N_members < min_members)):
    # if (N_members < min_members):
        # print("Out of range:",N_members, max_members)
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])

    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X)

    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        counts_per_halo[n] = find_mahalanobis_N_members(
            N_std, mean, covar, art_X[n])
        # print("SCALING COUNTS")
        counts_per_halo[n] = counts_per_halo[n] * \
            (len(X[:, 0])/len(art_X[n][:, 0]))

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])

def cut_expected_density_members(members, N_std, X, art_X, N_art, min_members):
    '''
    Returns:
    [members within region, mean_art_region_count, std_art_region_cound](np.array): The number of stars in the artificial halo
                                                in the region of cluster i, and the standard
                                                deviation in counts across the artificial
                                                halo data sets.
    '''

    # get a list of members (indices in our halo set) of the cluster C we are currently investigating

    N_members = len(members)

    # ignore these clusters, return placeholder (done for computational efficiency)
    if((N_members > max_members) or (N_members < min_members)):
    # if (N_members < min_members):
        # print("Out of range:",N_members, max_members)
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])

    xmins = np.min(X[members,:],axis=0)
    xmaxs = np.max(X[members,:],axis=0)
    eps=0.05
    filt = np.prod((X>xmins[None,:] - eps)*(X<xmaxs[None,:] + eps),axis=1).astype(bool)

    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X[filt])


    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        filt = np.prod((art_X[n]>xmins[None,:] - eps)*(art_X[n]<xmaxs[None,:] + eps),axis=1).astype(bool)
        counts_per_halo[n] = find_mahalanobis_N_members(
            N_std, mean, covar, art_X[n][filt,:])
        # print("SCALING COUNTS")
        counts_per_halo[n] = counts_per_halo[n] * \
            (len(X[:, 0])/len(art_X[n][:, 0]))

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])

def vec_expected_density_members(members, N_std, X, art_X_array, N_art, min_members):
    '''
    Returns:
    [members within region, mean_art_region_count, std_art_region_cound](np.array): The number of stars in the artificial halo
                                                in the region of cluster i, and the standard
                                                deviation in counts across the artificial
                                                halo data sets.
    '''

    # get a list of members (indices in our halo set) of the cluster C we are currently investigating

    N_members = len(members)

    # ignore these clusters, return placeholder (done for computational efficiency)
    if((N_members > max_members) or (N_members < min_members)):
    # if (N_members < min_members):
        # print("Out of range:",N_members, max_members)
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])

    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X)


    # Artificial
    counts_per_halo = find_mahalanobis_N_members( N_std, mean, covar, art_X_array)
    N_art_stars = np.array([len(art_X_array[n][:,0]) for n in range(N_art)])
    counts_per_halo = counts_per_halo * (len(X[:, 0])/N_art_stars)

    # print("My counts per halo:")
    # print(counts_per_halo)
    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])


# %% LOAD
def sig_load_data(param_file):
    params = dataf.read_param_file(param_file)
    data_params = params["data"]
    result_folder = data_params["result_folder"]
    cluster_file, sample_file, art_file = data_params["cluster"] ,data_params["sample"] , data_params["art"]
    save_name = data_params["sig"]

    cluster_params = params["cluster"]
    min_members, max_members = cluster_params["min_members"], cluster_params["max_members"]
    features = cluster_params["features"]
    N_art, N_std, N_process = cluster_params["N_art"] ,cluster_params["N_sigma_ellipse_axis"],cluster_params["N_process"]


    # LOADING all necessary Data
    sample_data = dicf.h5py_load(result_folder + sample_file)
    stars = sample_data["stars"]
    X = clusterf.find_X(features, stars)
    del stars

    art_data = dicf.h5py_load(result_folder + art_file)
    art_stars = art_data["stars"]
    art_X = clusterf.art_find_X(features, art_stars)
    del art_stars, art_data

    cluster_data = dicf.h5py_load(result_folder + cluster_file)
    Z = cluster_data["Z"]
    del cluster_data
    N_clusters = len(Z[:, 0])
    tree_members = clusterf.find_tree(Z, prune=True)
    list_tree_members = [tree_members[i + N_clusters+1] for i in range(N_clusters)]
    return save_name, result_folder, max_members, min_members, X, art_X, list_tree_members, N_clusters, N_process, N_art, N_std
