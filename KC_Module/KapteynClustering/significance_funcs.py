import numpy as np
from scipy.stats._multivariate import _PSD
try:
    from .mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian
    from . import data_funcs as dataf
    from . import param_funcs as paraf
    from . import linkage_funcs as linkf
except Exception:
    from KapteynClustering.mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian
    from KapteynClustering import data_funcs as dataf
    from KapteynClustering import param_funcs as paraf
    from KapteynClustering import linkage_funcs as linkf


def expected_density_members(members, N_std, X, art_X, N_art, min_members, max_members):
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
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])
    psd = _PSD(covar, allow_singular=True)
    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X, psd=psd)

    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        counts_per_halo[n] = find_mahalanobis_N_members(
            N_std, mean, covar, art_X[n], psd=psd)

    N_art_stars = np.array([len(art_X[n][:, 0]) for n in range(N_art)])
    counts_per_halo = counts_per_halo * (len(X[:, 0])/N_art_stars)

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])

def cut_expected_density_members(members, N_std, X, art_X, N_art, min_members, max_members):
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
    max_members = 25000
    if((N_members > max_members) or (N_members < min_members)):
        # if (N_members < min_members):
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])
    psd = _PSD(covar, allow_singular=False)

    xmins = np.min(X[members, :], axis=0)
    xmaxs = np.max(X[members, :], axis=0)
    x_range = 0.1*(xmaxs-xmins)
    xmins = xmins - x_range
    xmaxs = xmaxs + x_range

    # Count Fit
    filt = np.prod((X > xmins[None, :]) *
                   (X < xmaxs[None, :]), axis=1).astype(bool)

    region_count = find_mahalanobis_N_members(N_std, mean, covar, X[filt],psd=psd)

    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        filt = np.prod((art_X[n] > xmins[None, :]) *
                       (art_X[n] < xmaxs[None, :]), axis=1).astype(bool)
        counts_per_halo[n] = find_mahalanobis_N_members(
            N_std, mean, covar, art_X[n][filt, :], psd=psd)

    N_art_stars = np.array([len(art_X[n][:, 0]) for n in range(N_art)])
    counts_per_halo = counts_per_halo * (len(X[:, 0])/N_art_stars)

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])

def mask_expected_density_members(members, N_std, X, art_X, N_art, min_members):
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
    max_members = 25000
    if((N_members > max_members) or (N_members < min_members)):
        # if (N_members < min_members):
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])
    psd = _PSD(covar, allow_singular=False)

    xmins = np.min(X[members, :], axis=0)
    xmaxs = np.max(X[members, :], axis=0)
    x_range = 0.1*(xmaxs-xmins)
    xmins = xmins - x_range
    xmaxs = xmaxs + x_range
    # xmins = xmins - eps
    # xmaxs = xmaxs + eps
    # eps = 0.05

    # Count Fit
    filt = np.prod((X > xmins[None, :]) *
                   (X < xmaxs[None, :]), axis=1).astype(bool)

    region_count = find_mahalanobis_N_members(N_std, mean, covar, np.ma.masked_where(condition=filt,a = X),psd=psd)

    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        filt = np.prod((art_X[n] > xmins[None, :]) *
                       (art_X[n] < xmaxs[None, :]), axis=1).astype(bool)
        counts_per_halo[n] = find_mahalanobis_N_members(
            N_std, mean, covar, art_X[n][filt, :], psd=psd)

    N_art_stars = np.array([len(art_X[n][:, 0]) for n in range(N_art)])
    counts_per_halo = counts_per_halo * (len(X[:, 0])/N_art_stars)

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])


def vec_expected_density_members(members, N_std, X, art_X_array, N_art, min_members, max_members):
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
        return(np.array([0, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])

    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X)

    # Artificial
    counts_per_halo = find_mahalanobis_N_members(
        N_std, mean, covar, art_X_array)
    N_art_stars = np.array([len(art_X_array[n][:, 0]) for n in range(N_art)])
    counts_per_halo = counts_per_halo * (len(X[:, 0])/N_art_stars)

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])


# %% LOAD
def sig_load_data(param_file, scaled_force=False):
    if scaled_force:
        print("USING SCALED FEATURES manually")
    params = paraf.read_param_file(param_file)
    data_p = params["data"]

    link_p = params["linkage"]
    min_members = link_p["min_members"]
    features = link_p["features"]
    N_art, N_process = link_p["N_art"],  link_p["N_process"]
    N_std = paraf.find_Nstd_from_params(params)

    stars = dataf.read_data(data_p["sample"])
    print("Using features:")
    print(features)
    try:
        X = linkf.find_X(features, stars, scaled=scaled_force)
    except Exception:
        stars = linkf.scale_features(
            stars, features=features, scales=link_p["scales"])[0]
        X = linkf.find_X(features, stars, scaled=scaled_force)
    del stars

    max_members  = link_p.get("max_members",None)
    if max_members is None:
        max_members = len(X[:,0])/4
        print("auto max members, quarter of Nstars:",max_members)

    art_stars = dataf.read_data(data_p["art"])
    # Then using sofies, needs individual datasets split
    if 0 not in list(art_stars.keys()):
        import KapteynClustering.dic_funcs as dicf
        art_stars = dicf.groups(art_stars, group="index", verbose=True)
    art_X = linkf.art_find_X(features, art_stars, scaled=scaled_force)
    del art_stars

    link_data = dataf.read_data(data_p["linkage"])
    Z = link_data["Z"]
    del link_data
    N_clusters = len(Z[:, 0])
    tree_members = linkf.find_tree(Z, prune=True)
    list_tree_members = [tree_members[i + N_clusters+1] for i in range(N_clusters)]
    return data_p["sig"], max_members, min_members, X, art_X, list_tree_members, N_clusters, N_process, N_art, N_std
