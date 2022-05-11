import numpy as np
from .mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian

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
    # if((N_members > max_members) or (N_members < min_members)):
    if (N_members < min_members):
        # print("Out of range:",N_members, max_members)
        return(np.array([N_members, N_members, 0]))

    # Fit Gaussian
    mean, covar = fit_gaussian(X[members, :])

    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X)

    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        counts_per_halo[n] = find_mahalanobis_N_members(N_std, mean, covar, art_X[n])
        # print("SCALING COUNTS")
        counts_per_halo[n] = counts_per_halo[n]*(len(X[:,0])/len(art_X[n][:,0]))

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])

