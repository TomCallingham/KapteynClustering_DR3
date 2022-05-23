# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.significance_funcs as sigf

# %% [markdown]
# ## Load Data

# %%
import cluster_utils
import vaex
import time

# %%
params = dataf.read_param_file("gaia_params.yaml")
# params = dataf.read_param_file("sof_check_params.yaml")
cluster_p = params["cluster"]
features = cluster_p["features"]
print(features)

# %%
param_file = "../../../Params/gaia_params.yaml"
save_name, max_members, min_members, X, art_X, list_tree_members, N_clusters, N_process, N_art, N_std = sigf.sig_load_data(
    param_file)


# %%
param_file = "../../../Params/gaia_params.yaml"
save_name, max_members, min_members, s_X, s_art_X, list_tree_members, N_clusters, N_process, N_art, N_std = sigf.sig_load_data(
    param_file,scaled_force=False)


# %%
Nclusters = np.shape(scaled_X)[0]

# %% [markdown]
# # Find bad

# %%
from KapteynClustering import mahalanobis_funcs as mf

# %%
for i_test in np.random.choice(np.arange(Nclusters),size=50):
    members= list_tree_members[i_test]
    # print(i_test)
    if len(members)>5:
        
        mean, covar = mf.fit_gaussian(X[members, :])
        region_count = mf.find_mahalanobis_N_members(N_std, mean, covar, X)
        
        s_mean, s_covar = mf.fit_gaussian(s_X[members, :])
        s_region_count = mf.find_mahalanobis_N_members(N_std, s_mean, s_covar, s_X)
        
        print(region_count,s_region_count)
        
        print(mean, s_mean)
        
        counts_per_halo = np.zeros((N_art))
        s_counts_per_halo = np.zeros((N_art))
        for n in range(N_art):
            counts_per_halo[n] = mf.find_mahalanobis_N_members(N_std, mean, covar, art_X[n])
            s_counts_per_halo[n] = mf.find_mahalanobis_N_members(N_std, s_mean, s_covar, s_art_X[n])
            
        art_region_count = np.mean(counts_per_halo)
        art_region_count_std = np.std(counts_per_halo)
        
        s_art_region_count = np.mean(s_counts_per_halo)
        s_art_region_count_std = np.std(s_counts_per_halo)
        
        print(art_region_count, art_region_count_std)
        print(s_art_region_count, s_art_region_count_std)
        
    print("Finished")
        



# %%
