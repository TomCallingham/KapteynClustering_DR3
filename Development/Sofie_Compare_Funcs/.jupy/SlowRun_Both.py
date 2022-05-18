# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:py39]
#     language: python
#     name: conda-env-py39-py
# ---

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Setup

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.significance_funcs as sigf

from params import data_params, gaia2, cosma

# %% tags=[]
from KapteynClustering.default_params import  cluster_params0
min_members = cluster_params0["min_members"]
max_members = cluster_params0["max_members"]
# features = cluster_params0["features"]
N_sigma_ellipse_axis = cluster_params0["N_sigma_ellipse_axis"]

# %%
N_art = 100

# %%

# %% [markdown] tags=[]
# ## Load Data

# %%
import cluster_utils
import vaex
import time

# %%
import time

# %%
from params import gaia2

file_base= data_params["result_path"] + data_params["data_folder"]
cluster_file= data_params["cluster"]
sample_file= data_params["sample"]
art_file= data_params["art"]
save_name= data_params["sig"]


cluser_data = dicf.h5py_load(file_base + cluster_file)
Z = cluser_data["Z"]

N_clusters = len(Z[:,0])

# %%
if False:
    sample_data = dicf.h5py_load(file_base + sample_file)
    stars = sample_data["stars"]
    df  = vaex.from_dict(stars)

# %%
if False:
    def get_artificial_set():
        '''
        Creates a data set of N artificial halos

        Parameters:
        N(int): The number of artificial halos to be generated
        ds(vaex.DataFrame): A dataframe containing the halo stars and a little bit more, recommended: V-V_{lsr}>180.

        Returns:
        df_art_all(vaex.DataFrame): A dataframe containing each artificial halo, their integrals of motion, and data set index.
        '''
        temp=0
        N = len(art_stars.keys())
        for n in range(N):
            # art_stars[n]["scaled_Lperp"] = art_stars[n]["scaled_Lp"]
            df_art= vaex.from_dict(art_stars[n])
            # df_art = get_artificial_dataset(ds)
            df_art['index'] = np.array(int(df_art.count())*[n])

            if(temp==0):
                df_art_all = df_art
                temp=1
            else:    
                df_art_all=df_art_all.concat(df_art)

        return df_art_all


    df_artificial = get_artificial_set()
    art_data = dicf.h5py_load(file_base + art_file)
    art_stars = art_data["stars"]

# %% tags=[]
if True:
    import vaex
    s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
    df = vaex.open(f'{s_result_path}df.hdf5')
    features_to_be_scaled = ['En', 'Lperp', 'Lz'] #Clustering features that need scaling
    minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300], Lz=[-4500,4600]) #Fix origin for scaling of features
    df = cluster_utils.scaleData(df, features_to_be_scaled, minmax_values)
    
    stars = dataf.load_vaex_to_dic(f'{s_result_path}df.hdf5')
    dataf.scale_features(stars, features = ["En", "Lz", "Lperp"], plot=False)
        # del stars[k]

# %%
if True:
    import vaex
    s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
    df_artificial = vaex.open(f'{s_result_path}df_artificial_3D.hdf5')



    sof_art_stars = dataf.load_vaex_to_dic(f'{s_result_path}df_artificial_3D.hdf5')
    trans_pops = {"En":"E", "Lperp":"Lp"}
    # for k in trans_pops.keys():
    #     sof_art_stars[trans_pops[k]] = sof_art_stars[k]
        # del sof_art_stars[k]
    sof_art_stars = dicf.groups(sof_art_stars, group="index", verbose=True)
    art_stars = sof_art_stars

# %%

features = ["scaled_En", "scaled_Lperp", "scaled_Lz"]
s_features = features
my_features= features# ["E", "Lz", "Lp"]
# my_features=  ["scaled_E", "scaled_Lz", "scaled_Lp"]

# %%
X = clusterf.find_X(my_features, stars)
art_X = clusterf.art_find_X(my_features, art_stars)

# %%
import cluster_utils
N_datasets = N_art

# %% tags=[]
tree_members = clusterf.find_tree(Z, prune=True)
Nclusters = len(Z) +1

# %%
N_std = N_sigma_ellipse_axis

# %% [markdown]
# # Funcs

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### My Sig Find

# %%
from Sof_Sig_funcs import original_s_expected_density_parallel
from KapteynClustering.significance_funcs import cut_expected_density_members, expected_density_members

# %%
sof_dic = {"Z":Z, "min_members":min_members, "max_members":max_members,
          "df":df, "df_artificial":df_artificial,
          "N_sigma_ellipse_axis": N_sigma_ellipse_axis,
          "N_datasets":N_datasets}

# %%
print(Nclusters)

# %% tags=[]
s_region_count, s_art_region_count, s_art_region_count_std= np.zeros((Nclusters)), np.zeros((Nclusters)),np.zeros((Nclusters))
region_count, art_region_count, art_region_count_std= np.zeros((Nclusters)), np.zeros((Nclusters)),np.zeros((Nclusters))
for idx in np.arange(Nclusters):
    try:
        members= tree_members[idx+Nclusters]
        if (idx%500)==0:
            print(idx)
        N_members = len(members)
        s_art_region_count[idx] = N_members
        art_region_count[idx] = N_members
        if N_members>min_members:
            s_region_count[idx], s_art_region_count[idx], s_art_region_count_std[idx] = original_s_expected_density_parallel(idx, sof_dic)
            region_count[idx], art_region_count[idx], art_region_count_std[idx] = cut_expected_density_members(members, N_std, X, art_X, N_art, min_members)
            if ((s_region_count[idx] - region_count[idx])>1e-3):
                print("different!")
                print(s_region_count[idx])
                print(region_count[idx])

    except Exception as e:
        print(e)


# %%
print(idx)

# %%
print("Test")

# %%
f_match = ((np.abs(s_region_count-region_count)<1e-3).sum()/len(region_count))
print(f_match)

# %%
significance = (region_count - art_region_count)/np.sqrt( region_count + (art_region_count_std**2))
my_slow_data = {"region_count": region_count,
                "art_region_count": art_region_count,
                "art_region_count_std": art_region_count_std,
                "significance": significance}
file_base = data_params["result_path"] + data_params["data_folder"]
save_name = data_params["sig"] + "_slow"
dicf.h5py_save(file_base + save_name, my_slow_data, verbose=True, overwrite=True)

# %%
print("Test")

# %%
s_significance = (s_region_count - s_art_region_count)/np.sqrt( s_region_count + (s_art_region_count_std**2))
s_slow_data = {"region_count": s_region_count,
                "art_region_count": s_art_region_count,
                "art_region_count_std": s_art_region_count_std,
                "significance": s_significance}
file_base = data_params["result_path"] + data_params["data_folder"]
save_name = data_params["sig"] + "_slow_sof"
dicf.h5py_save(file_base + save_name, s_slow_data, verbose=True, overwrite=True)

# %%
region_diff = region_count - art_region_count
s_region_diff = s_region_count - s_art_region_count
f_match = ((np.abs(s_region_diff-region_diff)<1e-3).sum()/len(region_count))
print(f_match)

# %%
err_diff = np.sqrt( region_count + (art_region_count_std**2))
s_err_diff = np.sqrt( s_region_count + (s_art_region_count_std**2))
f_match = ((np.abs(s_err_diff-err_diff)<1e-3).sum()/len(region_count))
print(f_match)

# %%
print((region_count==0).sum()/len(region_count))

# %%
check_sig = region_diff/err_diff
check_sig[err_diff==0] = 0
check_s_sig = s_region_diff/s_err_diff
check_s_sig[s_err_diff==0] = 0
f_match = ((np.abs(check_sig-check_s_sig)<1e-3).sum()/len(region_count))
print(f_match)

# %%
s_significance[~np.isfinite(s_significance)] = -99
significance[~np.isfinite(significance)] = -99

# %%
f_match = ((np.abs(s_significance-significance)<1e-3).sum()/len(region_count))
print(f_match)

# %%
for k in my_slow_data.keys():
    print(k)
    f_match = ((np.abs(s_slow_data[k]-my_slow_data[k])<1e-3).sum()/len(region_count))
    print(f_match)

# %%
