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

# %%
%load_ext line_profiler

# %% tags=[]
from KapteynClustering.default_params import  cluster_params0
min_members = cluster_params0["min_members"]
max_members = cluster_params0["max_members"]
# features = cluster_params0["features"]
N_sigma_ellipse_axis = cluster_params0["N_sigma_ellipse_axis"]

# %%
print("Try N_art =100!")

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

    dataf.scale_features(stars, features = ["E", "Lz", "Lp"])
    stars["Lperp"] = stars["Lp"]
    stars["En"] = stars["E"]
    stars["scaled_En"] = stars["scaled_E"]
    stars["scaled_Lperp"] = stars["scaled_Lp"]
    df  = vaex.from_dict(stars)

# %% tags=[]
if True:
    import vaex
    s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
    df = vaex.open(f'{s_result_path}df.hdf5')
    features_to_be_scaled = ['En', 'Lperp', 'Lz'] #Clustering features that need scaling
    minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300], Lz=[-4500,4600]) #Fix origin for scaling of features
    df = cluster_utils.scaleData(df, features_to_be_scaled, minmax_values)
    
    stars = dataf.load_vaex_to_dic(f'{s_result_path}df.hdf5')
    dataf.scale_features(stars, features = ["En", "Lz", "Lperp"])
        # del stars[k]

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

# %% [markdown]
# # Funcs

# %%
# i_test = 14935
# i_test =  51002
# i_test =  15430
i_test =  35665
print(i_test)
members= tree_members[i_test+Nclusters]

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### My Sig Find

# %%
import scipy

# %%
from KapteynClustering.mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian
# def expected_density_members(members, N_std, X, art_X, N_art, min_members):
# Fit Gaussian
N_std = N_sigma_ellipse_axis
mean, covar = fit_gaussian(X[members, :])

my_evs = np.linalg.eigvals(covar)
my_s_evs = scipy.linalg.eigvals(covar)
my_eVs = np.linalg.eig(covar)

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


# %% [markdown]
# ### Sofie

# %% tags=[]

# def  members_s_expected_density_parallel(members):
    
df_members = df.take(np.array(members))

#Fit a PCA-object according to the members of cluster C
pca = vaex.ml.PCA(features=s_features, n_components=len(features))
pca.fit(df_members)

#TODO: This is still hardcoded according to four features
[[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
df_members.minmax(['scaled_En', 'scaled_Lperp', 'scaled_Lz'])

eps = 0.05 #large enough such that ellipse fits within


#Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
# region = df_artificial[(df_artificial.scaled_En>En_lower-eps) & (df_artificial.scaled_En<En_upper+eps) & 
#               (df_artificial.scaled_Lperp>Lperp_lower-eps) & (df_artificial.scaled_Lperp<Lperp_upper+eps) &
#               (df_artificial.scaled_Lz>Lz_lower-eps) & (df_artificial.scaled_Lz<Lz_upper+eps)]
region = df_artificial

#Map the stars in the artificial data set to the PCA-space defined by C
region = pca.transform(region)

#calculate the length of the axis in each dimension of the 4D-ellipsoid
n_std = N_sigma_ellipse_axis

#TODO: Here we could automate the expression and number of axes we need depending on which features we use.
r0 = n_std*np.sqrt(pca.eigen_values_[0])
r1 = n_std*np.sqrt(pca.eigen_values_[1])
r2 = n_std*np.sqrt(pca.eigen_values_[2])
sof_evs = pca.eigen_values_

#Extract the artificial halo stars that reside within the ellipsoid
within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                       (np.power(region.PCA_2, 2))/(np.power(r2, 2)) <=1)]



#Average count and standard deviation in this region.
#Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
s_art_region_count = within_region.count()/N_datasets
s_counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
s_art_region_count_std = np.std(s_counts_per_halo)

main_region = pca.transform(df)
# distances = np.sqrt((np.power(true_pca_test.PCA_0.values, 2))/(np.power(r0, 2)) + (np.power(true_pca_test.PCA_1.values, 2))/(np.power(r1, 2)) +
#                        (np.power(true_pca_test.PCA_2.values, 2))/(np.power(r2, 2)))
within_main_region = main_region[((np.power(main_region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(main_region.PCA_1, 2))/(np.power(r1, 2)) +
                       (np.power(main_region.PCA_2, 2))/(np.power(r2, 2)) <=1)]
main_s_region_count = within_main_region.count()
# print("sofie distances")
# print(np.shape(distances))
# print(distances[:10]*n_std)


# %%
print(len(members))

# %%

print("My counts per halo:")
# print(counts_per_halo)
print(art_region_count)
print(art_region_count_std)

print("Sofie count per halo")
# print(s_counts_per_halo)
print(s_art_region_count)
print(s_art_region_count_std)


# %% [markdown]
# # Find bad

# %%
def wrong_fit_gaussian(X):
    mean = np.mean(X, axis=0)
    covar = np.cov(X, rowvar=0)
    return mean, covar


# %% tags=[]
for i_test in np.random.choice(np.arange(Nclusters),size=500):
    members= tree_members[i_test+Nclusters]
    print(i_test)
    if len(members)>5:
        
        ### MEE
        N_std = N_sigma_ellipse_axis
        mean, covar = fit_gaussian(X[members, :])
        
        xmins = np.min(X[members,:],axis=0)
        xmaxs = np.max(X[members,:],axis=0)
        eps=0.05
        filt = np.prod((X>xmins[None,:] - eps)*(X<xmaxs[None,:] + eps),axis=1).astype(bool)
        # Count Fit
        region_count = find_mahalanobis_N_members(N_std, mean, covar, X[filt])
        
        # # Count Fit
        # region_count = find_mahalanobis_N_members(N_std, mean, covar, X[filt])
        

        # Artificial
        counts_per_halo = np.zeros((N_art))
        for n in range(N_art):
            filt = np.prod((art_X[n]>xmins[None,:] - eps)*(art_X[n]<xmaxs[None,:] + eps),axis=1).astype(bool)
            counts_per_halo[n] = find_mahalanobis_N_members(
                N_std, mean, covar, art_X[n][filt,:])
            # counts_per_halo[n] = find_mahalanobis_N_members(
            #     N_std, mean, covar, art_X[n])
            # print("SCALING COUNTS")
            # counts_per_halo[n] = counts_per_halo[n] * \
            #     (len(X[:, 0])/len(art_X[n][:, 0]))

        art_region_count = np.mean(counts_per_halo)
        art_region_count_std = np.std(counts_per_halo)
        
        ###SOFIE

        df_members = df.take(np.array(members))
        pca = vaex.ml.PCA(features=s_features, n_components=len(features))
        pca.fit(df_members)
        [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
        df_members.minmax(['scaled_En', 'scaled_Lperp', 'scaled_Lz'])
        eps = 0.05 #large enough such that ellipse fits within


        #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
        region = df_artificial[(df_artificial.scaled_En>En_lower-eps) & (df_artificial.scaled_En<En_upper+eps) & 
                      (df_artificial.scaled_Lperp>Lperp_lower-eps) & (df_artificial.scaled_Lperp<Lperp_upper+eps) &
                      (df_artificial.scaled_Lz>Lz_lower-eps) & (df_artificial.scaled_Lz<Lz_upper+eps)]
        # region = df_artificial

        #Map the stars in the artificial data set to the PCA-space defined by C
        region = pca.transform(region)

        #calculate the length of the axis in each dimension of the 4D-ellipsoid
        n_std = N_sigma_ellipse_axis

        #TODO: Here we could automate the expression and number of axes we need depending on which features we use.
        r0 = n_std*np.sqrt(pca.eigen_values_[0])
        r1 = n_std*np.sqrt(pca.eigen_values_[1])
        r2 = n_std*np.sqrt(pca.eigen_values_[2])
        sof_evs = pca.eigen_values_

        #Extract the artificial halo stars that reside within the ellipsoid
        within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                               (np.power(region.PCA_2, 2))/(np.power(r2, 2)) <=1)]



        #Average count and standard deviation in this region.
        #Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
        s_art_region_count = within_region.count()/N_datasets
        s_counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
        s_art_region_count_std = np.std(s_counts_per_halo)

        main_region = pca.transform(df)
        within_main_region = main_region[((np.power(main_region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(main_region.PCA_1, 2))/(np.power(r1, 2)) +
                               (np.power(main_region.PCA_2, 2))/(np.power(r2, 2)) <=1)]
        main_s_region_count = within_main_region.count()

        diff_counts =  counts_per_halo - s_counts_per_halo
        n_diff = (np.abs(diff_counts)>1e-3).sum()
        n_diff += (np.abs(s_art_region_count_std - art_region_count_std)>1e-3).sum()
        if n_diff>0:
            print("DIFFERENT")
            print(diff_counts)
            print(len(members))

            print("My counts per halo:")
            # print(counts_per_halo)
            print(art_region_count)
            print(art_region_count_std)

            print("Sofie count per halo")
            # print(s_counts_per_halo)
            print(s_art_region_count)
            print(s_art_region_count_std)
            break
print("Finished")


# %%
