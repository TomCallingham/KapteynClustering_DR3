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
art_X_array = clusterf.find_art_X_array(features, art_stars)

# %%
N_datasets = N_art

# %% tags=[]
tree_members = clusterf.find_tree(Z, prune=True)
Nclusters = len(Z) +1

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # My Sig Find

# %%
max_members =  25000
from KapteynClustering.mahalanobis_funcs import find_mahalanobis_N_members, fit_gaussian


# %%
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
    print(xmins,xmaxs)
    eps=0.05
    filt = np.prod((X>xmins[None,:] - eps)*(X<xmaxs[None,:] + eps),axis=1).astype(bool)
    print(filt.sum())

    # Count Fit
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X[filt])
    print("no filt count: ",region_count)
    region_count = find_mahalanobis_N_members(N_std, mean, covar, X)
    print("cut region_count: ", region_count)


    # Artificial
    counts_per_halo = np.zeros((N_art))
    for n in range(N_art):
        filt = np.prod((art_X[n]>xmins[None,:] - eps)*(art_X[n]<xmaxs[None,:] + eps),axis=1).astype(bool)
        counts_per_halo[n] = find_mahalanobis_N_members(
            N_std, mean, covar, art_X[n][filt,:])
        # print("SCALING COUNTS")
        # counts_per_halo[n] = counts_per_halo[n] * \
        #     (len(X[:, 0])/len(art_X[n][:, 0]))

    art_region_count = np.mean(counts_per_halo)
    art_region_count_std = np.std(counts_per_halo)

    return np.array([region_count, art_region_count, art_region_count_std])

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# # SofKapteynClusteringe Sig Find

# %%
import cluster_utils
def original_s_expected_density_parallel(idx):
    art_region_count, art_region_count_std = original_s_only_expected_density_parallel(idx)
    region_count = original_s_density_within_ellipse(idx)
    return np.array([region_count, art_region_count, art_region_count_std])
    
def original_s_only_expected_density_parallel(idx):
    features = ["scaled_En", "scaled_Lz", "scaled_Lperp"]
    '''
    The input parameter idx specifies the thread index of the parallel execution, and 
    thread i applies the function to the cluster formed at step i of the single linkage algorithm.
    
    The function applies PCA to a cluster C.
    We then define an ellipsoidal boundary around C, where the axes lengths of the ellipsoid 
    are chosen to be n standard deviations of spread along each axis (n=N_sigma_ellipse_axis).
    We then map stars from the artificial halo to the PCA-space defined by C and use the standard
    equation of an ellipse to check how many artificial halo stars fall within the ellipsoid defined by C.
    
    Parameters:
    idx(int): Thread index of the parallel execution
    
    Returns:
    [region_count, region_count_std](np.array): The number of stars in the artificial halo
                                                in the region of cluster i, and the standard
                                                deviation in counts across the artificial
                                                halo data sets.
    '''
    
    #get a list of members (indices in our halo set) of the cluster C we are currently investigating
    members = cluster_utils.get_members(idx, Z)
    
    #ignore these clusters, return placeholder (done for computational efficiency)
    if((len(members)>max_members) or (len(members)<min_members)):
        # print(len(members))
        # print(min_members,max_members)
        return(np.array([len(members), 0]))
    
    #Get a dataframe containing the members
    df_members = df.take(np.array(members))
    
    #Fit a PCA-object according to the members of cluster C
    pca = vaex.ml.PCA(features=features, n_components=len(features))
    pca.fit(df_members)
    
    #TODO: This is still hardcoded according to four features
    [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
    df_members.minmax(['scaled_En', 'scaled_Lperp', 'scaled_Lz'])

    eps = 0.05 #large enough such that ellipse fits within

    #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
    region = df_artificial[(df_artificial.scaled_En>En_lower-eps) & (df_artificial.scaled_En<En_upper+eps) & 
                  (df_artificial.scaled_Lperp>Lperp_lower-eps) & (df_artificial.scaled_Lperp<Lperp_upper+eps) &
                  (df_artificial.scaled_Lz>Lz_lower-eps) & (df_artificial.scaled_Lz<Lz_upper+eps)]
    
    #Map the stars in the artificial data set to the PCA-space defined by C
    region = pca.transform(region)
    
    #calculate the length of the axis in each dimension of the 4D-ellipsoid
    n_std = N_sigma_ellipse_axis
    
    #TODO: Here we could automate the expression and number of axes we need depending on which features we use.
    r0 = n_std*np.sqrt(pca.eigen_values_[0])
    r1 = n_std*np.sqrt(pca.eigen_values_[1])
    r2 = n_std*np.sqrt(pca.eigen_values_[2])
    #r3 = n_std*np.sqrt(pca.eigen_values_[3])

    #Extract the artificial halo stars that reside within the ellipsoid
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) #+ (np.power(region.PCA_3, 2))/(np.power(r3, 2))
                           ) <=1]
    
    #Average count and standard deviation in this region.
    #Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
    region_count = within_region.count()/N_datasets
    counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
    region_count_std = np.std(counts_per_halo)

    return(np.array([region_count, region_count_std]))

def original_s_density_within_ellipse(idx):
    features = ["scaled_En", "scaled_Lz", "scaled_Lperp"]
    '''
    Applies PCA to a cluster to find the principal components a_i.
    Then defines cluster region as the ellipse e defined by a_i, and counts the number of
    stars in this region in the artificial data set.
    '''
    
    members = cluster_utils.get_members(idx, Z)
    
    #ignore these clusters, return placeholder (done for computational efficiency)
    if((len(members)>max_members) or (len(members)<min_members)):
        region_count = 0
        return(region_count)
    
    pca = vaex.ml.PCA(features=features, n_components=len(features))
    df_members = df.take(np.array(members))

    [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
    df_members.minmax(['scaled_En', 'scaled_Lperp', 'scaled_Lz'])

    eps = 0.05 #large enough such that ellipse fits within

    #Take out minmax-region, so that we do not have to apply PCA to the full dataset.
    region = df[(df.scaled_En>En_lower-eps) & (df.scaled_En<En_upper+eps) & 
                (df.scaled_Lperp>Lperp_lower-eps) & (df.scaled_Lperp<Lperp_upper+eps) &
                (df.scaled_Lz>Lz_lower-eps) & (df.scaled_Lz<Lz_upper+eps)]

    pca.fit(df_members)
    region = pca.transform(region)
    
    n_std = N_sigma_ellipse_axis
    
    #calculate length of radius in each dimension of the 4D-ellipse
    r0 = n_std*np.sqrt(pca.eigen_values_[0])
    r1 = n_std*np.sqrt(pca.eigen_values_[1])
    r2 = n_std*np.sqrt(pca.eigen_values_[2])
    #r3 = n_std*np.sqrt(pca.eigen_values_[3])

    #the stars that actually reside within the ellipse
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) #+ (np.power(region.PCA_3, 2))/(np.power(r3, 2))
                           ) <=1]

    region_count = within_region.count()

    return(region_count)


# %% tags=[]
def s_expected_density_parallel(idx):
    '''
    The input parameter idx specifies the thread index of the parallel execution, and 
    thread i applies the function to the cluster formed at step i of the single linkage algorithm.
    
    The function applies PCA to a cluster C.
    We then define an ellipsoidal boundary around C, where the axes lengths of the ellipsoid 
    are chosen to be n standard deviations of spread along each axis (n=N_sigma_ellipse_axis).
    We then map stars from the artificial halo to the PCA-space defined by C and use the standard
    equation of an ellipse to check how many artificial halo stars fall within the ellipsoid defined by C.
    
    Parameters:
    idx(int): Thread index of the parallel execution
    
    Returns:
    [region_count, region_count_std](np.array): The number of stars in the artificial halo
                                                in the region of cluster i, and the standard
                                                deviation in counts across the artificial
                                                halo data sets.
    '''
    members = cluster_utils.get_members(idx, Z)
    return members_s_expected_density_parallel(members)

def  members_s_expected_density_parallel(members):
    features = ["scaled_En", "scaled_Lz", "scaled_Lperp"]
    
    #get a list of members (indices in our halo set) of the cluster C we are currently investigating
    
    #ignore these clusters, return placeholder (done for computational efficiency)
    # if((len(members)>max_members) or (len(members)<min_members)):
    #     return(np.array([len(members),len(members), 0]))
    # if  len(members)<min_members:
    if((len(members)>max_members) or (len(members)<min_members)):
        return(np.array([0,len(members), 0]))
    
    #Get a dataframe containing the members
    df_members = df.take(np.array(members))
    
    #Fit a PCA-object according to the members of cluster C
    pca = vaex.ml.PCA(features=features, n_components=len(features))
    pca.fit(df_members)
    
    #TODO: This is still hardcoded according to four features
    [[En_lower, En_upper], [Lperp_lower, Lperp_upper], [Lz_lower, Lz_upper]] = \
    df_members.minmax(['scaled_En', 'scaled_Lperp', 'scaled_Lz'])

    eps = 0.05 #large enough such that ellipse fits within
    

    #Extract neighborhood of C, so that we do not have to PCA-map the full artificial dataset  
    region = df_artificial[(df_artificial.scaled_En>En_lower-eps) & (df_artificial.scaled_En<En_upper+eps) & 
                  (df_artificial.scaled_Lperp>Lperp_lower-eps) & (df_artificial.scaled_Lperp<Lperp_upper+eps) &
                  (df_artificial.scaled_Lz>Lz_lower-eps) & (df_artificial.scaled_Lz<Lz_upper+eps)]
    
    #Map the stars in the artificial data set to the PCA-space defined by C
    region = pca.transform(region)
    
    #calculate the length of the axis in each dimension of the 4D-ellipsoid
    n_std = N_sigma_ellipse_axis
    
    #TODO: Here we could automate the expression and number of axes we need depending on which features we use.
    r0 = n_std*np.sqrt(pca.eigen_values_[0])
    r1 = n_std*np.sqrt(pca.eigen_values_[1])
    r2 = n_std*np.sqrt(pca.eigen_values_[2])

    #Extract the artificial halo stars that reside within the ellipsoid
    within_region = region[((np.power(region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(region.PCA_2, 2))/(np.power(r2, 2)) <=1)]
    
    
    
    #Average count and standard deviation in this region.
    #Vaex takes limits as non-inclusive in count(), so we add a small value to the default minmax limits
    region_count = within_region.count()/N_datasets
    counts_per_halo = within_region.count(binby="index", limits=[-0.01, N_datasets-0.99], shape=N_datasets)
    region_count_std = np.std(counts_per_halo)
    # print("Sofie count per halo")
    # print(counts_per_halo)
    # print(np.mean(counts_per_halo))
    
    main_region = pca.transform(df)
    # distances = np.sqrt((np.power(true_pca_test.PCA_0.values, 2))/(np.power(r0, 2)) + (np.power(true_pca_test.PCA_1.values, 2))/(np.power(r1, 2)) +
    #                        (np.power(true_pca_test.PCA_2.values, 2))/(np.power(r2, 2)))
    within_main_region = main_region[((np.power(main_region.PCA_0, 2))/(np.power(r0, 2)) + (np.power(main_region.PCA_1, 2))/(np.power(r1, 2)) +
                           (np.power(main_region.PCA_2, 2))/(np.power(r2, 2)) <=1)]
    main_region_count = within_main_region.count()
    # print("sofie distances")
    # print(np.shape(distances))
    # print(distances[:10]*n_std)

    return(np.array([main_region_count,region_count, region_count_std]))

# %% [markdown] tags=[]
# # Testing

# %% [markdown] tags=[]
# ## Single Test

# %% tags=[]
Nclusters = len(Z) +1
#i_test = 51002, 14935, 15430, 35665
for i_test in np.random.choice(np.arange(Nclusters),size=100):
    members_test = tree_members[i_test+Nclusters]
    
    N_members = len(members_test)
    if((N_members > max_members) or (N_members < min_members)):
        pass
    else:
        print(i_test)

        os_result = original_s_expected_density_parallel(i_test)
        # s_result = s_expected_density_parallel(i_test)
        s_m_result = members_s_expected_density_parallel(members_test)
        my_result = sigf.expected_density_members(members_test, N_std=N_sigma_ellipse_axis, X=X, art_X=art_X, N_art=N_art,
                                                            min_members=min_members)
        my_cut_result = cut_expected_density_members(members_test, N_std=N_sigma_ellipse_axis, X=X, art_X=art_X, N_art=N_art,
                                                            min_members=min_members)
        my_array_result = sigf.vec_expected_density_members(members_test, N_std=N_sigma_ellipse_axis, X=X, art_X_array=art_X_array, N_art=N_art,
                                                            min_members=min_members)

        print("org sof: ",os_result)
        # print(s_result)
        print("mem sof: ",s_m_result)
        print("my  cur: ",my_result)
        print("my _cut: ",my_cut_result)
        print("my  arr: ",my_array_result)

# %%

# %% [markdown]
# ## Multiple Test

# %%
N_test =100


# %% tags=[]
def o_sofie_test(N_test =100,tree_dic=None):
    print(f"{N_test}")
    n_select = np.floor(np.linspace(0,N_clusters-1,N_test)).astype(int)
    N_members = -np.ones((N_test))
    region_counts = -np.ones((N_test))
    region_counts_std = -np.ones((N_test))
    N_clusters1 = N_clusters+1
    times = -np.ones((N_test))
    for j,n in enumerate(n_select):
        t = time.time()
        N_members[j], region_counts[j], region_counts_std[j] = original_s_expected_density_parallel(n)
        times[j] = time.time() - t
        # print(N_members[j],times[j])
    print("Finished")
    t_data = {"N_members":N_members, "region_counts":region_counts,
              "region_counts_std":region_counts_std, "times":times}
    return t_data

def sofie_test(N_test =100,tree_dic=None):
    print(f"{N_test}")
    n_select = np.floor(np.linspace(0,N_clusters-1,N_test)).astype(int)
    N_members = -np.ones((N_test))
    region_counts = -np.ones((N_test))
    region_counts_std = -np.ones((N_test))
    N_clusters1 = N_clusters+1
    times = -np.ones((N_test))
    for j,n in enumerate(n_select):
        t = time.time()
        N_members[j], region_counts[j], region_counts_std[j] = s_expected_density_parallel(n)
        times[j] = time.time() - t
        # print(N_members[j],times[j])
    print("Finished")
    t_data = {"N_members":N_members, "region_counts":region_counts,
              "region_counts_std":region_counts_std, "times":times}
    return t_data


def test_func(N_test=100, tree_dic=None, members_function=sigf.expected_density_members):
    print(f"{N_test}")
    n_select = np.floor(np.linspace(0,N_clusters-1,N_test)).astype(int)
    N_members = -np.ones((N_test))
    N_clusters1 = N_clusters+1
    region_counts = -np.ones((N_test))
    region_counts_std = -np.ones((N_test))
    times = -np.ones((N_test))
    if tree_dic is None:
        print("No tree given, finding!")
        tree_dic = find_tree(Z)
    for j,i in enumerate(n_select):
        t = time.time()
        members = tree_dic[i + N_clusters1]
        N_members[j], region_counts[j], region_counts_std[j] = members_function(members)
        times[j] = time.time() - t
        # print(N_members[j],times[j])
    print("Finished")
    t_data = {"N_members":N_members, "region_counts":region_counts,
              "region_counts_std":region_counts_std, "times":times}
    return t_data

def sofie_members_test(N_test =100,tree_dic=None):
    return test_func(N_test, tree_dic, members_function= members_s_expected_density_parallel)

# def my_test(N_test =100,tree_dic=None):
#     return test_func(N_test, tree_dic, members_function= expected_density_members)


def my_test(N_test =100,tree_dic=None):
    print(f"{N_test}")
    n_select = np.floor(np.linspace(0,N_clusters-1,N_test)).astype(int)
    N_members = -np.ones((N_test))
    N_clusters1 = N_clusters+1
    region_counts = -np.ones((N_test))
    region_counts_std = -np.ones((N_test))
    times = -np.ones((N_test))
    if tree_dic is None:
        print("No tree given, finding!")
        tree_dic = find_tree(Z)
    for j,i in enumerate(n_select):
        t = time.time()
        members = tree_dic[i + N_clusters1]
        N_members[j], region_counts[j], region_counts_std[j]  = sigf.expected_density_members(members, N_std=N_sigma_ellipse_axis, X=X, art_X=art_X, N_art=N_art,
                                                    min_members=min_members)
        times[j] = time.time() - t
        # print(N_members[j],times[j])
    print("Finished")
    t_data = {"N_members":N_members, "region_counts":region_counts,
              "region_counts_std":region_counts_std, "times":times}
    return t_data

def my_cut_test(N_test =100,tree_dic=None):
    print(f"{N_test}")
    n_select = np.floor(np.linspace(0,N_clusters-1,N_test)).astype(int)
    N_members = -np.ones((N_test))
    N_clusters1 = N_clusters+1
    region_counts = -np.ones((N_test))
    region_counts_std = -np.ones((N_test))
    times = -np.ones((N_test))
    if tree_dic is None:
        print("No tree given, finding!")
        tree_dic = find_tree(Z)
    for j,i in enumerate(n_select):
        t = time.time()
        members = tree_dic[i + N_clusters1]
        N_members[j], region_counts[j], region_counts_std[j]  = cut_expected_density_members(members, N_std=N_sigma_ellipse_axis, X=X, art_X=art_X, N_art=N_art,
                                                    min_members=min_members)
        times[j] = time.time() - t
        # print(N_members[j],times[j])
    print("Finished")
    t_data = {"N_members":N_members, "region_counts":region_counts,
              "region_counts_std":region_counts_std, "times":times}
    return t_data



def my_vec_test(N_test =100,tree_dic=None):
    print(f"{N_test}")
    n_select = np.floor(np.linspace(0,N_clusters-1,N_test)).astype(int)
    N_members = -np.ones((N_test))
    N_clusters1 = N_clusters+1
    region_counts = -np.ones((N_test))
    region_counts_std = -np.ones((N_test))
    times = -np.ones((N_test))
    if tree_dic is None:
        print("No tree given, finding!")
        tree_dic = find_tree(Z)
    for j,i in enumerate(n_select):
        t = time.time()
        members = tree_dic[i + N_clusters1]
        N_members[j], region_counts[j], region_counts_std[j]  = sigf.vec_expected_density_members(members, N_std=N_sigma_ellipse_axis, X=X, art_X_array=art_X_array, N_art=N_art,
                                                    min_members=min_members)
        times[j] = time.time() - t
        # print(N_members[j],times[j])
    print("Finished")
    t_data = {"N_members":N_members, "region_counts":region_counts,
              "region_counts_std":region_counts_std, "times":times}
    return t_data



# %% tags=[]
o_sofie_td = o_sofie_test(N_test, tree_dic=tree_members)
sofie_td = sofie_test(N_test, tree_dic=tree_members)

# %% tags=[]
# %lprun -f   s_expected_density_parallel   sofie_test(N_test, tree_dic=tree_members)

# %% tags=[]
sofie_m_td = sofie_members_test(N_test, tree_dic=tree_members)

# %% tags=[]
# %lprun -f   members_s_expected_density_parallel   sofie_members_test(N_test, tree_dic=tree_members)

# %% tags=[]
my_td= my_test(N_test, tree_dic=tree_members)

# %% tags=[]
my_cut_td= my_cut_test(N_test, tree_dic=tree_members)

# %% tags=[]
%lprun -f   sigf.expected_density_members   my_test(N_test, tree_dic=tree_members)

# %% tags=[]
from KapteynClustering.mahalanobis_funcs import find_maha
%lprun -f   find_maha   my_test(N_test, tree_dic=tree_members)

# %% tags=[]
my_vec_td= my_vec_test(N_test, tree_dic=tree_members)

# %% tags=[]
%lprun -f   sigf.vec_expected_density_members   my_vec_test(N_test, tree_dic=tree_members)

# %% [markdown]
# # Plot Compare

# %%
# from TomScripts.SaveFig import gSaveFig
plt.figure(figsize=(8,8))
plt.scatter(o_sofie_td["N_members"], o_sofie_td["times"], label = "sofie original")
plt.scatter(sofie_td["N_members"], sofie_td["times"], label = "sofie")
plt.scatter(sofie_m_td["N_members"], sofie_m_td["times"], label = "sofie members")
plt.scatter(my_td["N_members"], my_td["times"], label = "my members")
plt.scatter(my_cut_td["N_members"], my_cut_td["times"], label = "my cut members")
plt.scatter(my_vec_td["N_members"], my_vec_td["times"], label = "my vec members")
plt.xlabel("N members")
plt.ylabel("Times")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title(f"Nart = {N_art}")
# SaveFig(f"Sig_Test/N_art_{N_art}")
plt.show()

# %%
dic_td = {"my_vec":my_vec_td, "my": my_td, "my_cut": my_cut_td, "sofie":sofie_td, "o_sofie":o_sofie_td}
for k in ["N_members",  "times", "region_counts",  "region_counts_std"]:
    for td_key in dic_td.keys():
        d = dic_td[td_key]
        diff = my_td[k] - d[k]
        n_diff = (np.abs(diff)<1e-3).sum()
        f_diff = n_diff/len(diff)
        print(td_key, k, f_diff)

# %%
