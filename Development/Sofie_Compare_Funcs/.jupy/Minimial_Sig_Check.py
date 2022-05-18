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

# %% jupyter={"source_hidden": true} tags=[]
import numpy as np
import sys,glob,os,time
import vaex, vaex.ml, vaex.ml.cluster
import importdata, cluster_utils, plotting_utils

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%config IPCompleter.use_jedi = False
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

from importlib import reload
reload(importdata)
reload(cluster_utils)
reload(plotting_utils)
np.random.seed(0)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Load and Params

# %%
'''PARAMETERS
Default values according to Lövdal et al. (2021),
but can be changed according to your specific preferences.
The default values should be solid for basic usage.
Change catalogue_path: Add path to where your raw data is stored.'''

N_datasets = 100 #Number of artificial datasets generated
linkage_method='single' #Linkage method for clustering
N_sigma_significance = 2 #Criterion for statistical significance of a cluster - want 2 sigma clusters info 22/10/21

features = ['scaled_En', 'scaled_Lperp', 'scaled_Lz'] #Clustering features
features_to_be_scaled = ['En', 'Lperp', 'Lz'] #Clustering features that need scaling
minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300], Lz=[-4500,4600]) #Fix origin for scaling of features

N_sigma_ellipse_axis = 2.83 #Length (in standard deviations) of axis of ellipsoidal cluster boundary
min_members = 10 #smallest cluster size that we are interested in checking (for computational efficiency)
max_members = 25000 #smallest cluster size that we are interested in checking (for computational efficiency)
N_processes = 20 #Number of processes for parallel execution

catalogue_path = '/net/gaia2/data/users/lovdal/datasample_vtoomre180.hdf5' #Path to raw data
result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored

# %%
print("CHANGED to artificial_3D")

# %%
'''Read in the data in case we already computed and stored these files
  (skipped if you run this notebook from scratch).'''

catalogues_exist = True

if(catalogues_exist):

    df = vaex.open(f'{result_path}df.hdf5')
    df = cluster_utils.scaleData(df, features_to_be_scaled, minmax_values)
    # df_artificial = vaex.open(f'{result_path}df_artificial.hdf5')
    df_artificial = vaex.open(f'{result_path}df_artificial_3D.hdf5')
    df_artificial = cluster_utils.scaleData(df_artificial, features_to_be_scaled, minmax_values)
    stats = vaex.open(f'{result_path}stats.hdf5')
    Z = cluster_utils.clusterData(df, features, linkage_method)
    
    if('region_counts' in stats.get_column_names()):
        stats['significance'] = (stats['within_ellipse_count']-stats['region_counts']) / \
                                 np.sqrt(stats['within_ellipse_count'] + np.power(stats['region_count_stds'], 2))


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Funcs

# %%
def expected_density_parallel(idx):
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


# %%
def density_within_ellipse(idx):
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


# %% [markdown]
# # Previous Sig Load

# %%
s_region_counts = stats["within_ellipse_count"].values
s_art_region_counts = stats["region_counts"].values
s_art_region_counts_std = stats["region_count_stds"].values

# %% [markdown]
# # Running

# %% tags=[]
art_region_counts = np.zeros((N_potential_clusters))
art_region_counts_std = np.zeros((N_potential_clusters))
region_counts = np.zeros((N_potential_clusters))
N_potential_clusters = stats.count()
for i in list(range(N_potential_clusters)):
    if i%500 ==0:
        print(i)
    art_region_counts[i], art_region_counts_std[i] = expected_density_parallel(i)
    region_counts[i] = density_within_ellipse(i)
    diff = np.abs(s_region_counts[i] - region_counts[i]) +  np.abs(s_art_region_counts[i] - s_art_region_counts[i])\
    +  np.abs(s_art_region_counts_std[i] - s_art_region_counts_std[i])
    if diff>1e-3:
        print("different")
        print(i)
        print(art_region_counts, art_region_counts_std)
        print(s_art_region_counts, s_art_region_counts_std)
        break


# %%
f_match = ((np.abs(s_region_counts-region_counts)<1e-3).sum()/len(region_counts))
print(f_match)

# %%
