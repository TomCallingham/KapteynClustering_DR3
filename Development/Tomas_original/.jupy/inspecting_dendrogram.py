# %%
from scipy.cluster.hierarchy import dendrogram
from scipy import stats
import pandas as pd  
import numpy as np
import sys,glob,os,time
import vaex
import vaex.ml
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from astropy.io import ascii
import matplotlib.cm as cmtime
import glob
from scipy.spatial import cKDTree
import itertools
from IPython.display import Image
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy import stats
# import seaborn as sns
from scipy.interpolate import splprep, splev, splrep
from matplotlib.patches import Ellipse
from scipy.stats import norm

import matplotlib as mpl
from scipy.cluster import hierarchy

from queue import Queue
from tqdm import tqdm

# sys.path.append(r'/net/gaia2/data/users/ruizlara/useful_codes/')
#import pot_koppelman2018 as Potential
# import H99Potential_sofie as Potential
# import lallement18_distant as l18

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import warnings
warnings.filterwarnings('ignore')

# For the dendrogram from Sofie
from scipy.cluster.hierarchy import linkage, dendrogram
from numba import jit

pi = math.pi

# %%
import time


# %%
def get_members(i, Z):
    '''
    Returns the list of members for a specific cluster included in the encoded linkage matrix
    
    Parameters:
    i(int): candidate cluster number
    Z(np.ndarray): the linkage matrix
    
    Returns:
    members(np.array): array of members corresponding to candidate cluster i
    '''
    members = []
    N = len(Z) + 1 #the number of original data points
    
    # Initializing a queue
    q = Queue(maxsize = N)
    q.put(i)
    
    while(not q.empty()):
        
        index = q.get()
        
        if(Z[index, 0]<N): 
            members.append(int(Z[index, 0])) #append original member
        else:
            q.put(int(Z[index, 0]-N)) #backtract one step, enqueue this branch
            
        if(Z[index,1]<N):
            members.append(int(Z[index,1])) #append original member
        else:
            q.put(int(Z[index, 1]-N)) #enqueue this branch
    
    return np.array(members)


# %% [markdown]
# # Now dendrogram in 3D, avoiding circularity!!!
#
# # Mahalanobis distance in 3D (probability from CDF) and re-doing everything from scratch!

# %%
# df = vaex.open('/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_3sigma.hdf5')
df = vaex.open('/net/gaia2/data/users/callingham/data/clustering/KapteynClustering_DR3/Development/emma_copy_df_labels_3D_3sigma.hdf5')
df

# %%

# %%
df.select('(labels==67)', name='bad')

# %%
plt.figure()
plt.hist(df.evaluate('dec_parallax_corr', selection='bad'), histtype='step', color='blue')
plt.hist(df.evaluate('parallax_pmdec_corr', selection='bad'), histtype= 'step', color= 'red')
plt.show()
# ((df.evaluate('dec_parallax_corr')<-0.15)|(df.evaluate('parallax_pmdec_corr')>0.15))

# %%
print(len(df.evaluate('labels', selection='bad')))
labels_array = np.array(df['labels'].values)
maxsig_array = np.array(df['maxsig'].values)
labels_array[(labels_array==67)] = 0.0
maxsig_array[(labels_array==67)] = 0.0

df.add_column('labels',labels_array)
df.add_column('maxsig',maxsig_array)
print(df.evaluate('labels', selection='bad'))
print(df.evaluate('maxsig', selection='bad'))

# %%


##########################################################################################################

def mahalanobis_distance(m1, m2, cov1, cov2):
    '''
    Computes the squared mahalanobis distance between two distributions.
    
    Parameters:
    m1(np.array): mean vector of first distribution
    m2(np.array): mean vector of second distribution
    cov1(np.ndarray): full covariance matrix of first distribution
    cov2(np.ndarray): full covariance matrix of second distribution
    
    Returns:
    md(float): the squared mahalanobis distance between the two distributions
    '''
    a = (m2-m1)
    b = np.linalg.inv((cov1+cov2))
    return a.dot(b).dot(a.T)

##########################################################################################################

# jit_md = jit(mahalanobis_distance, nopython=True, fastmath=True, cache=True)
jit_md = jit(mahalanobis_distance, nopython=True, fastmath=True, cache=False)

##########################################################################################################

def scaleData(ds, features, minmax_values):
    '''
    Scales data to the range [-1, 1].
    
    Parameters:
    ds(vaex.DataFrame): Dataframe containing the data to be scaled
    features([str]): List of features to be scaled
    minmax_values(vaex.DataFrame): Dataframe containing reference minimum and maximum 
                                     values for the scaling of the data (necessary in order to have 
                                     a fixed reference origin)
    
    Returns:
    ds(vaex.DataFrame): The original dataframe with appended columns containing the scaled features
    '''
    scaler = vaex.ml.MinMaxScaler(feature_range=[-1, 1], features=features, prefix='scaled_')
    scaler.fit(minmax_values)
    
    return scaler.transform(ds)

####################################################################################################

minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300], Lz=[-4500,4600]) 
features = ['scaled_En', 'scaled_Lperp', 'scaled_Lz']
df = scaleData(df, ['En', 'Lperp', 'Lz'], minmax_values)
three_sigma_labels = np.unique(df[df.maxsig>3].labels.values)
mapped_labels = np.array(range(len(three_sigma_labels)))


##########################################################################################################

def get_cluster_distance_matrix(df, distance_metric = 'euclidean'):
    start_time = time.time()
    
    '''
    Assumes that df has been passed as df[df.maxsig>3]
    The single linkage algorithm links groups with the smallest distance between any data point from 
    group one and two respectively. In order to remove the problem with overcrowding with data points
    in the dendrogram, we customize a distance matrix which reflects the relationship between the clusters
    according to your chosen distance metric.
    '''
    m = len(np.unique(df.labels.values)) #if you pass df[df.maxsig>3]
    dm = np.empty((m * (m - 1)) // 2, dtype=np.double)

    k = 0

    #Compute distance matrix either as mahalanobis distance between clusters, or as single linkage between clusters.
    for i in range(0, m-1):
        print(f"{i+1}/ {m-1}")
        for j in range(i + 1, m):
            
            cluster1 = df[df.labels==three_sigma_labels[i]]
            cluster2 = df[df.labels==three_sigma_labels[j]]

            U = cluster1[features].values
            V = cluster2[features].values
            
            if(distance_metric == 'euclidean'):
                mindist = 1000000
                for u in U:
                    for v in V:
                        dist = np.linalg.norm(u-v)
                        if(dist<mindist):
                            mindist = dist
                
            elif(distance_metric == 'mahalanobis'):
                #Computes Mahalanobis distance between clusters
                mindist = np.sqrt(jit_md(np.mean(U, axis=0), np.mean(V, axis=0),
                                         np.cov(U.T), np.cov(V.T)))

            dm[k] = mindist 
            k = k + 1
    end_time = time.time()
    time_taken = end_time - start_time
    print("Finished")
    print(f"Time taken {time_taken}")
        
    return dm

##########################################################################################################

def llf(id):
    '''For labelling the leaves in the dendrogram according to cluster index'''
    return str(int(three_sigma_labels[id]))

##########################################################################################################

def plot_cluster_relation(df, distance_metric):
    
    #Customize a distance matrix, as if each cluster would be a single data point.
    #You can either measure the distance between the clusters by their 'mahalanobis' distance
    #or 'euclidean' distance.
    dm = get_cluster_distance_matrix(df, distance_metric)
    
    #Perform single linkage with the custom made distance matrix
    Z = linkage(dm, 'single')

    fig, axs = plt.subplots(figsize=(20, 5))
    dendrogram(Z, leaf_label_func=llf, leaf_rotation=90, color_threshold=6)
    plt.title(f'Relationship between 3$\sigma$ clusters (single linkage by = {distance_metric} distance)')
    
    return Z, llf

##########################################################################################################
    
# Z_mahalanobis, labels_mahalanobis = plot_cluster_relation(df[df.maxsig>3], 'euclidean')
# Z_mahalanobis, labels_mahalanobis = plot_cluster_relation(df[df.maxsig>3], 'mahalanobis')


# %%
Z_mahalanobis, labels_mahalanobis = plot_cluster_relation(df[df.maxsig>3], 'mahalanobis')

# %%
# Option B:

N_struct = 6
x_pos = [74., 163.5, 200.5, 572.0, 640, 660]
y_pos = [4.3, 4.3, 4.3, 4.3, 4.3, 4.3]
x_label = ['A', 'B', 'C', 'D', 'E', 'F']

cmap = cm.jet(np.linspace(0, 1, N_struct))
cmap = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02', '#8f00ff']
hierarchy.set_link_color_palette([rgb for rgb in cmap])

fig,ax = plt.subplots(1,1,figsize=(30,6), facecolor='white')
asdf = dendrogram(Z_mahalanobis,leaf_label_func=labels_mahalanobis, color_threshold=4.0, above_threshold_color='grey')
ax.tick_params(labelsize=15)
ax.axhline(y=4.0, c='grey', lw=1, linestyle='dashed')
#plt.title(f'Relationship between 3$\sigma$ clusters (single linkage by mahalanobis distance)')
for i in np.arange(N_struct):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[i], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))#, weight='bold')

#ax.text(604, 6.5, 'E', size=16, color='grey', ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)), weight='bold')

        
ax.set_ylabel(r'$\rm Mahalanobis \; distance$', fontsize=22)
    
plt.xticks(rotation=45)

plt.ylim(0.0, 10.5)
    
plt.show()

# %%
print(asdf['ivl'])

# %%
# Option B + Gaia-Enceladus:

N_struct = 6
x_pos = [74., 163.5, 200.5, 572.0, 640, 660]
y_pos = [4.3, 4.3, 4.3, 4.3, 4.3, 4.3]
x_label = ['A', 'B', 'C', 'D', 'E', 'F']

cmap = cm.jet(np.linspace(0, 1, N_struct))
cmap = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02', '#8f00ff']
hierarchy.set_link_color_palette([rgb for rgb in cmap])

fig,ax = plt.subplots(1,1,figsize=(30,6), facecolor='white')
asdf = dendrogram(Z_mahalanobis,leaf_label_func=labels_mahalanobis, color_threshold=4.0, above_threshold_color='grey')
ax.tick_params(labelsize=15)
ax.axhline(y=4.0, c='grey', lw=1, linestyle='dashed')
#plt.title(f'Relationship between 3$\sigma$ clusters (single linkage by mahalanobis distance)')
for i in np.arange(N_struct):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[i], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))#, weight='bold')

#ax.text(604, 6.5, 'E', size=16, color='grey', ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)), weight='bold')
    
# 1:
x_pos = [53.5, 80]
y_pos = [3.4, 3.55]/np.sqrt(2)
x_label = ['A1', 'A2']
for i in np.arange(len(x_label)):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[0], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))#, weight='bold', alpha=0.8)
    
# Gaia-Enceladus labels:
x_pos = [200.5, 243, 304, 385, 502, 543]
y_pos = [4.5, 4.55, 4.55, 4.55, 3.7, 3.8]/np.sqrt(2)
x_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
for i in np.arange(len(x_label)):
    ax.text(x_pos[i], y_pos[i], x_label[i], size=16, color=cmap[2], ha="center", va="center", bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(1., 1., 1.)))#, weight='bold', alpha=0.8)

ax.set_ylabel(r'$\rm Mahalanobis \; distance$', fontsize=22)
    
plt.xticks(rotation=45)

plt.ylim(0.0, 10.5)
    
plt.show()

# %%
print(asdf['ivl'])

# %%
grouped_clusters = [[33, 12, 40, 34, 28, 32, 31, 15, 16], [42, 46, 38, 18, 30], [37, 8, 11], [6, 56, 59, 52, 14, 10, 7, 13, 20, 19, 57, 58, 54, 55, 50, 53, 47, 44, 51, 48, 45, 25, 29, 41, 49, 17, 43, 39, 36, 23, 5, 27, 22, 26, 21, 35, 9, 24], [64, 62, 63], [2, 1, 3], [60, 61], [65, 66]]
independent_clusters = [68, 4]
possible_independent_clusters = [14, ]

# %%
