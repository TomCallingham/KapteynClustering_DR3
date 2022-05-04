# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook takes a Gaia-style catalogue and performs clustering in integral of motion space according to Lövdal et al. (2021).
# Requires some libraries, most notably vaex, numpy and matplotlib. After changing the variable "catalogue_path" to indicate where your star catalogue is stored, you should be able to run the cells in order, and obtain a catalogue containing labels for significant clusters in integral of motion space.
#
#
# The input catalogue should contain heliocentric xyz coordinates and xyz velocity components, where necessary quality cuts has been readily imposed. The input should contain halo stars and a little bit more, as we use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

# %% [markdown]
# # Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

import sys
sys.path.append('../KapetynClustering/')
import data_funcs as dataf
import plot_funcs as plotf
import cluster_funcs as clusterf
import dic_funcs as dicf
import dynamics_funcs as dynf

from params import data_params

# %%
N_sigma_significance = 3

# %% [markdown]
# # Params

# %%
data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = data["stars"]
del stars["N_Part"]

# %%
cluster_data = dataf.read_data(fname=data_params["cluster"], data_params=data_params)
Z = cluster_data["Z"]

# %%
sig_data = dataf.read_data(fname=data_params["sig"], data_params=data_params)
significance = sig_data["significance"]

# %% [markdown]
# ## Load

# %%
tree_members = clusterf.find_tree(Z, prune=True)

# %%
N_clusters = len(Z[:, 0])
N_clusters1 = N_clusters + 1

# %% [markdown]
# ## Find Clusters

# %%
'''Select all clusters that satisfy your chosen minimum statistical significance level.
To distinguish between hierarchically overlapping significant clusters when extracting the final
cluster labels, select the cluster which displays the maximum statistical significance.'''


selected, statistical_significance = clusterf.select_maxsig_clusters_from_tree(
    significance, Z, minimum_significance=N_sigma_significance)


# %%
def get_cluster_labels_with_significance(selected, significance, tree_members, N_clusters):
    '''
    Extracts flat cluster labels given a list of statistically significant clusters
    in the linkage matrix Z, and also returns a list of their correspondning statistical significance.

    Parameters:
    significant(vaex.DataFrame): Dataframe containing statistics of the significant clusters.
    Z(np.ndarray): The linkage matrix outputted from single linkage

    Returns:
    labels(np.array): Array matching the indices of stars in the data set, where label 0 means noise
                      and other natural numbers encode significant structures.
    '''

    print("Extracting labels...")
    print(f"Initialy {len(significance)} clusters")
    N = len(tree_members.keys())
    print(N)

    labels = -np.ones((N_clusters+1))
    significance_list = np.zeros(N_clusters+1)
    # i_list = np.sort(significant.i.values)
    # Sort increasing, so last are includedIn the best cluster
    sig_sort = np.argsort(significance)
    s_significance = significance[sig_sort]
    s_index = selected[sig_sort]

    for i_cluster, sig_cluster in zip(s_index, s_significance) :
        members = tree_members[i_cluster+N_clusters+1]
        labels[members]=i_cluster
        significance_list[members] = sig_cluster

    Groups, pops = np.unique(labels, return_counts=True)
    p_sort = np.argsort(pops)[::-1] # Decreasing order
    Groups, pops = Groups[p_sort], pops[p_sort]

    for i,l in enumerate(Groups[Groups!=-1]):
        labels[labels==l] = i

    print(f'Number of clusters: {len(Groups)-1}')

    return labels, significance_list



# %%
labels, significance_list = get_cluster_labels_with_significance(selected, statistical_significance, tree_members, N_clusters)

# %%
# labels, significance_list = clusterf.get_cluster_labels_with_significance(selected, statistical_significance, tree_members,
labels, significance_list = get_cluster_labels_with_significance(selected, statistical_significance, tree_members,
                                                                         N_clusters)

# %%
import vaex
stars["labels"] = stars["group"]
del stars["N_Part"]
stars["Lperp"] = stars["Lp"]
stars["LPerp"] = stars["Lp"]
stars["Circ"] = stars["circ"]

# %%
stars["Lperp"] = stars["Lp"]
stars["Lz"] = - stars["Lz"]
stars["circ"] = -stars["Circ"]
df = vaex.from_dict(stars)
df['labels'] = labels
df['maxsig'] = significance_list

# %%
import plotting_utils as plotting_utils
from plotting_utils import get_cmap


# %%
def plot_IOM_subspaces(df, minsig=3, savepath=None, ):
    '''
    Plots the clusters for each combination of clustering features

    Parameters:
    df(vaex.DataFrame): Data frame containing the labelled sources.
    minsig(float): Minimum statistical significance level we want to consider.
    savepath(str): Path of the figure to be saved, if None the figure is not saved.
    '''

    x_axis = ['Lz/10e2', 'Lz/10e2', 'circ', 'Lperp/10e2', 'circ', 'circ']
    y_axis = ['En/10e4', 'Lperp/10e2', 'Lz/10e2', 'En/10e4', 'Lperp/10e2', 'En/10e4']

    xlabels = ['$L_z$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]', '$\eta$',
               '$L_{\perp}$ [$10^3$ kpc km/s]', '$\eta$', '$\eta$']

    ylabels = ['$E$ [$10^5$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]',
               '$E$ [$10^5$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$E$ [$10^5$ km$^2$/s$^2$]']

    fig, axs = plt.subplots(2,3, figsize = [14,8])
    plt.tight_layout()

    df_minsig = df[(df.labels>=0) & (df.maxsig>minsig)]

    unique_labels = np.unique(df_minsig.labels.values)
    cmap, norm = get_cmap(df_minsig)

    for i in range(6):
        plt.sca(axs[int(i/3), i%3])
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1,
                  length_check=False)#)#, length_limit=60000)
        df_minsig.scatter(x_axis[i], y_axis[i], s=1, c=df_minsig.labels.values,
                                cmap=cmap, norm=norm, alpha=0.6, length_check=False)

        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])

    #fig.subplots_adjust(top=0.95)
    plt.tight_layout(w_pad=1)

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()
from matplotlib import colors


# %%

def get_cmap(df_minsig):
    '''
    Returns a cmap and a norm which maps the three sigma labels to proper color bins.
    The last few colors of the colormap are very light, so we replace them for better
    visible colors.


    Parameters:
    df_minsig(vaex.DataFrame): The slice of the dataframe containing only the relevant clusters.

    Returns:
    cmap(matplotlib.colors.Colormap): The colormap we use for plotting clusters
    norm(matplotlib.colors.Normalize): Normalization bins such that unevenly spaced
                                       label values will get mapped
                                       to linearly spaced color bins
    '''
    unique_labels = np.unique(df_minsig.labels.values)
    print(unique_labels)
    cmap1 = plt.get_cmap('gist_ncar', len(unique_labels))
    cmaplist = [cmap1(i) for i in range(cmap1.N)]

    cmaplist[-1] = 'purple'
    cmaplist[-2] = 'blue'
    cmaplist[-3] = 'navy'
    cmaplist[-4] = 'gold'
    cmaplist[-5] = 'mediumvioletred'
    cmaplist[-6] = 'deepskyblue'
    cmaplist[-7] = 'indigo'

    cmap, norm = colors.from_levels_and_colors(unique_labels, cmaplist, extend='max')

    return cmap, norm


# %%
df['labels'] = labels
df['maxsig'] = significance_list

plot_IOM_subspaces(df, minsig=3, savepath=None)
# plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %% [markdown]
# # Fix Plotting

# %%
plot_IOM_subspaces(df, minsig=3, savepath=None, z_filt=True)
# plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %% [markdown]
# ## Fix Plotting Funcs

# %%
'''Add labels for subgroup membership in velocity space. Requires having the hdbscan library installed.'''

import velocity_space

if('vR' not in df.get_column_names()):
    # in case the polar coordinate velocities are missing, re-run it trough the get_gaia function
    df = importdata.get_GAIA(df)

df = velocity_space.apply_HDBSCAN_vspace(df)

# Plot an example: The subgroups for cluster nr 1
plotting_utils.plot_subgroup_vspace(df, clusterNr=2, savepath=None)

# %%
'''Compute membership probability for each star belonging to a cluster. Current method does not use XDGMM.'''

import membership_probability

df = cluster_utils.scaleData(df, features_to_be_scaled, minmax_values)
df = membership_probability.get_membership_indication_PCA(df, features=features)

# Plot example: Cluster nr 1 color coded by membership probability
plotting_utils.plot_membership_probability(df, clusterNr=2, savepath=None)

# %%
'''Get a probability matrix of any star belonging to any cluster'''

probability_table = membership_probability.get_probability_table_PCA(df, features=features)
probability_table['source_id'] = df.source_id.values

# %%
'''If desired: Export result catalogue and probability table'''

df.export_hdf5(f'{result_path}df_labels.hdf5')
probability_table.export_hdf5(f'{result_path}probability_table.hdf5')

# %%
# %%
print("Added in py")
# %%
