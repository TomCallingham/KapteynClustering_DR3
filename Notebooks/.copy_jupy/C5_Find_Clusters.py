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
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

from params import data_params

# %%
N_sigma_significance = 3

# %% [markdown]
# # Params

# %%
data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = data["stars"]

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
labels, significance_list = clusterf.get_cluster_labels_with_significance(selected, statistical_significance, tree_members, N_clusters)

# %%
import vaex


# %%
stars["Lperp"] = stars["Lp"]
stars["Lz"] = - stars["Lz"]
stars["circ"] = -stars["Circ"]
df = vaex.from_dict(stars)
df['labels'] = labels
df['maxsig'] = significance_list

# %%
from KapteynClustering import plotting_utils


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
    cmap, norm = plotting_utils.get_cmap(df_minsig)

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
df['labels'] = labels
df['maxsig'] = significance_list

plot_IOM_subspaces(df, minsig=3, savepath=None)
# plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %% [markdown]
# # Fix Plotting


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
