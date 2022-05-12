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
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf
import KapteynClustering.cluster_funcs as clusterf

from params import data_params, gaia2, auriga

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
labels, significance_list = clusterf.get_cluster_labels_with_significance(
    selected, statistical_significance, tree_members, N_clusters)

# %% [markdown]
# # RESULTS 

# %%
Groups, Pops = np.unique(labels[labels!=-1], return_counts=True)
G_sig = np.array([significance_list[labels==g][0] for g in Groups])
print(len(Groups))

# %%
plt.figure(figsize=(8,8))
plt.scatter(Pops[G_sig>3], G_sig)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Groups, Pops, G_sig):
    plt.text(p,s,g, size=15)
plt.show()

# %% [markdown]
# ## Plots

# %%
stars["groups"] = labels
stars["Lperp"] = stars["Lp"]
stars["circ"] = stars["Circ"]
print(stars.keys())

# %%
from matplotlib import colors
def plot_IOM_subspaces(stars, minsig=3, savepath=None, ):
    '''
    Plots the clusters for each combination of clustering features

    Parameters:
    df(vaex.DataFrame): Data frame containing the labelled sources.
    minsig(float): Minimum statistical significance level we want to consider.
    savepath(str): Path of the figure to be saved, if None the figure is not saved.
    
    '''
    xkeys =["Lz", "Lz", "Circ", "Lp", "Circ", "Circ"]
    ykeys =["En", "Lp", "Lz", "En", "Lp", "En"]


    fig, axs = plt.subplots(2, 3, figsize=[27,15])
    plt.tight_layout()
    size=10

    # cmap, norm = plotting_utils.get_cmap(df_minsig)

    for i,(x,y) in enumerate(zip(xkeys, ykeys)):
        plt.sca(axs[int(i / 3), i % 3])
        for j,(g,p,s) in enumerate(zip(Groups, Pops, G_sig)):
            g_filt = (stars["groups"] == g)
            label  = f"{g}|{s:.1f} : {p}"
            plt.scatter(stars[x][g_filt], stars[y][g_filt], label=label,
                       alpha=0.5,s=size, edgecolors="none", zorder=j)
        
#         df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1,
#                    length_check=False)  # )#, length_limit=60000)
#         df_minsig.scatter(x_axis[i], y_axis[i], s=1, c=df_minsig.labels.values,
#                           cmap=cmap, norm=norm, alpha=0.6, length_check=False)
        g_filt = (stars["groups"] == -1)
        p = Pops[Groups==-1]
    
        label  = f"{g}|Fluff : {p}"
        plt.scatter(stars[x][g_filt], stars[y][g_filt], label=label,
                   alpha=0.5,s=size, edgecolors="none", zorder=-1, c='grey')

        plt.xlabel(x)
        plt.ylabel(y)

    # fig.subplots_adjust(top=0.95)
    plt.tight_layout(w_pad=1)
    # plt.legend()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()
    return

# %%
plot_IOM_subspaces(stars, minsig=3, savepath=None)
# plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %% [markdown]
# # Original Plotting

# %%
import vaex
df = vaex.from_dict(stars)
df["labels"] = labels
df["maxsig"] = significance_list
from KapteynClustering import plotting_utils

# %%
plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)
