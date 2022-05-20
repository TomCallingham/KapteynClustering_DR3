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

# %% [markdown]
# # Params

# %%
N_sigma_significance = 3

# %% [markdown]
# ## Load

# %%
data = dataf.read_data(fname=data_params["sample"], data_params=data_params)
stars = data["stars"]

# %%
cluster_data = dataf.read_data(fname=data_params["cluster"], data_params=data_params)
Z = cluster_data["Z"]

# %%
sig_data = dataf.read_data(fname=data_params["sig"], data_params=data_params)
significance = sig_data["significance"]

# %%
emma_stats = dataf.load_vaex_to_dic("/net/gaia2/data/users/dodd/Clustering_EDR3/cluster_in_3D_test/results/stats")

# %%
print(emma_stats)
emma_Z = np.stack((emma_stats["index1"], emma_stats["index2"])).T

# %%
print(np.shape(Z))
print(np.shape(emma_Z))

# %%
print((Z[:,0] == emma_Z[:,0]).sum()/len(Z))
print((Z[:,1] == emma_Z[:,1]).sum()/len(Z))

# %%
emma_sig = emma_stats["significance"]

# %%
plt.figure()
plt.scatter(significance, emma_sig)
plt.xlabel("my_sig")
plt.ylabel("emma sig/ my")
plt.show()

# %%
print("Using Emma Sig")
significance=emma_sig

# %% [markdown]
# ## Find Clusters

# %%
tree_members = clusterf.find_tree(Z, prune=True)

# %%
N_clusters = len(Z[:, 0])
N_clusters1 = N_clusters + 1

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
# ### LOAD EMMAa

# %%
emma_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
emma_results = dataf.load_vaex_to_dic(f"{emma_result_path}df_labels_3D")

e_labels = emma_results["labels"]
e_sig_list = emma_results["maxsig"]

# %%
e_Groups, e_Pops = np.unique(e_labels, return_counts=True)
e_G_sig = np.array([e_sig_list[e_labels==g][0] for g in e_Groups])
print(len(e_Groups))

# %%
oe_labels = clusterf.order_labels(e_labels, fluff_label=0)

# %%
print((labels==oe_labels).sum()/len(labels))

# %% [markdown]
# # RESULTS 

# %%
 
Groups, Pops = np.unique(labels[labels!=-1], return_counts=True)
G_sig = np.array([significance_list[labels==g][0] for g in Groups])
print(len(Groups))
print(len(Groups[G_sig>N_sigma_significance]))

# %%
plt.figure(figsize=(8,8))
plt.scatter(Pops[G_sig>N_sigma_significance], G_sig[G_sig>N_sigma_significance])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Groups[G_sig>N_sigma_significance], Pops[G_sig>N_sigma_significance], G_sig[G_sig>N_sigma_significance]):
    plt.text(p,s,g, size=15)
plt.show()

# %% [markdown]
# ## Plots

# %%
stars["groups"] = labels

# %%
from matplotlib import colors
def plot_IOM_subspaces(stars, minsig=3, savepath=None, flip=False):
    '''
    Plots the clusters for each combination of clustering features

    Parameters:
    df(vaex.DataFrame): Data frame containing the labelled sources.
    minsig(float): Minimum statistical significance level we want to consider.
    savepath(str): Path of the figure to be saved, if None the figure is not saved.
    
    '''
    xkeys =["Lz", "Lz", "circ", "Lp", "circ", "circ"]
    ykeys =["E", "Lp", "Lz", "E", "Lp", "E"]


    # fig, axs = plt.subplots(2, 3, figsize=(12,6))#[27,15])
    fig, axs = plt.subplots(2, 3, figsize=[27,15])
    plt.tight_layout()
    size=20
    #original s=0.5, alpha=0.1 but vaex?

    # cmap, norm = plotting_utils.get_cmap(df_minsig)
    def prop_select(stars,g, xkey, ykey, flip):
        g_filt = (stars["groups"] == g)
        x = stars[xkey][g_filt]
        y = stars[ykey][g_filt]
        if flip:
            if xkey in ["Lz", "circ"]:
                x = -x
            if ykey in ["Lz", "circ"]:
                y=-y
        return x,y

    for i,(xkey,ykey) in enumerate(zip(xkeys, ykeys)):
        plt.sca(axs[int(i / 3), i % 3])
        for j,(g,p,s) in enumerate(zip(Groups, Pops, G_sig)):
            label  = f"{g}|{s:.1f} : {p}"
            x, y = prop_select(stars,g, xkey, ykey,flip)
            plt.scatter(x, y, label=label,
                       alpha=0.5,s=size, edgecolors="none", zorder=-j)
        
        g=-1
        fluff_pop = (stars["groups"]==g).sum()
        label  = f"fluff|{fluff_pop}"
        x, y= prop_select(stars,g, xkey, ykey,flip)
        plt.scatter(x, y, label=label,
                   alpha=0.2,s=size, edgecolors="none", zorder=-(j+1), c='grey')

        plt.xlabel(xkey)
        plt.ylabel(ykey)

    # fig.subplots_adjust(top=0.95)
    plt.tight_layout(w_pad=1)
    # plt.legend()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()
    return

# %%
plot_IOM_subspaces(stars, minsig=N_sigma_significance, savepath=None, flip=True)
# plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %% [markdown]
# # Original Plotting

# %%
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
    

# %%
emma_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
emma_results = dataf.load_vaex_to_dic(f"{emma_result_path}df_labels_3D")
e_labels = emma_results["labels"]
e_sig_list = emma_results["maxsig"]

# %%
from KapteynClustering.legacy.vaex_funcs import vaex_from_dict
df = vaex_from_dict(stars)
df["labels"] = e_labels
df["maxsig"] = e_sig_list
from KapteynClustering.legacy import plotting_utils

# %%
plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)

# %%
