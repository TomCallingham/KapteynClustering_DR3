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
# # Setup

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.label_funcs as labelf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = dataf.read_param_file("gaia_params.yaml")
data_params = params["data"]
result_folder = data_params["result_folder"]
min_sig = params["label"]["min_sig"]
min_sig = 3

# %% [markdown]
# ## Load

# %%
stars = dataf.read_data(fname=data_params["result_folder"] + data_params["sample"])

# %%
cluster_data = dataf.read_data(fname=data_params["result_folder"] + data_params["cluster"])
Z = cluster_data["Z"]

# %%
# sig_data = dataf.read_data(fname=result_folder+ data_params["sig"], extra="_SOF_ART_cut")
sig_data = dataf.read_data(fname=result_folder+ data_params["sig"], extra="_SOF_cut")
sig = sig_data["significance"]

# %% [markdown]
# # Find Clusters

# %%
label_data =labelf.find_group_data(sig, Z, minimum_significance=min_sig)

# %% [markdown]
# ## Save

# %%
dicf.h5py_save(fname=result_folder + data_params["label"], dic=label_data, verbose=True, overwrite=True)

# %% [markdown]
# # RESULTS

# %%
[labels, star_sig, Groups, Pops, G_sig] = [ label_data[p] for p in 
                                           ["labels","star_sig", "Groups", "Pops", "G_sig"]]

# %%
N_fluff = Pops[Groups==-1]
N_grouped = len(labels) - N_fluff
f_grouped = N_grouped/len(labels)

print(f"Fraction Grouped: {f_grouped}")

# %%
plt.figure(figsize=(8,8))
plt.scatter(Pops[G_sig>min_sig], G_sig[G_sig>min_sig])
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Population")
plt.ylabel("Significance")
for g,p,s in zip(Groups[G_sig>min_sig], Pops[G_sig>min_sig], G_sig[G_sig>min_sig]):
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
    xkeys =["Lz", "Lz", "circ", "Lperp", "circ", "circ"]
    ykeys =["En", "Lperp", "Lz", "En", "Lperp", "En"]


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
            if g!=-1:
                label  = f"{g}|{s:.1f} : {p}"
                x, y = prop_select(stars,g, xkey, ykey,flip)
                plt.scatter(x, y, label=label,
                           alpha=0.5,s=size, edgecolors="none", zorder=-j)

        g=-1
        fluff_pop = (stars["groups"]==g).sum()
        label  = f"fluff|{fluff_pop}"
        x, y= prop_select(stars,g, xkey, ykey,flip)
        plt.scatter(x, y, label=label,
                   alpha=0.2,s=size, edgecolors="none", zorder=-(j+1), c='lightgrey')

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
plot_IOM_subspaces(stars, minsig=min_sig, savepath=None, flip=True)
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
emma_results = dataf.read_data(f"{emma_result_path}df_labels_3D")
e_labels = emma_results["labels"]
e_sig_list = emma_results["maxsig"]

# %%
from KapteynClustering.legacy.vaex_funcs import vaex_from_dict
df = vaex_from_dict(stars)
df["labels"] = e_labels
df["maxsig"] = e_sig_list
from KapteynClustering.legacy import plotting_utils

# %%
plotting_utils.plot_IOM_subspaces(df, minsig=min_sig, savepath=None)

# %%