# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]

# %% [markdown]
# ## Load

# %%
stars = vaex.open(data_p["sample"])

# %%
label_data = dataf.read_data(fname= data_p["label"])
[labels, star_sig, Groups, Pops, G_sig] = [ label_data[p] for p in
                                           ["labels","star_sig", "Groups", "Pops", "G_sig"]]

# %% [markdown]
# ## Fix Plotting Funcs

# %% [markdown]
# # RESULTS

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
stars["labels"] = labels

# %%
from matplotlib import colors
def plot_IOM_subspaces(stars, minsig=3, savepath=None,g_key="labels"):
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
    def prop_select(stars,g, xkey, ykey,g_key):
        try:
            g_filt = (stars[g_key].values == g)
            x = stars[xkey].values[g_filt]
            y = stars[ykey].values[g_filt]
        except Exception as e:
            print(e)
            g_filt = (stars[g_key] == g)
            x = stars[xkey][g_filt]
            y = stars[ykey][g_filt]
        return x,y

    j=0
    for i,(xkey,ykey) in enumerate(zip(xkeys, ykeys)):
        plt.sca(axs[int(i / 3), i % 3])
        for j,(g,p,s) in enumerate(zip(Groups, Pops, G_sig)):
            if g!=-1:
                label  = f"{g}|{s:.1f} : {p}"
                x, y = prop_select(stars,g, g_key, xkey, ykey)
                plt.scatter(x, y, label=label,
                           alpha=0.5,s=size, edgecolors="none", zorder=-j)

        g=-1
        fluff_pop = (stars["groups"]==g).sum()
        label  = f"fluff|{fluff_pop}"
        x, y= prop_select(stars,g_key, xkey, ykey)
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
plot_IOM_subspaces(stars, minsig=min_sig, savepath=None)
# plotting_utils.plot_IOM_subspaces(df, minsig=N_sigma_significance, savepath=None)
