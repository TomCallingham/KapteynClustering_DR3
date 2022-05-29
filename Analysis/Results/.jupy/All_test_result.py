# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %% [markdown]
# # Plot Script

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps

def result_plots(param_file, title=None):
    params = paramf.read_param_file(param_file)
    data_p = params["data"]
    min_sig=params["label"]["min_sig"]

    stars = vaex.open(data_p["sample"])

    label_data = dataf.read_data(fname= data_p["label"])
    [labels, star_sig, Groups, Pops, G_sig] = [ label_data[p] for p in
                                               ["labels","star_sig", "Groups", "Pops", "G_sig"]]

    print(len(Groups))

    N_fluff = Pops[Groups==-1]
    N_grouped = len(labels) - N_fluff
    f_grouped = N_grouped/len(labels)

    print(f"Fraction Grouped: {f_grouped}")
    
    if (G_sig>min_sig).sum()>0:
        G_colours = ps.get_G_colours(Groups)
        G_col_list = np.array([G_colours[g] for g in Groups])
        plt.figure(figsize=(8,8))
        plt.scatter(Pops[G_sig>min_sig], G_sig[G_sig>min_sig], c=G_col_list[G_sig>min_sig])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Population")
        plt.ylabel("Significance")
        for g,p,s in zip(Groups[G_sig>min_sig], Pops[G_sig>min_sig], G_sig[G_sig>min_sig]):
            plt.text(p,s,g, size=15)
        if title is not None:
            plt.title(title)
        plt.show()

    xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"], 
               ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
    Fig, axs = plot_simple_scatter(stars, xy_keys, Groups, labels)
    if title is not None:
        Fig.suptitle(title)
    plt.show()
    return


# %% [markdown]
# # Load

# %% [markdown]
# # RESULTS

# %% [markdown]
# ## Gaia Test

# %%
param_file = "../../Params/gaia_params.yaml"
result_plots(param_file, title="Gaia Test H99")

# %% [markdown]
# ## Checks
# ### SOF Gaia Test

# %%
param_file = "../../Params/sof_check_params.yaml"
result_plots(param_file, title="Sofie Gaia check")

# %% [markdown]
# ### Art check

# %%
param_file = "../../Params/art_check_params.yaml"
result_plots(param_file, title="art check")

# %% [markdown]
# ## Artificial Changes
# ### Gaia Test N match

# %%
param_file = "../../Params/N_match_gaia_params.yaml"
result_plots(param_file, title="N match Gaia Test")

# %% [markdown]
# ### Test 1000

# %%
param_file = "../../Params/1000_gaia_params.yaml"
result_plots(param_file, title="Test gaia 1000 N_art")

# %% [markdown]
# ## Potentials

# %%
param_file = "../../Params/M17_gaia_params.yaml"
result_plots(param_file, title="M17 ")

# %%
param_file = "../../Params/C20_gaia_params.yaml"
result_plots(param_file, title="M17 ")

# %% [markdown]
# ## Spaces
# ### J_space

# %%
param_file = "../../Params/J_gaia_params.yaml"
result_plots(param_file, title="J space M17")

# %% [markdown]
# ### E, Lz, Lp, circ

# %%
param_file = "../../Params/circ_gaia_params.yaml"
result_plots(param_file, title="circ 4 space")

# %% [markdown]
# ## Scaling

# %%
param_file = "../../Params/E2_gaia_params.yaml"
result_plots(param_file, title="Energy scaled half")

# %%
