# %% [markdown]
# # Creating the Artificial Dataset
#
# To compare the sample stellar halo to, we generate a sample of artificially smooth halos. We use vtoomre>180 when scrambling velocity components to obtain an artificial reference representation of a halo.

# %% [markdown]
# # Setup

# %%
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.artificial_data_funcs as adf
import vaex

# %% [markdown]
# ## Params

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]

# %%
stars = vaex.open(data_p["sample"])

# %%
base_stars = vaex.open(data_p["base_data"])

# %%
art_stars = dataf.read_data(fname=data_p["art"], verbose=True)
art_stars0 = vaex.from_dict(art_stars[0])

# %% [markdown]
# # Plots comparing Shuffled Smooth dataset

# %%
print(art_stars.keys())
print(art_stars[0].keys())

# %%
from KapteynClustering.legacy import plotting_utils #, vaex_funcs

# %%
plotting_utils.plot_original_data(stars)

# %%
plotting_utils.plot_original_data(art_stars0)

# %%
for x in ["En", "Lz", "Lperp", "circ"]:
    plt.figure()
    plt.hist(stars[x].values, label="sample", bins="auto",histtype="step")
    plt.hist(base_stars[x].values, label="base", bins="auto",histtype="step")
    plt.hist(art_stars0[x].values, label="art", bins="auto", histtype="step")
    plt.legend()
    plt.xlabel(x)
    plt.show()

# %%

# %%

# %%

# %%

# %%

# %%
