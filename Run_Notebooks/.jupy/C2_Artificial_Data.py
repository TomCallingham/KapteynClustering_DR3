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
pot_name =  data_p["pot_name"]

solar_p = params["solar"]

art_p = params["art"]
N_art, additional_props = art_p["N_art"], art_p.get("additional",[])

features = params["linkage"]["features"]

# %% [markdown]
# ## Load Data
# Load the initial dynamics dataset, not the toomre cut setection.

# %%
stars = vaex.open(data_p["base_data"])
stars = dataf.create_galactic_posvel(stars, solar_params=solar_p)
stars = dynf.add_cylindrical(stars)
stars = dataf.apply_toomre_filt(stars, v_toomre_cut=180, solar_params=solar_p)

# %% [markdown]
# # Artificial DataSet
# Calculates N_art number of artificial halos by shuffling galactic vx, vz (keeping pos + vy). Recalculates dynamics. Can also propogate forward additional properties of the initial dataset.

# %%
import copy
dynamics_to_calc = copy.deepcopy(features)
dynamics_to_calc.append("circ")

print(additional_props)
print(dynamics_to_calc)

# %%
art_stars = adf.get_shuffled_artificial_set(N_art, stars, pot_name, art_v_toomre_cut=210, dynamics=dynamics_to_calc)

# %% [markdown]
# ## Save Artifical Dataset
# Note that we save and read this dataset with dataf.write_data and dataf.read_data. The artificial datasets can be different lengths, which vaex does not like. These functions write as hdf5 similar to vaex, but load all of the data into a nested dictionary and numpy array format.

# %%
dataf.write_data(fname=data_p["art"], dic=art_stars, verbose=True, overwrite=True)

# %% [markdown]
# # Plots comparing Shuffled Smooth dataset

# %%
print(art_stars.keys())
print(art_stars[0].keys())

# %%
from KapteynClustering.legacy import plotting_utils #, vaex_funcs
df_artificial = vaex.from_dict(art_stars[0])
df= stars

# %%
plotting_utils.plot_original_data(df)

# %%
plotting_utils.plot_original_data(df_artificial)

# %%
art_stars0 = art_stars[0]
sample_stars = dataf.read_data(fname=data_p["sample"])

for x in ["En", "Lz", "Lperp", "circ"]:
    plt.figure()
    plt.hist(stars[x], label="Original", bins="auto",histtype="step")
    plt.hist(sample_stars[x], label="sample", bins="auto",histtype="step")
    plt.hist(art_stars0[x], label="art", bins="auto", histtype="step")
    plt.legend()
    plt.xlabel(x)
    plt.show()

# %%

# %%

# %%

# %%

# %%

# %%
