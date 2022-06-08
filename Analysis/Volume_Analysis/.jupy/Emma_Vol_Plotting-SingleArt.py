# %% [markdown]
# # Scripts

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %%
param_file = "../../Params/emma_volumes_params_single_art.yaml"

# %%
volumes, vol_stars, vol_labels = paramf.load_volume_stars(param_file)

# %%
from KapteynClustering.plot_funcs import vol_simple_scatter


# %% [markdown]
# # Plots

# %%
available_props = list(vol_stars[volumes[0]].column_names)
print(available_props)

# %%
xy_key = ["Lz", "En"]
vol_simple_scatter(volumes, vol_stars, vol_labels, xy_key, NCol=3)
plt.show()


# %%
xy_key = ["Lz", "Lperp"]
vol_simple_scatter(volumes, vol_stars, vol_labels, xy_key, NCol=3)
plt.show()

# %%
xy_key = ["vx", "vy"]
vol_simple_scatter(volumes, vol_stars, vol_labels, xy_key, NCol=3)
plt.show()

# %%
xy_key = ["vR", "vT"]
vol_simple_scatter(volumes, vol_stars, vol_labels, xy_key, NCol=3)
plt.show()

# %%
xy_key = ["vx", "vz"]
vol_simple_scatter(volumes, vol_stars, vol_labels, xy_key, NCol=3)
plt.show()

# %%
