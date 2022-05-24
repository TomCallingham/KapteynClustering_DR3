# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.plot_funcs as plotf

from params import data_params, gaia2, auriga

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
