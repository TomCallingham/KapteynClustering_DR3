# %% [markdown]
# # Setup

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
# import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.cluster_funcs as clusterf

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
cluster_p = params["cluster"]
scales, features = cluster_p["scales"], cluster_p["features"]

# %% [markdown]
# # Load

# %%
stars = vaex.open( data_p["sample"])

# %%
print(stars.count())

# %% [markdown]
# # Linkage Clustering
# Takes ~10s on 5e4 sample
# Applies single linkage clustering.

# %%
print("Using the following features and scales:")
print(scales, "\n", features)

# %%
cluster_data = clusterf.clusterData(stars, features=features, scales = scales)

# %%
print(cluster_data.keys())

# %% [markdown]
# # Save

# %%
dataf.write_data(data_p["cluster"], cluster_data)

# %% [markdown]
# # Plots
# %%
Z = cluster_data["Z"]
print(np.shape(Z))
print(Z.dtype)
Z=Z.astype(int)

# %%
# Dendagram, Single linkage

# %%
from typing import Dict
def next_members(tree_dic: dict, Z: np.ndarray, i: int, N_cluster: int) -> dict:
    tree_dic[i + N_cluster] = tree_dic[Z[i, 0]] + tree_dic[Z[i, 1]]
    return tree_dic

def find_tree(Z, i_max=None, prune=True):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    # tree_dic: Dict[int, list] = {i: [i] for i in range(N_cluster1)}
    # tree_dic: Dict[int, list] = tree_dic| {i+N_cluster1: [] for i in range(N_cluster1)}
    tree_dic: Dict[int, list] = {i: [i] for i in range(2*N_cluster1)}
    # tree_dic: Dict[int, list] = tree_dic| {i+N_cluster1: [None]*i for i in range(N_cluster1)}
    # tree_dic = tree {i: [n for n in range(i)] for i in range(N_cluster1)}
    # tree_dic = {i: [i] for i in range(N_cluster1)}
    # members= {}
    if i_max is None:
        i_max = N_cluster1
    for i in range(N_cluster)[:i_max]:
        tree_dic = next_members(tree_dic, Z, i, N_cluster1)
    print("Tree found")
    if prune:
        for i in range(N_cluster1):
            del tree_dic[i]
    return tree_dic

def old_find_tree(Z, i_max=None, prune=True):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    # tree_dic: Dict[int, list] = {i: [i] for i in range(N_cluster1)}
    tree_dic: Dict[int, list] = {i: [i] for i in range(N_cluster1)}
    # tree_dic = tree {i: [n for n in range(i)] for i in range(N_cluster1)}
    # tree_dic = {i: [i] for i in range(N_cluster1)}
    # members= {}
    if i_max is None:
        i_max = N_cluster1
    for i in range(N_cluster)[:i_max]:
        tree_dic = next_members(tree_dic, Z, i, N_cluster1)
    print("Tree found")
    if prune:
        for i in range(N_cluster1):
            del tree_dic[i]
    return tree_dic


# %%
%load_ext line_profiler

# %%
%lprun -f  old_find_tree old_find_tree(Z)

# %%
%lprun -f find_tree new_tree = find_tree(Z)


# %%
def list_find_tree(Z, i_max=None, prune=True):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    # tree_list = [[i] for i in range(2*N_cluster1)]
    tree_list = [[i] for i in range(N_cluster1)]
    tree_list+= ([None]*N_cluster1)
    print(len(tree_list))
    print("made initial")
    for i in range(N_cluster):
        tree_list[i + N_cluster1] = tree_list[Z[i, 0]] + tree_list[Z[i, 1]]
    print("Tree found")
    tree_dic = {i+N_cluster1:tree_list[i+N_cluster1] for i in range(N_cluster1)}
    return tree_dic



# %%
%lprun -f list_find_tree list_tree = list_find_tree(Z)

# %%
check =  np.array([list_tree[i] == new_tree[i] for i in new_tree])
print(check)
print(check.all())

# %%
print(list(new_tree.keys())[-1])

# %%
print(new_t[102309])

# %%
