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
params = paramf.read_param_file("../Run_Notebooks/default_params.yaml")
data_p = params["data"]
cluster_p = params["cluster"]
scales, features = cluster_p["scales"], cluster_p["features"]

# %%
cluster_data = dataf.read_data(fname=data_p["cluster"])
Z = cluster_data["Z"]

# %%
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
    tree_dic: Dict[int, list] = {i: [i] for i in range(N_cluster1)}
    if i_max is None:
        i_max = N_cluster1
    for i in range(N_cluster)[:i_max]:
        tree_dic = next_members(tree_dic, Z, i, N_cluster1)
    print("Tree found")
    if prune:
        tree_dic = {i+N_cluster1:tree_dic[i+N_cluster1] for i in range(N_cluster)}
        # for i in range(N_cluster1):
        #     del tree_dic[i]
    return tree_dic


# %%
%load_ext line_profiler

# %%
%lprun -f find_tree new_tree = find_tree(Z)


# %%
def list_find_tree(Z, i_max=None, prune=True):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    # tree_list = [[i] for i in range(2*N_cluster1)]
    tree_list = [[i] for i in range(N_cluster1)]
    tree_list+= ([None]*N_cluster)
    print(len(tree_list))
    print("made initial")
    for i in range(N_cluster):
        tree_list[i + N_cluster1] = tree_list[Z[i, 0]] + tree_list[Z[i, 1]]
    print("Tree found")
    tree_dic = {i+N_cluster1:tree_list[i+N_cluster1] for i in range(N_cluster)}
    return tree_dic



# %%
# %lprun -f list_find_tree list_tree = list_find_tree(Z)

# %%
# check =  np.array([list_tree[i] == new_tree[i] for i in new_tree])
# print(check)
# print(check.all())

# %%
from multiprocessing import Pool
from itertools import repeat

def next_members(mZ1: list, mZ2:list) -> list:
    # return [mZ1 + mZ2]
    return mZ1 + mZ2

def multi_list_find_tree(Z):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    # tree_list = [[i] for i in range(2*N_cluster1)]
    tree_list = [[i] for i in range(N_cluster1)] + ([None]*N_cluster)
    # tree_list+= ([None]*N_cluster)
    # tree_list = np.array(tree_list, dtype=object)
    
    
    allowed = np.zeros((N_cluster1+N_cluster),dtype=bool)
    # allowed[:N_cluster1] = True
    allowed[:N_cluster1] = True
    previously_allowed = np.zeros((N_cluster), dtype=bool)
    N_todo = N_cluster
    print("starting multi")
    N_process=8
    with Pool(processes=N_process) as pool:
        print("Starting loop")
        while N_todo>1:
            Z_allowed = allowed[Z[:,0]] * allowed[Z[:,1]]
            todo = Z_allowed*(~previously_allowed)
            N_todo = todo.sum()
            print("N todo:", N_todo)

            members_Z1 = [[tree_list[z1]] for z1 in Z[todo,0]]
            members_Z2 = [[tree_list[z2]] for z2 in Z[todo,1]]
            index = np.arange(N_cluster).astype(int)[todo] + N_cluster1

            # result = np.array(pool.starmap(next_members, zip(members_Z1, members_Z2)),dtype=object)
            result = pool.starmap(next_members, zip(members_Z1, members_Z2))
            
            print("Unpacking result")
            # filt = np.zeros((len(tree_list)), dtype=bool)
            # filt[N_cluster1:] = todo
            # print(np.shape(tree_list[filt]))
            # print(np.shape(result))
            # tree_list[filt] = result

            # tree_list[index] = result[np.arange(N
            for i,ind in enumerate(index):
                tree_list[ind] = result[i]

            # for ind,mz1, mz2 in zip(index,members_Z1, members_Z2):
            #     tree_list[ind] = next_members(mz1,mz2)
                # result[i] = next_members(mZ)
            # print("found results")
            # tree_list[index] = result


            allowed[N_cluster1:][todo] = True

            previously_allowed = Z_allowed
            # print("Loop done")
        
    print("single path")
    allowed =  np.array([ t!=None for t in tree_list[N_cluster1:]])
    N_not_allowed = (~allowed).sum()
    print(f"N not allowed:{N_not_allowed}")
    index = np.arange(N_cluster)[~allowed]
    for i in index:
        tree_list[i+N_cluster1] = tree_list[Z[i,0]] + tree_list[Z[i,1]]
        
    print("Single done")
    return tree_list
    print("Writing to dic")
    tree_dic = {i+N_cluster1:tree_list[i+N_cluster1] for i in range(N_cluster)}
    print("Tree found")
    return tree_dic



# %%
# %lprun -f multi_list_find_tree multi_tree = multi_list_find_tree(Z)

# %%
# check =  np.array([multi_tree[i] == new_tree[i] for i in new_tree])
# print(check)
# print(check.all())

# %%
def climb_find_tree(Z):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1
    tree_dic: Dict[int, list] = {i: [i] for i in range(N_cluster1)}
    
    Z1 = Z[:,0]
    Z2 = Z[:,1]
    N = len(Z1)
    print("Creating index_dic")
    index_dic = {Z1[n]: n for n in range(N)} | {Z2[n]: n for n in range(N)}
    
    print("Finding intial indexes")
    index_to_do = np.arange(N_cluster)[(Z1<=N_cluster)*(Z2<=N_cluster)]
    
    print("Starting loop")
    # with Pool(processes=N_process) as pool:
    for index in index_to_do:
        while True:
            try:
                m1 = tree_dic[Z1[index]]
                m2 = tree_dic[Z2[index]]
            except Exception:
                break
            tree_dic[index+N_cluster1] = m1 + m2
            try:
                index = index_dic[index+N_cluster1]
            except KeyError:
                print("Top?")
                break
            # try:
            #     m1 = tree_dic[Z1[Z_index]]
            #     m2 = tree_dic[Z2[Z_index]]
            #     index = Z_index
            # except KeyError:
            #     # print("Not allowed")
            #     break
            # if Z_allowed[Z_index]:
            #     # print("continue")
            #     # index_to_do.put(Z_index)
            # else:
            #     # print("End found")
            #     Z_allowed[Z_index] = True
            
    print("Writing to dic")
    tree_dic = {i+N_cluster1:tree_dic[i+N_cluster1] for i in range(N_cluster)}
            
    return tree_dic



# %%
%lprun -f climb_find_tree climb_tree = climb_find_tree(Z)

# %%
check =  np.array([climb_tree[i] == new_tree[i] for i in new_tree])
print(check)
print(check.all())
del check


# %%

# %%
def array_climb_find_tree(Z):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1

    print("Creating Array")
    tree_array = np.zeros((N_cluster1+N_cluster,N_cluster), dtype=bool)
    tree_array[np.arange(N_cluster),np.arange(N_cluster)] = True
    print("Array Created")
    
    Z1 = Z[:,0]
    Z2 = Z[:,1]
    N = len(Z1)
    print("Finding intial indexes")
    index_to_do = np.arange(N_cluster)[(Z1<=N_cluster)*(Z2<=N_cluster)]
    print(len(index_to_do))
    # index_to_do = np.arange(N_cluster)
    # print(len(index_to_do))
    
    print("Creating index_dic")
    index_dic = {Z1[n]: n for n in range(N)} | {Z2[n]: n for n in range(N)}
    
    tree_allowed = np.zeros((N_cluster1+N_cluster), dtype=bool)
    tree_allowed[:N_cluster1] = True
    
    print("Starting loop")
    # with Pool(processes=N_process) as pool:
    for index in index_to_do:
        while True:
            z1_index =Z1[index]
            z2_index =Z2[index]
            if (not tree_allowed[z1_index]) or  (not tree_allowed[z2_index]):
                break
            tree_array[index+N_cluster1,:] =   tree_array[z1_index,:] +  tree_array[z2_index,:]
            tree_allowed[index+N_cluster1] = True
            try:
                index = index_dic[index+N_cluster1]
            except KeyError:
                # print("Top?")
                break
    print((~tree_allowed).sum())
            
    print("Writing to dic")
    indexes = np.arange(N_cluster)
    # tree_dic = {i+N_cluster1:np.argwhere(tree_array[i+N_cluster1,:]).reshape(-1).tolist() for i in range(N_cluster)}
    # tree_dic = {i+N_cluster1:np.argwhere(tree_array[i+N_cluster1,:]).reshape(-1) for i in range(N_cluster)}
    
    tree_dic = {i+N_cluster1:indexes[tree_array[i+N_cluster1,:]] for i in range(N_cluster)}
            
    return tree_dic



# %%
%lprun -f array_climb_find_tree array_tree = array_climb_find_tree(Z)

# %%
for i in np.random.choice(list(new_tree),size=5):
    # print(array_tree[i])
    # print(sorted(new_tree[i]))
    try:
        print((array_tree[i] == np.array(sorted(new_tree[i]))).all())
    except Exception:
        print(i)
        print(array_tree[i])
        # print(sorted(new_tree[i]))
        # print((array_tree[i] == np.array(sorted(new_tree[i]))).all())

# %%
# check =  np.array([array_tree[i] == new_tree[i] for i in new_tree])
# print(check)
# print(check.all())
# del check

# %%
from multiprocessing import Pool
from itertools import repeat
def share_array_climb_find_tree(Z):
    print("Finding Tree")
    N_cluster = len(Z)
    N_cluster1 = N_cluster + 1

    print("Creating Array")
    tree_array = np.zeros((N_cluster1+N_cluster,N_cluster), dtype=bool)
    tree_array[np.arange(N_cluster),np.arange(N_cluster)] = True
    
    Z1 = Z[:,0]
    Z2 = Z[:,1]
    N = len(Z1)
    print("Finding intial indexes")
    index_to_do = np.arange(N_cluster)[(Z1<N_cluster1)*(Z2<N_cluster)]
    print(len(index_to_do))
    
    print("Creating index_dic")
    index_dic = {Z1[n]: n for n in range(N)} | {Z2[n]: n for n in range(N)}
    
    tree_allowed = np.zeros((N_cluster1+N_cluster), dtype=bool)
    tree_allowed[:N_cluster1] = True
    
    def climb_tree(index, tree_array, index_dic, tree_allowed, Z1, Z2):
        indexes_done = []
        new_members = []
        index_todo = 0
        m1 = tree_array[Z1[index],:]  
        m2 = tree_array[Z2[index],:]
        while True:
            m1 =   m1 + m2
            indexes_done.append(index)
            new_members.append(m1)
            
            old_index = index
            try:
                index = index_dic[index+N_cluster1]
            except KeyError:
                break
            
            z_index =Z1[index]
            if z_index==old_index:
                z_index =Z2[index]
            if not tree_allowed[z_index]:
                index_todo = index
                break
                
            m2 = tree_array[z_index]
            
        return indexes_done, new_members, index_todo
            
    
    print("Starting loop")
    N_to_do = len(index_to_do)
    
    # N_process=8
    # with Pool(processes=N_process) as pool:
    # results =  pool.starmap(climb_tree, zip(index_to_do, repeat((tree_array, index_dic, tree_allowed))))
    while N_to_do>0:
        print(f"N to do {N_to_do}")
        results = []
        for i in index_to_do:
            results.append(climb_tree(i, tree_array, index_dic, tree_allowed, Z1, Z2))
        index_to_do = []
        for r in results:
            index_to_do.append(r[2])
            for i, m in zip(r[0], r[1]):
                tree_array[i+N_cluster1] = m
                tree_allowed[i] = True
        index_to_do = np.unique(np.array(index_to_do))
        N_to_do = len(index_to_do)
    # for index in index_to_do:
    #     climb_tree(index)
            
    print("Writing to dic")
    indexes = np.arange(N_cluster)
    # tree_dic = {i+N_cluster1:np.argwhere(tree_array[i+N_cluster1,:]).reshape(-1).tolist() for i in range(N_cluster)}
    # tree_dic = {i+N_cluster1:np.argwhere(tree_array[i+N_cluster1,:]).reshape(-1) for i in range(N_cluster)}
    tree_dic = {i+N_cluster1:indexes[tree_array[i+N_cluster1,:]] for i in range(N_cluster)}
            
    return tree_dic

# %%
# share_tree = share_array_climb_find_tree(Z)

# %%
