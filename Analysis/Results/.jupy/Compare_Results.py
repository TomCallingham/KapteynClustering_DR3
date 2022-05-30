# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %%
plt.set_cmap("viridis_r")

# %%
params = paramf.read_param_file("../../Params/gaia_params.yaml")
data_p = params["data"]

label_data = dataf.read_data(fname=data_p["label"])
[labels, star_sig, fGroups, Pops, G_sig] = [label_data[p] for p in
                                           ["labels", "star_sig", "Groups", "Pops", "G_sig"]]

# %%
# params2 = paramf.read_param_file("../../Params/1000_gaia_params.yaml")
params2 = paramf.read_param_file("../../Params/M17_gaia_params.yaml")
data_p2 = params2["data"]

label_data2 = dataf.read_data(fname=data_p2["label"])
[labels2, star_sig2, fGroups2, Pops2, G_sig2] = [label_data2[p] for p in
                                                ["labels", "star_sig", "Groups", "Pops", "G_sig"]]

# %%
Pop_dic = {g:p for (g,p) in zip(fGroups,Pops)}
Pop_dic2 = {g:p for (g,p) in zip(fGroups2,Pops2)}
# Groups = fGroups[fGroups!=-1]
# Groups2 = fGroups2[fGroups2!=-1]
Groups = np.roll(fGroups, shift=-1)
Groups2 = np.roll(fGroups2, shift=-1)


# %% [markdown]
# # Similarity Matrix

# %%
def member_indices(labels, Groups):
    member_dic = {}
    for g in Groups:
        member_dic[g] = np.where((labels == g))[0]
    return member_dic


# %%
member_dic = member_indices(labels, Groups)
member_dic2 = member_indices(labels2, Groups2)


# %%
def find_sim_cont_Matrix(Groups, Groups2, member_dic, member_dic2):
    Ng = len(Groups)
    Ng2 = len(Groups2)
    sim_M= np.zeros((Ng, Ng2))
    cont_M = np.zeros((Ng, Ng2))
    for i1, g1 in enumerate(Groups):
        for i2, g2 in enumerate(Groups2):
            m1 = member_dic[g1]
            p1 = len(m1)
            m2 = member_dic2[g2]
            p2 = len(m2)
            N_intersect = np.isin(m1, m2).sum()
            # print(m1)
            # print(N_intersect, p1, p2)
            # contained_Matrix[i1, i2] = N_intersect / np.min([p1, p2])
            cont_M[i1, i2] = N_intersect / p1
            sim_M[i1, i2] = N_intersect / np.mean([p1, p2])
    return sim_M, cont_M

def find_plot_bins(Groups, Pop_dic):
    Ng = len(Groups)
    pop_list = np.sqrt(np.array([Pop_dic[g] for g in Groups]))
    plot_bins = np.zeros((Ng + 1))
    plot_bins[1:] = np.cumsum(pop_list)
    plot_cents = 0.5*(plot_bins[1:] + plot_bins[:-1])
    # pop_list2 = np.log10(np.array([Pops2[g] for g in Groups2]))
    # pop_list2 = np.array([Pops2[g] for g in Groups2])
    return plot_bins, plot_cents

sim_M, cont_M = find_sim_cont_Matrix(Groups, Groups2, member_dic, member_dic2)
plot_bins, plot_cents = find_plot_bins(Groups, Pop_dic)
plot_bins2, plot_cents2 = find_plot_bins(Groups2, Pop_dic2)

# %%
plt.figure(figsize=(12,12))
mcont_M= np.ma.masked_where(cont_M==0,cont_M).T
plt.pcolor(plot_bins, plot_bins2, mcont_M)
for p in plot_bins:
    plt.axvline(p, c="lightgrey", alpha=0.5)
for p in plot_bins2:
    plt.axhline(p, c="lightgrey", alpha=0.5)
    
for x,g in zip(plot_cents,Groups):
    if g in [0,1,-1]:
        print(g,x)
        plt.text(x,0,g)
    
plt.colorbar()
plt.show()


# %% [markdown]
# # Ordering 1

# %%
def order_Groups(Groups, cont_M):
    g_index_dic = {g:i for i,g in enumerate(Groups)}
    g_filt = (Groups!=-1)
    N_unplaced = g_filt.sum()
    o_Groups = []
    while N_unplaced>0:
        g = Groups[g_filt][0]
        o_Groups.append(g)
        i = g_index_dic[g] 
        g_filt = g_filt*(Groups!=g)
        g_column = cont_M[i,:-1]
        g_column = (g_column>0).astype(float)
        shrank_cont_M = cont_M[g_filt,:-1]
        g_row = np.sum(g_column[None,:]*shrank_cont_M,axis=1)

        nonzero_filt = (g_row>0)
        g_row = g_row[nonzero_filt]
        add_groups= Groups[g_filt][nonzero_filt]
        i_order = np.argsort(g_row)[::-1]
        o_Groups.extend(add_groups)
        g_filt = g_filt* np.isin(Groups, np.array(o_Groups), invert=True)
        N_unplaced = g_filt.sum()

    o_Groups.append(-1)
    o_Groups = np.array(o_Groups)
    o_index = np.array([g_index_dic[g] for g in o_Groups])
    o_cont_M = cont_M[o_index,:]
    return o_Groups, o_cont_M


# %%
def row_block_rearange(Groups1, Groups2, cont_M):
    g2_index_dic = {g:i for i,g in enumerate(Groups2)}
    row_filt = (Groups2!=-1)
    o_G2 = []
    for i1,g1 in enumerate(Groups1[Groups1!=-1]):
        if row_filt.sum()>0:
            filtered_column = cont_M[i1,row_filt]
            # print(filtered_column)
            f_i2_max = np.argmax(filtered_column)
            o_G2.append(Groups2[row_filt][f_i2_max])
            row_filt = row_filt * np.isin(Groups2,np.array(o_G2),invert=True)
    o_G2.extend(Groups2[row_filt])
    o_G2.append(-1)
    o_G2 = np.array(o_G2)
    o_g2_index = np.array([g2_index_dic[g] for g in o_G2])
    o_cont_M = cont_M[:,o_g2_index]
        
    return o_G2, o_cont_M


# %%
Ng = len(Groups)
frac_grouped = 1- cont_M[:,-1]
frac_grouped[-1] = -99
i_sort = np.argsort(frac_grouped)[::-1]
o_Groups = Groups[i_sort]
o_cont_M = cont_M[i_sort,:]
o_Groups2= copy.deepcopy(Groups2)

# %%
import copy

# %%
o_Groups, o_cont_M = Groups, cont_M
o_Groups, o_cont_M = order_Groups(o_Groups, o_cont_M)
o_Groups2= copy.deepcopy(Groups2)

# %%
for i in range(3):
    o_Groups2, o_cont_M = row_block_rearange(o_Groups, o_Groups2, o_cont_M)
    o_Groups2, o_cont_M = order_Groups(o_Groups2, o_cont_M.T)
    o_cont_M = o_cont_M.T

# %%
o_plot_bins, o_plot_cents = find_plot_bins(o_Groups, Pop_dic)
o_plot_bins2, o_plot_cents2 = find_plot_bins(o_Groups2, Pop_dic2)

# %%
plt.figure(figsize=(12,12))
mo_cont_M= np.ma.masked_where(o_cont_M==0,o_cont_M).T
plt.pcolor(o_plot_bins, o_plot_bins2, mo_cont_M)
for p in o_plot_bins:
    plt.axvline(p, c="lightgrey", alpha=0.5)
for p in o_plot_bins2:
    plt.axhline(p, c="lightgrey", alpha=0.5)
    
for x,g in zip(o_plot_cents,o_Groups):
    if g in [0,1,-1]:
        print(g,x)
        plt.text(x,5,g)
    
plt.colorbar()
plt.show()

# %% [markdown]
# # Ordering 2
# Prioritise contained groups first

# %%
Groups, Groups2, cont_M

# %%
arg_max = np.unravel_index(np.argmax(cont_M), np.shape(cont_M))

# %%
print(cont_M[arg_max[0],arg_max[1]])

# %%

    print("loop")
