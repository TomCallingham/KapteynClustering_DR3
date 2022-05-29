# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

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
Groups = fGroups[fGroups!=-1]
Groups2 = fGroups2[fGroups2!=-1]


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
Ng = len(Groups)
Ng2 = len(Groups2)
sim_Matrix = np.zeros((Ng, Ng2))
contained_Matrix = np.zeros((Ng, Ng2))
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
        contained_Matrix[i1, i2] = N_intersect / p2
        sim_Matrix[i1, i2] = N_intersect / np.mean([p1, p2])
# pop_list = np.log10(np.array([Pops[g] for g in Groups]))
# pop_list = np.array([Pops[g] for g in Groups])
pop_list = np.sqrt(np.array([Pops[g] for g in Groups]))
plot_pop_list = np.zeros((Ng + 1))
plot_pop_list[1:] = np.cumsum(pop_list)
# pop_list2 = np.log10(np.array([Pops2[g] for g in Groups2]))
# pop_list2 = np.array([Pops2[g] for g in Groups2])
pop_list2 = np.sqrt(np.array([Pops2[g] for g in Groups2]))
plot_pop_list2 = np.zeros((Ng2 + 1))
plot_pop_list2[1:] = np.cumsum(pop_list2)

# %%
# plt.colormaps("cividis_r")
plt.figure()
# plt.imshow(sim_Matrix)
# plt.pcolor(np.arange(Ng), np.arange(Ng2),sim_Matrix.T)
msim_Matrix = np.ma.masked_where(sim_Matrix==0,sim_Matrix).T
plt.pcolor(plot_pop_list, plot_pop_list2, msim_Matrix,
          cmap=plt.get_cmap("viridis_r"))
for p in plot_pop_list:
    plt.axvline(p, c="lightgrey", alpha=0.3)
for p in plot_pop_list2:
    plt.axhline(p, c="lightgrey", alpha=0.3)
plt.colorbar()
plt.show()

# %%
# plt.colormaps("cividis_r")
plt.figure()
# plt.imshow(contained_Matrix)
# plt.pcolor(np.arange(Ng), np.arange(Ng2),contained_Matrix.T)
mcontained_Matrix = np.ma.masked_where(contained_Matrix==0,contained_Matrix).T
plt.pcolor(plot_pop_list, plot_pop_list2, mcontained_Matrix,
          cmap=plt.get_cmap("viridis_r"))
for p in plot_pop_list:
    plt.axvline(p, c="lightgrey", alpha=0.3)
for p in plot_pop_list2:
    plt.axhline(p, c="lightgrey", alpha=0.3)
plt.colorbar()
plt.show()

# %% [markdown]
# # Similarity Plot

# %%
