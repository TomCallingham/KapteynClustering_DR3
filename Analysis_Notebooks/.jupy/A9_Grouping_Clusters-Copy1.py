# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.cluster_funcs as clusterf

# %%
from KapteynClustering.plot_funcs import plot_simple_scatter
from KapteynClustering.plotting_utils import plotting_scripts as ps

# %% [markdown]
# # Load

# %%
params = paramf.read_param_file("default_params.yaml")
data_p = params["data"]
min_sig=params["cluster"]["min_sig"]

stars = vaex.open(data_p["labelled_sample"])

clusters_data = dataf.read_data(fname= data_p["clusters"])
[labels, star_sig, fClusters, cPops, C_sig] = [ clusters_data[p] for p in
                                           ["labels","star_sig", "Clusters", "cPops", "C_sig"]]

# %%
cPops_dic = {c:P for (c,P) in zip(fClusters,cPops)}

# %%
Clusters = fClusters[fClusters!=-1]

# %%
for c in stars.column_names:
    if "feh" in c:
        print(c)

# %%
FeHs = ["feh_rave", "feh_segue", "feh_galah", "feh_lamost_lrs", "feh_apogee"]
for f in FeHs:
    print(f,(~np.isnan(stars[f].values)).sum())


# %%
def get_cluster_feh(stars, feh_key = "feh_lamost_lrs", labels=None):
    ''' sorted, non nan fe_h'''
    if labels is None:
        labels = stars["label"].values
    feh = np.array(stars[feh_key].values)
    good_feh= ~np.isnan(feh)

    Clusters = np.unique(labels)
    c_feh = {}
    for c in Clusters:
        c_filt = (labels==c)
        c_feh[c] = np.sort(feh[c_filt*good_feh])
    return c_feh

from scipy.stats import ks_2samp
def calc_Cluster_KS(Clusters,  c_feh):
    
    N_Clusters = len(Clusters)
    N2_Clusters = int(N_Clusters*(N_Clusters-1)/2)
    KSs = np.zeros((N2_Clusters))
    pvals = np.zeros((N2_Clusters))
    cpair = np.empty((N2_Clusters,2), dtype=int)
    k=0
    for i in range(N_Clusters):
        c1 = Clusters[i]
        for j in range(i + 1, N_Clusters):
            c2 = Clusters[j]
            cpair[k,:] = [c1,c2]
            KSs[k], pvals[k] = ks_2samp(c_feh[c1], c_feh[c2], mode='exact')
            k+=1
    return KSs, pvals, cpair


# %%
c_feh = get_cluster_feh(stars)

# %%
N_feh = np.array([len(c_feh[c]) for c in Clusters])
plt.figure()
plt.scatter(Clusters, N_feh)
plt.xlabel("Cluster")
plt.ylabel("Number FeH")
plt.yscale("log")
plt.show()

# %%
c_feh = get_cluster_feh(stars)
KSs, pvals, cpair = calc_Cluster_KS(Clusters, c_feh)

# %%
from scipy.stats import chi2
dis = np.sqrt(chi2.ppf(1-pvals, 1))
dis[np.isinf(dis)] = 999

# %%

print(np.sqrt(chi2.ppf(1-0.05, 1)))

# %%
# for i,cpair in enumerate(cluster_pairs):
#     print(cpair)
for i in np.argsort(pvals)[::-1][:20]:
    [c1,c2] = cpair[i,:]
    print(c1,c2)
    p1, p2 = cPops[c1], cPops[c2]
    title = f"{c1}: {p1}| {c2} {p2} : {pvals[i]:.4f}, dis:{dis[i]:.3f}"
    plt.figure()
    plt.title(title)
    plt.plot(c_feh[c1],np.linspace(0,1,len(c_feh[c1])), label=c1)
    plt.plot(c_feh[c2],np.linspace(0,1,len(c_feh[c2])), label=c2)
    plt.legend()
    plt.ylim([0,1])
    plt.xlabel("Fe_H")
    plt.ylabel("CDF")
    plt.show()


# %%
# fisher combine pvals!

# %%
print(dis)

# %%
plt.figure()
plt.scatter(dis[dis!=999], pvals[dis!=999])
plt.xlabel("dis")
plt.ylabel("pval")
plt.show()

# %%
print(np.max(dis[dis<999]))
print(np.min(dis))
print((~np.isnan(dis)).sum())


# %%
plt.figure()
plt.hist(dis[dis!=999],cumulative=True)
plt.show()

# %%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# %% [markdown]
# ## Dis dendo

# %%

# %%
print(len(dis))
print(dis)
dis_Z = linkage(dis, 'single')

print(np.max(dis_Z[:,2]))

print(len(dis_Z))
d_filt = dis_Z[:,2]<999
print(d_filt.sum())
plt.figure()
plt.plot(dis_Z[d_filt,2], np.linspace(0,1,d_filt.sum()))
plt.show()

# %%
plt.subplots(figsize=(20,10))
d_result = dendrogram(dis_Z, leaf_rotation=90, color_threshold=4)
plt.minorticks_off()
# plt.title(f'p_val denda')
# plt.ylim(0,10)
# plt.yscale("log")
plt.show()

# %% [markdown]
# ## p dendo

# %%
print(len(dis))
p_Z = linkage(1-pvals, 'single')

print(np.max(p_Z[:,2]))

plt.figure()
plt.plot(p_Z[:,2], np.linspace(0,1,len(p_Z[:,2])))
plt.show()

# %%
plt.subplots(figsize=(20,10))
d_result = dendrogram(p_Z, leaf_rotation=90, color_threshold=4)
plt.minorticks_off()
# plt.title(f'p_val denda')
# plt.ylim(0,10)
# plt.yscale("log")
plt.show()

# %%
plt.figure()
plt.scatter(dis_Z[:,2], p_Z[:,2])
plt.show()

# %%
print(dis_Z[:,2]/p_Z[:,2])

# %%

# %% [markdown]
# # Combined Distances!

# %%
fit_data = dataf.read_data(fname= data_p["gaussian_fits"])
[Clusters, Pops, mean, covar] = [ fit_data[p] for p in
                                           ["Clusters", "Pops", "mean", "covariance"]]

# %%
from KapteynClustering import grouping_funcs as groupf
D, dm = groupf.get_cluster_distance_matrix(Clusters, cluster_mean=mean, cluster_cov=covar)

# %%
plt.figure()
plt.scatter(dm[dis<900], dis[dis<900])
plt.xlabel("maha dm")
plt.ylabel("met KS dis")
plt.show()

# %%
# combined_dis = np.sqrt((dis**2) + (dm**2))
combined_dis = np.sqrt((dis**2) + (dm**2))

# %%
combined_Z = linkage(combined_dis, 'single')

# %%

plt.subplots(figsize=(20,10))
d_result = dendrogram(combined_Z, leaf_rotation=90, color_threshold=4)
plt.minorticks_off()
plt.title(f"combined ")
# plt.ylim(0.94,1)
# plt.yscale("log")
plt.show()

# %%
Grouped_Clusters = fcluster(combined_Z,t=4, criterion="distance")

# %%
from KapteynClustering import grouping_funcs as groupf
groups = groupf.merge_clusters_labels(Clusters, Grouped_Clusters, labels)
groups = clusterf.order_labels(groups)[0]


# %%
Groups = np.unique(groups)

# %%
import KapteynClustering.plot_funcs as pf
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"],
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
pf.plot_simple_scatter(stars,xy_keys, Groups, groups)
plt.show()

pf.plot_simple_scatter(stars,xy_keys, Clusters, labels)
plt.show()


# %% [markdown]
# # No Greater than 0.05

# %%
import copy

# %%
ex_dis = copy.deepcopy(dis)
ex_dis[pvals<0.05]=10
ex_combined_dis = np.sqrt((ex_dis**2) + (dm**2))

# %%
ex_combined_Z = linkage(ex_combined_dis, 'single')

# %%

plt.subplots(figsize=(20,10))
d_result = dendrogram(ex_combined_Z, leaf_rotation=90, color_threshold=4)
plt.minorticks_off()
plt.title(f"combined ")
# plt.ylim(0.94,1)
# plt.yscale("log")
plt.show()

# %%
ex_Grouped_Clusters = fcluster(ex_combined_Z,t=4, criterion="distance")

# %%
from KapteynClustering import grouping_funcs as groupf
ex_groups = groupf.merge_clusters_labels(Clusters, ex_Grouped_Clusters, labels)
ex_groups = clusterf.order_labels(ex_groups)[0]


# %%
ex_Groups = np.unique(ex_groups)

# %%
import KapteynClustering.plot_funcs as pf
xy_keys = [["Lz", "En"], ["Lz", "Lperp"], ["circ", "Lz"],
           ["Lperp", "En"], ["circ", "Lperp"], ["circ", "En"]]
pf.plot_simple_scatter(stars,xy_keys, ex_Groups, ex_groups)
plt.show()

pf.plot_simple_scatter(stars,xy_keys, Clusters, labels)
plt.show()

