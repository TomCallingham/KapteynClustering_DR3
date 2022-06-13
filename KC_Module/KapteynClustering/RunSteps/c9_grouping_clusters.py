import KapteynClustering.data_funcs as dataf
# from KapteynClustering import grouping_funcs as groupf
from KapteynClustering import cluster_funcs as clusterf
from KapteynClustering import grouping_funcs as groupf
from scipy.cluster.hierarchy import linkage, fcluster
import vaex
import numpy as np
from scipy.stats import chi2

from scipy.stats import ks_2samp


def group_feh(params):
    print("finding dendogram")
    data_p = params["data"]
    group_p = params["group"]
    dendo_cut = group_p["dendo_cut"]

    stars = vaex.open(data_p["labelled_sample"])

    clusters_data = dataf.read_data(fname= data_p["clusters"])
    [labels, star_sig, fClusters, cPops, C_sig] = [ clusters_data[p] for p in
                                               ["labels","star_sig", "Clusters", "cPops", "C_sig"]]



    fit_data = dataf.read_data(fname= data_p["gaussian_fits"])
    [Clusters, Pops, mean, covar] = [ fit_data[p] for p in
                                               ["Clusters", "Pops", "mean", "covariance"]]
    # %%
    for c in stars.column_names:
        if "feh" in c:
            print(c)

    # %%
    FeHs = ["feh_rave", "feh_segue", "feh_galah", "feh_lamost_lrs", "feh_apogee"]
    for f in FeHs:
        print(f,(~np.isnan(stars[f].values)).sum())


    # %%
    # %%
    c_feh = get_cluster_feh(stars)

    # %%

    # %%
    KSs, pvals, cpair = calc_Cluster_KS(Clusters, c_feh)

    # %%
    dis = np.sqrt(chi2.ppf(1-pvals, 1))
    dis[np.isinf(dis)] = 999
    D, dm = groupf.get_cluster_distance_matrix(Clusters, cluster_mean=mean, cluster_cov=covar)

    combined_dis = np.sqrt((dis**2) + (dm**2))
    combined_Z = linkage(combined_dis, 'single')


    # %%
    Grouped_Clusters = fcluster(combined_Z,t=4, criterion="distance")

    # %%
    groups = groupf.merge_clusters_labels(Clusters, Grouped_Clusters, labels)
    groups = clusterf.order_labels(groups)[0]
    import copy

    # %%
    ex_dis = copy.deepcopy(dis)
    ex_dis[pvals<0.05]=10
    ex_combined_dis = np.sqrt((ex_dis**2) + (dm**2))
    ex_combined_Z = linkage(ex_combined_dis, 'single')
    ex_Grouped_Clusters = fcluster(ex_combined_Z,t=4, criterion="distance")
    ex_groups = groupf.merge_clusters_labels(Clusters, ex_Grouped_Clusters, labels)
    ex_groups = clusterf.order_labels(ex_groups)[0]

    # %%


    # %%

    stars["feh_groups"] = groups
    stars["feh_groups_strict"] = ex_groups
    dataf.vaex_overwrite(stars, data_p["labelled_sample"])

    return

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


