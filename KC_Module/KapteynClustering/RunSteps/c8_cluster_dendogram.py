import KapteynClustering.data_funcs as dataf
from KapteynClustering import grouping_funcs as groupf
from KapteynClustering import cluster_funcs as clusterf
from scipy.cluster.hierarchy import linkage, fcluster
import vaex
import numpy as np


def cluster_dendogram(params):
    print("finding dendogram")
    data_p = params["data"]
    group_p = params["group"]
    dendo_cut = group_p["dendo_cut"]

    # %%
    fit_data = dataf.read_data(fname= data_p["gaussian_fits"])
    [Clusters, Pops, mean, covar] = [ fit_data[p] for p in
                                               ["Clusters", "Pops", "mean", "covariance"]]


    # %%

    # %%


    # %%

    # %%
    D, dm = groupf.get_cluster_distance_matrix(Clusters, cluster_mean=mean, cluster_cov=covar)
    Z = linkage(dm, 'single')

    # %%

    # %%
    stars = vaex.open(data_p["labelled_sample"])
    labels = stars["label"].values

    # %%
    Grouped_Clusters = fcluster(Z,t=dendo_cut, criterion="distance")

    # %%
    groups = groupf.merge_clusters_labels(Clusters, Grouped_Clusters, labels)
    groups = clusterf.order_labels(groups)[0]

    # %%
    stars["dendo_groups"] = groups

    # %%
    dataf.vaex_overwrite(stars, data_p["labelled_sample"])

    # %%
    return
