import KapteynClustering.linkage_funcs as linkf
import KapteynClustering.data_funcs as dataf
import vaex


def cluster_fit_step(params):
    print("fitting clusters")
    data_p = params["data"]
    features = params["linkage"]["features"]

    stars = vaex.open(data_p["labelled_sample"])

    cluster_data = dataf.read_data(fname= data_p["clusters"])
    [labels, fClusters, fPops, fC_sig] = [ cluster_data[p] for p in
                                               ["labels", "Clusters", "cPops", "C_sig"]]

    Clusters, Pops, C_sig = fClusters[fClusters!=-1], fPops[fClusters!=-1], fC_sig[fClusters!=-1]

    mean_group, cov_group = linkf.fit_gaussian_clusters(stars, features, Clusters, labels=labels)

    fit_data = {"Clusters":Clusters, "Pops":Pops, "C_sig":C_sig,
                    "mean":mean_group, "covariance":cov_group,
                    "features":features}
    dataf.write_data(fname=data_p["gaussian_fits"], dic=fit_data)

    print("Cluster fits saved!")
    return
