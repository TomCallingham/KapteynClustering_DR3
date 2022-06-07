import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.data_funcs as dataf
import vaex


def cluster_fit_step(params):
    print("fitting clusters")
    data_p = params["data"]
    features = params["cluster"]["features"]

    stars = vaex.open(data_p["labelled_sample"])

    label_data = dataf.read_data(fname= data_p["label"])
    [labels, fGroups, fPops, fG_sig] = [ label_data[p] for p in
                                               ["labels", "Groups", "Pops", "G_sig"]]

    Groups, Pops, G_sig = fGroups[fGroups!=-1], fPops[fGroups!=-1], fG_sig[fGroups!=-1]

    mean_group, cov_group = clusterf.fit_clusters(stars, features, Groups, labels=labels)

    fit_data = {"Groups":Groups, "Pops":Pops, "G_sig":G_sig,
                    "mean":mean_group, "covariance:":cov_group,
                    "features":features}
    dataf.write_data(fname=data_p["gaussian_fits"], dic=fit_data)

    print("Cluster fits saved!")
    return
