import KapteynClustering.data_funcs as dataf
import KapteynClustering.cluster_funcs as clusterf
import vaex


def find_clusters(params):
    print("Finding clusters")
    data_p = params["data"]
    min_sig = params["cluster"]["min_sig"]

    link_data = dataf.read_data(fname=data_p["linkage"])
    Z = link_data["Z"]

    sig_data = dataf.read_data(fname=data_p["sig"])
    sig = sig_data["significance"]

    cluster_data = clusterf.find_cluster_data(sig, Z, minimum_significance=min_sig)
    print("clusters Found")
    fname = data_p["clusters"]
    dataf.write_data(fname=fname, dic=cluster_data, verbose=True, overwrite=True)
    print("clusters saved")

    print("Writing labelled sample file")
    stars = vaex.open(data_p["sample"])
    stars["label"] = cluster_data["labels"]
    stars["cluster_sig"] = cluster_data["star_sig"]

    stars.export(data_p["labelled_sample"])
    print("Labelled sample file saved")

    return
