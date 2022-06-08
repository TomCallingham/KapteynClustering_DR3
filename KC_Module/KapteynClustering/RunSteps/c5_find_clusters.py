import KapteynClustering.data_funcs as dataf
import KapteynClustering.label_funcs as labelf
import vaex


def find_labels(params):
    print("Finding Labels")
    data_p = params["data"]
    min_sig = params["label"]["min_sig"]

    cluster_data = dataf.read_data(fname=data_p["cluster"])
    Z = cluster_data["Z"]

    sig_data = dataf.read_data(fname=data_p["sig"])
    sig = sig_data["significance"]

    label_data = labelf.find_cluster_data(sig, Z, minimum_significance=min_sig)
    print("Labels Found")
    fname = data_p["label"]
    dataf.write_data(fname=fname, dic=label_data, verbose=True, overwrite=True)

    stars = vaex.open(data_p["sample"])
    stars["label"] = label_data["labels"]
    stars["cluster_sig"] = label_data["star_sig"]

    stars.export(data_p["labelled_sample"])

    print("Labels saved")
    return
