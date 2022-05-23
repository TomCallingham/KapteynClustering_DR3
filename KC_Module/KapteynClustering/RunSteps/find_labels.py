import KapteynClustering.data_funcs as dataf
import KapteynClustering.label_funcs as labelf


def find_labels(params):
    print("Finding Labels")
    data_params = params["data"]
    min_sig = params["label"]["min_sig"]

    cluster_data = dataf.read_data(fname=data_params["cluster"])
    Z = cluster_data["Z"]

    sig_data = dataf.read_data(fname=data_params["sig"])
    sig = sig_data["significance"]

    label_data = labelf.find_group_data(sig, Z, minimum_significance=min_sig)
    print("Labels Found")
    fname = data_params["label"]
    dataf.write_data(fname=fname, dic=label_data, verbose=True, overwrite=True)

    print("Labels saved")
    return
