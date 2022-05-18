import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
# import KapteynClustering.cluster_funcs as clusterf
import KapteynClustering.label_funcs as labelf

def find_labels(params):
    print("Finding Labels")
    data_params = params["data"]
    min_sig = params["label"]["min_sig"]
    result_folder = data_params["result_folder"]

    cluster_data = dataf.read_data(fname=result_folder + data_params["cluster"])
    Z = cluster_data["Z"]

    sig_data = dataf.read_data(fname=result_folder + data_params["sig"])
    sig = sig_data["significance"]

    label_data =labelf.find_group_data(sig, Z, minimum_significance=min_sig)
    print("Labels Found")
    folder = data_params["result_folder"]
    fname = data_params["label"]
    dicf.h5py_save(fname=folder + fname, dic=label_data, verbose=True, overwrite=True)

    print("Labels saved")
    return

