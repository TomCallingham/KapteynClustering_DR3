import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.cluster_funcs as clusterf
# from KapteynClustering.default_params import cluster_params0

def single_linkage(params):
    data_params = params["data"]
    cluster_params = params["cluster"]
    scales, features = cluster_params["scales"], cluster_params["features"]
    print("Applying single linkage clustering. Using the following features and scales:")
    print(scales, "\n", features)

    folder = data_params["result_folder"]
    stars = dataf.read_data(fname=folder+data_params["sample"])

    print("Now clustering")

    cluster_data = clusterf.clusterData(stars, features=features, scales = scales)

    dicf.h5py_save(data_params["result_folder"]+data_params["cluster"], cluster_data)
    print("Finished Linkage \n")
    return
