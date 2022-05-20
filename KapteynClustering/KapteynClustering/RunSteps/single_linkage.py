import KapteynClustering.data_funcs as dataf
import KapteynClustering.cluster_funcs as clusterf
import vaex
# from KapteynClustering.default_params import cluster_params0

def single_linkage(params):
    data_p = params["data"]
    cluster_p = params["cluster"]
    scales, features = cluster_p["scales"], cluster_p["features"]
    print("Applying single linkage clustering. Using the following features and scales:")
    print(scales, "\n", features)

    folder = data_p["result_folder"]
    stars = vaex.open(folder+data_p["sample"])

    print("Now clustering")

    cluster_data = clusterf.clusterData(stars, features=features, scales = scales)

    dataf.write_data(data_p["result_folder"]+data_p["cluster"], cluster_data)
    print("Finished Linkage \n")
    return
