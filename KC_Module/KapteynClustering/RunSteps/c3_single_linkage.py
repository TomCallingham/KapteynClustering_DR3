import KapteynClustering.data_funcs as dataf
import KapteynClustering.linkage_funcs as linkf
import vaex
# from Kapteynlinking.default_params import link_params0


def single_linkage(params):
    data_p = params["data"]
    link_p = params["linkage"]
    scales, features = link_p["scales"], link_p["features"]
    print("Applying single linkage linking. Using the following features and scales:")
    print(scales, "\n", features)
    print("Scaling...")
    stars = vaex.open(data_p["sample"])

    print("linking...")

    link_data = linkf.calc_linkage( stars, features=features, scales=scales)

    dataf.write_data(data_p["linkage"], link_data)
    print("Finished Linkage \n")
    return
