import KapteynClustering.data_funcs as dataf
import KapteynClustering.artificial_data_funcs as adf
import vaex


def art_data(params):
    print("Creating Artificial_data")
    data_p = params["data"]
    art_p = params["art"]
    features = params["cluster"]["features"]
    N_art = art_p["N_art"]
    print(f"{N_art} realisations")
    stars = vaex.open(data_p["base_dyn"])
    pot_name = data_p["pot_name"]
    # , features_to_scale=cluster_p["features"], scales= cluster_p["scales"])
    N_match = art_p.get("N_match", False)
    if N_match:
        N_original = vaex.open(data_p["sample"]).count()
    else:
        N_original=None

    art_stars = adf.get_shuffled_artificial_set(N_art, stars, pot_name,features=features,N_original=N_original)
    dataf.write_data(fname=data_p["art"],
                     dic=art_stars, verbose=True, overwrite=True)
    print("Finished \n")
    return
