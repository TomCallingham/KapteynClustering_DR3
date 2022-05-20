import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.artificial_data_funcs as adf
import vaex

def art_data(params):
    print("Creating Artificial_data")
    data_p = params["data"]
    art_p = params["art"]
    cluster_p = params["cluster"]
    folder = data_p["result_folder"]
    N_art = art_p["N_art"]
    print(f"{N_art} realisations")
    stars = vaex.open(folder+data_p["base_dyn"])
    pot_name = data_p["pot_name"]
    art_stars = adf.get_shuffled_artificial_set(N_art, stars, pot_name, features_to_scale=cluster_p["features"], scales= cluster_p["scales"])
    dataf.write_data(fname=folder + data_p["art"], dic=art_stars, verbose=True, overwrite=True)
    print("Finished \n")
    return
