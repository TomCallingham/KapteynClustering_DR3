import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.artificial_data_funcs as adf

def art_data(params):
    print("Creating Artificial_data")
    data_params = params["data"]
    art_params = params["art"]
    folder = data_params["result_folder"]
    N_art = art_params["N_art"]
    print(f"{N_art} realisations")
    stars = dataf.read_data(fname=folder+data_params["base_dyn"])
    pot_name = data_params["pot_name"]
    art_stars = adf.get_shuffled_artificial_set(N_art, stars, pot_name)
    dicf.h5py_save(fname=folder + data_params["art"], dic=art_stars, verbose=True, overwrite=True)
    print("Finished \n")
    return
