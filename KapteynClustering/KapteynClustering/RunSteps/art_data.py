import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.artificial_data_funcs as adf

def art_data(params):
    print("Creating Artificial_data")
    data_params = params["data"]
    art_params = params["art"]
    N_art = art_params["N_art"]
    print(f"{N_art} realisations")
    data = dataf.read_data(fname=data_params["base_dyn"], data_params=data_params)
    pot_name = data_params["pot_name"]
    art_data = adf.get_shuffled_artificial_set(N_art, data, pot_name)
    fname = data_params["art"]
    folder = data_params["result_folder"]
    dicf.h5py_save(fname=folder + fname, dic=art_data, verbose=True, overwrite=True)
    print("Finished")
    return
