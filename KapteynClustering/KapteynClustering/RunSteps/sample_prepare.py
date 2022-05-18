import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
import KapteynClustering.dic_funcs as dicf

# from params import data_params
def sample_prepare(params):
    data_params = params["data"]
    print("Finding Dynamics")
    save_dynamics(data_params)
    print("Finding Toomre Sample")
    create_toomre_example(data_params)
    print("Samples Created \n")
    return


def save_dynamics(data_params):
    stars = dataf.read_data(fname=data_params["base_data"], verbose=True)
    # Create dynamics
    pot_name = data_params["pot_name"]
    stars = dynf.add_dynamics(stars, pot_name, circ=True)
    stars = dynf.add_cylindrical(stars)
    #save dynamics
    folder = data_params["result_folder"]
    dataf.write_data(fname=folder + data_params["base_dyn"], dic=stars, verbose=True, overwrite=True)
    return

def create_toomre_example(data_params):
    folder = data_params["result_folder"]
    stars = dataf.read_data(fname=folder+data_params["base_dyn"])
    toomre_stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210)
    del stars
    toomre_stars = dicf.filt(toomre_stars, toomre_stars["En"]<0)
    dataf.write_data(fname=folder + data_params["sample"], dic=toomre_stars, verbose=True, overwrite=True)
    return
