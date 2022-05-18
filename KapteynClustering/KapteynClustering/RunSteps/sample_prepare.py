import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.dic_funcs as dicf
import KapteynClustering.data_funcs as dataf

# from params import data_params
def sample_prepare(params):
    data_params = params["data"]
    save_dynamics(data_params)
    create_toomre_example(data_params)
    print("Finished")
    return


def save_dynamics(data_params):
    data = dicf.h5py_load(fname=data_params["base_data"], verbose=True)
    selection = data["selection"]
    stars = data["stars"]
    # Create Dynamics
    pot_name = data_params["pot_name"]
    stars = dynf.add_dynamics(stars, pot_name, circ=True)
    stars = dynf.add_cylindrical(stars)
    # stars,scales = dataf.scale_features(stars)
    # data["scale"] = scales
    data["selection"] = selection
    data["stars"] = stars
    fname = data_params["base_dyn"]
    folder = data_params["result_folder"]
    dicf.h5py_save(fname=folder + fname, dic=data, verbose=True, overwrite=True)
    return

def create_toomre_example(data_params):
    data = dataf.read_data(fname=data_params["base_dyn"], data_params=data_params)
    data = dataf.apply_toomre_filt_dataset(data, v_toomre_cut=210)
    stars = data["stars"]
    stars = dicf.filt(stars, stars["En"]<0)
    data["stars"] = stars
    fname = data_params["sample"]
    folder = data_params["result_folder"]
    dicf.h5py_save(fname=folder + fname, dic=data, verbose=True, overwrite=True)
    return
