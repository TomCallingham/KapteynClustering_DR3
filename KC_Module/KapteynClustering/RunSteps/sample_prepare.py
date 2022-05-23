import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
# import KapteynClustering.cluster_funcs as clusterf
import vaex

# from params import data_params
def sample_prepare(params):
    print("Using base_data:")
    print(params["data"]["base_data"])

    print("Finding Dynamics")
    save_dynamics(params)
    print("Finding Toomre Sample")
    create_toomre_example(params)
    print("Samples Created \n")
    return


def save_dynamics(params):
    data_p = params["data"]
    # cluster_p = params["cluster"]

    #Load Base Data
    stars = vaex.open(data_p["base_data"])
    # Create dynamics
    pot_name = data_p["pot_name"]
    if "pos" not in stars.column_names:
        print("Creating Galactic Pos/Vel")
        solar_p = params["solar"]
        stars = dataf.create_galactic_posvel(stars, solar_params=solar_p)
    stars = dynf.add_dynamics(stars, pot_name, circ=True)
    stars = stars[stars["En"]<0]
    # stars = clusterf.scale_features(stars, features=cluster_p["features"], scales=cluster_p["scales"])[0]
    #save dynamics
    stars.export(data_p["base_dyn"])
    return

def create_toomre_example(params):
    data_p= params["data"]
    stars = vaex.open(data_p["base_dyn"])
    toomre_stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210)
    toomre_stars = toomre_stars[toomre_stars["En"]<0]
    toomre_stars.export(data_p["sample"])
    return
