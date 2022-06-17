import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.data_funcs as dataf
# import KapteynClustering.linkage_funcs as linkf
import vaex
import copy

# from params import data_params


def sample_prepare(params):
    print("Using base_data:")
    print(params["data"]["base_data"])
    data_p = params["data"]
    data_p = params["data"]
    pot_name = data_p["pot_name"]
    solar_p = params["solar"]
    dynamics_include = copy.deepcopy(params["linkage"]["features"])
    extra_dynamics = params.get("sample_dynamics",["circ"])
    dynamics_include.extend(extra_dynamics)
    print("Calculatin extra dynamics:")
    print(dynamics_include)

    # Load Base Data
    stars = vaex.open(data_p["base_data"])
    # Create dynamics
    stars = dataf.create_galactic_posvel(stars, solar_params=solar_p)
    stars = dynf.add_cylindrical(stars)
    stars = dataf.apply_toomre_filt(stars, v_toomre_cut=210, solar_params=solar_p)

    stars = dynf.add_dynamics(stars, pot_name, additional_dynamics=dynamics_include)
    stars = stars[stars["En"] < 0]
    # save dynamics
    print("Saving Sample...")
    stars.export(data_p["sample"])
    print("Saved!")
    return
