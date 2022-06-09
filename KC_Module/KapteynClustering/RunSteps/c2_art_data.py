import KapteynClustering.data_funcs as dataf
import KapteynClustering.dynamics_funcs as dynf
import KapteynClustering.artificial_data_funcs as adf
import vaex
import copy


def art_data(params):
    print("Creating Artificial_data")

    data_p = params["data"]
    pot_name =  data_p["pot_name"]
    solar_p = params["solar"]

    art_p = params["art"]
    N_art, additional_props = art_p["N_art"], art_p.get("additional",[])

    dynamics_include = copy.deepcopy(params["linkage"]["features"])


    print(f"{N_art} realisations")

    stars = vaex.open(data_p["base_data"])
    stars = dataf.create_galactic_posvel(stars, solar_params=solar_p)
    stars = dynf.add_cylindrical(stars)
    stars = dataf.apply_toomre_filt(stars, v_toomre_cut=180, solar_params=solar_p)

    # , features_to_scale=cluster_p["features"], scales= cluster_p["scales"])
    N_match = art_p.get("N_match", True)
    if N_match:
        N_original = vaex.open(data_p["sample"]).count()
    else:
        N_original=None


    art_stars = adf.get_shuffled_artificial_set(N_art, stars, pot_name,dynamics=dynamics_include,N_original=N_original, additional_props=additional_props)
    dataf.write_data(fname=data_p["art"],
                     dic=art_stars, verbose=True, overwrite=True)
    print("Finished \n")
    return
