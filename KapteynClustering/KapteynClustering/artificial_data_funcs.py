import numpy as np
import copy
from .default_params import solar_params0
from . import data_funcs as dataf
from . import dic_funcs as dicf
from . import dynamics_funcs as dynf

def get_shuffled_artificial_set(N_art, data, a_pot, additional_props=[], v_toomre_cut=180, solar_params=solar_params0):
    '''
    Creates a data set of N artificial halos

    Parameters:
    N(int): The number of artificial halos to be generated
    data(vaex.DataFrame): A dataframe containing the halo stars and a little bit more, recommended: V-V_{lsr}>180.

    Returns:
    df_art_all(vaex.DataFrame): A dataframe containing each artificial halo, their integrals of motion, and data set index.

    '''

    '''
    Generate artificial halos by scrambling the velocity components of the existing data and recomputing the integrals of motion.
    This artificial halo is used to assess overdensities we find in the real data
    '''
    print(f'Creating {N_art} artificial datasets...')
    art_stars_all = {}
    for n in range(N_art):
        art_stars = get_shuffled_artificial_dataset(
            data, a_pot, additional_props, v_toomre_cut, solar_params)
        N_stars = np.shape(art_stars["vx"])[0]
        art_stars['index'] = np.full((N_stars), n)
        art_stars_all[n] = art_stars
    selection = data["selection"]
    selection = np.append(selection, ["Art_shuffle", f"Toomre {v_toomre_cut}"])
    art_data = {"stars": art_stars_all,  "selection": selection}
    # scale = data["scale"] # "scale": scale,

    return art_data


def get_shuffled_artificial_dataset(o_data, pot_fname, additional_props=[], v_toomre_cut=180, solar_params=solar_params0):
    '''
    Returns an artificial dataset by shuffling the vy and vz-components of the original dataset.
    The artificial dataset is cropped to have the exact same number of halo-members as the original.

    Parameters:
    data(vaex.DataFrame): The original dataset (GAIA RSV-halo-sample).

    Returns:
    df_art: A dataframe containing the artificial dataset.
    '''
    props = ["pos", "vx", "vy", "vz", "phi"]
    props.extend(additional_props)

    art_stars = {p: copy.deepcopy(o_data["stars"][p]) for p in props}

    # params = o_data["params"]

    for p in ["vx", "vz"]:  # DON't SHUFFLE VY "vy",
        np.random.shuffle(art_stars[p])

    art_stars = dataf.create_galactic_vel(art_stars, solar_params=solar_params)
    art_stars = dataf.apply_toomre_filt(art_stars, v_toomre_cut=v_toomre_cut,
                                        solar_params=solar_params)
    art_stars = dynf.add_dynamics(art_stars, pot_fname=pot_fname, circ=True)
    art_stars = dicf.filt(art_stars, art_stars["En"] < 0, copy=False)
    # art_stars = dataf.scale_features(
    #     art_stars, scales=o_data["scale"], plot=False)[0]

    N_original = len(o_data["stars"]["vx"])
    N_shuffle = len(art_stars["vx"])

    return art_stars
#TODO: check numbers of catalogues
    if N_shuffle > num_stars:
        df_art = df_art.sample(num_stars)

    return df_art
