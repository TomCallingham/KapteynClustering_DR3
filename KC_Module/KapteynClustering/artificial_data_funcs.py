from . import dynamics_funcs as dynf
from . import dic_funcs as dicf
from . import data_funcs as dataf
from .default_params import solar_params0
import copy
import numpy as np
np.random.seed(0)


def get_shuffled_artificial_set(N_art, stars, pot_fname, features=[], additional_props=[], v_toomre_cut=210, solar_params=solar_params0, N_original=None):
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
    print(f"Using v_toomre_cut of {v_toomre_cut}")

    props = ["pos", "vel", "phi"]
    props.extend(additional_props)
    try:
        dic_stars = {p: np.copy(stars[p].values) for p in props}
    except Exception:
        dic_stars = {p: np.copy(stars[p]) for p in props}

    art_stars_all = {}
    for n in range(N_art):
        print(f" {n+1} / {N_art}")
        art_stars = get_shuffled_artificial_dataset(
            dic_stars, pot_fname,  v_toomre_cut, solar_params, features, N_original = N_original)
        N_stars = np.shape(art_stars["pos"])[0]
        art_stars['index'] = np.full((N_stars), n)
        art_stars_all[n] = art_stars
    # selection = data["selection"]
    # selection = np.append(selection, ["Art_shuffle", f"Toomre {v_toomre_cut}"])
    # art_data = {"stars": art_stars_all,  "selection": selection}
    # scale = data["scale"] # "scale": scale,

    return art_stars_all


def get_shuffled_artificial_dataset(stars_dic, pot_fname, v_toomre_cut=210, solar_params=solar_params0, features=[], N_original=None):
    '''
    Returns an artificial dataset by shuffling the vy and vz-components of the original dataset.
    The artificial dataset is cropped to have the exact same number of halo-members as the original.

    Parameters:
    data(vaex.DataFrame): The original dataset (GAIA RSV-halo-sample).

    Returns:
    df_art: A dataframe containing the artificial dataset.
    '''

    art_stars = copy.deepcopy(stars_dic)
    # shuffle vx,vz, don't shuffle vy!
    np.random.shuffle(art_stars["vel"][:, 0])  # vx
    np.random.shuffle(art_stars["vel"][:, 2])  # vz

    art_stars = dataf.apply_toomre_filt(art_stars, v_toomre_cut=v_toomre_cut, solar_params=solar_params)
    art_stars = dynf.add_dynamics(art_stars, pot_fname=pot_fname, additional_dynamics=features)
    if N_original is None:
        art_stars = dicf.filt(art_stars, art_stars["En"] < 0, copy=False)
        return art_stars
    print("Matching Number")
    filt = (art_stars["En"]<0)
    N_art_uncut = len(filt)
    N_shuffle = filt.sum()
    if N_shuffle < N_original:
        print("Not enough stars to match original")
        print(N_shuffle, N_original)
        raise SystemError("Not enough stars")
    index = np.arange(N_art_uncut)[filt]
    index = np.random.choice(index, size = N_original, replace=False)
    filt = np.zeros((N_art_uncut), dtype=bool)
    filt[index] = True
    art_stars = dicf.filt(art_stars, filt, copy=False)
    # N_check = len(art_stars["En"])
    # print("Check numbers:",N_check, N_original)

    return art_stars
