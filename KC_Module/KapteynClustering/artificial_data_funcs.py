from KapteynClustering.cluster_funcs import scale_features
from . import dynamics_funcs as dynf
from . import dic_funcs as dicf
from . import data_funcs as dataf
from .default_params import solar_params0
import copy
import numpy as np
np.random.seed(0)

def get_shuffled_artificial_set(N_art, stars, pot_fname, additional_props=[], v_toomre_cut=180, solar_params=solar_params0, scales={}, features_to_scale=[]):
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

    props = ["pos", "vel","phi"]
    props.extend(additional_props)
    try:
        dic_stars = {p: np.copy(stars[p].values) for p in props}
    except Exception:
        dic_stars = {p: np.copy(stars[p]) for p in props}

    art_stars_all = {}
    for n in range(N_art):
        print(f" {n+1} / {N_art}")
        art_stars = get_shuffled_artificial_dataset(
            dic_stars, pot_fname,  v_toomre_cut, solar_params) #, scales, features_to_scale)
        N_stars = np.shape(art_stars["pos"])[0]
        art_stars['index'] = np.full((N_stars), n)
        art_stars_all[n] = art_stars
    # selection = data["selection"]
    # selection = np.append(selection, ["Art_shuffle", f"Toomre {v_toomre_cut}"])
    # art_data = {"stars": art_stars_all,  "selection": selection}
    # scale = data["scale"] # "scale": scale,

    return art_stars_all


def get_shuffled_artificial_dataset(stars_dic, pot_fname, v_toomre_cut=180, solar_params=solar_params0):#, scales={}, features_to_scale=[]):
    '''
    Returns an artificial dataset by shuffling the vy and vz-components of the original dataset.
    The artificial dataset is cropped to have the exact same number of halo-members as the original.

    Parameters:
    data(vaex.DataFrame): The original dataset (GAIA RSV-halo-sample).

    Returns:
    df_art: A dataframe containing the artificial dataset.
    '''


    art_stars = copy.deepcopy(stars_dic)
    np.random.shuffle(art_stars["vel"][:,0]) #vx
    #don't shuffle vy!
    np.random.shuffle(art_stars["vel"][:,2]) #vz

    art_stars = dataf.apply_toomre_filt(art_stars, v_toomre_cut=v_toomre_cut,
                                        solar_params=solar_params)
    art_stars = dynf.add_dynamics(art_stars, pot_fname=pot_fname, circ=True)
    art_stars = dicf.filt(art_stars, art_stars["En"] < 0, copy=False)
    # art_stars = scale_features(art_stars, features =features_to_scale , scales=scales)[0]
    # print("Apply scale. (Shouldn't need to!)")
    # TODO: check numbers of catalogues
    # N_original = len(stars["vx"])
    # N_shuffle = len(art_stars["vx"])
    # if N_shuffle > num_stars:
    #     df_art = df_art.sample(num_stars)

    # return df_art

    return art_stars
