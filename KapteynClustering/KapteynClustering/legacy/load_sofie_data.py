from .. import data_funcs as dataf
from .. import dic_funcs as dicf
def load_sof_art():
    s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
    sof_art_stars = dataf.load_vaex_to_dic(f'{s_result_path}df_artificial_3D.hdf5')
    sof_art_stars = dicf.groups(sof_art_stars, group="index", verbose=True)
    return sof_art_stars

def load_sof_df():
    s_result_path = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/' #Path to folder where intermediate and final results are stored
    stars = dataf.load_vaex_to_dic(f'{s_result_path}df.hdf5')
    return stars
