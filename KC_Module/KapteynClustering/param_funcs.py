import os
import yaml
import numpy as np
# Paramater Load #


def read_param_file(param_file, check_data=True):
    params = read_yaml(param_file)
    if check_data:
        params["data"] = check_data_params(params["data"])
    params["solar"] = check_solar_params(params["solar"])
    return params

def read_yaml(param_file):
    param_path = os.path.abspath(param_file)
    with open(param_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def check_data_params(data_params):
    check_make_folder(data_params["result_folder"])
    data_params = check_sofie_data(data_params)
    data_params = check_file_path(data_params)
    return data_params


def check_make_folder(folder):
    if not os.path.exists(folder):
        print('Creating Folder: ', folder)
        os.makedirs(folder)
    return


def check_sofie_data(data_params):
    if data_params.get("sample",None) == "SOFIE":
        print("Sofies sample")
        s_result_folder = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/'
        data_params["sample"] = s_result_folder + "df.hdf5"
    if data_params.get("art",None) == "SOFIE":
        print("Sofies art sample")
        s_result_folder = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/'
        data_params["art"] = s_result_folder + "df_artificial_3D.hdf5"
    return data_params

def check_solar_params(solar_params):
    if "file" in list(solar_params.keys()):
        import KapteynClustering.dic_funcs as dicf
        solar_params = dicf.h5py_load(solar_params["file"])
    return solar_params


def check_file_path(data_params):
    props = np.array(list(data_params.keys()))
    props = props[np.isin(props, np.array(
        ["result_folder", "pot_name", "base_data"]), invert=True)]
    for p in props:
        file = data_params[p]
        if "/" not in file[:-1]:
            file = data_params["result_folder"] + data_params[p]
        if "/" == file[-1] and "/" == file[0]: # is a full path to a folder
            check_make_folder(file)
        data_params[p] = file
    return data_params


def multi_create_param_file(param_file, n_multi):
    params = read_param_file(param_file)
    data_p = params["data"]
    multi_stage = params["multiple"]["stage"]
    sample_name = list_files(
        data_p[multi_stage], ext=False, path=False, n_multi=n_multi)[0]
    print("making multi param file with sample name: ", sample_name)
    for p in ["base_data", "base_dyn","sample", "art", "cluster", "sig", "label"]:
        folder = data_p[p]
        if folder[-1] == "/":
            print("detected multiple in ", p)
            if p == multi_stage:
                data_p[p] = f"{folder}{sample_name}.hdf5"
            else:
                data_p[p] = f"{folder}{sample_name}_{p}.hdf5"
    params["data"] = data_p
    del params["multiple"]
    multi_param_file = param_file[:-5] + "_multi.yaml"
    write_param_file(file_name=multi_param_file, params=params)
    return multi_param_file


def write_param_file(file_name, params):
    with open(file_name, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
    return


def list_files(folder, ext=True, path=True, n_multi=None):
    file_list = os.listdir(folder)
    filetype_list = [os.path.splitext(file)[1] for file in file_list]
    file_list = np.array(file_list)
    filetype_list = np.array(filetype_list)
    file_list = file_list[filetype_list==".hdf5"]
    if path is False:
        file_list = [os.path.basename(file) for file in file_list]
    if ext is False:
        file_list = [os.path.splitext(file)[0] for file in file_list]

    if n_multi is not None:
        file_list = [file_list[n_multi]]
    return file_list

# Find N_std
def find_Nstd_from_params(params):
    cluster_p = params["cluster"]
    df = len(cluster_p["features"])
    Nstd = cluster_p.get("Nstd",cluster_p.get("N_sigma_ellipse_axis",None))
    sigma1d = cluster_p.get("sigma_1d",None)
    sigma_percent = cluster_p.get("sigma_percent",None)
    if Nstd is not None:
        print("Nstd given")
        pass
    elif sigma1d is not None:
        print("sigma1d given")
        Nstd = find_Nstd_from_1dSigma(sigma1d,df)
    elif sigma_percent is not None:
        print("sigma percent given")
        Nstd = find_Nstd_from_percent(sigma_percent,df)
    else:
        print(f"No Nstd given. Assume sigma1d = 2, using dof = {df}")
        Nstd = find_Nstd_from_1dSigma(2,df)

    print(f"Nstd: {Nstd}")
    return Nstd

def find_Nstd_from_percent(percent, df):
    from scipy.stats import chi2
    Nstd = np.sqrt(chi2.ppf(percent, df))
    return Nstd

def find_Nstd_from_1dSigma(sigma, df):
    from scipy.stats import chi2
    sigma_percent= chi2.cdf(sigma**2,df=1)
    return find_Nstd_from_percent(sigma_percent, df)
