import os
import yaml
import numpy as np
# Paramater Load #


def read_param_file(param_file, check_data=True):
    param_path = os.path.abspath(param_file)
    with open(param_path, 'r') as stream:
        params = yaml.safe_load(stream)
    if check_data:
        params["data"] = data_params_check(params["data"])
    return params


def data_params_check(data_params):
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
    if data_params["sample"] == "SOFIE":
        print("Sofies sample")
        s_result_folder = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/'
        data_params["sample"] = s_result_folder + "df.hdf5"
    if data_params["art"] == "SOFIE":
        print("Sofies art sample")
        s_result_folder = '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/'
        data_params["art"] = s_result_folder + "df_artificial_3D.hdf5"
    return data_params


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
    for p in ["sample", "art", "cluster", "sig", "label"]:
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
    if n_multi is not None:
        file_list = [file_list[n_multi]]
    if path is False:
        file_list = [os.path.basename(file) for file in file_list]
    if ext is False:
        file_list = [os.path.splitext(file)[0] for file in file_list]
    print(file_list)
    return file_list
