import os
import yaml
import numpy as np

try:
    from . import dic_funcs as dicf
    from .default_params import data_params0, solar_params0
except Exception:
    from KapteynClustering import dic_funcs as dicf
    from KapteynClustering.default_params import data_params0, solar_params0


def read_data(fname, verbose=True, extra=None):
    if extra is not None:
        fname += extra
    dic = dicf.h5py_load(fname, verbose=verbose)
    props = list(dic.keys())
    if props == ["table"]:
        dic = dic_from_vaex_dic(dic)
        props = list(dic.keys())
    return dic


def write_data(fname, dic, verbose=True, overwrite=True):
    dic = vaex_dic_from_dic(dic, delete=False)
    dicf.h5py_save(fname, dic, verbose=verbose, overwrite=overwrite)
    return

# TOOMRE SELECTION


def create_galactic_posvel(stars, solar_params=solar_params0):
    [vlsr, _U, _V, _W] = [solar_params[p] for p in
                          ["vslr", "_U", "_V", "_W"]]
    stars["_vx"] = stars["vx"] + _U
    stars["_vy"] = stars["vy"] + _V + vlsr
    stars["_vz"] = stars["vz"] + _W

    stars["_x"] = stars["x"] - solar_params["R0"]
    stars["_y"] = stars["y"]
    stars["_z"] = stars["z"]

    stars["vel"] = np.stack((stars["_vx"], stars["_vy"], stars["_vz"])).T
    stars["pos"] = np.stack((stars["_x"], stars["_y"], stars["_z"])).T

    return stars


def create_geo_vel(stars, solar_params=solar_params0):
    # print("Creating geo vel")
    [vlsr, _U, _V, _W] = [solar_params[p] for p in
                          ["vslr", "_U", "_V", "_W"]]
    stars["vx"] = stars["_vx"] - _U
    stars["vy"] = stars["_vy"] - (_V + vlsr)
    stars["vz"] = stars["_vz"] - _W

    return stars


def create_toomre(stars, solar_params=solar_params0):
    [vlsr, _U, _V, _W] = [solar_params[p] for p in
                          ["vslr", "_U", "_V", "_W"]]
    # Toomre velocity: velocity offset from being a disk orbit
    # print("Sofie Toomre, check!")
    sof_vx = stars["vx"] + _U - (vlsr * np.sin(stars["phi"]))
    sof_vy = stars["vy"] + _V - (vlsr * np.cos(stars["phi"]))
    sof_vz = stars["vz"] + _W
    stars["v_toomre"] = np.sqrt(
        (sof_vx**2) + (((sof_vy-232)**2) + (sof_vz**2)))
    # stars["v_toomre"] = np.sqrt(
    #     (stars["_vx"]**2) + (((stars["_vy"]-232)**2) + (stars["_vz"]**2)))
    # stars["v_toomre"] = np.sqrt(
    #     (stars["vR"]**2) + (((-stars["vT"]-232)**2) + (stars["vz"]**2)))
    return stars


def apply_toomre_filt(stars, v_toomre_cut=210, solar_params=solar_params0):
    # print("Applying toomre filt!")
    # The selection
    if "v_toomre" not in stars.keys():
        # print("No v_toomre defined, creating")
        stars = create_toomre(stars, solar_params)
    filt = (stars["v_toomre"] > v_toomre_cut)
    frac_filt = filt.sum() / len(filt)
    # print(f"Toomre filt cuts to {frac_filt}")

    stars = dicf.filt(dic=stars, filt=filt, copy=False)

    return stars


# def apply_toomre_filt_dataset(data, v_toomre_cut=210):#, solar_params=solar_params0):
#     print("Applying toomre filt!")

#     stars = data["stars"]
#     stars = create_toomre(stars)#, solar_params)
#     stars = apply_toomre_filt(stars, v_toomre_cut)#,  solar_params)

#     data["stars"] = stars
#     data["selection"] = np.append(
#         data["selection"], [f"Toomre {v_toomre_cut}"])
#     data["solar"] = "fixed default" #solar_params

#     return data


# LOAD Vaex_funcs

def dic_from_vaex_dic(vaex_dic):
    vaex_dic = vaex_dic["table"]["columns"]
    props = list(vaex_dic.keys())
    for p in props:
        try:
            vaex_dic[p] = vaex_dic[p]["data"]
        except Exception as e:
            pass
            # print(p, "\n",e)
    return vaex_dic


def vaex_dic_from_dic(dic, delete=True):
    props = list(dic.keys())
    columns = {}
    for p in props:
        columns[p] = {"data": dic[p]}
    save_dic = {"table": {"columns": columns}}
    if delete:
        del dic
    return save_dic


# Paramater Load


def read_param_file(param_file):
    param_path = os.path.abspath(param_file)
    with open(param_path, 'r') as stream:
        params = yaml.safe_load(stream)
    return params
