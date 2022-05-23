import numpy as np

try:
    from . import dic_funcs as dicf
    from .default_params import solar_params0
except Exception:
    from KapteynClustering import dic_funcs as dicf
    from KapteynClustering.default_params import solar_params0


# Read DATA

def read_data(fname, verbose=True, extra=None):
    if extra is not None:
        if fname[-5:] == ".hdf5":
            fname = fname[:-5] + "_" + extra + ".hdf5"
        else:
            fname += "_" + extra
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
                          ["vlsr", "_U", "_V", "_W"]]
    stars["_vx"] = stars["vx"] + _U
    stars["_vy"] = stars["vy"] + _V + vlsr
    stars["_vz"] = stars["vz"] + _W

    stars["_x"] = stars["x"] - solar_params["R0"]
    stars["_y"] = stars["y"]
    stars["_z"] = stars["z"]

    try:
        stars["vel"] = np.stack((stars["_vx"], stars["_vy"], stars["_vz"])).T
        stars["pos"] = np.stack((stars["_x"], stars["_y"], stars["_z"])).T
    except Exception:
        stars["vel"] = np.stack(
            (stars["_vx"].values, stars["_vy"].values, stars["_vz"].values)).T
        stars["pos"] = np.stack(
            (stars["_x"].values, stars["_y"].values, stars["_z"].values)).T

    return stars


def calc_toomre(vel, phi, vlsr=solar_params0["vlsr"]):
    # v_toomre = np.linalg.norm(vel - (vlsr*np.array([np.sin(phi),np.cos(phi),0])), axis=1)
    v_toomre = np.sqrt(((vel[:, 0] - vlsr*np.sin(phi))**2) +
                       ((vel[:, 1]+vlsr*np.cos(phi))**2) + (vel[:, 2]**2))
    return v_toomre


def sof_calc_toomre(vel, phi, vlsr=solar_params0["vlsr"]):
    # v_toomre = np.linalg.norm(vel - (vlsr*np.array([np.sin(phi),np.cos(phi),0])), axis=1)
    v_toomre = np.sqrt(((vel[:, 0] - vlsr*np.sin(phi))**2) +
                       ((vel[:, 1]-vlsr*(1+np.cos(phi))-232)**2) + (vel[:, 2]**2))
    return v_toomre


def create_toomre(stars, solar_params=solar_params0):
    # Toomre velocity: velocity offset from being a disk orbit
    try:
        stars["v_toomre"] = calc_toomre(
            stars["vel"].values, stars["phi"].values, vlsr=solar_params["vlsr"])
    except Exception:
        stars["v_toomre"] = calc_toomre(
            stars["vel"], stars["phi"], vlsr=solar_params["vlsr"])
    return stars


def apply_toomre_filt(stars, v_toomre_cut=210, solar_params=solar_params0):
    # The selection
    try:
        props = stars.keys()
    except Exception:
        props = stars.column_names
    if "v_toomre" not in props:
        stars = create_toomre(stars, solar_params)
    filt = (stars["v_toomre"] > v_toomre_cut)
    # frac_filt = filt.sum() / len(filt)
    # print(f"Toomre filt cuts to {frac_filt}")

    try:
        stars = dicf.filt(dic=stars, filt=filt, copy=False)
    except Exception:
        stars = stars[filt]

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


def create_geo_vel(stars, solar_params=solar_params0):
    [vlsr, _U, _V, _W] = [solar_params[p] for p in
                          ["vlsr", "_U", "_V", "_W"]]
    stars["vx"] = stars["_vx"] - _U
    stars["vy"] = stars["_vy"] - (_V + vlsr)
    stars["vz"] = stars["_vz"] - _W

    return stars