import numpy as np
import matplotlib.pyplot as plt
import copy
import dic_funcs as dicf

from default_params import data_params0, cluster_params0, solar_params0


def read_data(fname="stars", data_params=data_params0,extra=None):
    result_path = data_params["result_path"]
    data_folder = data_params["data_folder"]
    if extra is not None:
        fname += extra
    dic = dicf.h5py_load(result_path  + data_folder + '/' + fname)
    return dic
###

def scale_features(stars, features=cluster_params0["features"],
        scales=cluster_params0["scales"], plot=True):
    scale_data = {}
    for p in features:
        scale = scales.get(p, None)
        if scale is None:
            scale = get_scale(x=stars[p])
        stars["scaled_" + p] = apply_scale(stars, xkey=p, scale=scale, plot=plot)
        scale_data[p] = scale
    return stars, scale_data


def get_scale(x, p=5):
    scale = np.array(np.percentile(x, [p, 100 - p]))
    return scale

def apply_scale(data, xkey, scale, plot=False):
    x = data[xkey]
    scale_range = scale[1] - scale[0]
    scaled_x = ((x - scale[0]) / (scale_range / 2)) - 1

    if plot:
        n_x = len(x)
        f_below = (x < scale[0]).sum() / n_x
        f_above = (x > scale[1]).sum() / n_x
        print(xkey)
        print(f"Scale: below {f_below}, above{f_above}")
        x_sort = np.sort(copy.deepcopy(x))

        plt.figure()
        plt.plot(x_sort, np.linspace(0, 1, n_x))
        plt.axvline(scale[0], c='r')
        plt.axvline(scale[1], c='r')
        plt.xlabel(xkey)
        plt.ylabel("cdf")
        plt.show()

        plt.figure()
        plt.plot(x, scaled_x)
        plt.axvline(scale[0], c='r')
        plt.axvline(scale[1], c='r')
        plt.axhline(-1, c='r')
        plt.axhline(1, c='r')
        plt.xlabel(xkey)
        plt.ylabel("scaled " + xkey)
        plt.show()

    return scaled_x

### TOOMRE SELECTION

def create_toomre(stars, solar_params=solar_params0):
    # Toomre velocity: velocity offset from being a disk orbit
    [vlsr, _U, _V, _W] = [solar_params[p] for p in
            ["vslr", "_U", "_V", "_W"]]
    _vx = stars["vx"] + _U - (vlsr * np.sin(stars["RPhi"]))
    _vy = stars["vy"] + _V - (vlsr * np.cos(stars["RPhi"]))
    _vz = stars["vz"] + _W
    print("Fixed toomre")
    v_toomre = np.sqrt((_vx**2) + (((_vy-232)**2) + (_vz**2)))
    stars["v_toomre"] = v_toomre
    return stars

def apply_toomre_filt(stars,v_toomre_cut=210, solar_params=solar_params0):
    print("Applying toomre filt!")


    # The selection
    if "v_toomre" not in stars.keys():
        print("No v_toomre defined, creating")
        stars = create_toomre(stars, solar_params)
    filt = (stars["v_toomre"] > v_toomre_cut)
    frac_filt = filt.sum() / len(filt)
    print(f"Toomre filt cuts to {frac_filt}")

    stars = dicf.filt(dic=stars, filt=filt, copy=False)

    return stars


def apply_toomre_filt_dataset(data, v_toomre_cut=210, solar_params=solar_params0):
    print("Applying toomre filt!")

    stars = data["stars"]
    stars = create_toomre(stars, solar_params)
    stars = apply_toomre_filt(stars, v_toomre_cut,  solar_params)

    data["stars"] = stars
    data["selection"] = np.append(data["selection"],[f"Toomre {v_toomre_cut}"])
    data["solar"]=solar_params

    return data


## CUT
def load_cluster(Halo_n=5, Level=4,  fname="toomre_"):
    save_name = "clustered_" + fname + f"Lv{Level}Au{Halo_n}Stars"
    cluster_data = dicf.h5py_load(result_path + save_name)
    data =  read_auriga_data(fname=fname, Halo_n = Halo_n, Level=Level)
    data["cluster"] = cluster_data
    return data

###


def load_vaex_to_dic(vaex_fname):
    dic = dicf.h5py_load(vaex_fname)["table"]["columns"]
    props = list(dic.keys())
    for p in props:
        dic[p] = dic[p]["data"]
    return dic


def save_dic_to_vaex(fname, dic):
    props = list(dic.keys())
    columns = {}
    for p in props:
        columns[p] = {"data":dic[p]}

    save_dic = {"table":{"columns":columns}}
    dicf.h5py_save(fname, dic)
    return

def vaex_to_dic(vaex_df):
    return dic
def dic_to_vaex(dic):
    return vaex_df
