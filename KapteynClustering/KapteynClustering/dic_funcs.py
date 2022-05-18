import pickle
from copy import deepcopy
import os
import numpy as np
import h5py


def h5py_load(fname, verbose=False):
    ''' Loads hdf5 files structures into Nested Dictionaries'''
    max_nest = 10

    def load(hf, verbose=False, n_nest=0):
        if n_nest > max_nest:
            print('Error, max nesting exceeded!')
            raise SystemError
        n_nest += 1
        Data = {}
        for key in hf.keys():
            try:
                dic_key = int(key)
            except Exception:
                dic_key = key
            try:
                Data[dic_key] = hf[key]
                if isinstance(Data[dic_key], h5py._hl.group.Group):
                    Data[dic_key] = load(hf[key], verbose, n_nest)
                else:
                    Data[dic_key] = np.array(Data[dic_key])
                    if len(np.shape(Data[dic_key])) > 0:
                        Data[dic_key] = np.array(Data[dic_key])
                    p_type = Data[dic_key].dtype
                    if 'S' in str(p_type):
                        Data[dic_key] = Data[dic_key].astype('<U20')
            except AttributeError:
                print(f'Unidentified structure {key}')
                # raise SystemError'Attribute errors'
        return Data
    if ".hdf5" in fname:
        load_name = fname
    else:
        load_name = fname+".hdf5"
    if verbose:
        print('Loading file: ', load_name)
    with h5py.File(load_name, 'r') as hf:
        Data = load(hf, verbose)
    if verbose:
        print('Loaded')
    return Data


def h5py_save(fname, dic, verbose=False, overwrite=False):
    '''writes nested dictionarys to hdf5'''
    max_nest = 10

    def writer(hf, dic, root_name='', n_nest=0):
        if n_nest > max_nest:
            print('Error, max nesting exceeded!')
            raise SystemError
        for key in dic.keys():
            prop = dic[key]
            key = str(key)
            if type(prop) is dict:
                new_root_name = root_name + str(key) + '/'
                writer(hf, prop, new_root_name)
            else:
                if type(prop) is not np.ndarray:
                    try:
                        prop = np.array(prop)
                    except Exception as e:
                        print(key, e)
                        print("Can't be a numpy array")
                try:
                    ptype = prop.dtype
                except Exception:
                    ptype = type(prop)

                try:
                    if 'U' in str(ptype):
                        prop = prop.astype('|S10')
                        ptype = prop.dtype

                    hf.create_dataset(root_name + key, data=prop, dtype=ptype)
                except Exception as e:
                    print(e)
                    print(f'No {key} saved')
    if ".hdf5" in fname:
        save_name = fname
    else:
        save_name = fname+".hdf5"

    if verbose:
        print('Saving file: ',save_name)
    if not overwrite:
        if os.path.exists(save_name):
            print('File already here, moving old to _OLD')
            if os.path.exists(fname + '_OLD.hdf5'):
                print('OLD_File already here, deleting')
                os.remove(fname + '_OLD.hdf5')
            os.rename(fname + '.hdf5', fname + '_OLD.hdf5')

    with h5py.File(save_name, 'w') as hf:
        writer(hf, dic)
    if verbose:
        print('Saved')
    return


def filt(dic, filt, copy=True):
    if copy:
        new_dic = deepcopy(dic)
    else:
        new_dic = dic
    for p in dic.keys():
        try:
            new_dic[p] = dic[p][filt]
        except Exception:
            pass
    return new_dic


def groups(dic, group='group', verbose=False):
    Groups = np.unique(dic[group])
    gdic = {g: {} for g in Groups}
    if verbose:
        print(Groups)
    for g in Groups:
        filt = (dic[group] == g)
        gdic[g] = {}
        for p in dic.keys():
            try:
                gdic[g][p] = dic[p][filt]
            except Exception:
                pass
    return gdic


def pickle_save(a, fname):
    print(f"Pickle saving {fname}")
    with open(fname, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")
    return


def pickle_load(fname):
    print(f"Pickle loading {fname}")
    with open(fname, 'rb') as handle:
        b = pickle.load(handle)
    print("loaded")
    return b
