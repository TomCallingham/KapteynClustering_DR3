import vaex
import copy

def vaex_from_dict(dict, flip_sign=True):
    new_dict = copy.deepcopy(dict)
    translate = {"E":"En", "Circ":"circ", "Lp":"Lperp"}
    for k in translate.keys():
        try:
            new_dict[translate[k]] = new_dict[k]
            del new_dict[k]
        except Exception:
            pass
    if flip_sign:
        print("Flipping sign for vaex")
        new_dict["Lz"] = - new_dict["Lz"]
        new_dict["circ"] = - new_dict["circ"]
    return vaex.from_dict(new_dict)
