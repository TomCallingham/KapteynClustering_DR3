import numpy as np
def get_default_prop_unit_labels():
    prop_labels = {}
    prop_labels['Lz'] = f"$L_{{z}}$"
    prop_labels['Lx'] = f"$L_{{x}}$"
    prop_labels['Ly'] = f"$L_{{y}}$"
    prop_labels['E'] = f"$E$"
    prop_labels['Jz'] = f"$J_{{z}}$"
    prop_labels['JR'] = f"$J_{{R}}$"
    prop_labels['Lp'] = f"$L_{{p}}$"
    prop_labels['Circ'] = f"Circularity"
    prop_labels['Age'] = f"Age (Gyr)"
    prop_labels['Age_Err'] = f"Age Err (Gyr)"
    prop_labels['Fe_H'] = f"$\left[Fe/H\right]}}$"
    prop_labels['Lz_J'] = f"$L_{{z}}/J{{\mathrm{{Tot}}}}$"
    prop_labels['JR_J'] = f"$J_{{R}}/J{{\mathrm{{Tot}}}}$"
    prop_labels['Jz_J'] = f"$J_{{z}}/J{{\mathrm{{Tot}}}}$"
    prop_labels['r'] = f"$r$"

    unit_labels = {}
    for p in ["Lz", "Lx", "Ly", "Jz", "Jf", "Lp"]:
        unit_labels[p] = f"km kpc $\mathrm{{s}}^{{-1}}$"

    for p in ["E", "U"]:
        unit_labels[p] =  f"$\mathrm{{km}}^{{2}} / \mathrm{{s}}^{{2}}$"

    for p in ["f", "x", "y", "z"]:
        unit_labels[p] = f"$\mathrm{{kpc}}$"

    for p in ["vf", "vx", "vy", "vz", "vT", "vf"]:
        unit_labels[p] = f"$\mathrm{{km}} / \mathrm{{s}}$"

    prop_labels['En'], unit_labels['En'] = prop_labels["E"], unit_labels["E"]
    prop_labels["Lperp"], unit_labels["Lperp"] = prop_labels["Lp"], unit_labels["Lp"]
    prop_labels['circ']  = prop_labels["Circ"]

    return prop_labels, unit_labels

def get_default_lims():
    lims0 = {}
    for p in ["Jz", "JR", "Lperp", "Lp"]:
        lims0[p] = [0,None]
    lims0["circ"] = [-1,1]
    lims0["Circ"] = [-1,1]
    # lims0["En"] = [None,0]
    return lims0

def get_default_scales():
    scale_props0 = {"En":1e5}
    return scale_props0




