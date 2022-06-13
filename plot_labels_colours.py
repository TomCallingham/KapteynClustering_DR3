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
    prop_labels['distance_par'] = f"Distance (parallax)"

    unit_labels = {}
    for p in ["Lz", "Lx", "Ly", "Jz", "Jf", "Lp"]:
        unit_labels[p] = f"km kpc $\mathrm{{s}}^{{-1}}$"

    for p in ["E", "U"]:
        unit_labels[p] =  f"$\mathrm{{km}}^{{2}} / \mathrm{{s}}^{{2}}$"

    for p in ["f", "x", "y", "z", "distance_par", "R", "r", "X", "Y", "Z"]:
        unit_labels[p] = f"$\mathrm{{kpc}}$"

    for p in ["vf", "vx", "vy", "vz", "vT", "vf", "vX", "vY", "vZ", "vR"]:
        unit_labels[p] = f"$\mathrm{{km}} / \mathrm{{s}}$"


    unit_labels["phi"] = f"$Rad$"

    prop_labels["Lperp"], unit_labels["Lperp"] = prop_labels["Lp"], unit_labels["Lp"]
    prop_labels['En'], unit_labels['En'] = prop_labels["E"], unit_labels["E"]
    prop_labels['circ']  = prop_labels["Circ"]

    for pot in ["M17", "C20", "BoxChoc", "H99"]:
        for p in ["E", "En", "U", "Jz","JR", "circ"]:
            prop_labels[pot + "_" + p] = prop_labels[pot + "_" + p] + f" ({pot})"
            unit_labels[pot + "_" + p] =  unit_labels[p]

    return prop_labels, unit_labels
prop_labels0, unit_labels0  = get_default_prop_unit_labels()

scale_props0 = {"En":1e5}
for pot in ["M17", "C20", "BoxChoc", "H99"]:
    for p in ["En"]:
        scale_props0[pot + "_" p] = scale_props0[p]

def get_val(stars,xkey, scale=True, scale_props=scale_props0):
    # gets vals from vaex or dic
    xscale = scale_props.get(xkey,None)
    if "/" in xkey:
        xkey_stripped, xscale = xkey.split("/")
        x = get_val(stars, xkey_stripped, scale=False)
    else:
        try:
            x = stars[xkey].values
        except Exception:
            x = stars[xkey]

    if scale and xscale is not None:
        x = x/float(xscale)
    return x

def get_key_label(xkey, scale=True, scale_props=scale_props0, prop_labels=prop_labels0, unit_labels=unit_labels0):
    xscale = scale_props.get(xkey,None)
    if "/" in xkey:
        xkey_stripped, xscale = xkey.split("/")
        label = prop_labels.get(xkey_stripped,xkey_stripped)
        units = unit_labels.get(xkey_stripped)
    else:
        label = prop_labels.get(xkey,xkey)
        units = unit_labels.get(xkey)

    if scale and xscale is not None:
        xscale = float(xscale)
        lxscale = str(int(np.log10(xscale)))
        label = label + f" ($10^{{{lxscale}}}${units})"
        return label

    label = label + f" ({units})"
    return label

