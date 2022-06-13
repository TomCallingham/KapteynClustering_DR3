# import numpy as np
def get_default_prop_unit_labels():
    prop_labels = {}
    prop_labels['Lz'] = f"$L_{{z}}$"
    prop_labels['Lx'] = f"$L_{{x}}$"
    prop_labels['Ly'] = f"$L_{{y}}$"
    prop_labels['E'] = f"$E$"
    prop_labels['U'] = f"$U$, Potential"
    prop_labels['Jz'] = f"$J_{{z}}$"
    prop_labels['JR'] = f"$J_{{R}}$"
    prop_labels['Lp'] = f"$L_{{p}}$"
    prop_labels['Circ'] = f"Circularity"
    prop_labels['Age'] = f"Age (Gyr)"
    prop_labels['Age_Err'] = f"Age Err (Gyr)"
    prop_labels['Fe_H'] = f"$\\left[Fe/H\right]}}$"
    prop_labels['Lz_J'] = f"$L_{{z}}/J{{\\mathrm{{Tot}}}}$"
    prop_labels['JR_J'] = f"$J_{{R}}/J{{\\mathrm{{Tot}}}}$"
    prop_labels['Jz_J'] = f"$J_{{z}}/J{{\\mathrm{{Tot}}}}$"
    prop_labels['r'] = f"$r$"
    prop_labels['cdf'] = f"CDF, Cumulatitive"
    prop_labels['distance_par'] = f"Distance (parallax)"
    prop_labels['AR'] = f"Distance (parallax)"

    unit_labels = {}
    for p in ["Lz", "Lx", "Ly", "Jz", "JR", "Lp"]:
        unit_labels[p] = f"km kpc $\\mathrm{{s}}^{{-1}}$"

    for p in ["E", "U"]:
        unit_labels[p] = f"$\\mathrm{{km}}^{{2}} / \\mathrm{{s}}^{{2}}$"

    for p in ["f", "x", "y", "z", "distance_par", "R", "r", "X", "Y", "Z"]:
        unit_labels[p] = f"$\\mathrm{{kpc}}$"

    for p in ["vf", "vx", "vy", "vz", "vT", "vf", "vX", "vY", "vZ", "vR"]:
        unit_labels[p] = f"$\\mathrm{{km}} / \\mathrm{{s}}$"

    for p in ["l", "b"]:
        unit_labels[p] = f"$\\mathrm{{degrees}}$"
    unit_labels["phi"] = f"$\\mathrm{{Rad}}$"

    # Check Duplicates
    prop_labels['En'], unit_labels['En'] = prop_labels["E"], unit_labels["E"]
    prop_labels["Lperp"], unit_labels["Lperp"] = prop_labels["Lp"], unit_labels["Lp"]
    prop_labels['circ'] = prop_labels["Circ"]
    for pot in ["M17", "C20", "BoxChoc", "H99"]:
        for p in ["E","En", "U", "Jz","JR", "circ"]:
            try:
                prop_labels[pot + "_" + p] = prop_labels[p] + f" ({pot})"
            except Exception:
                pass
            try:
                unit_labels[pot + "_" + p] =  unit_labels[p]
            except Exception:
                pass

    return prop_labels, unit_labels


def get_default_lims():
    lims0 = {}
    for p in ["Jz", "JR", "Lperp", "Lp"]:
        lims0[p] = [0, None]
    lims0["circ"] = [-1, 1]
    lims0["Circ"] = lims0["circ"]
    lims0["l"] = [0,360]
    lims0["cdf"] = [0,1]
    # lims0["Jz"] = [0,1]
    # lims0["En"] = [None,0]
    return lims0


def get_default_scales():
    scale_props0 = {"En": 1e5}
    for pot in ["M17", "C20", "BoxChoc", "H99"]:
        for p in ["En"]:
            scale_props0[pot + "_" + p] = scale_props0[p]
    return scale_props0
