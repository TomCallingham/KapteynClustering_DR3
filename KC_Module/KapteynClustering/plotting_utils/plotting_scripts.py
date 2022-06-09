import numpy as np
from KapteynClustering.plotting_utils.property_labels import get_default_lims, get_default_scales, get_default_prop_unit_labels

prop_labels0, unit_labels0 = get_default_prop_unit_labels()
lims0 = get_default_lims()
scale_props0 = get_default_scales()


def get_val(stars, xkey, scale=True, scale_props=scale_props0):
    # gets vals from vaex or dic
    xscale = scale_props.get(xkey, None)
    if "/" in xkey:
        xkey_stripped, xscale = xkey.split("/")
        x = get_val(stars, xkey_stripped, scale=False)
    else:
        try:
            x = stars[xkey].values
        except Exception:
            x = stars[xkey]

    if scale and xscale is not None:
        x = x / float(xscale)
    return x


def get_key_label(xkey, scale=True, scale_props=scale_props0,
                  prop_labels=prop_labels0, unit_labels=unit_labels0):
    xscale = scale_props.get(xkey, None)
    if "/" in xkey:
        xkey_stripped, xscale = xkey.split("/")
        label = prop_labels.get(xkey_stripped, xkey_stripped)
        units = unit_labels.get(xkey_stripped)
    else:
        label = prop_labels.get(xkey, xkey)
        units = unit_labels.get(xkey)

    if scale and xscale is not None:
        xscale = float(xscale)
        lxscale = str(int(np.log10(xscale)))
        if units is not None:
            label = label + f" ($10^{{{lxscale}}}${units})"
        else:
            label = label + f" ($10^{{{lxscale}}}$)"
        return label

    if units is not None:
        label = label + f" ({units})"
    return label


import itertools


def get_G_colours(Groups, G_colours={}):
    cycle_colours = itertools.cycle(('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                     '#e377c3', '#bcbd22', '#17becf', 'grey', 'pink', 'blue',
                                     'orange', 'green'))
    dic_Groups = list(G_colours.keys())
    for g in Groups:
        if g not in dic_Groups:
            c = next(cycle_colours)
            current_colours = list(G_colours.values())
            i_max = 0
            while c in current_colours and i_max < len(current_colours):
                i_max += 1
                c = next(cycle_colours)
            G_colours[g] = c
    return G_colours


import matplotlib.pyplot as plt
import numpy as np


def check_xykeys(xy_keys):
    ''' returns new xkeys, shared, shape'''
    # At the moment, shared only works for shared y
    if isinstance(xy_keys, list):
        return xy_keys, False, None
    elif isinstance(xy_keys, tuple):
        propsX, propsY = xy_keys
        NRow = len(propsY)
        NCol = len(propsX)
        new_xy_keys = []
        for xkey in propsX:
            for ykey in propsY:
                new_xy_keys.append([xkey, ykey])
        return new_xy_keys, True, (NRow, NCol)
    print("Can't recognise xy_keys?")
    print(type(xy_keys))
    print(xy_keys)
    raise SystemError("xy keys error")


def create_axis(N_plots=None, shape=(None,None), sharex=False, sharey=False, xsize_scale=9, ysize_scale=7.5):
    if shape is None:
        NRow, NCol = None, None
    else:
        NRow,NCol = shape

    if (NRow!=None) and (NCol!=None) and (N_plots!=None):
        pass
    elif ((NRow, NCol) == ( None,None)) and N_plots!=None:
        if N_plots > 3:
            NRow= 2
        else:
            NRow= 1
        NCol = int(np.ceil(N_plots/NRow))
    elif ((N_plots, NCol) == ( None,None)) and NRow!=None:
        NCol =1
        N_plots = NRow
    elif ((N_plots, NRow) == ( None,None)) and NCol!=None:
        NRow =1
        N_plots = NCol
    elif (NRow == None) and (NCol!=None) and (N_plots!=None):
        NRow = int(np.ceil(N_plots/NCol))
    elif (NCol == None) and (NRow!=None) and (N_plots!=None):
        NCol = int(np.ceil(N_plots/NRow))
    else:
        raise ValueError("Not enough info to make axis")

    Fig, axs_array = plt.subplots(NRow, NCol, figsize=(NCol * xsize_scale, NRow * ysize_scale),
                                  sharex=sharex, sharey=sharey)
    if NRow == NCol == 1:
        axs_array = [axs_array]
    else:
        axs_array = axs_array.flatten()

    axs = {i:ax for i,ax in enumerate(axs_array)}

    imid = int(np.floor((NCol) / 2))
    axs["mid"] = axs_array[imid]

    return Fig, axs, axs_array


def set_ax_label(ax, xy_key):
    xkey, ykey = xy_key
    if xkey is not None:
        xlabel = get_key_label(xkey)
        ax.set_xlabel(xlabel)
    if ykey is not None:
        ylabel = get_key_label(ykey)
        ax.set_ylabel(ylabel)
    return



def set_ax_lims(ax, xy_key, Lims=lims0):
    xkey, ykey = xy_key
    try:
        ax.set_ylim(Lims[ykey])
    except Exception:
        pass
    try:
        ax.set_xlim(Lims[xkey])
    except Exception:
        pass
    return


def set_axs_pos(Fig, sharex=False, sharey=False):
    plt.tight_layout(w_pad=1)
    if sharex:
        Fig.subplots_adjust(wspace=0)
        plt.setp([a.get_yticklabels() for a in Fig.axes[1:]], visible=False)
    if sharey:
        Fig.subplots_adjust(vspace=0)
        plt.setp([a.get_xticklabels() for a in Fig.axes[1:]], visible=False)
    return


def set_G_labels(Fig, ax, Groups, G_colours, label_dic={},
                 below=True, leg_args={"ncol": 4}):
    group_legs = {}
    for g in Groups:
        label = label_dic.get(g, f"{g}")
        group_legs[g] = ax.scatter([None], [None], marker='o', label=label,
                                   color=G_colours[g], edgecolor='k')
    leg_args["title"] = leg_args.get("title", "Groups")
    if below:
        leg_args["loc"] = "upper center"
        leg_args["bbox_to_anchor"] = (0.5, 0.02)

    Fig.legend(
        handles=list(
            group_legs.values()),
        fancybox=True,
        shadow=True,
        **leg_args)
    return


def simple_scatter(ax, xy_key, stars, Groups, groups, G_colours, group_order={},
                   arg_dic={"alpha": 0.2, "s": 20, "edgecolors": "none"}, fluff=True):
    # Scatter plotting
    plot_Groups = Groups[Groups != -1]
    xkey, ykey = xy_key
    x, y = get_val(stars, xkey), get_val(stars, ykey)
    for i_g, g in enumerate(plot_Groups):
        g_filt = (groups == g)
        z_order = group_order.get(g, i_g)
        ax.scatter( x[g_filt], y[g_filt],
            c=G_colours[g], zorder=z_order,
            **arg_dic)

    if fluff:
        g_filt = (groups == -1)
        ax.scatter( x[g_filt], y[g_filt],
            c="lightgrey", zorder=-99,
            **arg_dic)

    return
