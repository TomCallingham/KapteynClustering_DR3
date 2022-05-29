import KapteynClustering.plotting_utils.plotting_scripts as ps
import numpy as np
import matplotlib.pyplot as plt

from KapteynClustering.plotting_utils.property_labels import get_default_lims, get_default_scales, get_default_prop_unit_labels

prop_labels0, unit_labels0  = get_default_prop_unit_labels()
lims0 = get_default_lims()
scale_props0 = get_default_scales()

def plot_simple_scatter(stars, xy_keys, Groups, groups,G_colours=None, Lims=lims0):
    xy_keys, shared, shape = ps.check_xykeys(xy_keys)
    Fig, axs = ps.create_axis(xy_keys, shape=shape, sharex=False, sharey=shared)
    if G_colours is None:
        G_colours = ps.get_G_colours(Groups)
    ps.simple_scatter(stars, xy_keys, Groups, groups, G_colours, Fig, axs)
    
    ps.set_ax_label(Fig, axs,xy_keys,sharex=False, sharey=shared )
    ps.set_ax_lims(Fig, axs,xy_keys, Lims=Lims)

    return Fig, axs

