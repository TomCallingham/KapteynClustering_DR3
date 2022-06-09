import KapteynClustering.plotting_utils.plotting_scripts as ps
import numpy as np
import matplotlib.pyplot as plt

from KapteynClustering.plotting_utils.property_labels import get_default_lims, get_default_scales, get_default_prop_unit_labels

prop_labels0, unit_labels0 = get_default_prop_unit_labels()
lims0 = get_default_lims()
scale_props0 = get_default_scales()


def plot_simple_scatter(stars, xy_keys, Groups, groups,
                        G_colours=None, Lims=lims0):
    xy_keys, shared, shape = ps.check_xykeys(xy_keys)
    N_plots = len(xy_keys)
    fig, axs, ax_list = ps.create_axis(N_plots=N_plots,shape=shape , sharex=False, sharey=shared)
    if G_colours is None:
        G_colours = ps.get_G_colours(Groups)
    for (ax, xy_key) in zip(ax_list, xy_keys):
        ps.simple_scatter(ax, xy_key, stars, Groups, groups, G_colours)
        ps.set_ax_label(ax, xy_key)#, sharex=False, sharey=shared)
        ps.set_ax_lims(ax, xy_key, Lims=Lims)

    return fig, axs

def vol_simple_scatter(volumes,vol_stars, vol_labels, xy_key, NCol=3):
    N_plots = len(volumes)
    fig, axs, ax_list = ps.create_axis(N_plots, NCol=NCol, sharex=False, sharey=False)
    scatter_arg_dic={"alpha": 0.5, "s": 20, "edgecolors": "none"}
    for (ax, v) in zip(ax_list, volumes):
        stars,labels = vol_stars[v], vol_labels[v]
        Clusters = np.unique(labels)
        G_colours = ps.get_G_colours(Clusters)
        ps.simple_scatter(ax, xy_key, stars, Clusters, labels, G_colours,
                         arg_dic = scatter_arg_dic)
        ps.set_ax_label(ax, xy_key, sharex=False, sharey=False)
        ps.set_ax_lims(ax, xy_key)
        ax.set_title(v)

        N_stars = len(labels)
        N_clustered = (labels>=0).sum()
        f_clustered = N_clustered/N_stars
        text = f"N_stars: {N_stars} \n {f_clustered:.2f} fraction in clusters"
        ax.text(0.5, 0.8, text, fontsize=20,
                   transform = ax.transAxes)
    return fig, axs
