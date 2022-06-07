# %% [markdown]
# # Scripts

# %%
import numpy as np
import vaex
import matplotlib.pyplot as plt
import KapteynClustering.data_funcs as dataf
import KapteynClustering.param_funcs as paramf
import KapteynClustering.label_funcs as labelf

# %%
param_file = "../../Params/emma_volumes_params.yaml"
params = paramf.read_param_file(param_file)


# %%
def get_list_multi_files(param_file):
    params = paramf.read_param_file(param_file)
    data_p = params["data"]
    multi_dic = {}
    for p in ["base_data", "base_dyn","sample", "art", "cluster", "sig", "label"]:
        folder = data_p[p]
        if folder[-1] == "/":
            print("detected multiple in ", p)
            files =   paramf.list_files(folder, ext=True, path=True)
        multi_dic[p] = np.array([folder+f for f in files])
    multi_stage = params["multiple"]["stage"]
    multi_folder = data_p[multi_stage]
    vols =   paramf.list_files(multi_folder, ext=False, path=False)
    multi_dic["volumes"] = vols
    return multi_dic

multi_dic = get_list_multi_files(param_file)

# %%
vol_stars = {}
vol_labels = {}
volumes = multi_dic["volumes"]
for i,v in enumerate(volumes):
    print(i)
    star_file = multi_dic["sample"][i]
    vol_stars[v] = vaex.open(star_file)
    label_file = multi_dic["label"][i]
    vol_labels[v]  = dataf.read_data(label_file)["labels"]

# %%
from KapteynClustering.plotting_utils.property_labels import get_default_lims, get_default_scales, get_default_prop_unit_labels

prop_labels0, unit_labels0 = get_default_prop_unit_labels()
lims0 = get_default_lims()
scale_props0 = get_default_scales()


# %%
from KapteynClustering.plotting_utils import plotting_scripts as ps
def simple_scatter(stars, xy_keys, Groups, groups, G_colours, axs, group_order={},
                   arg_dic={"alpha": 0.2, "s": 20, "edgecolors": "none"}, fluff=True):
    # Scatter plotting

    plot_Groups = Groups[Groups != -1]
    # for [xkey, ykey] in xy_keys:
    #     ax = axs[ykey][xkey]
    for i,[xkey, ykey] in enumerate(xy_keys):
        ax = axs[i]
        [x, y] = [get_val(stars, k) for k in [xkey, ykey]]
        for i_g, g in enumerate(plot_Groups):
            g_filt = (groups == g)
            z_order = group_order.get(g, i_g)
            ax.scatter(
                x[g_filt],
                y[g_filt],
                c=G_colours[g],
                zorder=z_order,
                **arg_dic)

        if fluff:
            g_filt = (groups == -1)
            ax.scatter(
                x[g_filt],
                y[g_filt],
                c="lightgrey",
                zorder=-99,
                **arg_dic)

    return

def vol_scatter(volumes,vol_stars, xy_key, vol_groups, axs, group_order={},
                   arg_dic={"alpha": 0.2, "s": 20, "edgecolors": "none"}, fluff=True):
    # Scatter plotting

    # for [xkey, ykey] in xy_keys:
    #     ax = axs[ykey][xkey]
    [xkey, ykey] = xy_key
    for v in volumes:
        ax = axs[v]
        stars,groups = vol_stars[v], vol_groups[v]
        [x, y] = [ps.get_val(stars, k) for k in [xkey, ykey]]
        Groups = np.unique(groups)
        plot_Groups = Groups[Groups != -1]
        G_colours = ps.get_G_colours(Groups)
        for i_g, g in enumerate(plot_Groups):
            g_filt = (groups == g)
            z_order = group_order.get(g, i_g)
            ax.scatter(
                x[g_filt],
                y[g_filt],
                c=G_colours[g],
                zorder=z_order,
                **arg_dic)
        if fluff:
            g_filt = (groups == -1)
            ax.scatter(
                x[g_filt],
                y[g_filt],
                c="lightgrey",
                zorder=-99,
                **arg_dic)

    return

def vol_set_ax_label(volumes,axs, xy_keys, sharex=False, sharey=False):
    [xkey, ykey] = xy_keys
    for v in volumes:
        ax = axs[v]
        xlabel = ps.get_key_label(xkey)
        ax.set_xlabel(xlabel)
        ylabel = ps.get_key_label(ykey)
        ax.set_ylabel(ylabel)
    return


def vol_set_ax_lims(volumes, axs, xy_keys, Lims=lims0):
    [xkey, ykey] = xy_keys
    for v in volumes:
        ax = axs[v]
        try:
            ax.set_ylim(Lims[ykey])
        except Exception:
            pass
        try:
            ax.set_xlim(Lims[xkey])
        except Exception:
            pass
    return



# %%
def volume_scatter_plot(xy_keys):
    xsize_scale=9
    ysize_scale=7.5
    NRow = 8
    NCol = 3
    Fig, axs_array = plt.subplots(NRow, NCol, figsize=(NCol * xsize_scale, NRow * ysize_scale))
    axs_array = axs_array.flatten()
    axs = {}
    for i,v in enumerate(volumes):
        axs[v] = axs_array[i]
    arg_dic={"alpha": 0.5, "s": 20, "edgecolors": "none"}
    vol_scatter(volumes, vol_stars,xy_keys, vol_labels, axs, arg_dic = arg_dic)
    for v in volumes:
        ax = axs[v]
        ax.set_title(v)
        stars = vol_stars[v]
        labels = vol_labels[v]
        N_stars = len(labels)
        N_clustered = (labels>=0).sum()
        f_clustered = N_clustered/N_stars
        text = f"N_stars: {N_stars} \n {f_clustered:.2f} fraction in clusters"
        ax.text(0.5, 0.8, text, fontsize=20,
                   transform = ax.transAxes)
    vol_set_ax_label(volumes, axs, xy_keys)
    vol_set_ax_lims(volumes, axs, xy_keys)

    plt.show()
    return


# %% [markdown]
# # Plots

# %%
available_props = list(vol_stars[volumes[0]].column_names)
print(available_props)

# %%
xy_keys = ["Lz", "En"]
volume_scatter_plot(xy_keys)


# %%
xy_keys = ["Lz", "Lperp"]
volume_scatter_plot(xy_keys)

# %%
xy_keys = ["vx", "vy"]
volume_scatter_plot(xy_keys)

# %%
xy_keys = ["vx", "vz"]
volume_scatter_plot(xy_keys)

# %%
