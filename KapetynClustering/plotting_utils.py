'''Utility functions for plotting

Contains helper functions visualization of the clustering results, both in integral of motion
space and in velocity space.

Sofie L
06/04/2021
'''

import vaex
import numpy as np
from scipy.stats import chi2

import matplotlib
from matplotlib import colors
import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
# import membership_probability

mpl.rcParams['figure.dpi'] = 100
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=16)  # fontsize of the figure title

####################################################################################################

def get_cmap(df_minsig):
    '''
    Returns a cmap and a norm which maps the three sigma labels to proper color bins.
    The last few colors of the colormap are very light, so we replace them for better
    visible colors.


    Parameters:
    df_minsig(vaex.DataFrame): The slice of the dataframe containing only the relevant clusters.

    Returns:
    cmap(matplotlib.colors.Colormap): The colormap we use for plotting clusters
    norm(matplotlib.colors.Normalize): Normalization bins such that unevenly spaced
                                       label values will get mapped
                                       to linearly spaced color bins
    '''
    unique_labels = np.unique(df_minsig.labels.values)
    cmap1 = plt.get_cmap('gist_ncar', len(unique_labels))
    cmaplist = [cmap1(i) for i in range(cmap1.N)]

    cmaplist[-1] = 'purple'
    cmaplist[-2] = 'blue'
    cmaplist[-3] = 'navy'
    cmaplist[-4] = 'gold'
    cmaplist[-5] = 'mediumvioletred'
    cmaplist[-6] = 'deepskyblue'
    cmaplist[-7] = 'indigo'

    #  cmaplist[1] = 'purple'
    #  cmaplist[2] = 'blue'
    #  cmaplist[3] = 'navy'
    #  cmaplist[4] = 'gold'
    #  cmaplist[5] = 'mediumvioletred'
    #  cmaplist[6] = 'deepskyblue'
    #  cmaplist[7] = 'indigo'

    cmap, norm = colors.from_levels_and_colors(unique_labels, cmaplist, extend='max')

    return cmap, norm

####################################################################################################

def plot_IOM_subspaces(df, minsig=3, savepath=None):
    '''
    Plots the clusters for each combination of clustering features

    Parameters:
    df(vaex.DataFrame): Data frame containing the labelled sources.
    minsig(float): Minimum statistical significance level we want to consider.
    savepath(str): Path of the figure to be saved, if None the figure is not saved.
    '''

    x_axis = ['Lz/10e2', 'Lz/10e2', 'circ', 'Lperp/10e2', 'circ', 'circ']
    y_axis = ['En/10e4', 'Lperp/10e2', 'Lz/10e2', 'En/10e4', 'Lperp/10e2', 'En/10e4']

    xlabels = ['$L_z$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]', '$\eta$',
               '$L_{\perp}$ [$10^3$ kpc km/s]', '$\eta$', '$\eta$']

    ylabels = ['$E$ [$10^5$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]',
               '$E$ [$10^5$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$E$ [$10^5$ km$^2$/s$^2$]']

    fig, axs = plt.subplots(2,3, figsize = [14,8])
    plt.tight_layout()

    df_minsig = df[(df.labels>0) & (df.maxsig>minsig)]

    cmap, norm = get_cmap(df_minsig)

    for i in range(6):
        plt.sca(axs[int(i/3), i%3])
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1)#)#, length_limit=60000)
        df_minsig.scatter(x_axis[i], y_axis[i], s=1, c=df_minsig.labels.values,
                                cmap=cmap, norm=norm, alpha=0.6)

        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])

    #fig.subplots_adjust(top=0.95)
    plt.tight_layout(w_pad=1)

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

####################################################################################################

def plot_cluster(df, clusterNr, savepath=None):
    '''
    Plots a single cluster for each combination of clustering features

    Parameters:
    df(vaex.DataFrame): Data frame containing the labelled sources.
    clusterNr(int): Label of cluster to be plotted
    savepath(str): Path of the figure to be saved, if None the figure is not saved.
    '''

    minsig = 3

    x_axis = ['Lz/10e2', 'Lz/10e2', 'circ', 'Lperp/10e2', 'circ', 'circ']
    y_axis = ['En/10e4', 'Lperp/10e2', 'Lz/10e2', 'En/10e4', 'Lperp/10e2', 'En/10e4']

    fig, axs = plt.subplots(2,3, figsize = [12,6])
    plt.tight_layout()

    unique_labels = np.unique(df.labels.values)
    cmap = plt.get_cmap('gist_ncar', len(unique_labels))

    cmap, norm = get_cmap(df[(df.labels>0) & (df.maxsig>minsig)])

    for i in range(6):
        plt.sca(axs[int(i/3), i%3])
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1)#)#, length_limit=60000)
        df[df.labels==clusterNr].scatter(x_axis[i], y_axis[i], s=2, c=df[df.labels==clusterNr].labels.values,
                                cmap=cmap, norm=norm, alpha=0.8)

    fig.subplots_adjust(top=0.95)

    plt.tight_layout()
    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

####################################################################################################

def plot_v_subspaces(df, minsig=3, savepath=None):
    ''''
    Plots the IOM-clusters in velocity space.

    Parameters:
    df(vaex.DataFrame): The halo set and its associated labels
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''
    x_axis = ['vR', 'vR', 'vz']
    y_axis = ['vphi', 'vz', 'vphi']

    xlabels = ['$v_R$ [km/s]', '$v_R$ [km/s]', '$v_{z}$ [km/s]']
    ylabels = ['$v_{\phi}$ [km/s]', '$v_z$ [km/s]', '$v_{\phi}$ [km/s]']

    fig, axs = plt.subplots(1,3, figsize = [14, 4]) #figsize = [20,9]

    unique_labels = np.unique(df.labels.values)

    df_minsig = df[(df.labels>0) & (df.maxsig>minsig)]
    cmap, norm = get_cmap(df_minsig)

    background = df[(df.vR>-500) & (df.vR<500) & (df.vphi>-500) & (df.vphi<500)]

    for i in range(3):
        plt.sca(axs[i])
        plt.xlim([-570, 570])
        plt.ylim([-570, 570])
        background.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1)# )#, length_limit=60000)
        df_minsig.scatter(x_axis[i], y_axis[i], s=2, c=df_minsig.labels.values,
                                cmap=cmap, norm=norm, alpha=0.8)

        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])

    plt.tight_layout(w_pad=1)
    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

####################################################################################################

def plot_membership_probability(df, clusterNr, savepath=None):
    ''''
    Plots all combinations of clustering features,
    color coded by membership probability for a specific cluster.
    Here darker red indicates lower membership probability.

    Parameters:
    df(vaex.DataFrame): The halo set and its associated labels and membership probabilities
    clusterNr(int): The label of the cluster we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    x_axis = ['Lz/10e3', 'Lz/10e3', 'circ', 'Lperp/10e3', 'circ', 'circ']
    y_axis = ['En/10e4', 'Lperp/10e3', 'Lz/10e3', 'En/10e4', 'Lperp/10e3', 'En/10e4']

    fig, axs = plt.subplots(2,3, figsize = [20,9])
    plt.tight_layout()

    cluster = df[df.labels==clusterNr]
    cmap = plt.get_cmap('YlOrRd_r')

    for i in range(6):
        plt.sca(axs[int(i/3), i%3])
        cluster.scatter(x_axis[i], y_axis[i], s=10, c=cluster.membership_probability.values,
                        cmap=cmap, alpha=0.8)

    fig.subplots_adjust(top=0.95)
    plt.suptitle(f'Cluster {clusterNr} color coded by membership probability')

    #plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

#############################################################################################################

def plot_outliers_vspace(df, clusterNr, savepath=None):
    '''
    Plots the outliers of a specific cluster over three panels:
    HDBSCAN (inliers in yellow, outliers in red)
    Membership probability (darker red is lower membership probability),
    Both (the data points that both methods agree are outliers)

    Parameters:
    df(vaex.DataFrame): The halo set and its associated labels and membership probabilities
    clusterNr(int): The label of the cluster we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(3,3, figsize = [12,8])
    fig.tight_layout()

    cluster = df[df.labels==clusterNr]

    cmaplist = ['firebrick', 'gold']

    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 2)
    bounds = np.linspace(0, 2, 3)
    norm = matplotlib.colors.BoundaryNorm(bounds, 2)#cmap.N)

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']
    cmap2 = plt.get_cmap('YlOrRd_r')

    for i in range(3):
        plt.sca(axs[0][i])
        cluster.scatter(x_axis[i], y_axis[i], s=5, c=cluster.v_space_labels.values, cmap=cmap1, norm=norm, xlabel=' ')

        plt.sca(axs[1][i])
        cluster.scatter(x_axis[i], y_axis[i], s=5, c=cluster.membership_probability.values, cmap=cmap2, xlabel = ' ')

        plt.sca(axs[2][i])
        cluster.scatter(x_axis[i], y_axis[i], s=5, c='lightgrey', alpha=0.5)
        cluster[(cluster.v_space_labels==0) & (cluster.membership_probability<0.05)].scatter(x_axis[i], y_axis[i], s=10, c='blue')

    axs[0][1].set_title(f'eDR3 cluster {clusterNr}\nHDBSCAN outliers in red')
    axs[1][1].set_title('Membership probability')
    axs[2][1].set_title('Outliers according to both')

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

######################################################################################################

def plot_clusters_over_subpanels(df, feature1, feature2, minsig=3, savepath=None):

    '''
    Plots the clusters over 15 subpanels in order to see them better

    Parameters:
    df(vaex.DataFrame): Dataframe containing the stars and their labels
    feature1(str): Feature to plot on the x-axis
    feature2(str): Feature to plot on the y-axis
    minsig(float): Minimum significance level of the clusters you want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(3,5, figsize = [25,13], sharex=True, sharey=True)

    df_minsig = df[(df.labels>0) & (df.maxsig>minsig)]
    cmap, norm = get_cmap(df_minsig)
    unique_labels = np.unique(df_minsig.labels.values)
    num_labels = len(unique_labels)

    for idx in range(15):
        plt.sca(axs[int(idx/5), idx%5])
        if(feature1 in ['vR', 'vphi', 'vz'] and feature2 in ['vR', 'vphi', 'vz']):
            plt.xlim([-430, 430])
            plt.ylim([-430, 430])
        labellist = np.where(np.array(range(1, num_labels+1))%15 == idx)
        labellist = unique_labels[labellist]
        indices = np.where(np.isin(df_minsig.labels.values, labellist)==True)
        v_slice = df_minsig.take(indices[0])
        df.scatter(feature1, feature2, s=0.5, c='lightgrey', alpha=0.1)# )#, length_limit = 60000)
        v_slice.scatter(feature1, feature2, s=5, c=v_slice.labels.values, cmap=cmap, norm=norm, alpha=0.8)

    plt.tight_layout()
    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

######################################################################################################

def plot_original_data(df, savepath=None):

    from matplotlib.cm import ScalarMappable as sm
    import matplotlib as mpl

    c_args = dict(colormap = 'afmhot', vmin = 0, vmax = 6)

    x_axis = ['Lz/10e2', 'Lz/10e2', 'circ', 'Lperp/10e2', 'circ', 'circ']
    y_axis = ['En/10e4', 'Lperp/10e2', 'Lz/10e2', 'En/10e4', 'Lperp/10e2', 'En/10e4']

    xlabels = ['$L_z$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]', '$\eta$',
               '$L_{\perp}$ [$10^3$ kpc km/s]', '$\eta$', '$\eta$']

    ylabels = ['$E$ [$10^5$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$L_z$ [$10^3$ kpc km/s]',
               '$E$ [$10^5$ km$^2$/s$^2$]', '$L_{\perp}$ [$10^3$ kpc km/s]', '$E$ [$10^5$ km$^2$/s$^2$]']

    limits =  [[[-4.5, 4.6], [-1.7, 0]],
               [[-4.5, 4.6], [-0, 4.3]],
               [[-1, 0.7], [-4.5, 4.6]],
               [[0, 4.3], [-1.7, 0]],
               [[-1, 0.7], [0, 4.3]],
               [[-1, 0.7], [-1.7, 0]]]

    fig, axs = plt.subplots(2,3, figsize = [14,8])

    for i in range(6):
        plt.sca(axs[int(i/3), i%3])
        im = df.plot(x_axis[i], y_axis[i], f='log', colorbar=False, limits=limits[i], **c_args)
        #im.set_clim(0, 6)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])


    fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.subplots_adjust(right=0.9)
    #plt.colorbar(im, ax=axs)

    cb = fig.add_axes([1.01, 0.08, 0.015, 0.9])
    #cb = plt.colorbar(ax1, cax = cbaxes)

    fig.colorbar(sm(norm=mpl.colors.Normalize(vmin=0, vmax=6), cmap='afmhot'), cax=cb, label = "log count (*)")

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    else:
        plt.show()

######################################################################################################

def plot_subgroup_vspace(df, clusterNr, zoom=False, savepath=None):
    '''
    Plots the subgroups of a specific cluster in velocity space.

    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    clusterNr(int): Label of cluster that we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(1,3, figsize = [13.5,3.5])
    fig.tight_layout()

    cluster = df[df.labels==clusterNr]

    cmaplist = ['lightgrey', 'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'yellow', 'deeppink']

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']

    cluster_colors = [cmaplist[col] for col in cluster.v_space_labels.values]

    for i in range(3):
        plt.sca(axs[i])
        if(zoom == False):
            axs[i].set_xlim(-420, 420)
            axs[i].set_ylim(-420, 420)
        cluster.scatter(x_axis[i], y_axis[i], s=15, c=cluster_colors, alpha=0.8)

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

######################################################################################################

def plot_cluster_vspace(df, clusterNr, savepath=None):
    '''
    Plots the subgroups of a specific cluster in velocity space.

    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    clusterNr(int): Label of cluster that we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(1,4, figsize = [13.5,3.5])
    fig.tight_layout()

    cluster = df[df.labels==clusterNr]

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']

    for i in range(3):
        plt.sca(axs[i])
        axs[i].set_xlim(-420, 420)
        axs[i].set_ylim(-420, 420)
        cluster.scatter(x_axis[i], y_axis[i], s=15, c='red', alpha=0.8)

    plt.tight_layout()

    cmap, norm = get_cmap(df[(df.labels>0) & (df.maxsig>3)])
    plt.sca(axs[3])
    axs[3].set_xlim(-3.2, 3.2)
    axs[3].set_ylim(-1.7, -0.6)
    df.scatter('Lz/10e2', 'En/10e4', s=0.5, c='lightgrey', alpha=0.1)#)#, length_limit=60000)
    cluster.scatter('Lz/10e2', 'En/10e4', s=2, c=cluster.labels.values, cmap=cmap, norm=norm, alpha=1, label=f'{clusterNr}')

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()


########################################################################################################

def plot_substructure_vspace(df, clusterNr, savepath=None):
    '''
    Plots the subgroups of a specific cluster in velocity space.

    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    clusterNr(int): Label of cluster that we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(1,3, figsize = [13.5,3.5])
    fig.tight_layout()

    cluster = df[df.labels_substructure==clusterNr]

    cmaplist = ['lightgrey', 'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'yellow', 'deeppink']

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']

    cluster_colors = [cmaplist[col] for col in cluster.v_space_labels_substructure.values]

    for i in range(3):
        plt.sca(axs[i])
        axs[i].set_xlim(-420, 420)
        axs[i].set_ylim(-420, 420)
        cluster.scatter(x_axis[i], y_axis[i], s=15, c=cluster_colors, alpha=0.8)

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

########################################################################################################

def plot_subgroup_vspace_En_Lz(df, clusterNr, savepath=None):
    '''
    Plots the subgroups in velocity space of a specific cluster.

    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    clusterNr(int): Label of cluster that we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(1,4, figsize = [20,4.5])
    #fig.tight_layout()

    cluster = df[df.labels==clusterNr]

    cmaplist = ['lightgrey', 'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkolivegreen', 'purple',
                'plum', 'darkgoldenrod', 'darkgrey', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue',
                'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue',
                'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue']

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']
    xlabels = ['$v_R$ [km/s]', '$v_R$ [km/s]', '$v_{\phi}$ [km/s]']
    ylabels = ['$v_{\phi}$ [km/s]', '$v_z$ [km/s]', '$v_z$ [km/s]']

    cluster_colors = [cmaplist[col] for col in cluster.v_space_labels.values]

    for i in range(3):
        plt.sca(axs[i])
        axs[i].set_xlim(-400, 400)
        axs[i].set_ylim(-400, 400)
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1)#)#, length_limit=60000)
        cluster.scatter(x_axis[i], y_axis[i], s=7, c=cluster_colors, alpha=0.8)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])

    cmap, norm = get_cmap(df[(df.labels>0) & (df.maxsig>3)])
    plt.sca(axs[3])
    axs[3].set_xlim(-3.2, 3.2)
    axs[3].set_ylim(-1.7, -0.6)
    df.scatter('Lz/10e2', 'En/10e4', s=0.5, c='lightgrey', alpha=0.1)#)#, length_limit=60000)
    cluster.scatter('Lz/10e2', 'En/10e4', s=2, c=cluster.labels.values, cmap=cmap, norm=norm, alpha=1, label=f'{clusterNr}')
    #plt.legend(loc='upper right')
    #cluster.scatter('Lz/10e3', 'En/10e4', s=2, c='red', alpha=0.7)
    plt.xlabel('$L_z$ [$10^3$ kpc km/s]')
    plt.ylabel('$E$ [$10^5$ km$^2$/s$^2$]')

    axs[3].text(2.5, -0.7, f'{clusterNr}',
         weight='bold', horizontalalignment='center',
         verticalalignment='center',
         fontsize=16, color='black')

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=100)
    else:
        plt.show()

########################################################################################################

def plot_substructure_vspace_En_Lz(df, clusterNr, name, savepath=None):
    '''
    Plots the subgroups in velocity space of a specific cluster.

    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    clusterNr(int): Label of cluster that we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(1,4, figsize = [20,4.5])
    cluster = df[df.labels_substructure==clusterNr]

    cmaplist = ['lightgrey', 'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue',
                'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue'
                'ForestGreen', 'DodgerBlue', 'MediumVioletRed', 'MediumBlue',
                'DarkCyan', 'gold', 'deeppink', 'yellow', 'red', 'lime', 'darkgrey', 'purple',
                'plum', 'darkgoldenrod', 'darkolivegreen', 'crimson', 'navy', 'hotpink', 'orange',
                'cadetblue', 'steelblue']

    x_axis = ['vR', 'vR', 'vphi']
    y_axis = ['vphi', 'vz', 'vz']
    xlabels = ['$v_R$ [km/s]', '$v_R$ [km/s]', '$v_{\phi}$ [km/s]']
    ylabels = ['$v_{\phi}$ [km/s]', '$v_z$ [km/s]', '$v_z$ [km/s]']

    cluster_colors = [cmaplist[col%len(cmaplist)] for col in cluster.v_space_labels_substructure.values]

    for i in range(3):
        plt.sca(axs[i])
        axs[i].set_xlim(-550, 550)
        axs[i].set_ylim(-550, 550)
        df.scatter(x_axis[i], y_axis[i], s=0.5, c='lightgrey', alpha=0.1)#)#, length_limit=60000)
        cluster.scatter(x_axis[i], y_axis[i], s=7, c=cluster_colors, alpha=0.8)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])

    cmap = plt.get_cmap('jet', len(np.unique(df.labels_substructure.values)))
    norm = mpl.colors.Normalize(vmin=0, vmax=len(np.unique(df.labels_substructure.values)))

    plt.sca(axs[3])
    axs[3].set_xlim(-0.5, 0.5)
    axs[3].set_ylim(-1.75, -0.15)
    df.scatter('Lz/10e2', 'En/10e4', s=0.5, c='lightgrey', alpha=0.1)#, length_limit=60000)
    cluster.scatter('Lz/10e2', 'En/10e4', s=2, c=cluster.labels_substructure.values, cmap=cmap, norm=norm, alpha=0.8, label=f'{name}')
    plt.xlabel('$L_z$ [$10^3$ kpc km/s]')
    plt.ylabel('$E$ [$10^5$ km$^2$/s$^2$]')

    axs[3].text(3, -3.5, f'{name}',
         weight='bold', horizontalalignment='center',
         verticalalignment='center',
         fontsize=16, color='black')

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

########################################################################################################

def plot_v_space_dispersions(dispersion_xyz, savepath=None):

    '''
    Plots velocity dispersion in vxyz as the standard deviation along the principal component for each cluster.

    Parameters:
    dispersion_xyz(np.ndarray): An array with three columns, where the 0th column contains the standard deviation
                                along the 0th principal component for each cluster, and correspondingly for
                                the 1st and 2nd principal component.
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(4, 1, figsize = (6.67, 8))
    plt.setp(axs, xlim=(0, 125), ylim=(0, 30))

    axs[0].hist(dispersion_xyz[:, 0], bins = 58, range=[0, 125], color='steelblue', alpha=0.5)
    axs[0].set_xlabel('$\sigma_1$ [km/s]')

    axs[1].hist(dispersion_xyz[:, 1], bins = 58, range=[0, 125], color='steelblue', alpha=0.5)
    axs[1].set_xlabel('$\sigma_2$ [km/s]')

    axs[2].hist(dispersion_xyz[:, 2], bins = 58, range=[0, 125], color='steelblue', alpha=0.5)
    axs[2].set_xlabel('$\sigma_3$ [km/s]')

    axs[3].hist(dispersion_xyz[:, 3], bins = 58, range=[0, 125], color='steelblue', alpha=0.5)
    axs[3].set_xlabel('$\sigma_{tot}$ [km/s]')

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

##########################################################################################################

def plot_chi2_vs_Dsquared(df, degrees_freedom=4, savepath=None):
    '''
    Plots the chi-square distribution with n degrees of freedom versus the squared Mahalanobis distances
    for all stars assigned to some cluster. Specifically, the Mahalanobis distance between each star
    being a cluster member, and the cluster distribution modelled as a Gaussian.

    Parameters:
    df(vaex.DataFrame): The halo set
    degrees_freedom: The number of degrees of freedom we want to use for the chi-square distribution
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(figsize = [6.67, 4.5]) #figsize = [20,9] for 2x3 panel

    k = degrees_freedom

    x = np.linspace(chi2.ppf(0.00, df=k), chi2.ppf(0.9999, df=k), 100)

    axs.hist(df[df.labels>0].D_squared.values, bins = 50, color='steelblue', alpha=0.5,
             density=True, label = r'$D^2$', range=[0,chi2.ppf(0.9999, df=k)])
    axs.plot(x, chi2.pdf(x, df=k), alpha=0.8, color='red', linestyle='dashed', label=f'$\chi^2$')
    axs.legend()
    axs.set_xlabel('Squared Mahalanobis distance')
    axs.set_ylabel('Probability density')

    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

##########################################################################################################

def plot_significance_levels(df, significance_levels = [3, 4, 5, 6, 8, 10], savepath=None):
    '''
    Plot clusters over subpanels depending on significance level.
    This is an artifict from when we selected clusters from the tree
    according to the largest structure with at least 3 sigma significance.
    Assumes that the input dataframe contains cluster labels according to
    "labels_3sigma", "labels_4sigma" etc, where nsigma indicates the clusters that
    were extracted according to that significance level.

    Parameters:
    df(vaex.DataFrame): Dataframe containing the stars and their labels
    significance_levels([int]): List of significance levels we want to plot
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''

    fig, axs = plt.subplots(2,3, figsize = [14,6], sharex=True, sharey=True)

    for idx, n_sigma in enumerate(significance_levels):
        colname = f'labels_{n_sigma}sigma'
        cmap = plt.get_cmap('gist_rainbow', max(df[colname].values))
        plt.sca(axs[int(idx/3), idx%3])
        plt.xlim([-4000, 4300])
        plt.ylim([-175000, 1000])
        df.scatter('Lz', 'En', s=0.5, c='lightgrey', alpha=0.1)#, length_limit = 60000, selection = df.En<-50000)
        df_slice = df[df[colname]>0]
        df_slice.scatter('Lz', 'En', s=0.5, c=df_slice[colname].values, cmap=cmap, alpha=0.6)
        plt.title(f'{n_sigma}$\sigma$')

##########################################################################################################

def plot_member_count_hist(df, minsig=3, derived_members=True, savepath=None):
    '''
    Plots the number of members per cluster

    Parameters:
    df(vaex.DataFrame): Dataframe containing stars and cluster labels
    minsig(float): Minimum significance level to consider in the clusters
    derived_members(Bool): If True, overplots a dashed histogram of the members which have been
                           associated according to some minimum membership probability.
    savepath(str): Location where the plot should be saved, if None the plot is not saved
    '''
    fig, axs = plt.subplots(figsize = [6.67, 4.5])

    df2 = df[df.maxsig>minsig]

    three_sigma_labels = np.unique(df2[df2.maxsig>minsig].labels.values)
    mapped_labels = np.array(range(len(three_sigma_labels)))

    unique_labels = np.unique(df2.labels.values)

    num_labels = len(unique_labels)

    labels = df2[df2.labels>0].labels.values
    for l in range(len(labels)):
        mapped_label = np.where(three_sigma_labels == labels[l])
        labels[l] = mapped_label[0]

    maxlabel = int(max(labels))
    n, bins, patches = axs.hist(labels, log=True, bins = num_labels,
                                color='steelblue', alpha=0.5, range=[0, maxlabel])

    print('Members accoring to single linkage process')
    print(f'Mean number of members: {np.mean(n):.3f}')
    print(f'Median number of members: {np.median(n)}')
    print(f'Max number of members: {np.max(n)}')
    print(f'Minimum number of members: {np.min(n)}')

    if(derived_members == True):

        derived_labels = df[df.derived_labels>0].derived_labels.values

        for l in range(len(derived_labels)):
            mapped_label = np.where(three_sigma_labels == derived_labels[l])
            derived_labels[l] = mapped_label[0]

        n, bins, patches = axs.hist(derived_labels, log=True, bins = num_labels, histtype='step',
                 color='red', linestyle='dashed', alpha=0.8, range = [0, maxlabel])

        print('\nDerived members accoring to membership probability')
        print(f'Mean number of members: {np.mean(n):.3f}')
        print(f'Median number of members: {np.median(n)}')
        print(f'Max number of members: {np.max(n)}')
        print(f'Minimum number of members: {np.min(n)}')

    axs.set_xticks([])
    plt.xlim([-0.5, maxlabel+0.5])
    plt.xlabel('Cluster ID')
    plt.ylabel('Member count (log)')
    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

##########################################################################################################

def plot_clusters_by_significance(df, minsig=3, savepath=None):

    '''
    Plots clusters with a significance higher than minsig in En-Lz space, color
    coded by their statistical significance.
    '''

    fig, ax = plt.subplots(figsize = (8.5, 6))

    cmap = plt.get_cmap('plasma_r')

    df_minsig = df[(df.labels>0) & (df.maxsig>minsig)]

    df[(df.Lz<4000) & (df.Lz>-4000)].scatter('Lz/10e2', 'En/10e4', s=0.5, c='lightgrey', alpha=0.1)#, length_limit=60000)
    df_minsig.scatter('Lz/10e2', 'En/10e4', s=1, c=df_minsig.maxsig.values,
                            cmap=cmap, alpha=0.6)

    plt.colorbar()
    #plt.title('Clusters - Maximized statistical significance', fontsize=7)
    plt.xlabel('$L_z$ [$10^3$ kpc km/s]')
    plt.ylabel('$E$ [$10^5$ km$^2$/s$^2$]')
    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

##########################################################################################################

def plot_substructures_En_Lz(df, savepath=None):
    '''
    Plots substructures in En-Lz, displaying their 95% extent and covariance in this 2D projection.

    Parameters:
    df(vaex.DataFrame): Our star catalogue containing substructure labels
    savepath(str): Path to save location of plot, if None the plot is only displayed
    '''
    fig, axs = plt.subplots(figsize = [4.5, 4])

    axs.scatter(df.Lz.values, df.En.values, alpha=0.1, c='lightgrey', s=0.5)

    for substructure in np.unique(df.labels_substructure.values):

        if(substructure==0):
            #let's not plot an ellipse for the background noise
            continue

        df_substructure = df[df.labels_substructure == substructure]

        mu = np.mean(df_substructure['Lz', 'En'].values.T, axis = 1)
        cov = np.cov(df_substructure['Lz', 'En'].values.T)

        # membership_probability.confidence_ellipse(mu, cov, 0, 1, ax=axs, n_std=2.0,
                           # alpha=0.8, edgecolor='black', zorder=2)


    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

##########################################################################################################
