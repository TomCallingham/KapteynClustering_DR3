# %%
import pandas as pd
import numpy as np
import sys
import glob
import os
import time
import vaex
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from astropy.io import ascii
import matplotlib.cm as cmtime
import glob
from scipy.spatial import cKDTree
import itertools
from IPython.display import Image
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy import stats
import seaborn as sns
from scipy.interpolate import splprep, splev, splrep
from matplotlib.patches import Ellipse
from scipy.stats import norm
from IPython.display import display
import tqdm
# import vaex.ml  # Un-comment this line for old vaex
from scipy import special, linalg

sys.path.append(r'/net/gaia2/data/users/ruizlara/useful_codes/')
#import pot_koppelman2018 as Potential
import H99Potential_sofie as Potential
import lallement18_distant as l18

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import warnings
warnings.filterwarnings('ignore')

pi = math.pi


# %% [markdown]
# # Identify possible disrupted star clusters
#
# In this notebook I will examine the results of the clustering algorithm
# in 3D and 4D as well as the outputs from HDBSCAN in velocities to
# identify possible disrupted cluster candidates.

# %% [markdown]
# ### Diagonalise matrix (info):
#
# http://eio.usc.es/eipc1/BASE/BASEMASTER/FORMULARIOS-PHP/MATERIALESMASTER/Mat_14_master0809multi-tema5.pdf
#
# https://docs.scipy.org/doc/scipy/tutorial/linalg.html

# %%
def inspect_star_cluster(df, cluster, prefix):

    limits1 = [[-3000.0 / 10**3, 3000.0 / 10**3], [-160000. / 10**4, -50000. / 10**4]]
    limits2 = [[-3000.0 / 10**3, 3000.0 / 10**3], [0. / 10**3, 4000. / 10**3]]

    df.select('(labels==' + str(cluster) + ')', name='this_cluster')
    df.select('(feh_lamost_lrs>-99.0)', name='lamost_good')

    LzLperp = df.count(binby=[df.Lz / 10**3, df.Lperp / 10**3], limits=[[limits2[0]
                       [0], limits2[0][1]], [limits2[1][0], limits2[1][1]]], shape=(200, 200))
    LzE = df.count(binby=[df.Lz / 10**3, df.En / 10**4], limits=[[limits1[0][0],
                   limits1[0][1]], [limits1[1][0], limits1[1][1]]], shape=(200, 200))

    f1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(
        8, figsize=(30, 12), sharex=True, facecolor='w')
    ax1 = plt.subplot(241)
    ax2 = plt.subplot(242)
    ax3 = plt.subplot(243)
    ax4 = plt.subplot(244)
    ax5 = plt.subplot(245)
    ax6 = plt.subplot(246)
    ax7 = plt.subplot(247)
    ax8 = plt.subplot(248)

    ax1.imshow(
        np.log10(
            LzLperp.T),
        origin='lower',
        extent=[
            limits2[0][0],
            limits2[0][1],
            limits2[1][0],
            limits2[1][1]],
        cmap='Greys',
        aspect='auto')

    ax1.plot(
        df.evaluate(
            'Lz',
            selection='this_cluster') /
        10**3,
        df.evaluate(
            'Lperp',
            selection='this_cluster') /
        10**3,
        'ro',
        ms=3,
        lw=0.0,
        alpha=0.2)

    ax1.set_ylabel(r'$\rm \sqrt{L_x^2+L_y^2}$', fontsize=25)
    ax1.set_xlabel(r'$\rm L_z$', fontsize=25)

    ax1.set_xlim(limits2[0])
    ax1.set_ylim(limits2[1])

    ax1.tick_params(labelsize=20)

    ax2.imshow(
        np.log(
            LzE.T),
        origin='lower',
        extent=[
            limits1[0][0],
            limits1[0][1],
            limits1[1][0],
            limits1[1][1]],
        cmap='Greys',
        aspect='auto')

    ax2.plot(
        df.evaluate(
            'Lz',
            selection='this_cluster') /
        10**3,
        df.evaluate(
            'En',
            selection='this_cluster') /
        10**4,
        'ro',
        ms=3,
        lw=0.0,
        alpha=0.2)

    ax2.set_xlabel(r'$\rm L_z$', fontsize=25)
    ax2.set_ylabel(r'$\rm E$', fontsize=25)

    ax2.set_xlim(limits1[0])
    ax2.set_ylim(limits1[1])

    ax2.tick_params(labelsize=20)

    limits = [[-450.0, 450.0], [-450.0, 450.0]]

    vxvy = df.count(binby=[df.vx, df.vy + 232.0], limits=[[limits[0][0], limits[0][1]],
                    [limits[1][0], limits[1][1]]], shape=(200, 200))
    vzvy = df.count(binby=[df.vz, df.vy + 232.0], limits=[[limits[0][0], limits[0][1]],
                    [limits[1][0], limits[1][1]]], shape=(200, 200))
    vxvz = df.count(binby=[df.vx, df.vz], limits=[[limits[0][0], limits[0][1]],
                    [limits[1][0], limits[1][1]]], shape=(200, 200))

    ax3.imshow(
        np.log10(
            vzvy.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')

    ax3.plot(
        df.evaluate(
            'vz',
            selection='this_cluster'),
        df.evaluate(
            'vy',
            selection='this_cluster') +
        232.0,
        'ro',
        ms=5,
        lw=0.0,
        alpha=0.2)

    ax4.imshow(
        np.log10(
            vxvy.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')

    ax4.plot(
        df.evaluate(
            'vx',
            selection='this_cluster'),
        df.evaluate(
            'vy',
            selection='this_cluster') +
        232.0,
        'ro',
        ms=5,
        lw=0.0,
        alpha=0.2)

    ax5.imshow(
        np.log10(
            vxvz.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')

    ax5.plot(df.evaluate('vx', selection='this_cluster'), df.evaluate(
        'vz', selection='this_cluster'), 'ro', ms=5, lw=0.0, alpha=0.2)

    Chist, Cbin_edges = np.histogram(df.evaluate(
        'feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), range=(-2.3, 0.1), bins=20)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax6.step(Cbin_edges, Chist, color='r', lw=1, where='mid')
    Nmax = np.float(np.amax(Chist))

    Chist, Cbin_edges = np.histogram(df.evaluate('feh_lamost_lrs'), range=(-2.3, 0.1), bins=20)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax6.step(Cbin_edges, Nmax * Chist / np.float(np.amax(Chist)), color='k', lw=1, where='mid')

    # feh_param computation
    values = ax7.hist(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'),
                      histtype='step', density=True, bins=20, range=(-2.3, 0.1), cumulative=True)
    centers = 0.5 * (values[1][1:] + values[1][:-1])
    ax7.plot(centers, values[0])
    tck = splrep(centers, values[0], s=0)
    xnew = np.linspace(-2.2, 0.0, 100)
    ynew = splev(xnew, tck, der=0)
    ynew_der = splev(xnew, tck, der=1)
    ax7.plot(xnew, ynew / np.nanmax(ynew))
    ax7.plot(xnew, ynew_der / np.nanmax(ynew_der))

    ax3.set_xlabel('vz', fontsize=25)
    ax3.set_ylabel('vy', fontsize=25)

    ax3.set_xlim(limits[0])
    ax3.set_ylim(limits[1])

    ax4.set_xlabel('vx', fontsize=25)
    ax4.set_ylabel('vy', fontsize=25)

    ax4.set_xlim(limits[0])
    ax4.set_ylim(limits[1])

    ax5.set_xlabel('vx', fontsize=25)
    ax5.set_ylabel('vz', fontsize=25)

    ax5.set_xlim(limits[0])
    ax5.set_ylim(limits[1])

    ax6.set_xlabel('[Fe/H]', fontsize=25)
    ax6.set_ylabel('N', fontsize=25)

    ax7.set_xlabel('[Fe/H]', fontsize=25)
    ax7.set_ylabel('*', fontsize=25)

    # l-b plot
    limits = [[np.nanmin(df.evaluate('l')), np.nanmax(df.evaluate('l'))], [
        np.nanmin(df.evaluate('b')), np.nanmax(df.evaluate('b'))]]
    lb_dens = df.count(binby=[df.l, df.b], limits=[[limits[0][0], limits[0][1]],
                       [limits[1][0], limits[1][1]]], shape=(200, 200))
    ax8.imshow(
        np.log10(
            lb_dens.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')
    ax8.plot(df.evaluate('l', selection='this_cluster'), df.evaluate(
        'b', selection='this_cluster'), 'ro', ms=25, lw=0.0, alpha=0.2)

    ax8.set_xlabel('l', fontsize=25)
    ax8.set_ylabel('b', fontsize=25)

    ax3.tick_params(labelsize=20)
    ax4.tick_params(labelsize=20)
    ax5.tick_params(labelsize=20)
    ax6.tick_params(labelsize=20)
    ax7.tick_params(labelsize=20)
    ax8.tick_params(labelsize=20)

    Ntotal, Nlamost = len(df.evaluate('feh_lamost_lrs', selection='this_cluster')), len(
        df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'))
    sigma_feh = round(np.nanstd(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)')), 2)
    #param_feh = round(np.nanpercentile(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), 66) - np.nanpercentile(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), 34), 2)
    param_feh = round(np.nanmax(ynew_der) / np.nanmedian(ynew_der), 3)

    # Computing covariances in E, Lz, Lperp, circ
    features = ['scaled_En*10e2', 'scaled_Lperp*10e2', 'scaled_Lz*10e2', 'circ*10e2']

    df_cluster = df[df.labels == cluster]
    #mean = np.mean(df_cluster[features].values, axis=0)
    cov = np.cov(df_cluster[features].values.T)
    cov_En, cov_Lperp, cov_Lz, cov_circ = round(cov[0, 0], 2), round(
        cov[1, 1], 2), round(cov[2, 2], 2), round(cov[3, 3], 2)

    eig_vals, eig_vects = linalg.eig(cov)
    D = linalg.inv(eig_vects).dot(cov).dot(eig_vects)

    eig00, eig11, eig22, eig33 = round(D[0, 0], 2), round(D[1, 1], 2), round(D[2, 2], 2), round(D[3, 3], 2)

    f1.suptitle(
        'Cluster #' +
        str(cluster) +
        '; ' +
        str(Ntotal) +
        ' stars; ' +
        str(Nlamost) +
        ' lamost; std_feh = ' +
        str(sigma_feh) +
        '; feh_param = ' +
        str(param_feh) +
        '; cov_En = ' +
        str(cov_En) +
        '; cov_Lperp = ' +
        str(cov_Lperp) +
        '; cov_Lz = ' +
        str(cov_Lz) +
        '; cov_circ = ' +
        str(cov_circ) +
        '; eig00 = ' +
        str(eig00) +
        '; eig11 = ' +
        str(eig11) +
        '; eig22 = ' +
        str(eig22) +
        '; eig33 = ' +
        str(eig33),
        fontsize=20)

    plt.tight_layout(w_pad=0.0, rect=[0.00, 0.00, 1, 0.95])

    plt.savefig(
        '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/' +
        str(prefix) +
        '_' +
        str(cluster) +
        '.png')

    plt.show()

    return cluster, Ntotal, Nlamost, sigma_feh, param_feh, cov_En, cov_Lperp, cov_Lz, cov_circ, float(
        eig00), float(eig11), float(eig22), float(eig33)


# %%
def inspect_star_cluster_noplot(df, cluster):

    df.select('(labels==' + str(cluster) + ')', name='this_cluster')
    df.select('(feh_lamost_lrs>-99.0)', name='lamost_good')

    Ntotal, Nlamost = len(df.evaluate('feh_lamost_lrs', selection='this_cluster')), len(
        df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'))
    sigma_feh = round(np.nanstd(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)')), 2)

    # feh_param computation
    plt.figure()
    values = plt.hist(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'),
                      histtype='step', density=True, bins=20, range=(-2.3, 0.1), cumulative=True)
    plt.close()
    centers = 0.5 * (values[1][1:] + values[1][:-1])
    tck = splrep(centers, values[0], s=0)
    xnew = np.linspace(-2.2, 0.0, 100)
    ynew = splev(xnew, tck, der=0)
    ynew_der = splev(xnew, tck, der=1)
    param_feh = np.nanmax(ynew_der) / np.nanmedian(ynew_der)

    # Computing covariances in E, Lz, Lperp, circ
    features = ['scaled_En*10e2', 'scaled_Lperp*10e2', 'scaled_Lz*10e2', 'circ*10e2']
    df_cluster = df[df.labels == cluster]
    #mean = np.mean(df_cluster[features].values, axis=0)
    cov = np.cov(df_cluster[features].values.T)
    cov_En, cov_Lperp, cov_Lz, cov_circ = cov[0, 0], cov[1, 1], cov[2, 2], cov[3, 3]

    eig_vals, eig_vects = linalg.eig(cov)
    D = linalg.inv(eig_vects).dot(cov).dot(eig_vects)

    eig00, eig11, eig22, eig33 = float(D[0, 0]), float(D[1, 1]), float(D[2, 2]), float(D[3, 3])

    return cluster, Ntotal, Nlamost, sigma_feh, param_feh, cov_En, cov_Lperp, cov_Lz, cov_circ, eig00, eig11, eig22, eig33


# %%
def get_cluster_description_table(df, minsig=3):
    '''
    Returns Table 1 of Lovdal et al. (2021),
    giving summary characteristics of the clusters extracted.
    Contains cluster ID, #core_members, #total_associated_members,
    cluster mean in non-scaled clustering space, and covariance matrix entries.
    [mu0, mu1, mu2, mu3] corresponds to ['En', 'Lperp', 'Lz', 'circ'] and similarly
    [cov00, cov01, cov02, cov03] corresponds to cov[EE, ELperp, ELz, Ecirc].
    Note that the covariance matrix is symmetric, so cov03=cov30 and we omit duplicate entries.

    Parameters:
    df(vaex.DataFrame): Star catalogue with cluster labels
    minsig(float): Minimum statistical significance level that we want to consider

    Returns:
    cluster_table(vaex.DataFrame): Dataframe containing cluster characteristics
    '''

    features = ['scaled_En*10e2', 'scaled_Lperp*10e2', 'scaled_Lz*10e2', 'circ*10e2']

    unique_labels = np.unique(df[df.maxsig > 3].labels.values)
    n_clusters = len(unique_labels)

    core_members = np.zeros(n_clusters)
    total_members = np.zeros(n_clusters)
    significance = np.zeros(n_clusters)

    # Mean of clusters across features E, Lperp, Lz, circ
    mu = np.zeros((n_clusters, 4))

    # Array of covariance matrices
    cov = np.zeros((n_clusters, 4, 4))

    # eigenvalues arrays (elements of the diagonal in the diagonal matrix)
    eig00, eig11, eig22, eig33 = [], [], [], []
    for idx, label in enumerate(unique_labels):

        cluster = df[df.labels == label]
        mean = np.mean(cluster[features].values, axis=0)
        covariance = np.cov(cluster[features].values.T)

        eig_vals, eig_vects = linalg.eig(covariance)
        D = linalg.inv(eig_vects).dot(covariance).dot(eig_vects)

        eig00.append(float(D[0, 0]))
        eig11.append(float(D[1, 1]))
        eig22.append(float(D[2, 2]))
        eig33.append(float(D[3, 3]))

        mu[idx, :] = mean
        cov[idx, :, :] = covariance
        significance[idx] = cluster.maxsig.values[0]
        core_members[idx] = cluster.count()
        total_members[idx] = df[df.labels == label].count()

    cluster_table = vaex.from_arrays(label=unique_labels, significance=significance,
                                     core_members=core_members, total_members=total_members,
                                     mu0=mu[:, 0], mu1=mu[:, 1], mu2=mu[:, 2], mu3=mu[:, 3],
                                     cov00=cov[:, 0, 0], cov01=cov[:, 0, 1], cov02=cov[:, 0, 2],
                                     cov03=cov[:, 0, 3],
                                     cov11=cov[:, 1, 1], cov12=cov[:, 1, 2], cov13=cov[:, 1, 3],
                                     cov22=cov[:, 2, 2], cov23=cov[:, 2, 3],
                                     cov33=cov[:, 3, 3], eig00=eig00, eig11=eig11, eig22=eig22, eig33=eig33)

    return cluster_table


# %%
def scaleData(ds, features, minmax_values):
    '''
    Scales data to the range [-1, 1].

    Parameters:
    ds(vaex.DataFrame): Dataframe containing the data to be scaled
    features([str]): List of features to be scaled
    minmax_values(vaex.DataFrame): Dataframe containing reference minimum and maximum
                                     values for the scaling of the data (necessary in order to have
                                     a fixed reference origin)

    Returns:
    ds(vaex.DataFrame): The original dataframe with appended columns containing the scaled features
    '''
    scaler = vaex.ml.MinMaxScaler(feature_range=[-1, 1], features=features, prefix='scaled_')
    scaler.fit(minmax_values)

    return scaler.transform(ds)


# %%
def inspect_star_cluster_velocities_noplot(df, cluster, subcluster):

    df.select('(labels==' + str(cluster) + ')', name='this_cluster_all')
    df.select('(labels==' + str(cluster) + ')&(v_space_labels==' + str(subcluster) + ')', name='this_cluster')
    df.select('(feh_lamost_lrs>-99.0)', name='lamost_good')

    # feh_param computation
    plt.figure()
    values = plt.hist(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'),
                      histtype='step', density=True, bins=20, range=(-2.3, 0.1), cumulative=True)
    plt.close()
    centers = 0.5 * (values[1][1:] + values[1][:-1])
    tck = splrep(centers, values[0], s=0)
    xnew = np.linspace(-2.2, 0.0, 100)
    ynew = splev(xnew, tck, der=0)
    ynew_der = splev(xnew, tck, der=1)

    Ntotal, Nlamost = len(df.evaluate('feh_lamost_lrs', selection='this_cluster')), len(
        df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'))
    sigma_feh = round(np.nanstd(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)')), 2)
    #param_feh = round(np.nanpercentile(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), 66) - np.nanpercentile(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), 34), 2)
    param_feh = round(np.nanmax(ynew_der) / np.nanmedian(ynew_der), 3)

    # Computing sigmas in vx, vy, vz
    sig_vx, sig_vy, sig_vz = np.nanstd(
        df.evaluate(
            'vx', selection='this_cluster')), np.nanstd(
        df.evaluate(
            'vy', selection='this_cluster')), np.nanstd(
        df.evaluate(
            'vz', selection='this_cluster'))

    # Computing eigenvalues
    features = ['vx', 'vy', 'vz']
    df_cluster = df[(df.labels == cluster) & (df.v_space_labels == subcluster)]
    # print(len(df_cluster))
    if len(df_cluster) < 2:
        eig00, eig11, eig22 = np.nan, np.nan, np.nan
    else:
        cov = np.cov(df_cluster[features].values.T)
        eig_vals, eig_vects = linalg.eig(cov)
        D = linalg.inv(eig_vects).dot(cov).dot(eig_vects)
        eig00, eig11, eig22 = float(D[0, 0]), float(D[1, 1]), float(D[2, 2])

    return str(cluster) + '-' + \
        str(subcluster), Ntotal, Nlamost, sigma_feh, param_feh, sig_vx, sig_vy, sig_vz, eig00, eig11, eig22


# %%
def inspect_star_cluster_velocities(df, cluster, subcluster, prefix):

    limits1 = [[-3000.0 / 10**3, 3000.0 / 10**3], [-160000. / 10**4, -50000. / 10**4]]
    limits2 = [[-3000.0 / 10**3, 3000.0 / 10**3], [0. / 10**3, 4000. / 10**3]]

    df.select('(labels==' + str(cluster) + ')', name='this_cluster_all')
    df.select('(labels==' + str(cluster) + ')&(v_space_labels==' + str(subcluster) + ')', name='this_cluster')
    df.select('(feh_lamost_lrs>-99.0)', name='lamost_good')

    LzLperp = df.count(binby=[df.Lz / 10**3, df.Lperp / 10**3], limits=[[limits2[0]
                       [0], limits2[0][1]], [limits2[1][0], limits2[1][1]]], shape=(200, 200))
    LzE = df.count(binby=[df.Lz / 10**3, df.En / 10**4], limits=[[limits1[0][0],
                   limits1[0][1]], [limits1[1][0], limits1[1][1]]], shape=(200, 200))

    f1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(
        8, figsize=(30, 12), sharex=True, facecolor='w')
    ax1 = plt.subplot(241)
    ax2 = plt.subplot(242)
    ax3 = plt.subplot(243)
    ax4 = plt.subplot(244)
    ax5 = plt.subplot(245)
    ax6 = plt.subplot(246)
    ax7 = plt.subplot(247)
    ax8 = plt.subplot(248)

    ax1.imshow(
        np.log10(
            LzLperp.T),
        origin='lower',
        extent=[
            limits2[0][0],
            limits2[0][1],
            limits2[1][0],
            limits2[1][1]],
        cmap='Greys',
        aspect='auto')

    ax1.plot(
        df.evaluate(
            'Lz',
            selection='this_cluster_all') /
        10**3,
        df.evaluate(
            'Lperp',
            selection='this_cluster_all') /
        10**3,
        'bo',
        ms=3,
        lw=0.0,
        alpha=0.2)
    ax1.plot(
        df.evaluate(
            'Lz',
            selection='this_cluster') /
        10**3,
        df.evaluate(
            'Lperp',
            selection='this_cluster') /
        10**3,
        'ro',
        ms=3,
        lw=0.0,
        alpha=0.2)

    ax1.set_ylabel(r'$\rm \sqrt{L_x^2+L_y^2}$', fontsize=25)
    ax1.set_xlabel(r'$\rm L_z$', fontsize=25)

    ax1.set_xlim(limits2[0])
    ax1.set_ylim(limits2[1])

    ax1.tick_params(labelsize=20)

    ax2.imshow(
        np.log(
            LzE.T),
        origin='lower',
        extent=[
            limits1[0][0],
            limits1[0][1],
            limits1[1][0],
            limits1[1][1]],
        cmap='Greys',
        aspect='auto')

    ax2.plot(
        df.evaluate(
            'Lz',
            selection='this_cluster_all') /
        10**3,
        df.evaluate(
            'En',
            selection='this_cluster_all') /
        10**4,
        'bo',
        ms=3,
        lw=0.0,
        alpha=0.2)
    ax2.plot(
        df.evaluate(
            'Lz',
            selection='this_cluster') /
        10**3,
        df.evaluate(
            'En',
            selection='this_cluster') /
        10**4,
        'ro',
        ms=3,
        lw=0.0,
        alpha=0.2)

    ax2.set_xlabel(r'$\rm L_z$', fontsize=25)
    ax2.set_ylabel(r'$\rm E$', fontsize=25)

    ax2.set_xlim(limits1[0])
    ax2.set_ylim(limits1[1])

    ax2.tick_params(labelsize=20)

    limits = [[-450.0, 450.0], [-450.0, 450.0]]

    vxvy = df.count(binby=[df.vx, df.vy + 232.0], limits=[[limits[0][0], limits[0][1]],
                    [limits[1][0], limits[1][1]]], shape=(200, 200))
    vzvy = df.count(binby=[df.vz, df.vy + 232.0], limits=[[limits[0][0], limits[0][1]],
                    [limits[1][0], limits[1][1]]], shape=(200, 200))
    vxvz = df.count(binby=[df.vx, df.vz], limits=[[limits[0][0], limits[0][1]],
                    [limits[1][0], limits[1][1]]], shape=(200, 200))

    ax3.imshow(
        np.log10(
            vzvy.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')

    ax3.plot(
        df.evaluate(
            'vz',
            selection='this_cluster_all'),
        df.evaluate(
            'vy',
            selection='this_cluster_all') +
        232.0,
        'bo',
        ms=5,
        lw=0.0,
        alpha=0.2)
    ax3.plot(
        df.evaluate(
            'vz',
            selection='this_cluster'),
        df.evaluate(
            'vy',
            selection='this_cluster') +
        232.0,
        'ro',
        ms=5,
        lw=0.0,
        alpha=0.2)

    ax4.imshow(
        np.log10(
            vxvy.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')

    ax4.plot(
        df.evaluate(
            'vx',
            selection='this_cluster_all'),
        df.evaluate(
            'vy',
            selection='this_cluster_all') +
        232.0,
        'bo',
        ms=5,
        lw=0.0,
        alpha=0.2)
    ax4.plot(
        df.evaluate(
            'vx',
            selection='this_cluster'),
        df.evaluate(
            'vy',
            selection='this_cluster') +
        232.0,
        'ro',
        ms=5,
        lw=0.0,
        alpha=0.2)

    ax5.imshow(
        np.log10(
            vxvz.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')

    ax5.plot(df.evaluate('vx', selection='this_cluster_all'), df.evaluate(
        'vz', selection='this_cluster_all'), 'bo', ms=5, lw=0.0, alpha=0.2)
    ax5.plot(df.evaluate('vx', selection='this_cluster'), df.evaluate(
        'vz', selection='this_cluster'), 'ro', ms=5, lw=0.0, alpha=0.2)

    Chist, Cbin_edges = np.histogram(df.evaluate(
        'feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), range=(-2.3, 0.1), bins=20)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax6.step(Cbin_edges, Chist, color='r', lw=1, where='mid')
    Nmax = np.float(np.amax(Chist))

    Chist, Cbin_edges = np.histogram(df.evaluate('feh_lamost_lrs'), range=(-2.3, 0.1), bins=20)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax6.step(Cbin_edges, Nmax * Chist / np.float(np.amax(Chist)), color='k', lw=1, where='mid')

    # feh_param computation
    values = ax7.hist(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'),
                      histtype='step', density=True, bins=20, range=(-2.3, 0.1), cumulative=True)
    centers = 0.5 * (values[1][1:] + values[1][:-1])
    ax7.plot(centers, values[0])
    tck = splrep(centers, values[0], s=0)
    xnew = np.linspace(-2.2, 0.0, 100)
    ynew = splev(xnew, tck, der=0)
    ynew_der = splev(xnew, tck, der=1)
    ax7.plot(xnew, ynew / np.nanmax(ynew))
    ax7.plot(xnew, ynew_der / np.nanmax(ynew_der))

    ax3.set_xlabel('vz', fontsize=25)
    ax3.set_ylabel('vy', fontsize=25)

    ax3.set_xlim(limits[0])
    ax3.set_ylim(limits[1])

    ax4.set_xlabel('vx', fontsize=25)
    ax4.set_ylabel('vy', fontsize=25)

    ax4.set_xlim(limits[0])
    ax4.set_ylim(limits[1])

    ax5.set_xlabel('vx', fontsize=25)
    ax5.set_ylabel('vz', fontsize=25)

    ax5.set_xlim(limits[0])
    ax5.set_ylim(limits[1])

    ax6.set_xlabel('[Fe/H]', fontsize=25)
    ax6.set_ylabel('N', fontsize=25)

    ax7.set_xlabel('[Fe/H]', fontsize=25)
    ax7.set_ylabel('*', fontsize=25)

    # l-b plot
    limits = [[np.nanmin(df.evaluate('l')), np.nanmax(df.evaluate('l'))], [
        np.nanmin(df.evaluate('b')), np.nanmax(df.evaluate('b'))]]
    lb_dens = df.count(binby=[df.l, df.b], limits=[[limits[0][0], limits[0][1]],
                       [limits[1][0], limits[1][1]]], shape=(200, 200))
    ax8.imshow(
        np.log10(
            lb_dens.T),
        origin='lower',
        extent=[
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1]],
        cmap='Greys',
        aspect='auto')
    ax8.plot(df.evaluate('l', selection='this_cluster_all'), df.evaluate(
        'b', selection='this_cluster_all'), 'bo', ms=25, lw=0.0, alpha=0.2)
    ax8.plot(df.evaluate('l', selection='this_cluster'), df.evaluate(
        'b', selection='this_cluster'), 'ro', ms=25, lw=0.0, alpha=0.2)

    ax8.set_xlabel('l', fontsize=25)
    ax8.set_ylabel('b', fontsize=25)

    ax3.tick_params(labelsize=20)
    ax4.tick_params(labelsize=20)
    ax5.tick_params(labelsize=20)
    ax6.tick_params(labelsize=20)
    ax7.tick_params(labelsize=20)
    ax8.tick_params(labelsize=20)

    Ntotal, Nlamost = len(df.evaluate('feh_lamost_lrs', selection='this_cluster')), len(
        df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'))
    sigma_feh = round(np.nanstd(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)')), 2)
    #param_feh = round(np.nanpercentile(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), 66) - np.nanpercentile(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(lamost_good)'), 34), 2)
    param_feh = round(np.nanmax(ynew_der) / np.nanmedian(ynew_der), 3)

    # Computing sigmas in vx, vy, vz
    sig_vx, sig_vy, sig_vz = round(
        np.nanstd(
            df.evaluate(
                'vx', selection='this_cluster')), 2), round(
        np.nanstd(
            df.evaluate(
                'vy', selection='this_cluster')), 2), round(
        np.nanstd(
            df.evaluate(
                'vz', selection='this_cluster')), 2)

    # Computing eigenvalues
    features = ['vx', 'vy', 'vz']
    df_cluster = df[(df.labels == cluster) & (df.v_space_labels == subcluster)]
    # print(len(df_cluster))
    if len(df_cluster) < 2:
        print('oio')
        eig00, eig11, eig22 = np.nan, np.nan, np.nan
    else:
        cov = np.cov(df_cluster[features].values.T)
        eig_vals, eig_vects = linalg.eig(cov)
        D = linalg.inv(eig_vects).dot(cov).dot(eig_vects)
        eig00, eig11, eig22 = round(D[0, 0], 2), round(D[1, 1], 2), round(D[2, 2], 2)

    f1.suptitle(
        'Cluster #' +
        str(cluster) +
        '-' +
        str(subcluster) +
        '; ' +
        str(Ntotal) +
        ' stars; ' +
        str(Nlamost) +
        ' stars lamost; std_feh = ' +
        str(sigma_feh) +
        '; feh_param = ' +
        str(param_feh) +
        '; sig_vx = ' +
        str(sig_vx) +
        '; sig_vy = ' +
        str(sig_vy) +
        '; sig_vz = ' +
        str(sig_vz) +
        '; eig00 = ' +
        str(eig00) +
        '; eig11 = ' +
        str(eig11) +
        '; eig22 = ' +
        str(eig22),
        fontsize=20)

    plt.tight_layout(w_pad=0.0, rect=[0.00, 0.00, 1, 0.95])

    plt.savefig('/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/' +
                str(prefix) + '_' + str(cluster) + '-' + str(subcluster) + '.png')

    plt.show()

    return str(cluster) + '-' + str(subcluster), Ntotal, Nlamost, sigma_feh, param_feh, sig_vx, sig_vy, sig_vz, float(
        eig00), float(eig11), float(eig22)


# %% [markdown]
# ## A) 4D analysis (no HDBSCAN):

# %% [markdown]
# ### A.1) Getting scaled quantities to 4D catalogue (old vaex):

# %%
df_4D_aux = vaex.open('/net/gaia2/data/users/lovdal/df_labels.hdf5')
df_4D_aux

# %%
minmax_values = vaex.from_arrays(En=[-170000, 0], Lperp=[0, 4300], Lz=[-4500, 4600])
df_aux = scaleData(df_4D_aux, ['En', 'Lperp', 'Lz'], minmax_values)
df_aux

# %%
os.system('rm /data/users/ruizlara/halo_clustering/df_labels_scaled.hdf5')
df_aux.export_hdf5('/net/gaia2/data/users/ruizlara/halo_clustering/df_labels_scaled.hdf5', virtual=True)

# %% [markdown]
# ### 4.2) Start doing things (back to new vaex):

# %%
df_4D = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/df_labels_scaled.hdf5')
df_4D

# %%
# Checking eigenvalues and eigenvectors and diagonalize covariance matrix.

cluster = df_4D[df_4D.labels == 69.0]
features = ['scaled_En*10e2', 'scaled_Lperp*10e2', 'scaled_Lz*10e2', 'circ*10e2']

covariance = np.cov(cluster[features].values.T)

print(covariance)

eig_vals, eig_vects = linalg.eig(covariance)

D = linalg.inv(eig_vects).dot(covariance).dot(eig_vects)

print(D)

print(D[0, 0])
print(D[1, 1])
print(D[2, 2])
print(D[3, 3])

print(eig_vals)


# %%
df_des = get_cluster_description_table(df_4D, minsig=3)
df_des

# %%
df_4D.select('maxsig>3.0', name='sign')
clusters = list(set(df_4D.evaluate('labels', selection='sign')))
cluster_array, Ntotal_array, Nlamost_array, sigma_feh_array, param_feh_array, cov_En_array, cov_Lperp_array, cov_Lz_array, cov_circ_array, eig00_array, eig11_array, eig22_array, eig33_array = [
], [], [], [], [], [], [], [], [], [], [], [], []
for cluster in clusters:
    cluster, Ntotal, Nlamost, sigma_feh, param_feh, cov_En, cov_Lperp, cov_Lz, cov_circ, eig00, eig11, eig22, eig33 = inspect_star_cluster_noplot(
        df_4D, cluster)
    cluster_array.append(cluster)
    Ntotal_array.append(Ntotal)
    Nlamost_array.append(Nlamost)
    sigma_feh_array.append(sigma_feh)
    param_feh_array.append(param_feh)
    cov_En_array.append(cov_En)
    cov_Lperp_array.append(cov_Lperp)
    cov_Lz_array.append(cov_Lz)
    cov_circ_array.append(cov_circ)
    eig00_array.append(eig00)
    eig11_array.append(eig11)
    eig22_array.append(eig22)
    eig33_array.append(eig33)

cluster_table = vaex.from_arrays(
    cluster=cluster_array,
    Ntotal=Ntotal_array,
    Nlamost=Nlamost_array,
    sigma_feh=sigma_feh_array,
    param_feh=param_feh_array,
    cov_En=cov_En_array,
    cov_Lperp=cov_Lperp_array,
    cov_Lz=cov_Lz_array,
    cov_circ=cov_circ_array,
    eig00=eig00_array,
    eig11=eig11_array,
    eig22=eig22_array,
    eig33=eig33_array)
os.system('rm /net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_IoM_4D.hdf5')
cluster_table.export_hdf5(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_IoM_4D.hdf5')

# %%
cluster_table = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_IoM_4D.hdf5')
cluster_table

# %%
f1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = plt.subplots(
    12, figsize=(32, 12), sharex=True, facecolor='w')
ax1 = plt.subplot(261)
ax2 = plt.subplot(262)
ax3 = plt.subplot(263)
ax4 = plt.subplot(264)
ax5 = plt.subplot(265)
ax6 = plt.subplot(266)
ax7 = plt.subplot(267)
ax8 = plt.subplot(268)
ax9 = plt.subplot(269)
ax10 = plt.subplot(2, 6, 10)
ax11 = plt.subplot(2, 6, 11)
ax12 = plt.subplot(2, 6, 12)

axs = [ax1, ax2, ax3, ax4, ax5, ax6]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22', 'eig33']
for ax, param in zip(axs, params):
    ax.plot(cluster_table.evaluate('cluster'), cluster_table.evaluate(param), '*')
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

axs = [ax7, ax8, ax9, ax10, ax11, ax12]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22', 'eig33']
low_percs = [0.0, 80.0, 0.0, 0.0, 0.0, 0.0]
up_percs = [15.0, 98.0, 15.0, 15.0, 15.0, 15.0]
sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr, eig33_arr = [], [], [], [], [], []
arrays = [sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr, eig33_arr]

for ax, param, low_perc, up_perc, array in zip(axs, params, low_percs, up_percs, arrays):
    param_array = cluster_table.evaluate(param)
    ax.plot(cluster_table.evaluate('cluster'), param_array, '*')
    for i in np.arange(len(cluster_table.evaluate('cluster'))):
        if (param_array[i] > np.nanpercentile(param_array, low_perc)) & (
                param_array[i] < np.nanpercentile(param_array, up_perc)):
            ax.text(
                cluster_table.evaluate('cluster')[i],
                param_array[i],
                s=str(
                    cluster_table.evaluate('cluster')[i]))
            array.append(cluster_table.evaluate('cluster')[i])
    ax.set_ylim(np.nanpercentile(param_array, low_perc), np.nanpercentile(param_array, up_perc))
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

print(arrays)

plt.show()

# %%
newfile = open('/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/4D_IoM_reasons.txt', 'w')
prefix = '4D_IoM'
tight_clusters = []
clusters = set(df_4D.evaluate('labels', selection='sign'))
for cluster in clusters:
    count = 0
    for i in np.arange(len(arrays)):
        if cluster in arrays[i]:
            count += 1
    if count > 1:
        tight_clusters.append(cluster)
print(str(len(tight_clusters)) + ' potential tight clusters')
print(tight_clusters)
reason = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22', 'eig33']
for pot_cluster in tight_clusters:
    print('Cluster #' + str(pot_cluster))
    newfile.write('Cluster #' + str(pot_cluster) + ' \n')
    for i in np.arange(len(arrays)):
        if pot_cluster in arrays[i]:
            print(reason[i])
            newfile.write(str(reason[i]) + ' \n')
    inspect_star_cluster(df_4D, pot_cluster, prefix)
newfile.close()

# %% [markdown]
# ## B) 4D analysis (HDBSCAN):
#
# Is it vx,vy,vz enough for our purposes or do we need to apply a principal component analysis?
#
# ### NO, we need PCA here as well, eigenvalues and eigenvectors

# %%
df_4D = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/df_labels_scaled.hdf5')
df_4D

# %%
df_4D.select('maxsig>3.0', name='sign')
clusters = list(set(df_4D.evaluate('labels', selection='sign')))
cluster_array, Ntotal_array, Nlamost_array, sigma_feh_array, param_feh_array, sig_vx_array, sig_vy_array, sig_vz_array, eig00_array, eig11_array, eig22_array = [
], [], [], [], [], [], [], [], [], [], []
for cluster in clusters:
    # print(cluster)
    df_4D.select('(labels==' + str(cluster) + ')', name='this_cluster_all')
    subclusters = list(set(df_4D.evaluate('v_space_labels', selection='this_cluster_all')))
    # print(subclusters)
    for subcluster in subclusters:
        cluster_name, Ntotal, Nlamost, sigma_feh, param_feh, sig_vx, sig_vy, sig_vz, eig00, eig11, eig22 = inspect_star_cluster_velocities_noplot(
            df_4D, cluster, subcluster)
        cluster_array.append(cluster_name)
        Ntotal_array.append(Ntotal)
        Nlamost_array.append(Nlamost)
        sigma_feh_array.append(sigma_feh)
        param_feh_array.append(param_feh)
        sig_vx_array.append(sig_vx)
        sig_vy_array.append(sig_vy)
        sig_vz_array.append(sig_vz)
        eig00_array.append(eig00)
        eig11_array.append(eig11)
        eig22_array.append(eig22)

cluster_table = vaex.from_arrays(
    cluster=cluster_array,
    Ntotal=Ntotal_array,
    Nlamost=Nlamost_array,
    sigma_feh=sigma_feh_array,
    param_feh=param_feh_array,
    sig_vx=sig_vx_array,
    sig_vy=sig_vy_array,
    sig_vz=sig_vz_array,
    eig00=eig00_array,
    eig11=eig11_array,
    eig22=eig22_array)
cluster_table['cluster'] = cluster_table['cluster'].astype('string')
os.system('rm /net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_vel_4D.hdf5')
cluster_table.export_hdf5(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_vel_4D.hdf5')

# %%
cluster_table = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_vel_4D.hdf5')
cluster_table

# %%
f1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(
    10, figsize=(32, 12), sharex=True, facecolor='w')
ax1 = plt.subplot(251)
ax2 = plt.subplot(252)
ax3 = plt.subplot(253)
ax4 = plt.subplot(254)
ax5 = plt.subplot(255)
ax6 = plt.subplot(256)
ax7 = plt.subplot(257)
ax8 = plt.subplot(258)
ax9 = plt.subplot(259)
ax10 = plt.subplot(2, 5, 10)

axs = [ax1, ax2, ax3, ax4, ax5]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22']
for ax, param in zip(axs, params):
    ax.plot(np.arange(len(cluster_table.evaluate('cluster'))), cluster_table.evaluate(param), '*')
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

axs = [ax6, ax7, ax8, ax9, ax10]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22']
low_percs = [0.0, 80.0, 0.0, 0.0, 0.0]
up_percs = [15.0, 98.0, 15.0, 15.0, 15.0]
sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr = [], [], [], [], []
arrays = [sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr]

for ax, param, low_perc, up_perc, array in zip(axs, params, low_percs, up_percs, arrays):
    param_array = cluster_table.evaluate(param)
    ax.plot(np.arange(len(cluster_table.evaluate('cluster'))), param_array, '*')
    for i in np.arange(len(cluster_table.evaluate('cluster'))):
        if (param_array[i] > np.nanpercentile(param_array, low_perc)) & (
                param_array[i] < np.nanpercentile(param_array, up_perc)):
            ax.text(np.arange(len(cluster_table.evaluate('cluster')))[
                    i], param_array[i], s=str(cluster_table.evaluate('cluster')[i]))
            array.append(str(cluster_table.evaluate('cluster')[i]))
    ax.set_ylim(np.nanpercentile(param_array, low_perc), np.nanpercentile(param_array, up_perc))
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

plt.show()

# %%
newfile = open('/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/4D_vel_reasons.txt', 'w')
prefix = '4D_vel'
tight_clusters = []
clusters = set(df_4D.evaluate('labels', selection='sign'))
all_clusters = []
for cluster in clusters:
    df_4D.select('(labels==' + str(cluster) + ')', name='this_cluster_all')
    subclusters = list(set(df_4D.evaluate('v_space_labels', selection='this_cluster_all')))
    for subcluster in subclusters:
        all_clusters.append(str(cluster) + '-' + str(subcluster))

print('All clusters -->' + str(len(all_clusters)))

for cluster in all_clusters:
    count = 0
    for i in np.arange(len(arrays)):
        if cluster in arrays[i]:
            count += 1
    if count > 1:
        tight_clusters.append(cluster)
print(str(len(tight_clusters)) + ' potential tight clusters')
print(tight_clusters)
reason = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22']
for pot_cluster in tight_clusters:
    print('Cluster #' + str(pot_cluster))
    newfile.write('Cluster #' + str(pot_cluster) + ' \n')
    for i in np.arange(len(arrays)):
        if pot_cluster in arrays[i]:
            print(reason[i])
            newfile.write(str(reason[i]) + ' \n')
    inspect_star_cluster_velocities(
        df_4D,
        float(
            pot_cluster.split('-')[0]),
        float(
            pot_cluster.split('-')[1]),
        prefix)
newfile.close()

# %% [markdown]
# ## C) 3D analysis (no HDBSCAN):

# %%
df_3D = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_3sigma.hdf5')
df_3D

# %%
df_des = get_cluster_description_table(df_3D, minsig=3)
df_des

# %%
df_3D.select('maxsig>3.0', name='sign')
clusters = list(set(df_3D.evaluate('labels', selection='sign')))
cluster_array, Ntotal_array, Nlamost_array, sigma_feh_array, param_feh_array, cov_En_array, cov_Lperp_array, cov_Lz_array, cov_circ_array, eig00_array, eig11_array, eig22_array, eig33_array = [
], [], [], [], [], [], [], [], [], [], [], [], []
for cluster in clusters:
    cluster, Ntotal, Nlamost, sigma_feh, param_feh, cov_En, cov_Lperp, cov_Lz, cov_circ, eig00, eig11, eig22, eig33 = inspect_star_cluster_noplot(
        df_3D, cluster)
    cluster_array.append(cluster)
    Ntotal_array.append(Ntotal)
    Nlamost_array.append(Nlamost)
    sigma_feh_array.append(sigma_feh)
    param_feh_array.append(param_feh)
    cov_En_array.append(cov_En)
    cov_Lperp_array.append(cov_Lperp)
    cov_Lz_array.append(cov_Lz)
    cov_circ_array.append(cov_circ)
    eig00_array.append(eig00)
    eig11_array.append(eig11)
    eig22_array.append(eig22)
    eig33_array.append(eig33)

cluster_table = vaex.from_arrays(
    cluster=cluster_array,
    Ntotal=Ntotal_array,
    Nlamost=Nlamost_array,
    sigma_feh=sigma_feh_array,
    param_feh=param_feh_array,
    cov_En=cov_En_array,
    cov_Lperp=cov_Lperp_array,
    cov_Lz=cov_Lz_array,
    cov_circ=cov_circ_array,
    eig00=eig00_array,
    eig11=eig11_array,
    eig22=eig22_array,
    eig33=eig33_array)
os.system('rm /net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_IoM_3D.hdf5')
cluster_table.export_hdf5(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_IoM_3D.hdf5')

# %%
cluster_table = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_IoM_3D.hdf5')
cluster_table

# %%
f1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = plt.subplots(
    12, figsize=(32, 12), sharex=True, facecolor='w')
ax1 = plt.subplot(261)
ax2 = plt.subplot(262)
ax3 = plt.subplot(263)
ax4 = plt.subplot(264)
ax5 = plt.subplot(265)
ax6 = plt.subplot(266)
ax7 = plt.subplot(267)
ax8 = plt.subplot(268)
ax9 = plt.subplot(269)
ax10 = plt.subplot(2, 6, 10)
ax11 = plt.subplot(2, 6, 11)
ax12 = plt.subplot(2, 6, 12)

axs = [ax1, ax2, ax3, ax4, ax5, ax6]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22', 'eig33']
for ax, param in zip(axs, params):
    ax.plot(cluster_table.evaluate('cluster'), cluster_table.evaluate(param), '*')
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

axs = [ax7, ax8, ax9, ax10, ax11, ax12]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22', 'eig33']
low_percs = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
up_percs = [20.0, 100.0, 20.0, 20.0, 20.0, 20.0]
sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr, eig33_arr = [], [], [], [], [], []
arrays = [sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr, eig33_arr]

for ax, param, low_perc, up_perc, array in zip(axs, params, low_percs, up_percs, arrays):
    param_array = cluster_table.evaluate(param)
    ax.plot(cluster_table.evaluate('cluster'), param_array, '*')
    for i in np.arange(len(cluster_table.evaluate('cluster'))):
        if (param_array[i] > np.nanpercentile(param_array, low_perc)) & (
                param_array[i] < np.nanpercentile(param_array, up_perc)):
            ax.text(
                cluster_table.evaluate('cluster')[i],
                param_array[i],
                s=str(
                    cluster_table.evaluate('cluster')[i]))
            array.append(cluster_table.evaluate('cluster')[i])
    ax.set_ylim(np.nanpercentile(param_array, low_perc), np.nanpercentile(param_array, up_perc))
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

print(arrays)

plt.show()

# %%
newfile = open('/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/3D_IoM_reasons.txt', 'w')
prefix = '3D_IoM'
tight_clusters = []
clusters = set(df_3D.evaluate('labels'))
for cluster in clusters:
    count = 0
    for i in np.arange(len(arrays)):
        if cluster in arrays[i]:
            count += 1
    if count > 1:
        tight_clusters.append(cluster)
print(str(len(tight_clusters)) + ' potential tight clusters')
print(tight_clusters)
reason = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22', 'eig33']
for pot_cluster in tight_clusters:
    print('Cluster #' + str(pot_cluster))
    newfile.write('Cluster #' + str(pot_cluster) + ' \n')
    for i in np.arange(len(arrays)):
        if pot_cluster in arrays[i]:
            print(reason[i])
            newfile.write(str(reason[i]) + ' \n')
    inspect_star_cluster(df_3D, pot_cluster, prefix)
newfile.close()

# %% [markdown]
# ## D) 3D analysis (HDBSCAN):

# %%
df_3D = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_3sigma.hdf5')
df_3D

# %%
df_3D.select('maxsig>3.0', name='sign')
clusters = list(set(df_3D.evaluate('labels', selection='sign')))
cluster_array, Ntotal_array, Nlamost_array, sigma_feh_array, param_feh_array, sig_vx_array, sig_vy_array, sig_vz_array, eig00_array, eig11_array, eig22_array = [
], [], [], [], [], [], [], [], [], [], []
for cluster in clusters:
    # print(cluster)
    df_3D.select('(labels==' + str(cluster) + ')', name='this_cluster_all')
    subclusters = list(set(df_3D.evaluate('v_space_labels', selection='this_cluster_all')))
    # print(subclusters)
    for subcluster in subclusters:
        cluster_name, Ntotal, Nlamost, sigma_feh, param_feh, sig_vx, sig_vy, sig_vz, eig00, eig11, eig22 = inspect_star_cluster_velocities_noplot(
            df_3D, cluster, subcluster)
        cluster_array.append(cluster_name)
        Ntotal_array.append(Ntotal)
        Nlamost_array.append(Nlamost)
        sigma_feh_array.append(sigma_feh)
        param_feh_array.append(param_feh)
        sig_vx_array.append(sig_vx)
        sig_vy_array.append(sig_vy)
        sig_vz_array.append(sig_vz)
        eig00_array.append(eig00)
        eig11_array.append(eig11)
        eig22_array.append(eig22)

cluster_table = vaex.from_arrays(
    cluster=cluster_array,
    Ntotal=Ntotal_array,
    Nlamost=Nlamost_array,
    sigma_feh=sigma_feh_array,
    param_feh=param_feh_array,
    sig_vx=sig_vx_array,
    sig_vy=sig_vy_array,
    sig_vz=sig_vz_array,
    eig00=eig00_array,
    eig11=eig11_array,
    eig22=eig22_array)
cluster_table['cluster'] = cluster_table['cluster'].astype('string')
os.system('rm /net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_vel_3D.hdf5')
cluster_table.export_hdf5(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_vel_3D.hdf5')

# %%
cluster_table = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/cluster_table_vel_3D.hdf5')
cluster_table

# %%
f1, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(
    10, figsize=(32, 12), sharex=True, facecolor='w')
ax1 = plt.subplot(251)
ax2 = plt.subplot(252)
ax3 = plt.subplot(253)
ax4 = plt.subplot(254)
ax5 = plt.subplot(255)
ax6 = plt.subplot(256)
ax7 = plt.subplot(257)
ax8 = plt.subplot(258)
ax9 = plt.subplot(259)
ax10 = plt.subplot(2, 5, 10)

axs = [ax1, ax2, ax3, ax4, ax5]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22']
for ax, param in zip(axs, params):
    ax.plot(np.arange(len(cluster_table.evaluate('cluster'))), cluster_table.evaluate(param), '*')
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

axs = [ax6, ax7, ax8, ax9, ax10]
params = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22']
low_percs = [0.0, 80.0, 0.0, 0.0, 0.0]
up_percs = [15.0, 98.0, 15.0, 15.0, 15.0]
sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr = [], [], [], [], []
arrays = [sigma_feh_arr, param_feh_arr, eig00_arr, eig11_arr, eig22_arr]

for ax, param, low_perc, up_perc, array in zip(axs, params, low_percs, up_percs, arrays):
    param_array = cluster_table.evaluate(param)
    ax.plot(np.arange(len(cluster_table.evaluate('cluster'))), param_array, '*')
    for i in np.arange(len(cluster_table.evaluate('cluster'))):
        if (param_array[i] > np.nanpercentile(param_array, low_perc)) & (
                param_array[i] < np.nanpercentile(param_array, up_perc)):
            ax.text(np.arange(len(cluster_table.evaluate('cluster')))[
                    i], param_array[i], s=str(cluster_table.evaluate('cluster')[i]))
            array.append(str(cluster_table.evaluate('cluster')[i]))
    ax.set_ylim(np.nanpercentile(param_array, low_perc), np.nanpercentile(param_array, up_perc))
    ax.set_xlabel('Cluster id')
    ax.set_ylabel(param)

plt.show()

# %%
newfile = open('/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/3D_vel_reasons.txt', 'w')
prefix = '3D_vel'
tight_clusters = []
clusters = set(df_3D.evaluate('labels', selection='sign'))
all_clusters = []
for cluster in clusters:
    df_3D.select('(labels==' + str(cluster) + ')', name='this_cluster_all')
    subclusters = list(set(df_3D.evaluate('v_space_labels', selection='this_cluster_all')))
    for subcluster in subclusters:
        all_clusters.append(str(cluster) + '-' + str(subcluster))

print('All clusters -->' + str(len(all_clusters)))

for cluster in all_clusters:
    count = 0
    for i in np.arange(len(arrays)):
        if cluster in arrays[i]:
            count += 1
    if count > 1:
        tight_clusters.append(cluster)
print(str(len(tight_clusters)) + ' potential tight clusters')
print(tight_clusters)
reason = ['sigma_feh', 'param_feh', 'eig00', 'eig11', 'eig22']
for pot_cluster in tight_clusters:
    print('Cluster #' + str(pot_cluster))
    newfile.write('Cluster #' + str(pot_cluster) + ' \n')
    for i in np.arange(len(arrays)):
        if pot_cluster in arrays[i]:
            print(reason[i])
            newfile.write(str(reason[i]) + ' \n')
    inspect_star_cluster_velocities(
        df_3D,
        float(
            pot_cluster.split('-')[0]),
        float(
            pot_cluster.split('-')[1]),
        prefix)
newfile.close()

# %% [markdown]
# # Summary so far:
#
# ### Possible tight clusters 4D, no HDBSCAN
#
# 21 potential tight clusters
#
# [1.0, 2.0, 14.0, 18.0, 536.0, 555.0, 69.0, 130.0, 205.0, 287.0, 294.0, 352.0, 370.0, 391.0, 394.0, 397.0, 405.0, 410.0, 463.0, 477.0, 480.0]
#
#
# ### Possible tight clusters 4D, HDBSCAN
#
# 49 potential tight clusters
# ['1.0-1', '516.0-2', '2.0-1', '522.0-2', '522.0-3', '14.0-1', '530.0-1', '536.0-2', '536.0-3', '555.0-1', '557.0-2', '558.0-1', '560.0-1', '560.0-5', '563.0-1', '568.0-1', '69.0-2', '69.0-3', '69.0-4', '327.0-2', '327.0-3', '344.0-0', '344.0-4', '352.0-1', '352.0-2', '369.0-3', '369.0-4', '371.0-1', '383.0-5', '385.0-2', '390.0-2', '391.0-0', '391.0-1', '392.0-1', '394.0-2', '394.0-3', '407.0-0', '407.0-5', '407.0-6', '410.0-1', '415.0-1', '455.0-6', '455.0-7', '458.0-4', '463.0-0', '463.0-1', '471.0-3', '479.0-2', '500.0-2']
#
# ### Possible tight clusters 3D, no HDBSCAN
#
# 20 potential tight clusters
# [3.0, 7.0, 8.0, 9.0, 10.0, 13.0, 16.0, 17.0, 23.0, 24.0, 29.0, 30.0, 37.0, 44.0, 48.0, 52.0, 53.0, 54.0, 57.0, 58.0]
#
# ### Possible tight clusters 3D, HDBSCAN
#
# 50 potential tight clusters
# ['1.0-0', '1.0-1', '2.0-0', '2.0-1', '2.0-2', '2.0-3', '4.0-3', '5.0-5', '5.0-6', '5.0-7', '8.0-2', '8.0-4', '9.0-4', '9.0-5', '11.0-0', '11.0-3', '11.0-5', '14.0-2', '16.0-2', '16.0-3', '17.0-3', '17.0-5', '17.0-6', '17.0-7', '17.0-8', '17.0-9', '18.0-3', '18.0-4', '18.0-5', '21.0-2', '23.0-4', '30.0-2', '39.0-3', '39.0-4', '40.0-1', '40.0-3', '40.0-4', '42.0-4', '45.0-0', '45.0-2', '47.0-4', '53.0-5', '54.0-3', '62.0-1', '64.0-4', '64.0-5', '64.0-8', '64.0-9', '66.0-1', '68.0-3']
#
#
# ### Small clusters left out of paper II (3D):
#
# #1, #2, #4, #28, #40, #65, #66, and #68
#
#
# ## check what cluster we really want to analyse, obtain their source_ids and send to Eduardo.

# %% [markdown]
# # ####

# %% [markdown]
# # Second summary, eigenvalues
#
# ### Possible tight clusters 4D, no HDBSCAN
#
# 18 potential tight clusters
#
# [1.0, 2.0, 14.0, 69.0, 205.0, 352.0, 370.0, 394.0, 410.0, 458.0, 462.0, 463.0, 477.0, 480.0, 536.0, 555.0, 558.0, 562.0]
#
#
# ### Possible tight clusters 4D, HDBSCAN
#
#
# 46 potential tight clusters
#
# ['2.0-1', '14.0-1', '69.0-2', '69.0-3', '69.0-4', '327.0-2', '327.0-3', '344.0-4', '352.0-1', '352.0-2', '369.0-4', '371.0-1', '390.0-2', '391.0-0', '391.0-1', '392.0-1', '394.0-2', '394.0-3', '407.0-0', '407.0-6', '410.0-1', '415.0-1', '455.0-6', '458.0-1', '458.0-4', '463.0-0', '463.0-1', '471.0-1', '471.0-3', '475.0-2', '475.0-4', '479.0-2', '500.0-2', '516.0-2', '522.0-2', '522.0-3',  '536.0-2', '555.0-0', '555.0-1', '557.0-2', '557.0-3', '558.0-1', '560.0-1', '560.0-5', '563.0-1', '568.0-1']
#
#
#
# ### Possible tight clusters 3D, no HDBSCAN
#
# 15 potential tight clusters
#
# [2.0, 4.0, 9.0, 16.0, 18.0, 19.0, 20.0, 28.0, 30.0, 33.0, 40.0, 45.0, 51.0, 53.0, 57.0]
#
#
# ### Possible tight clusters 3D, HDBSCAN
#
#
# 52 potential tight clusters
#
# ['1.0-0', '1.0-1', '2.0-0', '2.0-2', '2.0-3', '5.0-5', '8.0-2', '8.0-4', '9.0-4', '11.0-0', '11.0-1', '11.0-5', '14.0-2', '14.0-4', '16.0-2', '16.0-3', '17.0-2', '17.0-3', '17.0-4', '17.0-5', '17.0-7', '17.0-8', '17.0-9', '18.0-3', '18.0-4', '18.0-5', '21.0-2', '23.0-0', '23.0-4', '30.0-2', '39.0-3', '40.0-1', '40.0-2', '40.0-3', '40.0-4', '42.0-4', '45.0-0', '45.0-2', '47.0-4', '50.0-3', '51.0-1', '51.0-4', '53.0-0', '53.0-4', '53.0-5', '54.0-3', '57.0-0', '62.0-1', '62.0-3', '64.0-4', '64.0-5', '68.0-3']
#

# %% [markdown]
# # Preparing things for Eduardo
#
# ### Working with dictionaries and plots to show the intersection

# %%
# 4D, IoM:

df_4D = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/df_labels_scaled.hdf5')
tight_clusters_4D_IoM = [1.0, 2.0, 14.0, 69.0, 205.0, 352.0, 370.0, 394.0,
                         410.0, 458.0, 462.0, 463.0, 477.0, 480.0, 536.0, 555.0, 558.0, 562.0]
dict_4D_IoM = {}
for cluster in tight_clusters_4D_IoM:
    df_4D.select('(labels==' + str(cluster) + ')', name='this_cluster')
    dict_4D_IoM[cluster] = df_4D.evaluate('source_id', selection='this_cluster')

newfile = open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/4D_IoM_source_ids.txt',
    'w')
newfile.write(str(dict_4D_IoM))
newfile.close()

# %%
# 3D, IoM:

df_3D = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_3sigma.hdf5')
tight_clusters_3D_IoM = [2.0, 4.0, 9.0, 16.0, 18.0, 19.0,
                         20.0, 28.0, 30.0, 33.0, 40.0, 45.0, 51.0, 53.0, 57.0]
dict_3D_IoM = {}
for cluster in tight_clusters_3D_IoM:
    df_3D.select('(labels==' + str(cluster) + ')', name='this_cluster')
    dict_3D_IoM[cluster] = df_3D.evaluate('source_id', selection='this_cluster')

newfile = open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/3D_IoM_source_ids.txt',
    'w')
newfile.write(str(dict_3D_IoM))
newfile.close()

# %%
# 4D, vel:

df_4D = vaex.open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/df_labels_scaled.hdf5')
tight_clusters_4D_vel = [
    '2.0-1',
    '14.0-1',
    '69.0-2',
    '69.0-3',
    '69.0-4',
    '327.0-2',
    '327.0-3',
    '344.0-4',
    '352.0-1',
    '352.0-2',
    '369.0-4',
    '371.0-1',
    '390.0-2',
    '391.0-0',
    '391.0-1',
    '392.0-1',
    '394.0-2',
    '394.0-3',
    '407.0-0',
    '407.0-6',
    '410.0-1',
    '415.0-1',
    '455.0-6',
    '458.0-1',
    '458.0-4',
    '463.0-0',
    '463.0-1',
    '471.0-1',
    '471.0-3',
    '475.0-2',
    '475.0-4',
    '479.0-2',
    '500.0-2',
    '516.0-2',
    '522.0-2',
    '522.0-3',
    '536.0-2',
    '555.0-0',
    '555.0-1',
    '557.0-2',
    '557.0-3',
    '558.0-1',
    '560.0-1',
    '560.0-5',
    '563.0-1',
    '568.0-1']
dict_4D_vel = {}
for subcluster_id in tight_clusters_4D_vel:
    df_4D.select('(labels==' +
                 str(subcluster_id.split('-')[0]) +
                 ')&(v_space_labels==' +
                 str(subcluster_id.split('-')[1]) +
                 ')', name='this_cluster')
    dict_4D_vel[subcluster_id] = df_4D.evaluate('source_id', selection='this_cluster')

newfile = open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/4D_vel_source_ids.txt',
    'w')
newfile.write(str(dict_4D_vel))
newfile.close()

# %%
# 3D, vel:

df_3D = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_3sigma.hdf5')
tight_clusters_3D_vel = [
    '1.0-0',
    '1.0-1',
    '2.0-0',
    '2.0-2',
    '2.0-3',
    '5.0-5',
    '8.0-2',
    '8.0-4',
    '9.0-4',
    '11.0-0',
    '11.0-1',
    '11.0-5',
    '14.0-2',
    '14.0-4',
    '16.0-2',
    '16.0-3',
    '17.0-2',
    '17.0-3',
    '17.0-4',
    '17.0-5',
    '17.0-7',
    '17.0-8',
    '17.0-9',
    '18.0-3',
    '18.0-4',
    '18.0-5',
    '21.0-2',
    '23.0-0',
    '23.0-4',
    '30.0-2',
    '39.0-3',
    '40.0-1',
    '40.0-2',
    '40.0-3',
    '40.0-4',
    '42.0-4',
    '45.0-0',
    '45.0-2',
    '47.0-4',
    '50.0-3',
    '51.0-1',
    '51.0-4',
    '53.0-0',
    '53.0-4',
    '53.0-5',
    '54.0-3',
    '57.0-0',
    '62.0-1',
    '62.0-3',
    '64.0-4',
    '64.0-5',
    '68.0-3']
dict_3D_vel = {}
for subcluster_id in tight_clusters_3D_vel:
    df_3D.select('(labels==' +
                 str(subcluster_id.split('-')[0]) +
                 ')&(v_space_labels==' +
                 str(subcluster_id.split('-')[1]) +
                 ')', name='this_cluster')
    dict_3D_vel[subcluster_id] = df_3D.evaluate('source_id', selection='this_cluster')

newfile = open(
    '/net/gaia2/data/users/ruizlara/halo_clustering/disrupted_clusters_project/3D_vel_source_ids.txt',
    'w')
newfile.write(str(dict_3D_vel))
newfile.close()

# %%
for key_3D in dict_3D_vel.keys():
    for key_4D in dict_4D_vel.keys():
        this_3D, this_4D = dict_3D_vel[key_3D], dict_4D_vel[key_4D]
        intersect = np.intersect1d(this_3D, this_4D)
        if len(intersect) > 0:
            print(str(round(100.0 *
                            float(len(intersect)) /
                            float(len(this_3D)), 2)) +
                  '%: ' +
                  str(len(intersect)) +
                  ' stars out of the ' +
                  str(len(this_3D)) +
                  ' stars in ' +
                  str(key_3D) +
                  ' (3D) are also part of ' +
                  str(key_4D) +
                  ' (4D, encompassing ' +
                  str(len(this_4D)) +
                  ' stars)')

# %%
for key_4D in dict_4D_vel.keys():
    for key_3D in dict_3D_vel.keys():
        this_3D, this_4D = dict_3D_vel[key_3D], dict_4D_vel[key_4D]
        intersect = np.intersect1d(this_3D, this_4D)
        if len(intersect) > 0:
            print(str(round(100.0 *
                            float(len(intersect)) /
                            float(len(this_4D)), 2)) +
                  '%: ' +
                  str(len(intersect)) +
                  ' stars out of the ' +
                  str(len(this_4D)) +
                  ' stars in ' +
                  str(key_4D) +
                  ' (4D) are also part of ' +
                  str(key_3D) +
                  ' (3D, encompassing ' +
                  str(len(this_3D)) +
                  ' stars)')

# %%
for key_3D in dict_3D_IoM.keys():
    for key_4D in dict_4D_IoM.keys():
        this_3D, this_4D = dict_3D_IoM[key_3D], dict_4D_IoM[key_4D]
        intersect = np.intersect1d(this_3D, this_4D)
        if len(intersect) > 0:
            print(str(round(100.0 *
                            float(len(intersect)) /
                            float(len(this_3D)), 2)) +
                  '%: ' +
                  str(len(intersect)) +
                  ' stars out of the ' +
                  str(len(this_3D)) +
                  ' stars in ' +
                  str(key_3D) +
                  ' (3D) are also part of ' +
                  str(key_4D) +
                  ' (4D, encompassing ' +
                  str(len(this_4D)) +
                  ' stars)')

# %%
for key_4D in dict_4D_IoM.keys():
    for key_3D in dict_3D_IoM.keys():
        this_3D, this_4D = dict_3D_IoM[key_3D], dict_4D_IoM[key_4D]
        intersect = np.intersect1d(this_3D, this_4D)
        if len(intersect) > 0:
            print(str(round(100.0 *
                            float(len(intersect)) /
                            float(len(this_4D)), 2)) +
                  '%: ' +
                  str(len(intersect)) +
                  ' stars out of the ' +
                  str(len(this_4D)) +
                  ' stars in ' +
                  str(key_4D) +
                  ' (4D) are also part of ' +
                  str(key_3D) +
                  ' (3D, encompassing ' +
                  str(len(this_3D)) +
                  ' stars)')

# %%
