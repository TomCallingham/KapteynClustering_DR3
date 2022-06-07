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
import matplotlib.transforms as transforms

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
# # Definitions:

# %%
# Reddening computation:

def reddening_computation(self):

    # Compute ebv from one of the published 3D maps:
    if dust_map == 'l18':
        # We use the dust map from lallement18. It uses
        # dust_maps_3d/lallement18.py and the file lallement18_nside64.fits.gz
        EBV = l18.ebv(self.evaluate('l'), self.evaluate('b'), self.evaluate('distance*1000.0'))
    elif dust_map == 'p19':
        # We use the dust map from pepita19. It would use dust_maps_3d/pepita19.py
        # and the file pepita19.fits.gz
        EBV = p19.ebv(self.evaluate('l'), self.evaluate('b'), self.evaluate('distance'))

    # From those E(B-V) compute the A_G and E_BP_RP values

    if ext_rec == 'cas18':
        # Casagrande & VandenBerg 2018 --> http://aselfabs.harvard.edu/abs/2018MNRAS.479L.102C (Table 2)
        # A_G ~ 2.740*E(B-V) --> We are not using this at all. This is just an average.
        # R_G = 1.4013 + xco*(3.1406-1.5626*xco)  --> xco = T4 = 10^−4 T_{eff}
        # R_BP = 1.7895 + xco*(4.2355-2.7071*xco)
        # R_RP = 1.8593 + xco*(0.3985-0.1771*xco)

        # Colour--Teff relation determined from Gaia DR2 data --> Computed using color_teff_GDR2.ipynb
        poly = np.array([62.55257114, -975.10845442, 4502.86260828, -8808.88745592, 10494.72444183])
        Teff = np.poly1d(poly)(self.evaluate('bp_rp')) / 1e4

        # We enter in an iterative process of 10 steps to refine the A_G and
        # E_BP_RP computation. 'Cause we do not have intrinsic colour.
        for i in range(10):
            E_BP_RP = (-0.0698 + Teff * (3.837 - 2.530 * Teff)) * EBV  # A_BP - A_RP
            A_G = (1.4013 + Teff * (3.1406 - 1.5626 * Teff)) * EBV
            Teff = np.poly1d(poly)(self.evaluate('bp_rp') - E_BP_RP) / 1e4

    elif ext_rec == 'bab18':
        # Babusiaux et al. 2018 --> http://aselfabs.harvard.edu/abs/2018A%26A...616A..10G (eq 1, Table 1)

        A0 = 3.1 * EBV
        bp_rp_0 = self.evaluate('bp_rp')

        # We enter in an iterative process of 10 steps to refine the kBP, kRP, and
        # kG values. 'Cause we do not have intrinsic colour.
        for i in range(10):
            kG = 0.9761 - 0.1704 * bp_rp_0 + 0.0086 * bp_rp_0**2 + 0.0011 * \
                bp_rp_0**3 - 0.0438 * A0 + 0.00130 * A0**2 + 0.0099 * bp_rp_0 * A0
            kBP = 1.1517 - 0.0871 * bp_rp_0 - 0.0333 * bp_rp_0**2 + 0.0173 * \
                bp_rp_0**3 - 0.0230 * A0 + 0.00060 * A0**2 + 0.0043 * bp_rp_0 * A0
            kRP = 0.6104 - 0.0170 * bp_rp_0 - 0.0026 * bp_rp_0**2 - 0.0017 * \
                bp_rp_0**3 - 0.0078 * A0 + 0.00005 * A0**2 + 0.0006 * bp_rp_0 * A0
            # 0 -- intrinsic, otherwise observed, Ax = mx - m0x --> m0x = mx - Ax = mx
            # - kx*A0  ==> m0BP - m0RP = mbp - mrp - (kbp - krp)*A0 = bp_rp
            bp_rp_0 = bp_rp_0 - (kBP - kRP) * A0

        E_BP_RP = (kBP - kRP) * A0  # A_BP - A_RP
        A_G = kG * A0

        COL_bp_rp = self.evaluate('bp_rp') - E_BP_RP

        M_G = self.evaluate('phot_g_mean_mag') - A_G + 5. - 5. * \
            np.log10(1000.0 / self.evaluate('new_parallax'))

    return E_BP_RP, A_G, COL_bp_rp, M_G


# %%
def pval_red_or_green(val):
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s' % color


# %%
def create_tables_defining_CaMD(df, path, name, main):

    iso_dict = {}
    ages_iso, mets_iso = [], []
    names_isos = glob.glob('/data/users/ruizlara/isochrones/alpha_enhanced/*.isc_gaia-dr3')
    for i, name_iso in enumerate(names_isos):
        lines_iso = open(names_isos[i], 'r').readlines()
        iso_age, iso_mh = lines_iso[4].split('=')[5].split()[0], lines_iso[4].split('=')[2].split()[0]
        ages_iso.append(iso_age)
        mets_iso.append(float(iso_mh))
        iso_dict['age' + str(iso_age) + '-met' + str(iso_mh)] = {'name_iso': name_iso}

    best_isochrone, df = apply_CaMD_fitting(
        df=df, clusters=main, path_CMD=path, name=name, iso_dict=iso_dict, mah_dist_lim=2.13)

    tablefile_csv = open(str(path) + str(name) + '_definition_table_CaMD.csv', 'w')
    tablefile_csv_percs_same_dec = open(
        str(path) +
        str(name) +
        '_definition_table_percs_same_dec_CaMD.csv',
        'w')
    tablefile_csv_percs_less = open(str(path) + str(name) + '_definition_table_percs_less_CaMD.csv', 'w')
    chain1 = ','
    for cluster in main:
        df.select('(def_cluster_id==' + str(cluster) + ')', name='this_cluster')
        if len(df.evaluate('source_id', selection='(this_cluster)&(quality)')) < 20.0:
            chain1 = chain1 + str(cluster) + '*, '
        else:
            chain1 = chain1 + str(cluster) + ', '

    chain1 = chain1[0:-2] + ' \n'
    tablefile_csv.write(chain1)
    tablefile_csv_percs_same_dec.write(chain1)
    tablefile_csv_percs_less.write(chain1)

    chain2 = str(name) + ', '
    chain3 = str(name) + ', '
    chain4 = str(name) + ', '
    for cluster in main:
        coords = np.where(np.array(main) != cluster)
        orig_pval, comp_percentage, less_percentage = compare_MDFs_CMDs_low_Nstars_CaMDs(
            df, clusters_CM=[
                np.array(main)[coords], [cluster]], labels=[
                str(name), str(cluster)], colours=[
                'red', 'lightblue'], zorder=[
                    1, 2], Nstars_lim=20, NMC=100, path_step2=path, name_step2=name, best_isochrone=best_isochrone)
        chain2 = chain2 + str(orig_pval) + ', '
        chain3 = chain3 + str(comp_percentage) + ', '
        chain4 = chain4 + str(less_percentage) + ', '
    chain2 = chain2[0:-2] + ' \n'
    chain3 = chain3[0:-2] + ' \n'
    chain4 = chain4[0:-2] + ' \n'
    tablefile_csv.write(chain2)
    tablefile_csv_percs_same_dec.write(chain3)
    tablefile_csv_percs_less.write(chain4)

    tablefile_csv.close()
    tablefile_csv_percs_same_dec.close()
    tablefile_csv_percs_less.close()

    df = pd.read_csv(str(path) + str(name) + '_definition_table_CaMD.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(name))
            .background_gradient(cmap=cm).applymap(pval_red_or_green))

    df = pd.read_csv(str(path) + str(name) + '_definition_table_percs_same_dec_CaMD.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(name)))

    df = pd.read_csv(str(path) + str(name) + '_definition_table_percs_less_CaMD.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(name)))


# %%
def apply_CaMD_fitting(df, clusters, path_CMD, name, iso_dict, mah_dist_lim):

    # A) isochrone fitting:
    COL, COL_err, MAG, MAG_err, labels_input, mah_dists_input, ls_input, bs_input, AGs_input, mets_input, mets_lamost_good = [
    ], [], [], [], [], [], [], [], [], [], []
    for cluster in clusters:
        df.select('(def_cluster_id==' + str(cluster) + ')', name='this_cluster')
        COL.extend(df.evaluate('COL_bp_rp', selection='(this_cluster)&(quality)'))
        COL_err.extend(df.evaluate('err_COL_bp_rp', selection='(this_cluster)&(quality)'))
        MAG.extend(df.evaluate('M_G', selection='(this_cluster)&(quality)'))
        MAG_err.extend(df.evaluate('err_MG', selection='(this_cluster)&(quality)'))
        labels_input.extend(df.evaluate('def_cluster_id', selection='(this_cluster)&(quality)'))
        mah_dists_input.extend(df.evaluate('def_distances', selection='(this_cluster)&(quality)'))
        ls_input.extend(df.evaluate('l', selection='(this_cluster)&(quality)'))
        bs_input.extend(df.evaluate('b', selection='(this_cluster)&(quality)'))
        AGs_input.extend(df.evaluate('A_G', selection='(this_cluster)&(quality)'))
        mets_input.extend(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(quality)'))
        mets_lamost_good.extend(df.evaluate('feh_lamost_lrs', selection='(this_cluster)&(quality)'))

    COL, COL_err, MAG, MAG_err, labels_input, mah_dists_input, ls_input, bs_input, AGs_input, mets_input, mets_lamost_good = np.array(COL), np.array(COL_err), np.array(MAG), np.array(
        MAG_err), np.array(labels_input), np.array(mah_dists_input), np.array(ls_input), np.array(bs_input), np.array(AGs_input), np.array(mets_input), np.array(mets_lamost_good)

    Nlamost = len(mets_lamost_good[np.isnan(mets_lamost_good) == False])
    if Nlamost > 10:
        mean_feh, std_feh = np.nanmean(mets_lamost_good), np.nanstd(mets_lamost_good)
        Age, errA, FeH, errFeH, d2_param, d2_perc, best_isochrone = iso_fit_prob(
            name, COL, COL_err, MAG, MAG_err, labels_input, mah_dists_input, ls_input, bs_input, AGs_input, mets_input, [
                0.0, 14.0], [
                mean_feh - std_feh, mean_feh + std_feh], 7, 7, 15, 1, mah_dist_lim, path_CMD)
    else:
        mean_feh, std_feh = np.nan, np.nan
        Age, errA, FeH, errFeH, d2_param, d2_perc, best_isochrone = iso_fit_prob(
            name, COL, COL_err, MAG, MAG_err, labels_input, mah_dists_input, ls_input, bs_input, AGs_input, mets_input, [
                0.0, 14.0], [
                -4.0, 1.0], 7, 7, 15, 1, mah_dist_lim, path_CMD)

    # B) Computing distance to isochrone:

    # B.1) Defining curve for best isochrone.

    f0, (ax1, ax2) = plt.subplots(2, figsize=(6, 3), facecolor='w')
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # isochrones:
    iso = ascii.read(best_isochrone)
    MG, GBP, GRP = iso['col5'], iso['col6'], iso['col7']
    ax1.plot(GBP - GRP, MG, 'b-')
    ax1.plot(GBP[0:1280] - GRP[0:1280], MG[0:1280], 'r--')
    col, mag = GBP[0:1280] - GRP[0:1280], MG[0:1280]

    new_mag = np.arange(np.nanmin(mag), np.nanmax(mag), 0.001)
    new_col = new_mag.copy()
    for i in np.arange(len(new_mag) - 1):
        new_col[i] = np.nanmean(col[(new_mag[i] < mag) & (mag < new_mag[i + 1])])
    new_col[-1] = col[0]

    coords = np.where(np.isnan(new_col) == False)
    new_col = new_col[coords]
    new_mag = new_mag[coords]

    tck2 = splrep(new_mag, new_col, k=5)
    xx = np.arange(np.min(new_mag), np.max(new_mag), 0.1)
    yy = splev(xx, tck2)
    ax2.plot(mag, col, 'bo')
    ax2.plot(xx, yy, 'r-', lw=4)
    ax2.plot(MG[0:1280], GBP[0:1280] - GRP[0:1280], 'c--')

    mag_pred = [6.0, 5.0, 3.5, 0.1]
    col_pred = splev(mag_pred, tck2)

    ax1.plot(col_pred, mag_pred, '*r')
    ax2.plot(mag_pred, col_pred, '*r')

    ax1.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax1.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax1.set_xlim(-0.2, 2.5)
    ax1.set_ylim(9.0, -4.0)
    ax1.tick_params(labelsize=20)

    ax2.set_ylim(0.0, 3.5)

    ax1.legend()

    plt.tight_layout()
    plt.show()

    # B.2) Adding a column to df that is distance in colour to isochrone (iso_dist_col_NAME)

    mag_pred = df_clusters.evaluate('M_G')
    col_pred = splev(mag_pred, tck2)
    iso_dist = df_clusters.evaluate('COL_bp_rp') - col_pred
    df_clusters.add_column('iso_dist_col_' + name, iso_dist)

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    f0, (ax1) = plt.subplots(1, figsize=(4, 4), facecolor='w')
    ax1 = plt.subplot(1, 1, 1)

    # isochrones:
    iso = ascii.read(best_isochrone)
    MG, GBP, GRP = iso['col5'], iso['col6'], iso['col7']
    ax1.plot(GBP - GRP, MG, 'b-')
    ax1.plot(GBP[0:1280] - GRP[0:1280], MG[0:1280], 'r--')

    scatter1 = ax1.scatter(
        df_clusters.evaluate(
            'COL_bp_rp',
            selection='(structure)'),
        df_clusters.evaluate(
            'M_G',
            selection='(structure)'),
        c=df_clusters.evaluate(
            'iso_dist_col_' + name,
            selection='(structure)'),
        s=8.0,
        alpha=1.0,
        cmap='jet',
        vmin=-0.75,
        vmax=0.75)
    f0.colorbar(scatter1, ax=ax1, label=r'$iso dist$', pad=0.0)

    ax1.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax1.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax1.set_xlim(-0.2, 2.5)
    ax1.set_ylim(9.0, -4.0)
    ax1.tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()

    return best_isochrone, df


# %%
def compare_MDFs_CMDs_low_Nstars_CaMDs(df, clusters_CM, labels, colours,
                                       zorder, Nstars_lim, NMC, path_step2, name_step2, best_isochrone):

    # Compare the colour-distribution of two structures:
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = str()
        for cluster in clusters[:-1]:
            condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
        condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
        df.select(condition, name='structure')
        distr_dict[str(system)] = {'colour_distr': df.evaluate('iso_dist_col_' + name_step2, selection='(structure)'),
                                   'mag': df.evaluate('M_G', selection='(structure)'),
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)'),
                                   'en': df.evaluate('En', selection='(structure)') / 10**4,
                                   'lz': df.evaluate('Lz', selection='(structure)') / 10**3,
                                   'lperp': df.evaluate('Lperp', selection='(structure)') / 10**3}

    main_colour, second_colour = distr_dict[labels[0]]['colour_distr'], distr_dict[labels[1]]['colour_distr']
    main_colour = main_colour[np.isnan(main_colour) == False]
    second_colour = second_colour[np.isnan(second_colour) == False]

    if len(second_colour) < Nstars_lim:
        f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, figsize=(18, 15), facecolor='w')
        ax1 = plt.subplot(331)
        ax2 = plt.subplot(332)
        ax3 = plt.subplot(333)
        ax4 = plt.subplot(334)
        ax5 = plt.subplot(335)
        ax6 = plt.subplot(336)
        ax7 = plt.subplot(337)
        ax8 = plt.subplot(338)
    else:
        f0, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(18, 11), facecolor='w')
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(232)
        ax3 = plt.subplot(233)
        ax4 = plt.subplot(234)
        ax5 = plt.subplot(235)
        ax6 = plt.subplot(236)

    ax4.scatter(df.evaluate('Lz') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(df.evaluate('Lz') / (10**3), df.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)

    for i in np.arange(len(labels)):
        ax1.hist(distr_dict[labels[i]]['colour_distr'], range=(-1.5, 1.5), histtype='step',
                 bins=25, density=True, color=colours[i], lw=2, label=labels[i])
        ax2.hist(distr_dict[labels[i]]['colour_distr'], range=(-1.5, 1.5), histtype='step',
                 bins=25, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['colour_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])
        ax3.plot(distr_dict[labels[i]]['col'], distr_dict[labels[i]]
                 ['mag'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        # best_isochrone
        iso = ascii.read(best_isochrone)
        MG, GBP, GRP = iso['col5'], iso['col6'], iso['col7']
        ax3.plot(GBP - GRP, MG, 'k-')

        ax4.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]
                 ['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax5.plot(distr_dict[labels[i]]['lperp'], distr_dict[labels[i]]
                 ['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax6.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]
                 ['lperp'], 'o', color=colours[i], zorder=zorder[i], ms=3)

    indexes = np.arange(len(clusters_CM))
    pairs = list(itertools.combinations(indexes, 2))
    count = 0
    for pair in pairs:
        colour_x, colour_y = distr_dict[labels[pair[0]]
                                        ]['colour_distr'], distr_dict[labels[pair[1]]]['colour_distr']
        colour_x = colour_x[np.isnan(colour_x) == False]
        colour_y = colour_y[np.isnan(colour_y) == False]
        colour_x = colour_x[(colour_x > -100.0) & (colour_x < 100.0)]
        colour_y = colour_y[(colour_y > -100.0) & (colour_y < 100.0)]
        KS, pval = stats.ks_2samp(colour_x, colour_y, mode='exact')
        if len(second_colour) < Nstars_lim:
            ax8.axhline(pval, ls='--', color='black')
            orig_pval = pval
        else:
            orig_pval = pval
            comp_percentage = 999.9
            less_percentage = 999.9

        ax2.text(0.95,
                 0.05 + 0.05 * count,
                 str(labels[pair[0]]) + '-' + str(labels[pair[1]]) + ': ' + str(round(pval,
                                                                                      3)),
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform=ax2.transAxes,
                 fontsize=20)
        count = count + 1

    ax1.text(0.04, 0.95, 'N = ' + str(len(main_colour)),
             color=colours[0], transform=ax1.transAxes, fontsize=14)
    ax1.text(0.04, 0.90, 'N = ' + str(len(second_colour)),
             color=colours[1], transform=ax1.transAxes, fontsize=14)

    ax1.legend()
    ax1.set_xlabel(r'$\rm \Delta Colour$', fontsize=20)
    ax1.set_ylabel(r'$\rm Normalised \; N$', fontsize=20)
    ax1.tick_params(labelsize=20)

    ax2.set_xlabel(r'$\rm \Delta Colour$', fontsize=20)
    ax2.set_ylabel(r'$\rm Cumulative \; Distribution$', fontsize=20)
    ax2.tick_params(labelsize=20)

    ax3.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax3.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax3.set_xlim(-0.2, 3.0)
    ax3.set_ylim(9.0, -4.0)
    ax3.tick_params(labelsize=20)

    ax4.set_ylabel(r'$En/10^4$', fontsize=20)
    ax4.set_xlabel(r'$Lz/10^3$', fontsize=20)
    ax4.tick_params(labelsize=20)

    ax5.set_ylabel(r'$En/10^4$', fontsize=20)
    ax5.set_xlabel(r'$Lperp/10^3$', fontsize=20)
    ax5.tick_params(labelsize=20)

    ax6.set_ylabel(r'$Lperp/10^3$', fontsize=20)
    ax6.set_xlabel(r'$Lz/10^3$', fontsize=20)
    ax6.tick_params(labelsize=20)

    plt.tight_layout()

    if len(second_colour) < Nstars_lim:

        ax7.hist(main_colour, range=(-1.5, 1.5), histtype='step', bins=25,
                 density=True, color=colours[0], lw=2, label=labels[0], zorder=1)
        ax7.hist(second_colour, range=(-1.5, 1.5), histtype='step', bins=25,
                 density=True, color=colours[1], lw=2, label=labels[1], zorder=1)

        NNs, pvals = [], []

        count = 0
        while count < NMC:
            second_colour = second_colour[(second_colour > -100.0) & (second_colour < 100.0)]
            main_colour = main_colour[(main_colour > -100.0) & (main_colour < 100.0)]
            this_second_colour = np.random.choice(main_colour, len(second_colour))
            this_second_colour = this_second_colour[(
                this_second_colour > -100.0) & (this_second_colour < 100.0)]
            KS, pval = stats.ks_2samp(main_colour, this_second_colour, mode='exact')
            NNs.append(count)
            pvals.append(pval)
            ax7.hist(
                this_second_colour,
                histtype='step',
                bins=25,
                density=True,
                color='gray',
                lw=2,
                zorder=0,
                alpha=0.01)
            count = count + 1

        coords = np.where((np.array(pvals) < orig_pval))
        less_percentage = float(len(np.array(pvals)[coords])) / float(len(np.array(pvals)))

        coords = np.where((np.array(pvals) < orig_pval + 0.05) & (np.array(pvals) > orig_pval - 0.05))
        compatibility = 100.0 * float(len(np.array(pvals)[coords])) / float(len(np.array(pvals)))
        ax7.text(0.05, 0.94, 'Compatible pvals ' +
                 str(round(compatibility, 2)) +
                 ' %', fontsize=14, transform=ax7.transAxes)

        coords = np.where((np.array(pvals) > 0.05))
        same_distr = 100.0 * float(len(np.array(pvals)[coords])) / float(len(np.array(pvals)))
        ax7.text(0.05, 0.88, 'Same distr ' + str(round(same_distr, 2)) +
                 ' %', fontsize=14, transform=ax7.transAxes)

        coords = np.where((np.array(pvals) < 0.05))
        diff_distr = 100.0 * float(len(np.array(pvals)[coords])) / float(len(np.array(pvals)))
        ax7.text(0.05, 0.82, 'Different distr ' +
                 str(round(diff_distr, 2)) +
                 ' %', fontsize=14, transform=ax7.transAxes)

        ax7.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', label='MC')

        ax8.plot(NNs, pvals, 'ko')
        ax8.axhline(np.nanmean(pvals), ls='--', alpha=0.4, color='black')
        coords = np.where(np.array(pvals) < 0.05)
        ax8.plot(np.array(NNs)[coords], np.array(pvals)[coords], 'ro')
        ax8.axhline(0.05, ls='--', alpha=0.4, color='red')
        ax8.text(0.0, np.nanmean(pvals) - 0.05, 'pval = ' + str(round(np.nanmean(pvals), 2)), fontsize=14)

        ax7.legend()
        ax7.set_xlabel(r'$\rm \Delta Colour$', fontsize=20)
        ax7.set_ylabel(r'$\rm Normalised \; N$', fontsize=20)
        ax7.tick_params(labelsize=20)

        ax8.set_xlabel(r'$\rm N_{boot}$', fontsize=20)
        ax8.set_ylabel(r'$\rm pval$', fontsize=20)
        ax8.tick_params(labelsize=20)

        if orig_pval < 0.05:
            comp_percentage = diff_distr
        else:
            comp_percentage = same_distr

    plt.show()

    return orig_pval, comp_percentage, less_percentage


# %%
# kdtree to compute distances from stars to the closest point in the isochrone.
def do_kdtree(observed_data, isochrone, k, error_weights):
    mytree = cKDTree(observed_data)
    dist, indexes = mytree.query(isochrone, k=k)
    #parameter = np.nanmean(dist)/float(len(observed_data))
    # Minimisation similar to https://ui.adsabs.harvard.edu/abs/2019ApJS..245...32L/abstract
    parameter = np.sqrt(np.average(dist**2, weights=error_weights))
    return parameter, dist

# Compute surface based on the fitting in order to compute the minimum (age and metallicity):
# w = (Phi^T Phi)^{-1} Phi^T t
# where Phi_{k, j + i (m_2 + 1)} = x_k^i y_k^j,
#       t_k = z_k,
#           i = 0, 1, ..., m_1,
#           j = 0, 1, ..., m_2,
#           k = 0, 1, ..., n - 1


def polyfit2d(x, y, z, m_1, m_2):
    # Generate Phi by setting Phi as x^i y^j
    nrows = x.size
    ncols = (m_1 + 1) * (m_2 + 1)
    Phi = np.zeros((nrows, ncols))
    ij = itertools.product(range(m_1 + 1), range(m_2 + 1))
    for h, (i, j) in enumerate(ij):
        Phi[:, h] = x ** i * y ** j
    # Generate t by setting t as Z
    t = z
    # Generate w by solving (Phi^T Phi) w = Phi^T t
    w = np.linalg.solve(Phi.T.dot(Phi), (Phi.T.dot(t)))
    return w

# Evaluating the previously computed surface:
# t' = Phi' w
# where Phi'_{k, j + i (m_2 + 1)} = x'_k^i y'_k^j
#       t'_k = z'_k,
#           i = 0, 1, ..., m_1,
#           j = 0, 1, ..., m_2,
#           k = 0, 1, ..., n' - 1


def polyval2d(x_, y_, w, m_1, m_2):
    # Generate Phi' by setting Phi' as x'^i y'^j
    nrows = x_.size
    ncols = (m_1 + 1) * (m_2 + 1)
    Phi_ = np.zeros((nrows, ncols))
    ij = itertools.product(range(m_1 + 1), range(m_2 + 1))
    for h, (i, j) in enumerate(ij):
        Phi_[:, h] = x_ ** i * y_ ** j
    # Generate t' by setting t' as Phi' w
    t_ = Phi_.dot(w)
    # Generate z_ by setting z_ as t_
    z_ = t_
    return z_

# Set of isochrones to consider:


def define_set_of_isochrones(age_range, met_range, alpha):
    names_aux_iso, ages_aux_iso, mets_aux_iso = [], [], []
    if alpha == 0:
        auxnames_iso = glob.glob('/data/users/ruizlara/isochrones/solar_scaled/*.isc_gaia-dr3')
    elif alpha == 1:
        auxnames_iso = glob.glob('/data/users/ruizlara/isochrones/alpha_enhanced/*.isc_gaia-dr3')
    for i, name_iso in enumerate(auxnames_iso):
        lines_iso = open(auxnames_iso[i], 'r').readlines()
        iso_age, iso_mh = lines_iso[4].split('=')[5].split()[0], lines_iso[4].split('=')[2].split()[0]
        ages_aux_iso.append(float(iso_age) / 10**3)
        if alpha == 0:
            mets_aux_iso.append(float(iso_mh))
        elif alpha == 1:
            mets_aux_iso.append(float(iso_mh) - 0.31)
        names_aux_iso.append(name_iso)

    names_aux_iso, ages_aux_iso, mets_aux_iso = np.array(
        names_aux_iso), np.array(ages_aux_iso), np.array(mets_aux_iso)
    coords = np.where(
        ((age_range[0] < ages_aux_iso) & (
            ages_aux_iso < age_range[1]) & (
            met_range[0] < mets_aux_iso) & (
                mets_aux_iso < met_range[1])))
    ages_iso, mets_iso, names_iso = ages_aux_iso[coords], mets_aux_iso[coords], names_aux_iso[coords]
    return np.array(ages_iso), np.array(mets_iso), np.array(names_iso)


# Fitting and determination of age and metallicity:
def iso_fit_prob(name, colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, cluster_labels_all, cluster_mah_dists_all,
                 l_array_all, b_array_all, AG_array_all, met_array_all, iso_age_range, iso_met_range, deg_surfx, deg_surfy, percentile_solution, alpha, mah_dist_limit, path_CMD):

    coords = np.where(
        (cluster_labels_all > 0.5) & (
            (colour_array_all > 0.65) | (
                (colour_array_all > 0.3) & (
                    colour_array_all < 0.65) & (
                    magnitude_array_all > 2.0))) & (
                        cluster_mah_dists_all < mah_dist_limit) & (
                            AG_array_all < 0.7))
    colour_array, colour_error_array, magnitude_array, magnitude_error_array, cluster_labels, cluster_probs, l_array, b_array, met_array, AG_array = colour_array_all[coords], colour_error_array_all[
        coords], magnitude_array_all[coords], magnitude_error_array_all[coords], cluster_labels_all[coords], cluster_mah_dists_all[coords], l_array_all[coords], b_array_all[coords], met_array_all[coords], AG_array_all[coords]

    coords_bad = np.where(cluster_mah_dists_all > mah_dist_limit)
    colour_array_BAD, colour_error_array_BAD, magnitude_array_BAD, magnitude_error_array_BAD, l_array_BAD, b_array_BAD, met_array_BAD, AG_array_BAD = colour_array_all[coords_bad], colour_error_array_all[
        coords_bad], magnitude_array_all[coords_bad], magnitude_error_array_all[coords_bad], l_array_all[coords_bad], b_array_all[coords_bad], met_array_all[coords_bad], AG_array_all[coords_bad]

    if len(colour_array) == 0:
        print('No stars after applying all cuts and conditions')
        Age, errA, FeH, errFeH, d2_param, d2_perc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        start_time = time.time()

        print('###############################')
        print('## iso_fit to ' + str(name) + ' ##')
        print('###############################')

        # Preparing observed CMD:
        cluster_CMD, cluster_weights = [], []
        for i in np.arange(len(colour_array)):
            cluster_CMD.append([colour_array[i], magnitude_array[i]])
            cluster_weights.append(1.0 / (colour_error_array[i] + magnitude_error_array[i]))
        cluster_CMD, cluster_weights = np.array(cluster_CMD), np.array(cluster_weights)

        # Preparing set of isochrones and fit to observed data:
        ages_iso, mets_iso, names_iso = define_set_of_isochrones(iso_age_range, iso_met_range, alpha)
        if alpha == 0:
            print('## Using ' + str(len(names_iso)) + ' isochrones (solar-scaled)')
            print('...')
        elif alpha == 1:
            print('## Using ' + str(len(names_iso)) + ' isochrones (alpha-enhanced)')
            print('...')
        ages_fit, mets_fit, parameters_fit = [], [], []
        # isochrones
        for iso_name, iso_age, iso_met in zip(names_iso, ages_iso, mets_iso):
            iso = ascii.read(iso_name)
            iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
            isochrone_CMD = []
            for j in np.arange(len(iso_color)):
                isochrone_CMD.append([iso_color[j], iso_mag[j]])
            isochrone_CMD = np.array(isochrone_CMD)
            parameter, aux2 = do_kdtree(isochrone_CMD, cluster_CMD, 1, cluster_weights)
            ages_fit.append(float(iso_age))
            mets_fit.append(float(iso_met))
            parameters_fit.append(parameter)

        print('## Plotting results')
        print('...')

        # PLOT

        x = np.array(ages_fit)
        y = np.array(mets_fit)
        z_plot = np.array(np.log10(parameters_fit))
        z = np.array(parameters_fit)

        # Fitting surface to the points:
        # Generate w
        w = polyfit2d(x, y, z, m_1=deg_surfx, m_2=deg_surfy)
        # Generate x', y', z'
        n_ = 1000
        x_, y_ = np.meshgrid(np.linspace(x.min(), x.max(), n_), np.linspace(y.min(), y.max(), n_))
        z_ = np.zeros((n_, n_))
        for i in range(n_):
            z_[i, :] = polyval2d(x_[i, :], y_[i, :], w, m_1=deg_surfx, m_2=deg_surfy)

        # Compute age-Metallicity values and their error:
        condition = (z < np.percentile(z, percentile_solution))
        XX1, XXerr, YY1, YYerr = np.nanmean(
            x[condition]), np.nanstd(
            x[condition]), np.nanmean(
            y[condition]), np.nanstd(
                y[condition])
        XX2, YY2 = x_[np.where(z_ == np.min(z_))], y_[np.where(z_ == np.min(z_))]
        XX, YY = XX1, YY1
        #XX, YY = (XX1+XX2)/2.0, (YY1+YY2)/2.0

        f1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(21, 12), facecolor='w')
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(232)
        ax3 = plt.subplot(233)
        ax4 = plt.subplot(234)
        ax5 = plt.subplot(235)
        ax6 = plt.subplot(236)

        # choose 20 contour levels, just to show how good its interpolation is
        ax1.tricontourf(x, y, z_plot, 20, cmap='viridis_r')
        ax1.errorbar(
            XX,
            YY,
            xerr=XXerr,
            yerr=YYerr,
            marker='o',
            mfc='black',
            mec='black',
            ecolor='black',
            ms=10)
        ax1.scatter(ages_fit, mets_fit, c=np.log10(parameters_fit), cmap='viridis_r')
        ax1.scatter(x[condition], y[condition], c='black', alpha=0.3, s=5)

        ax1.set_xlabel(r'$\rm Age \; [Gyr]$', fontsize=25)
        ax1.set_ylabel(r'$\rm [Fe/H]$', fontsize=25)
        ax1.set_xlim(np.nanmin(ages_fit), np.nanmax(ages_fit))
        ax1.set_ylim(np.nanmin(mets_fit), np.nanmax(mets_fit))

        ax1.tick_params(labelsize=20)

        #ax2.plot(colour_array, magnitude_array, 'bo', alpha = 0.8)
        ax2.errorbar(
            colour_array_all,
            magnitude_array_all,
            xerr=colour_error_array_all,
            yerr=magnitude_error_array_all,
            fmt='none',
            ecolor='blue',
            alpha=0.4)
        ax2.errorbar(
            colour_array,
            magnitude_array,
            xerr=colour_error_array,
            yerr=magnitude_error_array,
            fmt='none',
            ecolor='blue')
        scatter2 = ax2.scatter(
            colour_array,
            magnitude_array,
            c=cluster_weights,
            cmap='viridis',
            s=23,
            zorder=3,
            vmin=np.percentile(
                cluster_weights,
                10),
            vmax=np.percentile(
                cluster_weights,
                90))
        f1.colorbar(scatter2, ax=ax2, label=r'$Weights$', pad=0.0)
        ax2.plot([np.nan, np.nan], [np.nan, np.nan], 'bo', label='Cluster #' +
                 str(name) + ', ' + str(len(colour_array)) + ' stars')
        ax2.plot(colour_array_BAD, magnitude_array_BAD, 'kx', ms=10, alpha=0.6)

        # isochrones:
        whole_age_dist = []
        for k in np.arange(len(names_iso)):
            whole_age_dist.append([ages_iso[k], mets_iso[k]])
            if ((XX - XXerr < ages_iso[k]) & (ages_iso[k] < XX + XXerr) &
                    (YY - YYerr < mets_iso[k]) & (mets_iso[k] < YY + YYerr)):
                iso = ascii.read(names_iso[k])
                iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
                ax2.plot(iso_color, iso_mag, 'k-', lw=1, alpha=0.3)
        ax2.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw=1, alpha=0.3, label='compatible isochrones')

        # Closest isochrone:
        whole_age_distr_weights = np.ones((len(whole_age_dist)))
        age_dist = [[XX, YY]]
        aux1, distances = do_kdtree(age_dist, whole_age_dist, 1, whole_age_distr_weights)
        min_dist_index = np.where(distances == np.min(distances))[0][0]

        best_isochrone = names_iso[min_dist_index]
        iso = ascii.read(best_isochrone)
        iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        ax2.plot(iso_color, iso_mag, 'r-', lw=1, label='Best fit: Age = ' +
                 str(XX.round(2)) + '; [Fe/H] = ' + str(YY.round(2)))

        ax2.plot([0.3, 0.3, 0.65, 0.65], [7.5, 2.0, 2.0, -6.0], 'k--', alpha=0.2)

        ax2.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax2.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax2.set_xlim(-0.5, 2.5)
        ax2.set_ylim(7.5, -6.0)
        ax2.tick_params(labelsize=20)

        #ax3.plot(colour_array, magnitude_array, 'bo', alpha = 0.8)
        scatter3 = ax3.scatter(
            colour_array_all,
            magnitude_array_all,
            c=cluster_labels_all,
            cmap='gist_rainbow',
            s=23,
            vmin=np.min(cluster_labels_all) - 0.5,
            vmax=np.max(cluster_labels_all) + 0.5,
            alpha=0.4)
        scatter3 = ax3.scatter(
            colour_array,
            magnitude_array,
            c=cluster_labels,
            cmap='gist_rainbow',
            s=23,
            vmin=np.min(cluster_labels) - 0.5,
            vmax=np.max(cluster_labels) + 0.5)
        f1.colorbar(scatter3, ax=ax3, label=r'$Membership$', pad=0.0)
        ax3.plot([np.nan, np.nan], [np.nan, np.nan], 'bo', label='Cluster #' +
                 str(name) + ', ' + str(len(colour_array)) + ' stars')

        # isochrones:
        whole_age_dist = []
        for k in np.arange(len(names_iso)):
            whole_age_dist.append([ages_iso[k], mets_iso[k]])
            if ((XX - XXerr < ages_iso[k]) & (ages_iso[k] < XX + XXerr) &
                    (YY - YYerr < mets_iso[k]) & (mets_iso[k] < YY + YYerr)):
                iso = ascii.read(names_iso[k])
                iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
                ax3.plot(iso_color, iso_mag, 'k-', lw=1, alpha=0.3)
        ax3.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw=1, alpha=0.3, label='compatible isochrones')

        iso = ascii.read(best_isochrone)
        iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        ax3.plot(iso_color, iso_mag, 'r-', lw=1, label='Best fit: Age = ' +
                 str(XX.round(2)) + '; [Fe/H] = ' + str(YY.round(2)))

        ax3.plot(colour_array_BAD, magnitude_array_BAD, 'kx', ms=10, alpha=0.6)

        ax3.plot([0.3, 0.3, 0.65, 0.65], [7.5, 2.0, 2.0, -6.0], 'k--', alpha=0.2)

        ax3.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax3.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax3.set_xlim(-0.5, 2.5)
        ax3.set_ylim(7.5, -6.0)
        ax3.tick_params(labelsize=20)

        ax4.scatter(colour_array_all, magnitude_array_all, c=met_array_all, cmap='viridis', s=23, alpha=0.4)
        scatter4 = ax4.scatter(colour_array, magnitude_array, c=met_array, cmap='viridis', s=23)
        f1.colorbar(scatter4, ax=ax4, label=r'$[Fe/H]$', pad=0.0)

        # isochrones:
        whole_age_dist = []
        for k in np.arange(len(names_iso)):
            whole_age_dist.append([ages_iso[k], mets_iso[k]])
            if ((XX - XXerr < ages_iso[k]) & (ages_iso[k] < XX + XXerr) &
                    (YY - YYerr < mets_iso[k]) & (mets_iso[k] < YY + YYerr)):
                iso = ascii.read(names_iso[k])
                iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
                ax4.plot(iso_color, iso_mag, 'k-', lw=1, alpha=0.3)

        iso = ascii.read(best_isochrone)
        iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        ax4.plot(iso_color, iso_mag, 'r-', lw=1)

        ax4.plot(colour_array_BAD, magnitude_array_BAD, 'kx', ms=10, alpha=0.6)

        ax4.plot([0.3, 0.3, 0.65, 0.65], [7.5, 2.0, 2.0, -6.0], 'k--', alpha=0.2)

        ax4.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax4.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax4.set_xlim(-0.5, 2.5)
        ax4.set_ylim(7.5, -6.0)
        ax4.tick_params(labelsize=20)

        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        axx4 = plt.axes([0, 0, 1, 2])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax4, [0.12, 0.67, 0.4, 0.3])  # [left, bottom, width, height]
        axx4.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
        #mark_inset(ax1, axx0, loc1=2, loc2=4, fc="none", ec='0.5')
        axx4.patch.set_alpha(0.0)
        axx4.hist(met_array_all, range=(-3.0, 0.7), bins=20, histtype='step', color='black', alpha=0.4)
        axx4.hist(met_array, range=(-3.0, 0.7), bins=20, histtype='step', color='black')
        axx4.axvline(YY, color='red', ls='--')
        axx4.set_xlabel(r'$[Fe/H]$')
        axx4.set_ylabel(r'$N$')

        ax5.scatter(
            colour_array_all,
            magnitude_array_all,
            c=AG_array_all,
            cmap='viridis',
            s=23,
            alpha=0.4,
            vmin=np.nanmin(AG_array_all),
            vmax=np.nanmax(AG_array_all))
        scatter5 = ax5.scatter(
            colour_array,
            magnitude_array,
            c=AG_array,
            cmap='viridis',
            s=23,
            vmin=np.nanmin(AG_array_all),
            vmax=np.nanmax(AG_array_all))
        f1.colorbar(scatter5, ax=ax5, label=r'$AG$', pad=0.0)

        # isochrones:
        whole_age_dist = []
        for k in np.arange(len(names_iso)):
            whole_age_dist.append([ages_iso[k], mets_iso[k]])
            if ((XX - XXerr < ages_iso[k]) & (ages_iso[k] < XX + XXerr) &
                    (YY - YYerr < mets_iso[k]) & (mets_iso[k] < YY + YYerr)):
                iso = ascii.read(names_iso[k])
                iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
                ax5.plot(iso_color, iso_mag, 'k-', lw=1, alpha=0.3)

        iso = ascii.read(best_isochrone)
        iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        ax5.plot(iso_color, iso_mag, 'r-', lw=1)

        ax5.plot(colour_array_BAD, magnitude_array_BAD, 'kx', ms=10, alpha=0.6)

        ax5.plot([0.3, 0.3, 0.65, 0.65], [7.5, 2.0, 2.0, -6.0], 'k--', alpha=0.2)

        ax5.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax5.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax5.set_xlim(-0.5, 2.5)
        ax5.set_ylim(7.5, -6.0)
        ax5.tick_params(labelsize=20)

        ax6.scatter(
            l_array_all,
            b_array_all,
            c=AG_array_all,
            cmap='viridis',
            s=23,
            alpha=0.4,
            vmin=np.nanmin(AG_array_all),
            vmax=np.nanmax(AG_array_all))
        scatter6 = ax6.scatter(
            l_array,
            b_array,
            c=AG_array,
            cmap='viridis',
            s=23,
            vmin=np.nanmin(AG_array_all),
            vmax=np.nanmax(AG_array_all))
        f1.colorbar(scatter6, ax=ax6, label=r'$AG$', pad=0.0)
        ax6.plot(l_array_BAD, b_array_BAD, 'kx', ms=10, alpha=0.6)

        ax6.set_xlabel(r'$\rm l$', fontsize=25)
        ax6.set_ylabel(r'$\rm b$', fontsize=25)
        ax6.set_xlim(0.0, 360.0)
        ax6.set_ylim(-90.0, 90.0)
        ax6.tick_params(labelsize=20)

        # Goodness of fit and plot:

        coord = np.where((mets_fit == mets_iso[min_dist_index]) & (ages_fit == ages_iso[min_dist_index]))
        fit_param = parameters_fit[coord[0][0]]
        coords = (fit_param < parameters_fit)
        fit_param_perc = 100.0 * float(len(coords[coords == True])) / float(len(parameters_fit))

        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        axx0 = plt.axes([0, 0, 1, 1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax1, [0.03, 0.05, 0.4, 0.3])  # [left, bottom, width, height]
        axx0.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
        #mark_inset(ax1, axx0, loc1=2, loc2=4, fc="none", ec='0.5')
        axx0.patch.set_alpha(0.0)
        axx0.hist(parameters_fit, bins=20, histtype='step', color='white')
        axx0.axvline(fit_param, ls='--', color='white')
        # axx0.axes.get_xaxis().set_ticks([])
        axx0.axes.get_yaxis().set_ticks([])
        axx0.spines['right'].set_visible(False)
        axx0.spines['top'].set_visible(False)
        axx0.spines['bottom'].set_color('white')
        axx0.spines['left'].set_color('white')
        axx0.tick_params(axis='x', colors='white')
        ax1.text(0.45, 0.02, r'$\rm d_2 = ' + str(fit_param.round(3)) + ' \\; (' +
                 str(round(fit_param_perc, 2)) + '\\%)$', transform=ax1.transAxes, color='white', fontsize=15)

        ax3.legend(fontsize='x-large')

        plt.tight_layout()

        plt.savefig(str(path_CMD) + '/' + str(name) + '_iso_fit_DEF_3D_CaMD.png')

        plt.show()

        print('###############################')
        print('## iso_fit results: ##')
        print('## Age = ' + str(XX.round(2)) + ' +/- ' + str(XXerr.round(2)) + ' Gyr ##')
        print('## [Fe/H] = ' + str(YY.round(2)) + ' +/- ' + str(YYerr.round(2)) + ' ##')
        print('## Closest isochrone is ' +
              str(best_isochrone) +
              ' (age,[Fe/H]) = (' +
              str(ages_iso[min_dist_index]) +
              ', ' +
              str(mets_iso[min_dist_index]) +
              ')')
        print('## Goodness of fit: ' + str(fit_param.round(3)) +
              ' (' + str(round(fit_param_perc, 2)) + '%) ##')
        print('###############################')
        print('Execution time: ' + str((time.time() - start_time)) + ' seconds')

        Age, errA, FeH, errFeH, d2_param, d2_perc = XX, XXerr, YY, YYerr, fit_param, fit_param_perc

    return Age, errA, FeH, errFeH, d2_param, d2_perc, best_isochrone


# %%
def compare_MDFs_CMDs(df, clusters_CM, labels, colours, P, zorder):
    df.select('(feh_lamost_lrs>-10000.0)', name='lamost')

    # Compare the MDF of two structures:
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = str()
        for cluster in clusters[:-1]:
            condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
        condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
        df.select(condition, name='structure')
        distr_dict[str(system)] = {'feh_distr': df.evaluate('feh_lamost_lrs', selection='(structure)&(lamost)'),
                                   'mag': df.evaluate('M_G', selection='(structure)'),
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)'),
                                   'en': df.evaluate('En', selection='(structure)') / 10**4,
                                   'lz': df.evaluate('Lz', selection='(structure)') / 10**3,
                                   'lperp': df.evaluate('Lperp', selection='(structure)') / 10**3}

    for system, clusters in zip(labels, clusters_CM):
        print(system)
        print(str(len(distr_dict[system]['feh_distr'])) + ' good stars with LAMOST LRS [Fe/H]')

    f0, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(18, 11), facecolor='w')
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    ax6 = plt.subplot(236)

    ax4.scatter(df.evaluate('Lz') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(df.evaluate('Lz') / (10**3), df.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)

    for i in np.arange(len(labels)):
        ax1.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7),
                 bins=25, density=True, color=colours[i], lw=2, label=labels[i])
        ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7),
                 bins=25, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])
        ax3.plot(distr_dict[labels[i]]['col'], distr_dict[labels[i]]
                 ['mag'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax4.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]
                 ['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax5.plot(distr_dict[labels[i]]['lperp'], distr_dict[labels[i]]
                 ['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax6.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]
                 ['lperp'], 'o', color=colours[i], zorder=zorder[i], ms=3)

    indexes = np.arange(len(clusters_CM))
    pairs = list(itertools.combinations(indexes, 2))
    count = 0
    for pair in pairs:
        feh_x, feh_y = distr_dict[labels[pair[0]]]['feh_distr'], distr_dict[labels[pair[1]]]['feh_distr']
        feh_x = feh_x[np.isnan(feh_x) == False]
        feh_y = feh_y[np.isnan(feh_y) == False]
        if ((len(feh_x) > 10) & (len(feh_y) > 10)):
            KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
        else:
            KS, pval = 999, 999

        ax2.text(0.95,
                 0.05 + 0.05 * count,
                 str(labels[pair[0]]) + '-' + str(labels[pair[1]]) + ': ' + str(round(pval,
                                                                                      3)),
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform=ax2.transAxes,
                 fontsize=20)
        count = count + 1

    ax1.legend()
    ax1.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
    ax1.set_ylabel(r'$\rm Normalised \; N$', fontsize=20)
    ax1.tick_params(labelsize=20)

    ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
    ax2.set_ylabel(r'$\rm Cumulative \; Distribution$', fontsize=20)
    ax2.tick_params(labelsize=20)

    ax3.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax3.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax3.set_xlim(-0.2, 3.0)
    ax3.set_ylim(9.0, -4.0)
    ax3.tick_params(labelsize=20)

    ax4.set_ylabel(r'$En/10^4$', fontsize=20)
    ax4.set_xlabel(r'$Lz/10^3$', fontsize=20)
    ax4.tick_params(labelsize=20)

    ax5.set_ylabel(r'$En/10^4$', fontsize=20)
    ax5.set_xlabel(r'$Lperp/10^3$', fontsize=20)
    ax5.tick_params(labelsize=20)

    ax6.set_ylabel(r'$Lperp/10^3$', fontsize=20)
    ax6.set_xlabel(r'$Lz/10^3$', fontsize=20)
    ax6.tick_params(labelsize=20)

    plt.suptitle(' perc' + str(P), fontsize=20)

    plt.tight_layout()

    plt.show()


# %%
def create_tables_adding_extra_members_CaMD(df, path, tab_name, main_names, mains, extra_clusters):

    tablefile_csv = open(str(path) + str(tab_name) + '_addition_table_CaMD.csv', 'w')
    tablefile_csv_percs_same_dec = open(
        str(path) +
        str(tab_name) +
        '_addition_table_percs_same_dec_CaMD.csv',
        'w')
    tablefile_csv_percs_less = open(str(path) + str(tab_name) + '_addition_table_percs_less_CaMD.csv', 'w')

    chain1 = ','
    for cluster in extra_clusters:
        df.select('(def_cluster_id==' + str(cluster) + ')', name='this_cluster')
        if len(df.evaluate('source_id', selection='(this_cluster)')) < 20.0:
            chain1 = chain1 + str(cluster) + '*, '
        else:
            chain1 = chain1 + str(cluster) + ', '

    chain1 = chain1[0:-2] + ' \n'
    tablefile_csv.write(chain1)
    tablefile_csv_percs_same_dec.write(chain1)
    tablefile_csv_percs_less.write(chain1)

    iso_dict = {}
    ages_iso, mets_iso = [], []
    names_isos = glob.glob('/data/users/ruizlara/isochrones/alpha_enhanced/*.isc_gaia-dr3')
    for i, name_iso in enumerate(names_isos):
        lines_iso = open(names_isos[i], 'r').readlines()
        iso_age, iso_mh = lines_iso[4].split('=')[5].split()[0], lines_iso[4].split('=')[2].split()[0]
        ages_iso.append(iso_age)
        mets_iso.append(float(iso_mh))
        iso_dict['age' + str(iso_age) + '-met' + str(iso_mh)] = {'name_iso': name_iso}

    for main_name, main in zip(main_names, mains):

        best_isochrone, df = apply_CaMD_fitting(
            df=df, clusters=main, path_CMD=path, name=main_name, iso_dict=iso_dict, mah_dist_lim=2.13)

        chain2 = str(main_name) + ', '
        chain3 = str(main_name) + ', '
        chain4 = str(main_name) + ', '
        for cluster in extra_clusters:
            coords = np.where(np.array(main) != cluster)
            orig_pval, comp_percentage, less_percentage = compare_MDFs_CMDs_low_Nstars_CaMDs(
                df, clusters_CM=[
                    np.array(main)[coords], [cluster]], labels=[
                    str(main_name), str(cluster)], colours=[
                    'red', 'lightblue'], zorder=[
                    1, 2], Nstars_lim=20, NMC=100, path_step2=path, name_step2=main_name, best_isochrone=best_isochrone)
            chain2 = chain2 + str(orig_pval) + ', '
            chain3 = chain3 + str(comp_percentage) + ', '
            chain4 = chain4 + str(less_percentage) + ', '
        chain2 = chain2[0:-2] + ' \n'
        chain3 = chain3[0:-2] + ' \n'
        chain4 = chain4[0:-2] + ' \n'
        tablefile_csv.write(chain2)
        tablefile_csv_percs_same_dec.write(chain3)
        tablefile_csv_percs_less.write(chain4)

    tablefile_csv.close()
    tablefile_csv_percs_same_dec.close()
    tablefile_csv_percs_less.close()

    df = pd.read_csv(str(path) + str(tab_name) + '_addition_table_CaMD.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(tab_name))
            .background_gradient(cmap=cm).applymap(pval_red_or_green))

    df = pd.read_csv(str(path) + str(tab_name) + '_addition_table_percs_same_dec_CaMD.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(tab_name)))

    df = pd.read_csv(str(path) + str(tab_name) + '_addition_table_percs_less_CaMD.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(tab_name)))


# %%
def nice_table_from_csv(path, name, table_type):

    df = vaex.open(str(path) + str(name) + str(table_type))

    columns = df.column_names
    Ncols = len(columns)

    header = ''
    for col in columns[:-1]:
        header = header + '{\\bf ' + str(col).replace('Unnamed: 0', '') + '} & '
    header = header + '{\\bf ' + str(columns[-1]) + '} \\\\ \n'
    # print(header)

    caption = name

    f = open(str(path) + str(name) + str(table_type), 'r')
    lines = f.readlines()
    Nlines = len(lines)
    f.close()
    # print(Nlines)

    rows = []

    for i in np.arange(1, Nlines):
        this_row = ''
        this_line = lines[i]
        # print(this_line)
        elements = this_line.split(',')
        for k, element in enumerate(elements):
            if k == 0:
                this_row = this_row + '{\\bf ' + str(element) + '} & '
            else:
                if float(element) < 0.05:
                    this_row = this_row + '\\cellcolor[HTML]{' + str(
                        matplotlib.colors.to_hex(
                            plt.cm.Greens(
                                float(element)),
                            keep_alpha=True)).strip('#').replace(
                        'ff',
                        'f') + '}\\color{red}' + str(
                        round(
                            float(element),
                            2)) + ' & '
                elif float(element) > 0.75:
                    this_row = this_row + '\\cellcolor[HTML]{' + str(
                        matplotlib.colors.to_hex(
                            plt.cm.Greens(
                                float(element)),
                            keep_alpha=True)).strip('#').replace(
                        'ff',
                        'f') + '}\\color{white}' + str(
                        round(
                            float(element),
                            2)) + ' & '
                else:
                    this_row = this_row + '\\cellcolor[HTML]{' + str(
                        matplotlib.colors.to_hex(
                            plt.cm.Greens(
                                float(element)),
                            keep_alpha=True)).strip('#').replace(
                        'ff',
                        'f') + '}\\color{black}' + str(
                        round(
                            float(element),
                            2)) + ' & '
        rows.append(this_row)

    # Start writing on the .tex file:

    latex_table = open(str(path) + 'temp.tex', 'w')

    tab_type = ''
    for i in np.arange(Ncols):
        if i == 0:
            tab_type = tab_type + 'l'
        else:
            tab_type = tab_type + 'r'

    chain = '\\begin{table} \n'
    latex_table.write(chain)
    chain = '{\\centering \n'
    latex_table.write(chain)
    chain = '\\small \n'
    latex_table.write(chain)
    chain = '\\begin{tabular}{' + str(tab_type) + '} \n'
    latex_table.write(chain)
    chain = '\\hline \n'
    latex_table.write(chain)

    # Writing header:
    chain = header
    latex_table.write(chain)

    # Now the rest of rows with the correct format
    for row in rows:
        chain = row[0:-2] + ' \\\\ '
        latex_table.write(chain)

    chain = '\\hline \\hline \n'
    latex_table.write(chain)
    chain = '\\end{tabular} \n'
    latex_table.write(chain)
    chain = '\\caption{' + str(caption) + '} \n'
    latex_table.write(chain)
    chain = '} \n'
    latex_table.write(chain)
    chain = '\\end{table} \n'
    latex_table.write(chain)
    latex_table.close()

    h = open(str(path) + 'temp.tex', "r")
    file_contents = h.read()
    print(file_contents)
    h.close()

    os.system('rm ' + str(path) + 'temp.tex')


# %% [markdown]
# # Preparing catalogue

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

dust_map = 'l18'
ext_rec = 'bab18'
# new_parallax in 'mas', 1/parallax --> distance in kpc.
df_clusters.add_virtual_column('new_parallax', 'parallax+0.017')
# new_parallax in 'mas', 1/parallax --> distance in kpc.
df_clusters.add_virtual_column('distance', '1.0/new_parallax')
E_BP_RP, A_G, COL_bp_rp, M_G = reddening_computation(df_clusters)
df_clusters.add_virtual_columns_cartesian_velocities_to_polar(
    x='(x-8.20)',
    y='(y)',
    vx='(vx)',
    vy='(vy)',
    vr_out='_vR',
    vazimuth_out='_vphi',
    propagate_uncertainties=False)
df_clusters.add_virtual_column('vphi', '-_vphi')  # flip sign
df_clusters.add_virtual_column('vR', '-_vR')  # flip sign
df_clusters.add_column('E_BP_RP', E_BP_RP)
df_clusters.add_column('A_G', A_G)
df_clusters.add_column('COL_bp_rp', COL_bp_rp)
df_clusters.add_column('M_G', M_G)
df_clusters.add_column('err_MG', 2.17 / df_clusters.evaluate('parallax_over_error'))
df_clusters.add_column('err_COL_bp_rp', 1.0857 *
                       np.sqrt((df_clusters.evaluate('phot_bp_mean_flux_error') /
                                df_clusters.evaluate('phot_bp_mean_flux'))**2 +
                               (df_clusters.evaluate('phot_rp_mean_flux_error') /
                                df_clusters.evaluate('phot_rp_mean_flux'))**2))  # Coming from error propagation theory (derivative).

df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')  # no segue within quality.

df_clusters

# %% [markdown]
# # Substructure A

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='A1', main=[34, 28, 32])

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='A2', main=[31, 15, 16])

# %%
create_tables_adding_extra_members_CaMD(df=df_clusters, path=path, tab_name='A', main_names=['A1', 'A2'], mains=[
                                        [34, 32], [31, 15, 16]], extra_clusters=[28, 33, 12, 40, 42, 46, 38, 18, 30])

# %%
create_tables_defining_CaMD(
    df=df_clusters,
    path=path,
    name='A',
    main=[
        28,
        33,
        12,
        40,
        34,
        28,
        32,
        31,
        15,
        16,
        42,
        38,
        18,
         30])

# %%
create_tables_adding_extra_members_CaMD(df=df_clusters, path=path, tab_name='A_def', main_names=['A'], mains=[
                                        [15, 16, 46, 31, 32, 30, 42, 18, 33, 34]], extra_clusters=[12, 38, 28, 40])

# %% [markdown]
# ## Conclusion A:
#
# The conclusion is similar as for the MDF... 'A' forms a single structure
# putting together all clusters. In this case, the only one standing out
# would be cluster #12 (which indeed shows a quite low p-value as well in
# terms of MDFs...

# %%
compare_MDFs_CMDs(df_clusters, clusters_CM=[[34, 28, 32, 31, 15, 16, 33, 40, 42, 46, 38, 18, 30], [12], [
                  38]], labels=['A', '12', '38'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1, 2, 3])

# %%
compare_MDFs_CMDs(df_clusters, clusters_CM=[[34, 28, 32, 31, 15, 16, 33, 40, 42, 46, 38, 18, 30], [12], [
                  38]], labels=['A', '12', '38'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[2, 1, 3])

# %% [markdown]
# # Thamnos (substructure B)

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='B', main=[37, 8, 11])

# %% [markdown]
# ## Conclusion B:
#
# Everything seems compatible in MDF... but the point here is that the
# differences in MDF are not huge either. Given that in the end, comparing
# CaMDs with this number of stars we lose some 'separation' power... it is
# normal that we find this. So, let's keep Thamnos1 and Thamnos2 divided.

# %% [markdown]
# # GE (substructure C)

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='C3a', main=[57, 58])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C3b', main=[50, 53])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C3c', main=[47, 44, 51])

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='C1', main=[56, 59])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C2', main=[10, 7, 13])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C3', main=[57, 58, 54, 55, 50, 53, 47, 44, 51])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C4', main=[48, 45, 25, 29, 41, 49])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C5', main=[27, 22, 26])
create_tables_defining_CaMD(df=df_clusters, path=path, name='C6', main=[35, 9, 24])

# %%
create_tables_adding_extra_members_CaMD(
    df=df_clusters, path=path, tab_name='C', main_names=[
        'C1', 'C2', 'C3', 'C4', 'C5'], mains=[
            [
                56, 59], [
                    10, 7, 13], [
                        57, 58, 54, 50, 53, 47, 44, 51], [
                            48, 45, 25, 29, 41, 49], [
                                27, 22, 26]], extra_clusters=[
                                    6, 52, 55, 14, 20, 19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24])

# %% [markdown]
# ## Conclusion C
#
# C1 to C6 display compatible CaMDs.
#
# Regarding addition of aparently independent clusters... all compatible as well.
#
# I would suggest, GE, 6 and 14.

# %%

# %% [markdown]
# # Sequoia (substructure D):

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='D', main=[64, 63, 62])

# %% [markdown]
# ## Conclusion D
#
# All compatible, the same as using the MDF... and finally, and based on
# IoM, the three could be separated or not.

# %%

# %% [markdown]
# # Helmi (Substructure E)

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='E', main=[60, 61])

# %% [markdown]
# ## Conclusion E:
#
# Same as MDF... the two belong together as part of the Helmi streams.

# %%

# %% [markdown]
# # High-energy blob (F):

# %%
create_tables_defining_CaMD(df=df_clusters, path=path, name='F', main=[65, 66])

# %% [markdown]
# ## Conclusions F
#
# Different CaMD as well... and same problem, low number of stars here as well... So, no idea...

# %%

# %% [markdown]
# # Tables in latex format:

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
name = 'B'
table_type = '_definition_table_CaMD.csv'
nice_table_from_csv(path, name, table_type)

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
name = 'A_def'
table_type = '_addition_table_CaMD.csv'
nice_table_from_csv(path, name, table_type)

# %%


# %%

# %%

# %%

# %%

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

dust_map = 'l18'
ext_rec = 'bab18'
# new_parallax in 'mas', 1/parallax --> distance in kpc.
df_clusters.add_virtual_column('new_parallax', 'parallax+0.017')
# new_parallax in 'mas', 1/parallax --> distance in kpc.
df_clusters.add_virtual_column('distance', '1.0/new_parallax')
E_BP_RP, A_G, COL_bp_rp, M_G = reddening_computation(df_clusters)
df_clusters.add_virtual_columns_cartesian_velocities_to_polar(
    x='(x-8.20)',
    y='(y)',
    vx='(vx)',
    vy='(vy)',
    vr_out='_vR',
    vazimuth_out='_vphi',
    propagate_uncertainties=False)
df_clusters.add_virtual_column('vphi', '-_vphi')  # flip sign
df_clusters.add_virtual_column('vR', '-_vR')  # flip sign
df_clusters.add_column('E_BP_RP', E_BP_RP)
df_clusters.add_column('A_G', A_G)
df_clusters.add_column('COL_bp_rp', COL_bp_rp)
df_clusters.add_column('M_G', M_G)
df_clusters.add_column('err_MG', 2.17 / df_clusters.evaluate('parallax_over_error'))
df_clusters.add_column('err_COL_bp_rp', 1.0857 *
                       np.sqrt((df_clusters.evaluate('phot_bp_mean_flux_error') /
                                df_clusters.evaluate('phot_bp_mean_flux'))**2 +
                               (df_clusters.evaluate('phot_rp_mean_flux_error') /
                                df_clusters.evaluate('phot_rp_mean_flux'))**2))  # Coming from error propagation theory (derivative).

df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')  # no segue within quality.

df_clusters

# %%
compare_distributions_CaMD(df=df_clusters, path=path, main_names=['A1', 'A2'], mains=[
                           [34, 32], [31, 15, 16]], tab_name='substructures_CaMD.csv')
df_clusters

# %%
name = 'substructures'
table_type = '_CaMD.csv'
nice_table_from_csv(path, name, table_type)


# %%
def compare_distributions_CaMD(df, path, main_names, mains, tab_name):

    iso_dict = {}
    ages_iso, mets_iso = [], []
    names_isos = glob.glob('/data/users/ruizlara/isochrones/alpha_enhanced/*.isc_gaia-dr3')
    for i, name_iso in enumerate(names_isos):
        lines_iso = open(names_isos[i], 'r').readlines()
        iso_age, iso_mh = lines_iso[4].split('=')[5].split()[0], lines_iso[4].split('=')[2].split()[0]
        ages_iso.append(iso_age)
        mets_iso.append(float(iso_mh))
        iso_dict['age' + str(iso_age) + '-met' + str(iso_mh)] = {'name_iso': name_iso}

    newfile = open(path + tab_name, 'w')
    chain_head = ' , '

    for main_name, main in zip(main_names, mains):
        print(main_name)
        print(main)
        best_isochrone, df = apply_CaMD_fitting(
            df=df, clusters=main, path_CMD=path, name=main_name, iso_dict=iso_dict, mah_dist_lim=2.13)
        chain_head = chain_head + main_name + ' , '
    chain_head = chain_head[0:-2] + ' \n'
    print(chain_head)
    newfile.write(chain_head)

    for main_name1, main1 in zip(main_names, mains):
        chain = main_name1 + ' , '
        for main_name2, main2 in zip(main_names, mains):
            pval = compare_CMDs(
                df, clusters_CM=[
                    main1, main2], labels=[
                    main_name1, main_name2], colours=[
                    'blue', 'red'], zorder=[
                    0, 1])
            chain = chain + str(round(pval, 2)) + ' , '
        chain = chain[0:-2] + ' \n'
        print(chain)
        newfile.write(chain)

    newfile.close()


# %%
def compare_CMDs(df, clusters_CM, labels, colours, zorder):

    # Compare the MDF of two structures:
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = str()
        for cluster in clusters[:-1]:
            condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
        condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
        df.select(condition, name='structure')
        distr_dict[str(system)] = {'colour_distr': df.evaluate('iso_dist_col_' + system, selection='(structure)'),
                                   'mag': df.evaluate('M_G', selection='(structure)'),
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)'),
                                   'en': df.evaluate('En', selection='(structure)') / 10**4,
                                   'lz': df.evaluate('Lz', selection='(structure)') / 10**3,
                                   'lperp': df.evaluate('Lperp', selection='(structure)') / 10**3}

    for system, clusters in zip(labels, clusters_CM):
        print(system)
        print(str(len(distr_dict[system]['colour_distr'])) + ' good stars with colour')

    f0, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(18, 11), facecolor='w')
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    ax6 = plt.subplot(236)

    ax4.scatter(df.evaluate('Lz') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(df.evaluate('Lz') / (10**3), df.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)

    for i in np.arange(len(labels)):
        ax1.hist(distr_dict[labels[i]]['colour_distr'], histtype='step', range=(-1.5, 1.5),
                 bins=25, density=True, color=colours[i], lw=2, label=labels[i])
        ax2.hist(distr_dict[labels[i]]['colour_distr'], histtype='step', range=(-1.5, 1.5),
                 bins=25, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        ax3.plot(distr_dict[labels[i]]['col'], distr_dict[labels[i]]
                 ['mag'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax4.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]
                 ['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax5.plot(distr_dict[labels[i]]['lperp'], distr_dict[labels[i]]
                 ['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax6.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]
                 ['lperp'], 'o', color=colours[i], zorder=zorder[i], ms=3)

    indexes = np.arange(len(clusters_CM))
    pairs = list(itertools.combinations(indexes, 2))
    count = 0
    for pair in pairs:
        feh_x, feh_y = distr_dict[labels[pair[0]]
                                  ]['colour_distr'], distr_dict[labels[pair[1]]]['colour_distr']
        feh_x = feh_x[np.isnan(feh_x) == False]
        feh_y = feh_y[np.isnan(feh_y) == False]
        if ((len(feh_x) > 10) & (len(feh_y) > 10)):
            KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
        else:
            KS, pval = 999, 999

        ax2.text(0.95,
                 0.05 + 0.05 * count,
                 str(labels[pair[0]]) + '-' + str(labels[pair[1]]) + ': ' + str(round(pval,
                                                                                      3)),
                 horizontalalignment='right',
                 verticalalignment='center',
                 transform=ax2.transAxes,
                 fontsize=20)
        count = count + 1

    ax1.legend()
    ax1.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
    ax1.set_ylabel(r'$\rm Normalised \; N$', fontsize=20)
    ax1.tick_params(labelsize=20)

    ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
    ax2.set_ylabel(r'$\rm Cumulative \; Distribution$', fontsize=20)
    ax2.tick_params(labelsize=20)

    ax3.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax3.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax3.set_xlim(-0.2, 3.0)
    ax3.set_ylim(9.0, -4.0)
    ax3.tick_params(labelsize=20)

    ax4.set_ylabel(r'$En/10^4$', fontsize=20)
    ax4.set_xlabel(r'$Lz/10^3$', fontsize=20)
    ax4.tick_params(labelsize=20)

    ax5.set_ylabel(r'$En/10^4$', fontsize=20)
    ax5.set_xlabel(r'$Lperp/10^3$', fontsize=20)
    ax5.tick_params(labelsize=20)

    ax6.set_ylabel(r'$Lperp/10^3$', fontsize=20)
    ax6.set_xlabel(r'$Lz/10^3$', fontsize=20)
    ax6.tick_params(labelsize=20)

    plt.tight_layout()

    plt.show()

    return pval


# %%
main_names = ['a', 'b', 'c']
mains = [10, 20, 30]
for i, (main_name1, main1) in enumerate(zip(main_names, mains)):
    print(i, main_name1, main1)

# %%
systems = [
    'Helmi',
    '64',
    'Sequoia',
    '62',
    'Thamnos1',
    'Thamnos2',
    '14',
    'GE',
    '6',
    '3',
    'A',
    '12',
    '28',
    '38']
grouped_clusters_systs = [[60,
                           61],
                          [64],
                          [63],
                          [62],
                          [37],
                          [8,
                           11],
                          [14],
                          [56,
                           59,
                           10,
                           7,
                           13,
                           57,
                           58,
                           54,
                           50,
                           53,
                           47,
                           44,
                           51,
                           48,
                           45,
                           25,
                           29,
                           41,
                           49,
                           27,
                           22,
                           26,
                           52,
                           55,
                           20,
                           19,
                           17,
                           43,
                           39,
                           36,
                           23,
                           5,
                           21,
                           35,
                           9,
                           24],
                          [6],
                          [3],
                          [15,
                           33,
                           34,
                           32,
                           31,
                           16,
                           42,
                           46,
                           18,
                           30],
                          [12],
                          [28],
                          [38]]
compare_distributions_CaMD(
    df=df_clusters,
    path=path,
    main_names=systems,
    mains=grouped_clusters_systs,
    tab_name='substructures_ALL_CaMD.csv')
df_clusters

# %%
name = 'substructures_ALL'
table_type = '_CaMD.csv'
nice_table_from_csv(path, name, table_type)

# %%
