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
# # Definitions

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
def make_IoM_vels_plot_def_cluster_id(df, clusters, alpha, name):

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df.select(condition, name='structure')

    print(condition)
    f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,
         ax12) = plt.subplots(12, figsize=(30, 30), facecolor='w')

    ax1 = plt.subplot(4, 3, 1)
    ax2 = plt.subplot(4, 3, 2)
    ax3 = plt.subplot(4, 3, 3)
    ax4 = plt.subplot(4, 3, 4)
    ax5 = plt.subplot(4, 3, 5)
    ax6 = plt.subplot(4, 3, 6)
    ax7 = plt.subplot(4, 3, 7)
    ax8 = plt.subplot(4, 3, 8)
    ax9 = plt.subplot(4, 3, 9)
    ax10 = plt.subplot(4, 3, 10)
    ax11 = plt.subplot(4, 3, 11)
    ax12 = plt.subplot(4, 3, 12)

    ax1.scatter(df.evaluate('Lz') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax2.scatter(df.evaluate('Lz') / (10**3), df.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)
    ax3.scatter(df.evaluate('vR'), df.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
    ax4.scatter(df.evaluate('vz'), df.evaluate('vR'), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('vz'), df.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(
        df.evaluate('vphi'),
        np.sqrt(
            df.evaluate('vR')**2 +
            df.evaluate('vz')**2),
        c='grey',
        s=1.0,
        alpha=0.2)
    ax7.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    #ax8.scatter(df.evaluate('Lz')/(10**3), df.evaluate('Ltotal')/(10**3), c='grey', s=1.0, alpha=0.2)
    ax9.scatter(df.evaluate('En') / (10**4), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax10.scatter(df.evaluate('Lz') / (10**3), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax11.scatter(df.evaluate('Ltotal') / (10**3), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax12.scatter(df.evaluate('l'), df.evaluate('b'), c='grey', s=1.0, alpha=0.2)

    scatter1 = ax1.scatter(df.evaluate('Lz',
                                       selection='structure') / (10**3),
                           df.evaluate('En',
                                       selection='structure') / (10**4),
                           c=df.evaluate('def_cluster_id',
                                         selection='structure'),
                           s=18.0,
                           alpha=alpha,
                           cmap='jet')
    f0.colorbar(scatter1, ax=ax1, label=r'$Cluster$', pad=0.0)
    scatter1 = ax2.scatter(df.evaluate('Lz',
                                       selection='structure') / (10**3),
                           df.evaluate('Lperp',
                                       selection='structure') / (10**3),
                           c=df.evaluate('def_cluster_id',
                                         selection='structure'),
                           s=18.0,
                           alpha=alpha,
                           cmap='jet')
    f0.colorbar(scatter1, ax=ax2, label=r'$Cluster$', pad=0.0)
    scatter1 = ax3.scatter(
        df.evaluate(
            'vR', selection='structure'), df.evaluate(
            'vphi', selection='structure'), c=df.evaluate(
                'def_cluster_id', selection='structure'), s=18.0, alpha=alpha, cmap='jet')
    f0.colorbar(scatter1, ax=ax3, label=r'$Cluster$', pad=0.0)
    scatter1 = ax4.scatter(
        df.evaluate(
            'vz', selection='structure'), df.evaluate(
            'vR', selection='structure'), c=df.evaluate(
                'def_cluster_id', selection='structure'), s=18.0, alpha=alpha, cmap='jet')
    f0.colorbar(scatter1, ax=ax4, label=r'$Cluster$', pad=0.0)
    scatter1 = ax5.scatter(
        df.evaluate(
            'vz', selection='structure'), df.evaluate(
            'vphi', selection='structure'), c=df.evaluate(
                'def_cluster_id', selection='structure'), s=18.0, alpha=alpha, cmap='jet')
    f0.colorbar(scatter1, ax=ax5, label=r'$Cluster$', pad=0.0)
    scatter1 = ax6.scatter(
        df.evaluate(
            'vphi',
            selection='structure'),
        np.sqrt(
            df.evaluate(
                'vR',
                selection='structure')**2 +
            df.evaluate(
                'vz',
                selection='structure')**2),
        c=df.evaluate(
            'def_cluster_id',
            selection='structure'),
        s=18.0,
        alpha=alpha,
        cmap='jet')
    f0.colorbar(scatter1, ax=ax6, label=r'$Cluster$', pad=0.0)
    scatter1 = ax7.scatter(df.evaluate('Lperp',
                                       selection='structure') / (10**3),
                           df.evaluate('En',
                                       selection='structure') / (10**4),
                           c=df.evaluate('def_cluster_id',
                                         selection='structure'),
                           s=18.0,
                           alpha=alpha,
                           cmap='jet')
    f0.colorbar(scatter1, ax=ax7, label=r'$Cluster$', pad=0.0)
    #scatter1 = ax8.scatter(df.evaluate('Lz', selection='structure')/(10**3), df.evaluate('Ltotal', selection='structure')/(10**3), c=df.evaluate('def_cluster_id', selection='structure'), s=18.0, alpha=alpha, cmap='jet')
    #f0.colorbar(scatter1, ax=ax8, label=r'$Cluster$', pad=0.0)
    scatter1 = ax9.scatter(df.evaluate('En',
                                       selection='structure') / (10**4),
                           df.evaluate('circ',
                                       selection='structure'),
                           c=df.evaluate('def_cluster_id',
                                         selection='structure'),
                           s=18.0,
                           alpha=alpha,
                           cmap='jet')
    f0.colorbar(scatter1, ax=ax9, label=r'$Cluster$', pad=0.0)
    scatter1 = ax10.scatter(df.evaluate('Lz',
                                        selection='structure') / (10**3),
                            df.evaluate('circ',
                                        selection='structure'),
                            c=df.evaluate('def_cluster_id',
                                          selection='structure'),
                            s=18.0,
                            alpha=alpha,
                            cmap='jet')
    f0.colorbar(scatter1, ax=ax10, label=r'$Cluster$', pad=0.0)
    scatter1 = ax11.scatter(
        df.evaluate(
            'Ltotal',
            selection='structure') / (
            10**3),
        df.evaluate(
            'circ',
                selection='structure'),
        c=df.evaluate(
            'def_cluster_id',
            selection='structure'),
        s=18.0,
        alpha=alpha,
        cmap='jet')
    f0.colorbar(scatter1, ax=ax11, label=r'$Cluster$', pad=0.0)
    scatter1 = ax12.scatter(
        df.evaluate(
            'l', selection='structure'), df.evaluate(
            'b', selection='structure'), c=df.evaluate(
                'def_cluster_id', selection='structure'), s=32.0, alpha=0.3, cmap='jet')
    f0.colorbar(scatter1, ax=ax12, label=r'$Cluster$', pad=0.0)

    ids = []  # This is the first star belonging to each subcluster
    for i in clusters:
        for j in np.arange(len(df.evaluate('def_cluster_id', selection='structure'))):
            if df.evaluate('def_cluster_id', selection='structure')[j] == i:
                ids.append(j)
                break
    colors = scatter1.to_rgba(df.evaluate('def_cluster_id', selection='structure')[ids])

    '''
    # Create a set of inset Axes: these should fill the bounding box allocated to them.
    axx8 = plt.axes([0,0,1,8])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax8, [0.27,0.55,0.48,0.41]) #[left, bottom, width, height]
    axx8.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
    #mark_inset(axs[8], axx8, loc1=2, loc2=4, fc="none", ec='0.5')
    axx8.patch.set_alpha(0.0)
    axx8.set_xlabel('[Fe/H]')

    for i, cluster in enumerate(clusters):
        condition = '(def_cluster_id=='+str(cluster)
        df.select('(def_cluster_id=='+str(cluster)+')', name = 'this_cluster')
        axx8.hist(df.evaluate('feh_lamost_lrs', selection='this_cluster'), bins = 20, histtype='step', density=True, color=colors[i])
    '''

    for i, cluster in enumerate(clusters):
        condition = '(def_cluster_id==' + str(cluster)
        df.select('(def_cluster_id==' + str(cluster) + ')', name='this_cluster')
        if len(clusters) == 1:
            ax8.hist(
                df.evaluate(
                    'feh_lamost_lrs',
                    selection='this_cluster'),
                bins=15,
                histtype='step',
                density=True,
                color='red')
        else:
            ax8.hist(
                df.evaluate(
                    'feh_lamost_lrs',
                    selection='this_cluster'),
                bins=15,
                histtype='step',
                density=True,
                color=colors[i])

    ax8.set_xlabel('[Fe/H]', fontsize=25)

    ax1.set_ylabel(r'$En/10^4$', fontsize=25)
    ax1.set_xlabel(r'$Lz/10^3$', fontsize=25)
    ax2.set_ylabel(r'$Lperp/10^3$', fontsize=25)
    ax2.set_xlabel(r'$Lz/10^3$', fontsize=25)
    ax3.set_xlabel('vR', fontsize=25)
    ax3.set_ylabel('vphi', fontsize=25)
    ax4.set_xlabel('vz', fontsize=25)
    ax4.set_ylabel('vR', fontsize=25)
    ax5.set_xlabel('vz', fontsize=25)
    ax5.set_ylabel('vphi', fontsize=25)
    ax6.set_xlabel('vphi', fontsize=25)
    ax6.set_ylabel(r'$\rm \sqrt{vR^2+vz^2}$', fontsize=25)
    ax7.set_ylabel(r'$En/10^4$', fontsize=25)
    ax7.set_xlabel(r'$Lperp/10^3$', fontsize=25)
    #ax8.set_ylabel(r'$Ltotal/10^3$', fontsize=25)
    #ax8.set_xlabel(r'$Lz/10^3$', fontsize=25)
    ax9.set_xlabel(r'$En/10^4$', fontsize=25)
    ax9.set_ylabel(r'$circ$', fontsize=25)
    ax10.set_xlabel(r'$Lz/10^3$', fontsize=25)
    ax10.set_ylabel(r'$circ$', fontsize=25)
    ax11.set_xlabel(r'$Ltotal/10^3$', fontsize=25)
    ax11.set_ylabel(r'$circ$', fontsize=25)
    ax12.set_xlabel(r'$l$', fontsize=25)
    ax12.set_ylabel(r'$b$', fontsize=25)

    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)
    ax4.tick_params(labelsize=20)
    ax5.tick_params(labelsize=20)
    ax6.tick_params(labelsize=20)
    ax7.tick_params(labelsize=20)
    ax8.tick_params(labelsize=20)
    ax9.tick_params(labelsize=20)
    ax10.tick_params(labelsize=20)
    ax11.tick_params(labelsize=20)
    ax12.tick_params(labelsize=20)

    plt.tight_layout()

    # plt.savefig('/data/users/ruizlara/halo_clustering/plots_paper/'+str(name)+'.png')

    plt.show()


# %%
def compare_MDF_structures(df, structures_names, structures, path, name):

    # KS table NO cut in met
    df.select('(feh_lamost_lrs>-10000.0)', name='lamost')
    KSs_array_all, pvals_array_all = [], []
    columns, indexes = [], []
    for structure_y_name in structures_names:
        columns.append(str(structure_y_name))
    for structure_x_name, structure_x_clusters in zip(structures_names, structures):
        KSs_array, pvals_array = [], []
        indexes.append(structure_x_name)
        for structure_y_name, structure_y_clusters in zip(structures_names, structures):

            condition = str()
            for cluster in structure_x_clusters[:-1]:
                condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
            condition = condition + '(def_cluster_id==' + str(structure_x_clusters[-1]) + ')'
            df.select(condition, name='structure_x')

            condition = str()
            for cluster in structure_y_clusters[:-1]:
                condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
            condition = condition + '(def_cluster_id==' + str(structure_y_clusters[-1]) + ')'
            df.select(condition, name='structure_y')

            feh_x, feh_y = df.evaluate(
                'feh_lamost_lrs', selection='(structure_x)&(lamost)'), df.evaluate(
                'feh_lamost_lrs', selection='(structure_y)&(lamost)')
            feh_x = feh_x[np.isnan(feh_x) == False]
            feh_y = feh_y[np.isnan(feh_y) == False]
            if ((len(feh_x) > 10) & (len(feh_y) > 10)):
                KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
                KSs_array.append(KS)
                pvals_array.append(pval)
            else:
                KSs_array.append(1.01)
                pvals_array.append(1.01)
        KSs_array_all.append(KSs_array)
        pvals_array_all.append(pvals_array)
    pvals_df = pd.DataFrame(pvals_array_all, columns=columns, index=indexes)
    pvals_df.to_csv(path + name + '_met_table.csv')


# %%
# GOOD coordinate transformation:

# Definition from Helmer's paper (escape velocity paper)
def pcart_TRL(vl, vb, l, b, vlos, U=11.1, V=12.24 + 232.8, W=7.25):
    from numpy import sin, cos

    _l = np.deg2rad(l)
    _b = np.deg2rad(b)

    vlsun = -U * sin(_l) + V * cos(_l)
    vbsun = W * cos(_b) - sin(_b) * (U * cos(_l) + V * sin(_l))

    _vl = vl + vlsun
    _vb = vb + vbsun

    vlos_sun = W * sin(_b) + cos(_b) * (U * cos(_l) + V * sin(_l))

    _vlos = vlos + vlos_sun

    vx = _vlos * cos(_l) * cos(_b) - _vl * sin(_l) - _vb * cos(_l) * sin(_b)
    vy = _vlos * sin(_l) * cos(_b) + _vl * cos(_l) - _vb * sin(_l) * sin(_b)
    vz = _vlos * sin(_b) + _vb * cos(_b)

    return vx, vy, vz


# Updated coordinate transformation using Helmer's recipes (velocity escape paper).
def compute_coordinates_new(self):

    # Coordinates transformation and computation of everything for self
    # (radial velocity is zero for _vir and a measured vlos for normal):

    self.add_virtual_column('radial_velocity_zero', '0.0*ra')
    self.add_variable('k', 4.74057)
    k = 4.74057
    self.add_variable('R0', 8.20)
    R0 = 8.2
    self.add_variable('Z0', 0.0)
    Z0 = 0.0
    self.add_variable('vlsr', 232.8)
    vlsr = 232.8
    self.add_variable('_U', 11.1)
    _U = 11.1
    self.add_variable('_V', 12.24)
    _V = 12.24
    self.add_variable('_W', 7.25)
    _W = 7.25

    # new_parallax in 'mas', 1/parallax --> distance in kpc.
    self.add_column('new_parallax', self.evaluate('parallax') + 0.017)
    # new_parallax in 'mas', 1/parallax --> distance in kpc.
    self.add_column('distance', 1.0 / self.evaluate('new_parallax'))

    # Adding Cartesian coordinates (tilde X, Y, Z), centred at the Sun
    self.add_virtual_columns_spherical_to_cartesian(
        alpha='l',
        delta='b',
        distance='distance',
        xname='tildex',
        yname='tildey',
        zname='tildez')

    # Adding Cylindrical (polar) coordinates, with the Sun at the centre. --> tilde Rho, tilde Phi
    self.add_virtual_columns_cartesian_to_polar(
        x='(tildex)',
        y='(tildey)',
        radius_out='tildeRho',
        azimuth_out='tildePhi',
        radians=True)

    # Adding Cartesian coordinates centred at the galactic centre. --> xyz
    self.add_column('x', self.evaluate('tildex-R0'))
    self.add_column('y', self.evaluate('tildey'))
    self.add_column('z', self.evaluate('tildez') + Z0)

    # Adding Cylindrical (polar) coordinates, with the Galactic centre at the centre. --> Rho, Phi
    self.add_virtual_columns_cartesian_to_polar(
        x='(x)', y='(y)', radius_out='Rho', azimuth_out='Phi', radians=True)

    # Adding Spherical coordinates, with the Galactic centre at the centre. alpha, delta, sphR
    self.add_virtual_columns_cartesian_to_spherical(
        x='(x)',
        y='(y)',
        z='z',
        alpha='alpha',
        delta='delta',
        distance='sphR',
        radians=True)

    # Equatorial proper motions to galactic proper motions:
    self.add_virtual_columns_proper_motion_eq2gal(
        pm_long='pmra', pm_lat='pmdec', pm_long_out='pml', pm_lat_out='pmb')

    # Computing vl and vb (linear)
    vl = k * self.evaluate('pml') * self.evaluate('distance')
    vb = k * self.evaluate('pmb') * self.evaluate('distance')
    self.add_column('vl', vl)  # distance in kpc; mu in mas/yr
    self.add_column('vb', vb)

    # computing _vir, as if no radial velocity.
    vir_vx, vir_vy, vir_vz = pcart_TRL(self.evaluate('vl'), self.evaluate(
        'vb'), self.evaluate('l'), self.evaluate('b'), self.evaluate('radial_velocity_zero'))
    self.add_column('vir_vx', vir_vx)
    self.add_column('vir_vy', vir_vy)
    self.add_column('vir_vz', vir_vz)

    self.add_virtual_columns_cartesian_velocities_to_polar(
        x='(x)',
        y='(y)',
        vx='(vir_vx)',
        vy='(vir_vy)',
        vr_out='vir_vRho',
        vazimuth_out='_vir_vPhi',
        propagate_uncertainties=False)
    self.add_virtual_column('vir_vPhi', '-_vir_vPhi')  # flip sign

    # Energies and so on using Helmer's code
    En, Lx, Ly, Lz, Lperp = Potential.En_L_calc(self.evaluate('x'), self.evaluate('y'), self.evaluate(
        'z'), self.evaluate('vir_vx'), self.evaluate('vir_vy'), self.evaluate('vir_vz'))
    Lz = -Lz  # As always, change of sign according to Koppelman's papers.
    self.add_column('vir_En_99', En)
    self.add_column('vir_Lx', Lx)
    self.add_column('vir_Ly', Ly)
    self.add_column('vir_Lz', Lz)
    self.add_column('vir_Lperp', Lperp)  # sqrt(Lx**2+Ly**2)
    self.add_column(
        'vir_Ltot',
        np.sqrt(
            self.evaluate('vir_Lx')**2. +
            self.evaluate('vir_Ly')**2. +
            self.evaluate('vir_Lz')**2.))

    self.add_column('vir_vtoomre', np.sqrt(self.evaluate('vir_vx')**2. + self.evaluate('vir_vz')**2.))

    # computing normal velocities, using the value of radial velocity if one exists
    vx, vy, vz = pcart_TRL(self.evaluate('vl'), self.evaluate(
        'vb'), self.evaluate('l'), self.evaluate('b'), self.evaluate('any_vlos'))
    self.add_column('vx', vx)
    self.add_column('vy', vy)
    self.add_column('vz', vz)

    self.add_virtual_columns_cartesian_velocities_to_polar(
        x='(x)',
        y='(y)',
        vx='(vx)',
        vy='(vy)',
        vr_out='vRho',
        vazimuth_out='_vPhi',
        propagate_uncertainties=False)
    self.add_virtual_column('vPhi', '-_vPhi')  # flip sign

    # Energies and so on using Helmer's code
    En, Lx, Ly, Lz, Lperp = Potential.En_L_calc(self.evaluate('x'), self.evaluate(
        'y'), self.evaluate('z'), self.evaluate('vx'), self.evaluate('vy'), self.evaluate('vz'))
    Lz = -Lz  # As always, change of sign according to Koppelman's papers.
    self.add_column('En_99', En)
    self.add_column('Lx', Lx)
    self.add_column('Ly', Ly)
    self.add_column('Lz_new', Lz)
    self.add_column('Lperp_new', Lperp)  # sqrt(Lx**2+Ly**2)
    self.add_column(
        'Ltotal',
        np.sqrt(
            self.evaluate('Lx')**2. +
            self.evaluate('Ly')**2. +
            self.evaluate('Lz')**2.))

    self.add_column('vtoomre', np.sqrt(self.evaluate('vx')**2. + self.evaluate('vz')**2.))

    self


# %%
def pval_red_or_green(val):
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s' % color


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
        #axx0.axvline(fit_param, ls='--', color='white')
        # axx0.axes.get_xaxis().set_ticks([])
        axx0.axes.get_yaxis().set_ticks([])
        axx0.spines['right'].set_visible(False)
        axx0.spines['top'].set_visible(False)
        axx0.spines['bottom'].set_color('white')
        axx0.spines['left'].set_color('white')
        axx0.tick_params(axis='x', colors='white')
        #ax1.text(0.45, 0.02, r'$\rm d_2 = '+str(fit_param.round(3))+' \; ('+str(round(fit_param_perc, 2))+'\%)$', transform=ax1.transAxes, color='white', fontsize=15)
        ax1.text(0.45, 0.02, r'$\rm d_2 = ' + str(fit_param.round(3)) +
                 '$', transform=ax1.transAxes, color='white', fontsize=15)

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
def substr_IoM_paper_GOOD(df, clusters, name, colour):

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df.select(condition, name='structure1')

    f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, figsize=(20, 18.5))

    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)
    ax9 = plt.subplot(3, 3, 9)

    ax1.scatter(df.evaluate('Lz') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax2.scatter(df.evaluate('circ'), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax3.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax4.scatter(df.evaluate('Lz') / (10**3), df.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('l'), df.evaluate('b'), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(df.evaluate('vy'), df.evaluate('vtoomre'), c='grey', s=1.0, alpha=0.2)
    ax7.scatter(df.evaluate('vR'), df.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
    ax8.scatter(df.evaluate('vz'), df.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
    ax9.scatter(df.evaluate('vz'), df.evaluate('vR'), c='grey', s=1.0, alpha=0.2)

    colour = df.evaluate('def_cluster_id', selection='structure1')

    scatter1 = ax1.scatter(df.evaluate('Lz', selection='structure1') / (10**3),
                           df.evaluate('En', selection='structure1') / (10**4), c=colour, s=12.0)
    scatter1 = ax2.scatter(df.evaluate('circ', selection='structure1'), df.evaluate(
        'En', selection='structure1') / (10**4), c=colour, s=12.0)
    scatter1 = ax3.scatter(df.evaluate('Lperp', selection='structure1') / (10**3),
                           df.evaluate('En', selection='structure1') / (10**4), c=colour, s=12.0)
    scatter1 = ax4.scatter(df.evaluate('Lz', selection='structure1') / (10**3),
                           df.evaluate('Lperp', selection='structure1') / (10**3), c=colour, s=12.0)
    scatter1 = ax5.scatter(
        df.evaluate(
            'l',
            selection='structure1'),
        df.evaluate(
            'b',
            selection='structure1'),
        c=colour,
        s=32.0,
        alpha=0.4)
    scatter1 = ax6.scatter(
        df.evaluate(
            'vy', selection='structure1'), df.evaluate(
            'vtoomre', selection='structure1'), c=colour, s=12.0)
    scatter1 = ax7.scatter(
        df.evaluate(
            'vR', selection='structure1'), df.evaluate(
            'vphi', selection='structure1'), c=colour, s=12.0)
    scatter1 = ax8.scatter(
        df.evaluate(
            'vz', selection='structure1'), df.evaluate(
            'vphi', selection='structure1'), c=colour, s=12.0)
    scatter1 = ax9.scatter(
        df.evaluate(
            'vz', selection='structure1'), df.evaluate(
            'vR', selection='structure1'), c=colour, s=12.0)

    ax1.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=25)
    ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=25)
    ax2.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=25)
    ax2.set_xlabel(r'$\eta$', fontsize=25)
    ax3.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=25)
    ax3.set_xlabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=25)
    ax4.set_ylabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=25)
    ax4.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=25)
    ax5.set_xlabel(r'$l \; [degrees]$', fontsize=25)
    ax5.set_ylabel(r'$b \; [degrees]$', fontsize=25)
    ax6.set_xlabel(r'$v_{y} \; [km/s]$', fontsize=25)
    ax6.set_ylabel(r'$\rm \sqrt{v_{x}^2+v_{z}^2} \; [km/s]$', fontsize=25)
    ax7.set_xlabel(r'$v_{R} \; [km/s]$', fontsize=25)
    ax7.set_ylabel(r'$v_{\phi} \; [km/s]$', fontsize=25)
    ax8.set_xlabel(r'$v_{z} \; [km/s]$', fontsize=25)
    ax8.set_ylabel(r'$v_{\phi} \; [km/s]$', fontsize=25)
    ax9.set_xlabel(r'$v_{z} \; [km/s]$', fontsize=25)
    ax9.set_ylabel(r'$v_{R} \; [km/s]$', fontsize=25)

    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax3.tick_params(labelsize=20)
    ax4.tick_params(labelsize=20)
    ax5.tick_params(labelsize=20)
    ax6.tick_params(labelsize=20)
    ax7.tick_params(labelsize=20)
    ax8.tick_params(labelsize=20)
    ax9.tick_params(labelsize=20)

    plt.tight_layout()

    plt.savefig(
        '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D_80/plots_appendix/subs_' +
        str(name) +
        '_2.png')

    plt.show()


# %%
def detailed_CaMD_comparison(df, path, main_names, mains, extra_clusters):

    chain1 = ','
    for cluster in extra_clusters:
        df.select('(def_cluster_id==' + str(cluster) + ')', name='this_cluster')

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

        for cluster in extra_clusters:
            coords = np.where(np.array(main) != cluster)
            orig_pval = create_CaMD_comp_plot(
                df, clusters_CM=[
                    np.array(main)[coords], [cluster]], labels=[
                    str(main_name), str(cluster)], colours=[
                    'red', 'blue'], zorder=[
                    1, 2], name_step2=main_name, best_isochrone=best_isochrone)
            print(str(main_name) + '-' + str(cluster) + ' p-value is ' + str(orig_pval))


def create_CaMD_comp_plot(df, clusters_CM, labels, colours, zorder, name_step2, best_isochrone):

    # Compare the colour-distribution of two structures:
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = str()
        for cluster in clusters[:-1]:
            condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
        condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
        df.select(condition, name='structure')
        distr_dict[str(system)] = {'colour_distr': df.evaluate('iso_dist_col_' + name_step2, selection='(structure)&(quality)&(good_CaMD)'),
                                   'mag': df.evaluate('M_G', selection='(structure)&(quality)&(good_CaMD)'),
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)&(quality)&(good_CaMD)'),
                                   'en': df.evaluate('En', selection='(structure)&(quality)&(good_CaMD)') / 10**4,
                                   'lz': df.evaluate('Lz', selection='(structure)&(quality)&(good_CaMD)') / 10**3,
                                   'lperp': df.evaluate('Lperp', selection='(structure)&(quality)&(good_CaMD)') / 10**3,
                                   'feh': df.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(good_CaMD)')}

    main_colour, second_colour = distr_dict[labels[0]]['colour_distr'], distr_dict[labels[1]]['colour_distr']
    main_colour = main_colour[np.isnan(main_colour) == False]
    second_colour = second_colour[np.isnan(second_colour) == False]

    f0, (ax1) = plt.subplots(1, figsize=(6.5, 7), facecolor='w')
    ax1 = plt.subplot(111)

    axx3 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.1, 0.585, 0.4, 0.4])  # [left, bottom, width, height]
    axx3.set_axes_locator(ip)
    #axx3.hist(df_clusters.evaluate('feh_lamost_lrs', selection='(HelmiA)'), histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colour_helmiA, lw=2, cumulative=True)
    #axx3.hist(df_clusters.evaluate('feh_lamost_lrs', selection='(HelmiB)'), histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colour_helmiB, alpha=0.7, lw=2, cumulative=True)
    # axx3.axes.get_xaxis().set_ticks([])
    #axx3.axes.get_yaxis().set_ticks([0.5, 1.0])
    axx3.set_xlabel(r'$\rm \Delta(G_{BP}-G_{RP})$', labelpad=-0.04, fontsize=14)
    axx3.set_ylabel(r'$\rm Normalised \; N$', labelpad=-0.04, fontsize=14)
    # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
    #mark_inset(ax3, axx3, loc1=2, loc2=4, fc="none", ec='0.5')
    # axx1.patch.set_alpha(0.0)

    for i in np.arange(len(labels)):
        axx3.hist(distr_dict[labels[i]]['colour_distr'], range=(-0.8, 0.8),
                  histtype='step', bins=35, density=True, color=colours[i], lw=2, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['colour_distr'], range=(-1.1, 1.1), histtype='step', bins=25, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['colour_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])
        ax1.plot(distr_dict[labels[i]]['col'], distr_dict[labels[i]]
                 ['mag'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        # best_isochrone
        iso = ascii.read(best_isochrone)
        MG, GBP, GRP = iso['col5'], iso['col6'], iso['col7']
        ax1.plot(GBP - GRP, MG, 'k-')

    ax1.plot(distr_dict[labels[1]]['col'][distr_dict[labels[1]]['feh'] < -
                                          1.4], distr_dict[labels[1]]['mag'][distr_dict[labels[1]]['feh'] < -
                                                                             1.4], 'x', color=colours[1], zorder=zorder[1], ms=10)

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
        orig_pval = pval
        count = count + 1

    ax1.text(
        0.05,
        0.05,
        r'$\rm \bf Cluster \; \#12$',
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes,
        fontsize=16,
        color=colours[1])
    ax1.text(
        0.05,
        0.1,
        r'$\rm \bf Substructure \; A$',
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes,
        fontsize=16,
        color=colours[0])

    ax1.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax1.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax1.set_xlim(0.0, 1.7)
    ax1.set_ylim(7.0, -3.7)
    ax1.tick_params(labelsize=20)

    plt.tight_layout()

    return orig_pval


# %% [markdown]
# # Science:

# %% [markdown]
# ## a) IoM characterisation of the six big structures

# %%
#del df_clusters
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

# Adding angular momemtum:

Lx_n = df_clusters.evaluate('y') * df_clusters.evaluate('vz') - \
    df_clusters.evaluate('z') * df_clusters.evaluate('vy')
Ly_n = df_clusters.evaluate('z') * df_clusters.evaluate('vx') - \
    df_clusters.evaluate('x') * df_clusters.evaluate('vz')
Lz_n = df_clusters.evaluate('x') * df_clusters.evaluate('vy') - \
    df_clusters.evaluate('y') * df_clusters.evaluate('vx')
Ltotal = np.sqrt(Lx_n**2. + Ly_n**2. + Lz_n**2.)

df_clusters.add_column('Lx_n', Lx_n)
df_clusters.add_column('Ly_n', Ly_n)
df_clusters.add_column('Lz_n', Lz_n)
df_clusters.add_column('Ltotal', Ltotal)

df_clusters

# %%
# Initial grouping, just for IoM plot
names_IoM_plot = ['A', 'B', 'C', 'D', 'E', 'F']
grouped_clusters_IoM = [[33,
                         12,
                         40,
                         34,
                         28,
                         32,
                         31,
                         15,
                         16,
                         42,
                         46,
                         38,
                         18,
                         30],
                        [37,
                         8,
                         11],
                        [6,
                         56,
                         59,
                         52,
                         14,
                         10,
                         7,
                         13,
                         20,
                         19,
                         57,
                         58,
                         54,
                         55,
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
                         17,
                         43,
                         39,
                         36,
                         23,
                         5,
                         27,
                         22,
                         26,
                         21,
                         35,
                         9,
                         24],
                        [64,
                         62,
                         63],
                        [60,
                         61],
                        [65,
                         66]]
individual_clusters = [1, 2, 3, 4, 68]
#colours = []
#colours.extend(cm.jet(np.linspace(0, 1, len(names_IoM_plot))))
colours = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02', '#8f00ff']
print(colours)

# %%
f0, ax = plt.subplots(8, figsize=(20, 10), facecolor='w')  # figsize (x,y)
axs = []
for i in range(2):  # Rows
    for j in range(4):  # Cols
        ax_aux = plt.subplot2grid((2, 4), (i, j))
        axs.append(ax_aux)

for j in np.arange(len(names_IoM_plot)):
    clusters = grouped_clusters_IoM[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    axs[j].scatter(df_clusters.evaluate('Lz') /
                   (10**3), df_clusters.evaluate('En') /
                   (10**4), c='grey', s=1.0, alpha=0.1)
    axs[j].scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                   df_clusters.evaluate('En', selection='(structure)') / (10**4), c=colours[j], s=14)
    axs[j].set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=20)
    axs[j].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)

    axs[j].set_title(names_IoM_plot[j], fontsize=20)

    axs[j].tick_params(labelsize=17)

# Independent clusters:
axs[-2].scatter(df_clusters.evaluate('Lz') / (10**3),
                df_clusters.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
axs[-2].set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=20)
axs[-2].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)
axs[-2].set_title(r'$\rm Independent \; Clusters$', fontsize=20)
axs[-2].tick_params(labelsize=17)
#mean, width, height, angle, color  = [ 3.9 ,  -2.8], 5.5, 1.2, 85, 'black'
#ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle = angle, lw=2, color=color, fill=False)
# axs[7].add_patch(ell)
# axs[7].autoscale()
axs[-2].set_xlim(-4.5, 4.5)

# Individual clusters:

ind_colours = []
ind_colours.extend(cm.tab20b(np.linspace(0, 1, len(individual_clusters))))

condition = str()
for i, cluster in enumerate(individual_clusters):
    df_clusters.select('def_cluster_id==' + str(cluster), name='this_cluster')

    axs[-2].plot(df_clusters.evaluate('Lz', selection='this_cluster') / (10**3),
                 df_clusters.evaluate('En', selection='this_cluster') / (10**4), 'o', ms=3, color=ind_colours[i])
    axs[-2].text(0.03,
                 0.95 - 0.06 * i,
                 r'$\rm \bf Cluster \; \#' + str(cluster) + '$',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=axs[-2].transAxes,
                 fontsize=16,
                 color=ind_colours[i])

axs[-1].axis('off')

plt.tight_layout(w_pad=1, h_pad=1)

plt.show()

# %%
f0, ax = plt.subplots(8, figsize=(20, 10), facecolor='w')  # figsize (x,y)
axs = []
for i in range(2):  # Rows
    for j in range(4):  # Cols
        ax_aux = plt.subplot2grid((2, 4), (i, j))
        axs.append(ax_aux)

for j in np.arange(len(names_IoM_plot)):
    clusters = grouped_clusters_IoM[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    axs[j].scatter(df_clusters.evaluate('Lz') /
                   (10**3), df_clusters.evaluate('Lperp') /
                   (10**3), c='grey', s=1.0, alpha=0.1)
    axs[j].scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                   df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=14)
    axs[j].set_ylabel(r'$L_\perp \; [10^3 \; kpc \; km/s]$', fontsize=20)
    axs[j].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)

    axs[j].set_title(names_IoM_plot[j], fontsize=20)

    axs[j].tick_params(labelsize=17)

# Independent clusters:
axs[-2].scatter(df_clusters.evaluate('Lz') / (10**3),
                df_clusters.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)
axs[-2].set_ylabel(r'$L_\perp \; [10^3 \; kpc \; km/s]$', fontsize=20)
axs[-2].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)
axs[-2].set_title(r'$\rm Independent \; Clusters$', fontsize=20)
axs[-2].tick_params(labelsize=17)
#mean, width, height, angle, color  = [ 3.9 ,  -2.8], 5.5, 1.2, 85, 'black'
#ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle = angle, lw=2, color=color, fill=False)
# axs[7].add_patch(ell)
# axs[7].autoscale()
axs[-2].set_xlim(-4.5, 4.5)

# Individual clusters:

ind_colours = []
ind_colours.extend(cm.tab20b(np.linspace(0, 1, len(individual_clusters))))

condition = str()
for i, cluster in enumerate(individual_clusters):
    df_clusters.select('def_cluster_id==' + str(cluster), name='this_cluster')

    axs[-2].plot(df_clusters.evaluate('Lz', selection='this_cluster') / (10**3),
                 df_clusters.evaluate('Lperp', selection='this_cluster') / (10**3), 'o', ms=3, color=ind_colours[i])
    axs[-2].text(0.03,
                 0.95 - 0.06 * i,
                 r'$\rm \bf Cluster \; \#' + str(cluster) + '$',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=axs[-2].transAxes,
                 fontsize=16,
                 color=ind_colours[i])

axs[-1].axis('off')

plt.tight_layout(w_pad=1, h_pad=1)

plt.show()

# %% [markdown]
# ## B) Counting stars in case I need to:

# %%
#del df_clusters
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')
df_clusters.select('feh_lamost_lrs>-20.0', name='lamost_good')

# %%
# Computing number of stars and average metallicity of the different structures:

names = ['1', '2', '4', '68']
grouped_clusters = [[1], [2], [4], [68]]

for name, clusters in zip(names, grouped_clusters):
    print(name)

    if len(clusters) == 1:
        df_clusters.select('(def_cluster_id==' + str(clusters[0]) + ')', name='structure')
    else:
        for cluster in clusters[:-1]:
            condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
        condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
        df_clusters.select(condition, name='structure')
        print(condition)

    Nlamost = df_clusters['source_id'].count(selection='(structure)&(quality)&(lamost_good)')
    Nlamost2 = df_clusters['source_id'].count(selection='(structure)&(lamost_good)')
    Ncluster = df_clusters['source_id'].count(selection='(structure)&(quality)')
    Ncluster_all = df_clusters['source_id'].count(selection='(structure)')

    plt.figure()
    plt.hist(df_clusters.evaluate('feh_lamost_lrs',
             selection='(structure)&(quality)&(lamost_good)'), range=(-2.2, 0.2), bins=40)
    plt.show()
    print('Structure ' + str(name))
    print(str(Ncluster_all) + ' stars in total')
    print(str(Ncluster) + ' stars (after quality cuts)')
    print(str(Nlamost) + ' stars (good with lamost LRS [Fe/H])')
    print(str(Nlamost2) + ' stars (all with lamost LRS [Fe/H])')
    print(df_clusters.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(lamost_good)'))
    print('Average metallicity')
    print(np.nanmean(df_clusters.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(lamost_good)')))
    print('std metallicity')
    print(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(lamost_good)')))

# %%
clusters_array = np.arange(1.0, 69.0, 1.0)
for name, clusters in zip(clusters_array, clusters_array):
    print(clusters.size)
    if clusters.size == 1:
        df_clusters.select('(def_cluster_id==' + str(clusters) + ')', name='structure')
    else:
        for cluster in clusters[:-1]:
            condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
        condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
        df_clusters.select(condition, name='structure')
        print(condition)

    Nlamost = df_clusters['source_id'].count(selection='(structure)&(quality)&(lamost_good)')
    Nlamost2 = df_clusters['source_id'].count(selection='(structure)&(lamost_good)')
    Ncluster = df_clusters['source_id'].count(selection='(structure)&(quality)')
    Ncluster_all = df_clusters['source_id'].count(selection='(structure)')

    plt.figure()
    plt.hist(df_clusters.evaluate('feh_lamost_lrs',
             selection='(structure)&(quality)&(lamost_good)'), range=(-2.2, 0.2), bins=40)
    plt.show()
    print('Cluster ' + str(name))
    print(str(Ncluster_all) + ' stars in total')
    print(str(Ncluster) + ' stars (after quality cuts)')
    print(str(Nlamost) + ' stars (good with lamost LRS [Fe/H])')
    print(str(Nlamost2) + ' stars (all with lamost LRS [Fe/H])')
    print(df_clusters.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(lamost_good)'))
    print('Average metallicity')
    print(np.nanmean(df_clusters.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(lamost_good)')))
    print('std metallicity')
    print(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='(structure)&(quality)&(lamost_good)')))

# %% [markdown]
# ## C) Final configuration of A:

# %%
#del df_clusters
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
# New plot, new C2:
grouped_clusters_C = [[15, 16, 46, 31, 32, 30, 42, 18, 33, 34], [12], [38], [28], [40]]
colours = ['blue', 'lightblue', 'red', 'magenta', 'lightgreen']
names = ['A', 'Cluster \\#12', 'Cluster \\#38', 'Cluster \\#28', 'Cluster \\#40']

f0, (ax1, ax2) = plt.subplots(2, figsize=(14, 6), facecolor='w')  # figsize (x,y)

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('Lperp') /
            (10**3), c='grey', s=1.0, alpha=0.2)

axx1 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.58, 0.55, 0.40, 0.43])  # [left, bottom, width, height]
axx1.set_axes_locator(ip)
axx1.scatter(df_clusters.evaluate('Lz') /
             (10**3), df_clusters.evaluate('Lperp') /
             (10**3), c='grey', s=1.0, alpha=0.2)
axx1.axes.get_xaxis().set_ticks([])
axx1.axes.get_yaxis().set_ticks([])
axx1.set_xlim(0.85, 1.82)
axx1.set_ylim(0.0, 1.0)
# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
mark_inset(ax1, axx1, loc1=2, loc2=4, fc="none", ec='0.0')
# axx1.patch.set_alpha(0.0)

for j in np.arange(len(grouped_clusters_C)):
    print(j)
    clusters = grouped_clusters_C[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    ax1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=3)
    if j == 1:
        axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                     df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=10)
    else:
        axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                     df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=3)

ax1.set_ylabel(r'$L_{\perp} \; [10^3 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=24)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=24)

ax1.tick_params(labelsize=19)

ax1.text(-4.5, 4.2, r'$\rm \bf A \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[0])
ax1.text(-4.5,
         3.9,
         r'$\rm \bf ' + str(names[1]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[1])
ax1.text(-4.5,
         3.6,
         r'$\rm \bf ' + str(names[2]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[2])
ax1.text(-4.5,
         3.3,
         r'$\rm \bf ' + str(names[3]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[3])
ax1.text(-4.5,
         3.0,
         r'$\rm \bf ' + str(names[4]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[4])


clusters_CM, labels, colours = [[15, 16, 46, 31, 32, 30, 42, 18, 33, 34], [12], [38], [28], [40]], [
    'A', 'Cluster \\#12', 'Cluster \\#38', 'Cluster \\#28', 'Cluster \\#40'], ['blue', 'lightblue', 'red', 'magenta', 'lightgreen']
# Compare the MDF of two structures:
distr_dict = {}
for system, clusters in zip(labels, clusters_CM):
    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')
    distr_dict[str(system)] = {'feh_distr': df_clusters.evaluate('feh_lamost_lrs', selection='(structure)')}

for i in np.arange(len(labels)):
    Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.5, 0.7), bins=24)
    error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
    ax2.errorbar(
        Cbin_centres,
        Chist /
        np.nanmax(Chist),
        yerr=error /
        np.nanmax(Chist),
        fmt='o',
        ecolor=colours[i],
        ms=0)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax2.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post')
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])

if len(clusters_CM) == 2:
    feh_x, feh_y = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    feh_x = feh_x[np.isnan(feh_x) == False]
    feh_y = feh_y[np.isnan(feh_y) == False]
    KS, pval = stats.ks_2samp(distr_dict[labels[0]]['feh_distr'],
                              distr_dict[labels[1]]['feh_distr'], mode='exact')
    #ax1.text(0.82, 0.83, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
    #ax2.text(0.2, 0.95, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
    print(pval)

# axx1.text(0.8, 0.6, '#418', horizontalalignment='center', verticalalignment='center', transform=axx1.transAxes, fontsize=20)
# axx1.text(0.7, 0.25, '#479', horizontalalignment='center',
# verticalalignment='center', transform=axx1.transAxes, fontsize=20)

# ax2.legend()
ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax2.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax2.tick_params(labelsize=20)

ax2.set_xlim(-2.2, 0.5)

plt.tight_layout(w_pad=2)

plt.show()


plt.tight_layout()

plt.show()

# %%
# New plot, new C2:
grouped_clusters_C = [[15, 16, 46, 31, 32, 30, 42, 18, 33, 34], [12], [38], [28], [40]]
colours = ['blue', 'lightblue', 'red', 'magenta', 'lightgreen']
names = ['A', 'Cluster \\#12', 'Cluster \\#38', 'Cluster \\#28', 'Cluster \\#40']

f0, (ax1, ax2) = plt.subplots(2, figsize=(14, 6), facecolor='w')  # figsize (x,y)

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('Lperp') /
            (10**3), c='grey', s=1.0, alpha=0.2)

axx1 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.58, 0.55, 0.40, 0.43])  # [left, bottom, width, height]
axx1.set_axes_locator(ip)
axx1.scatter(df_clusters.evaluate('Lz') /
             (10**3), df_clusters.evaluate('Lperp') /
             (10**3), c='grey', s=1.0, alpha=0.2)
axx1.axes.get_xaxis().set_ticks([])
axx1.axes.get_yaxis().set_ticks([])
axx1.set_xlim(0.85, 1.82)
axx1.set_ylim(0.0, 1.0)
# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
mark_inset(ax1, axx1, loc1=2, loc2=4, fc="none", ec='0.0')
# axx1.patch.set_alpha(0.0)

for j in np.arange(len(grouped_clusters_C)):
    print(j)
    clusters = grouped_clusters_C[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    ax1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=3)
    if j == 1:
        axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                     df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=10)
    else:
        axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                     df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=3)

ax1.set_ylabel(r'$L_{\perp} \; [10^3 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=24)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=24)

ax1.tick_params(labelsize=19)

ax1.text(-4.5, 4.2, r'$\rm \bf A \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[0])
ax1.text(-4.5,
         3.9,
         r'$\rm \bf ' + str(names[1]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[1])
ax1.text(-4.5,
         3.6,
         r'$\rm \bf ' + str(names[2]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[2])
ax1.text(-4.5,
         3.3,
         r'$\rm \bf ' + str(names[3]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[3])
ax1.text(-4.5,
         3.0,
         r'$\rm \bf ' + str(names[4]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[4])


clusters_CM, labels, colours = [[15, 16, 46, 31, 32, 30, 42, 18, 33, 34], [12], [38], [28], [40]], [
    'A', 'Cluster \\#12', 'Cluster \\#38', 'Cluster \\#28', 'Cluster \\#40'], ['blue', 'lightblue', 'red', 'magenta', 'lightgreen']
# Compare the MDF of two structures:
distr_dict = {}
for system, clusters in zip(labels, clusters_CM):
    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')
    distr_dict[str(system)] = {'feh_distr': df_clusters.evaluate('feh_lamost_lrs', selection='(structure)')}

for i in np.arange(len(labels)):
    Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.5, 0.7), bins=24)
    error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
    ax2.errorbar(
        Cbin_centres,
        Chist /
        np.nanmax(Chist),
        yerr=error /
        np.nanmax(Chist),
        fmt='o',
        ecolor=colours[i],
        ms=0)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax2.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post')
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])

if len(clusters_CM) == 2:
    feh_x, feh_y = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    feh_x = feh_x[np.isnan(feh_x) == False]
    feh_y = feh_y[np.isnan(feh_y) == False]
    KS, pval = stats.ks_2samp(distr_dict[labels[0]]['feh_distr'],
                              distr_dict[labels[1]]['feh_distr'], mode='exact')
    #ax1.text(0.82, 0.83, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
    #ax2.text(0.2, 0.95, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
    print(pval)

axx3 = plt.axes([0, 0, 1, 3])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax2, [0.1, 0.585, 0.4, 0.4])  # [left, bottom, width, height]
axx3.set_axes_locator(ip)
Chist, Cbin_edges = np.histogram(distr_dict['Cluster \\#12']['feh_distr'], range=(-2.3, 0.1), bins=20)
error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
axx3.errorbar(
    Cbin_centres,
    Chist /
    np.nanmax(Chist),
    yerr=error /
    np.nanmax(Chist),
    fmt='o',
    ecolor='black',
    ms=0)
Chist = np.append(Chist, 0)
Chist = np.append(0, Chist)
Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
axx3.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='black', lw=2, where='post')
axx3.text(-2.5, 1.15, r'$\rm Cluster \#12$', color='black')
# axx3.axes.get_xaxis().set_ticks([])
# axx3.axes.get_yaxis().set_ticks([])
axx3.set_xlabel(r'$\rm Fe/H$', labelpad=-0.00)
axx3.set_ylabel(r'$\rm N$', labelpad=-0.00)
axx3.patch.set_alpha(0.01)

# axx1.text(0.8, 0.6, '#418', horizontalalignment='center', verticalalignment='center', transform=axx1.transAxes, fontsize=20)
# axx1.text(0.7, 0.25, '#479', horizontalalignment='center',
# verticalalignment='center', transform=axx1.transAxes, fontsize=20)

# ax2.legend()
ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax2.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax2.tick_params(labelsize=20)

axx3.axvline(-1.4)

ax2.set_xlim(-2.2, 0.5)

plt.tight_layout(w_pad=2)

plt.show()


plt.tight_layout()

plt.show()

# %%
# Referee version:

# New plot, new C2:
grouped_clusters_C = [[15, 16, 46, 31, 32, 30, 42, 18, 33, 34], [12], [38], [28], [40]]
colours = ['blue', 'lightblue', 'red', 'magenta', 'lightgreen']
names = ['A', 'Cluster \\#12', 'Cluster \\#38', 'Cluster \\#28', 'Cluster \\#40']

f0, (ax1, ax2, ax3) = plt.subplots(3, figsize=(21, 6), facecolor='w')  # figsize (x,y)

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

axx3 = plt.axes([0, 0, 1, 3])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax2, [0.11, 0.585, 0.38, 0.38])  # [left, bottom, width, height]
axx3.set_axes_locator(ip)

ax1.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('Lperp') /
            (10**3), c='grey', s=1.0, alpha=0.2)

axx1 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.58, 0.55, 0.40, 0.43])  # [left, bottom, width, height]
axx1.set_axes_locator(ip)
axx1.scatter(df_clusters.evaluate('Lz') /
             (10**3), df_clusters.evaluate('Lperp') /
             (10**3), c='grey', s=1.0, alpha=0.2)
axx1.axes.get_xaxis().set_ticks([])
axx1.axes.get_yaxis().set_ticks([])
axx1.set_xlim(0.85, 1.82)
axx1.set_ylim(0.0, 1.0)
# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
mark_inset(ax1, axx1, loc1=2, loc2=4, fc="none", ec='0.0')
# axx1.patch.set_alpha(0.0)

for j in np.arange(len(grouped_clusters_C)):
    print(j)
    clusters = grouped_clusters_C[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    ax1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=3)
    if j == 1:
        axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                     df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=10)
    else:
        axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                     df_clusters.evaluate('Lperp', selection='(structure)') / (10**3), c=colours[j], s=3)

ax1.set_ylabel(r'$L_{\perp} \; [10^3 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=24)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=24)

ax1.tick_params(labelsize=19)

ax1.text(-4.5, 4.2, r'$\rm \bf A \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[0])
ax1.text(-4.5,
         3.9,
         r'$\rm \bf ' + str(names[1]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[1])
ax1.text(-4.5,
         3.6,
         r'$\rm \bf ' + str(names[2]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[2])
ax1.text(-4.5,
         3.3,
         r'$\rm \bf ' + str(names[3]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[3])
ax1.text(-4.5,
         3.0,
         r'$\rm \bf ' + str(names[4]) + '$',
         horizontalalignment='left',
         verticalalignment='center',
         fontsize=16,
         color=colours[4])


clusters_CM, labels, colours = [[15, 16, 46, 31, 32, 30, 42, 18, 33, 34], [12], [38], [28], [40]], [
    'A', 'Cluster \\#12', 'Cluster \\#38', 'Cluster \\#28', 'Cluster \\#40'], ['blue', 'lightblue', 'red', 'magenta', 'lightgreen']
# Compare the MDF of two structures:
distr_dict = {}
for system, clusters in zip(labels, clusters_CM):
    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')
    distr_dict[str(system)] = {'feh_distr': df_clusters.evaluate('feh_lamost_lrs', selection='(structure)')}

for i in np.arange(len(labels) - 2):
    if i == 0:
        Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.5, 0.7), bins=24)
        error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
        ax2.fill_between(
            Cbin_centres,
            np.zeros(
                (len(Chist))),
            Chist /
            np.nanmax(Chist),
            step='mid',
            color=colours[i],
            alpha=0.5)
        ax2.errorbar(
            Cbin_centres,
            Chist /
            np.nanmax(Chist),
            yerr=error /
            np.nanmax(Chist),
            fmt='o',
            ecolor=colours[i],
            ms=0,
            alpha=0.8)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
        ax2.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post')
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])
    else:
        Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.5, 0.7), bins=24)
        error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
        ax2.errorbar(
            Cbin_centres,
            Chist /
            np.nanmax(Chist),
            yerr=error /
            np.nanmax(Chist),
            fmt='o',
            ecolor=colours[i],
            ms=0,
            alpha=0.8)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
        ax2.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post')
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])

for i in np.arange(len(labels)):
    if i == 0:
        Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.5, 0.7), bins=24)
        error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
        ax3.errorbar(
            Cbin_centres,
            Chist /
            np.nanmax(Chist),
            yerr=error /
            np.nanmax(Chist),
            fmt='o',
            ecolor=colours[i],
            ms=0,
            alpha=0.8)
        #axx3.errorbar(Cbin_centres, Chist/np.nanmax(Chist), yerr=error/np.nanmax(Chist), fmt='o', ecolor=colours[i], ms=0, alpha=0.8)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
        ax3.step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color=colours[i],
            ls='--',
            lw=2,
            where='post',
            alpha=0.8)
        #axx3.step(Cbin_edges,Chist/np.float(np.amax(Chist)), color=colours[i], ls='--', lw=2,  where='post', alpha=0.8)
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])
    if i > 2:
        Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.5, 0.7), bins=24)
        error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
        ax3.fill_between(
            Cbin_centres,
            np.zeros(
                (len(Chist))),
            Chist /
            np.nanmax(Chist),
            step='mid',
            color=colours[i],
            alpha=0.5)
        ax3.errorbar(
            Cbin_centres,
            Chist /
            np.nanmax(Chist),
            yerr=error /
            np.nanmax(Chist),
            fmt='o',
            ecolor=colours[i],
            ms=0,
            alpha=0.8)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
        ax3.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post')
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])

if len(clusters_CM) == 2:
    feh_x, feh_y = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    feh_x = feh_x[np.isnan(feh_x) == False]
    feh_y = feh_y[np.isnan(feh_y) == False]
    KS, pval = stats.ks_2samp(distr_dict[labels[0]]['feh_distr'],
                              distr_dict[labels[1]]['feh_distr'], mode='exact')
    #ax1.text(0.82, 0.83, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
    #ax2.text(0.2, 0.95, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
    print(pval)

Chist, Cbin_edges = np.histogram(distr_dict['Cluster \\#12']['feh_distr'], range=(-2.3, 0.1), bins=20)
error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
axx3.errorbar(
    Cbin_centres,
    Chist /
    np.nanmax(Chist),
    yerr=error /
    np.nanmax(Chist),
    fmt='o',
    ecolor='black',
    ms=0)
Chist = np.append(Chist, 0)
Chist = np.append(0, Chist)
Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
axx3.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='black', lw=2, where='post')
axx3.text(-2.5, 1.15, r'$\rm Cluster \#12$', color='black')
# axx3.axes.get_xaxis().set_ticks([])
# axx3.axes.get_yaxis().set_ticks([])
axx3.set_xlabel(r'$\rm Fe/H$', labelpad=-0.00)
axx3.set_ylabel(r'$\rm N$', labelpad=-0.00)
axx3.patch.set_alpha(0.01)

# axx1.text(0.8, 0.6, '#418', horizontalalignment='center', verticalalignment='center', transform=axx1.transAxes, fontsize=20)
# axx1.text(0.7, 0.25, '#479', horizontalalignment='center',
# verticalalignment='center', transform=axx1.transAxes, fontsize=20)

# ax2.legend()
ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax2.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax2.tick_params(labelsize=20)

# axx3.axvline(-1.4)

ax2.set_xlim(-2.2, 0.5)

ax3.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax3.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax3.tick_params(labelsize=20)
ax3.set_xlim(-2.2, 0.5)


plt.tight_layout(w_pad=2)

plt.show()


plt.tight_layout()

plt.show()

# %% [markdown]
# ## D) Final configuration of B:

# %%
#del df_clusters
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
# Referee report suggestion:

grouped_clusters_E = [[37], [8, 11]]
colours = 'blue', 'lightblue'

f0, (ax1, ax2) = plt.subplots(2, figsize=(14, 6), facecolor='w')  # figsize (x,y)

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('En') /
            (10**4), c='grey', s=1.0, alpha=0.2)

axx1 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.58, 0.55, 0.40, 0.43])  # [left, bottom, width, height]
axx1.set_axes_locator(ip)
axx1.scatter(df_clusters.evaluate('Lz') /
             (10**3), df_clusters.evaluate('En') /
             (10**4), c='grey', s=1.0, alpha=0.2)
axx1.axes.get_xaxis().set_ticks([])
axx1.axes.get_yaxis().set_ticks([])
axx1.set_xlim(-1.5, -0.6)
axx1.set_ylim(-15.4, -13.1)
# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
mark_inset(ax1, axx1, loc1=2, loc2=4, fc="none", ec='0.0')
# axx1.patch.set_alpha(0.0)

for j in np.arange(len(grouped_clusters_E)):
    clusters = grouped_clusters_E[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    ax1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                df_clusters.evaluate('En', selection='(structure)') / (10**4), c=colours[j], s=3)
    axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                 df_clusters.evaluate('En', selection='(structure)') / (10**4), c=colours[j], s=3)

ax1.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=24)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=24)

ax1.tick_params(labelsize=19)

ax1.text(-4.5, -0.5, r'$\rm \bf B1 \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[0])
ax1.text(-4.5, -1.5, r'$\rm \bf B2 \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[1])


clusters_CM, labels, colours = [[37], [8, 11]], ['B1', 'B2'], ['blue', 'lightblue']
# Compare the MDF of two structures:
distr_dict = {}
for system, clusters in zip(labels, clusters_CM):
    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')
    distr_dict[str(system)] = {'feh_distr': df_clusters.evaluate('feh_lamost_lrs', selection='(structure)')}

for i in np.arange(len(labels)):
    Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.3, 0.1), bins=20)
    error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
    if i == 1:
        ax2.fill_between(
            Cbin_centres,
            np.zeros(
                (len(Chist))),
            Chist /
            np.nanmax(Chist),
            step='mid',
            color=colours[i],
            alpha=0.5)
    ax2.errorbar(
        Cbin_centres,
        Chist /
        np.nanmax(Chist),
        yerr=error /
        np.nanmax(Chist),
        fmt='o',
        ecolor=colours[i],
        ms=0,
        alpha=0.8)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax2.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post', alpha=0.8)
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])

if len(clusters_CM) == 2:

    feh_x, feh_y = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    feh_x = feh_x[np.isnan(feh_x) == False]
    feh_y = feh_y[np.isnan(feh_y) == False]

    KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
    #ax1.text(0.82, 0.83, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
    #ax2.text(0.2, 0.95, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
    print(pval)

# ax2.legend()
ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax2.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax2.tick_params(labelsize=20)

plt.tight_layout(w_pad=2)

plt.show()


plt.tight_layout()

plt.show()

# %%
grouped_clusters_E = [[37], [8, 11]]
colours = 'blue', 'lightblue'

f0, (ax1, ax2) = plt.subplots(2, figsize=(14, 6), facecolor='w')  # figsize (x,y)

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('En') /
            (10**4), c='grey', s=1.0, alpha=0.2)

axx1 = plt.axes([0, 0, 1, 1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.58, 0.55, 0.40, 0.43])  # [left, bottom, width, height]
axx1.set_axes_locator(ip)
axx1.scatter(df_clusters.evaluate('Lz') /
             (10**3), df_clusters.evaluate('En') /
             (10**4), c='grey', s=1.0, alpha=0.2)
axx1.axes.get_xaxis().set_ticks([])
axx1.axes.get_yaxis().set_ticks([])
axx1.set_xlim(-1.5, -0.6)
axx1.set_ylim(-15.4, -13.1)
# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
mark_inset(ax1, axx1, loc1=2, loc2=4, fc="none", ec='0.0')
# axx1.patch.set_alpha(0.0)

for j in np.arange(len(grouped_clusters_E)):
    clusters = grouped_clusters_E[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    ax1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                df_clusters.evaluate('En', selection='(structure)') / (10**4), c=colours[j], s=3)
    axx1.scatter(df_clusters.evaluate('Lz', selection='(structure)') / (10**3),
                 df_clusters.evaluate('En', selection='(structure)') / (10**4), c=colours[j], s=3)

ax1.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=24)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=24)

ax1.tick_params(labelsize=19)

ax1.text(-4.5, -0.5, r'$\rm \bf B1 \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[0])
ax1.text(-4.5, -1.5, r'$\rm \bf B2 \; substructure$', horizontalalignment='left',
         verticalalignment='center', fontsize=16, color=colours[1])


clusters_CM, labels, colours = [[37], [8, 11]], ['B1', 'B2'], ['blue', 'lightblue']
# Compare the MDF of two structures:
distr_dict = {}
for system, clusters in zip(labels, clusters_CM):
    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')
    distr_dict[str(system)] = {'feh_distr': df_clusters.evaluate('feh_lamost_lrs', selection='(structure)')}

for i in np.arange(len(labels)):
    Chist, Cbin_edges = np.histogram(distr_dict[labels[i]]['feh_distr'], range=(-2.3, 0.1), bins=20)
    error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
    ax2.errorbar(
        Cbin_centres,
        Chist /
        np.nanmax(Chist),
        yerr=error /
        np.nanmax(Chist),
        fmt='o',
        ecolor=colours[i],
        ms=0)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    ax2.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color=colours[i], lw=2, where='post')
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.3, -0.3), bins=20, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
    #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])

if len(clusters_CM) == 2:

    feh_x, feh_y = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    feh_x = feh_x[np.isnan(feh_x) == False]
    feh_y = feh_y[np.isnan(feh_y) == False]

    KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
    #ax1.text(0.82, 0.83, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=20)
    #ax2.text(0.2, 0.95, 'pval = '+str(round(pval,3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
    print(pval)

# ax2.legend()
ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax2.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax2.tick_params(labelsize=20)

plt.tight_layout(w_pad=2)

plt.show()


plt.tight_layout()

plt.show()

# %% [markdown]
# ## E) GE in IoM:

# %%
#del df_clusters
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

# Adding angular momemtum:

Lx_n = df_clusters.evaluate('y') * df_clusters.evaluate('vz') - \
    df_clusters.evaluate('z') * df_clusters.evaluate('vy')
Ly_n = df_clusters.evaluate('z') * df_clusters.evaluate('vx') - \
    df_clusters.evaluate('x') * df_clusters.evaluate('vz')
Lz_n = df_clusters.evaluate('x') * df_clusters.evaluate('vy') - \
    df_clusters.evaluate('y') * df_clusters.evaluate('vx')
Ltotal = np.sqrt(Lx_n**2. + Ly_n**2. + Lz_n**2.)

df_clusters.add_column('Lx_n', Lx_n)
df_clusters.add_column('Ly_n', Ly_n)
df_clusters.add_column('Lz_n', Lz_n)
df_clusters.add_column('Ltotal', Ltotal)

df_clusters

# %%
grouped_clusters_IoM = [[56, 59], [10, 7, 13, 20, 19, 43], [57, 58, 54, 50, 53, 47, 44,
                                                            51, 39, 35, 9, 24, 5, 23, 36, 21, 17], [48, 45, 25, 29, 41, 49, 55], [27, 22, 26]]
GE_not_ind = [56, 59, 10, 7, 13, 20, 19, 43, 57, 58, 54, 50, 53, 47, 44, 51,
              39, 35, 9, 24, 5, 23, 36, 21, 17, 48, 45, 25, 29, 41, 49, 55, 27, 22, 26]
GE_all = [56, 59, 10, 7, 13, 20, 19, 43, 57, 58, 54, 50, 53, 47, 44, 51, 39, 35,
          9, 24, 5, 23, 36, 21, 17, 48, 45, 25, 29, 41, 49, 55, 27, 22, 26, 6, 14, 52]
names_IoM_plot = ['C1', 'C2', 'C3', 'C4', 'C5']
individual_clusters = [6, 14, 52]
#colours = []
#colours.extend(cm.jet(np.linspace(0, 1, len(names_IoM_plot))))
colours = ['#01befe', '#ffdd00', '#ff7d00', '#ff006d', '#adff02']
print(colours)

# %%
f0, ax = plt.subplots(6, figsize=(15, 9), facecolor='w')  # figsize (x,y)
axs = []
for i in range(2):  # Rows
    for j in range(3):  # Cols
        ax_aux = plt.subplot2grid((2, 3), (i, j))
        axs.append(ax_aux)

clusters = GE_all
condition = str()
for cluster in clusters[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
df_clusters.select(condition, name='all_GE')

axs[0].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('En', selection='all_GE') /
               (10**4), c='silver', s=7, zorder=1)
axs[1].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('En', selection='all_GE') /
               (10**4), c='silver', s=7, zorder=1)
axs[2].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('En', selection='all_GE') /
               (10**4), c='silver', s=7, zorder=1)
axs[3].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('En', selection='all_GE') /
               (10**4), c='silver', s=7, zorder=1)
axs[4].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('En', selection='all_GE') /
               (10**4), c='silver', s=7, zorder=1)
axs[5].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('En', selection='all_GE') /
               (10**4), c='silver', s=7, zorder=1)

for j in np.arange(len(names_IoM_plot)):
    clusters = grouped_clusters_IoM[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    axs[j].scatter(df_clusters.evaluate('Lz') /
                   (10**3), df_clusters.evaluate('En') /
                   (10**4), c='grey', s=1.0, alpha=0.1, zorder=0)
    axs[j].scatter(df_clusters.evaluate('Lz', selection='(structure)') /
                   (10**3), df_clusters.evaluate('En', selection='(structure)') /
                   (10**4), c=colours[j], s=7, zorder=2)
    axs[j].set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=20)
    axs[j].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)

    axs[j].set_title(names_IoM_plot[j], fontsize=20)

    axs[j].tick_params(labelsize=17)

# Independent clusters:
axs[-1].scatter(df_clusters.evaluate('Lz') / (10**3), df_clusters.evaluate('En') /
                (10**4), c='grey', s=1.0, alpha=0.2, zorder=0)
axs[-1].set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=20)
axs[-1].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)
axs[-1].set_title(r'$\rm Independent \; Clusters$', fontsize=20)
axs[-1].tick_params(labelsize=17)
#mean, width, height, angle, color  = [ 3.9 ,  -2.8], 5.5, 1.2, 85, 'black'
#ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle = angle, lw=2, color=color, fill=False)
# axs[7].add_patch(ell)
# axs[7].autoscale()
axs[-1].set_xlim(-4.5, 4.5)

# Individual clusters:

ind_colours = []
ind_colours.extend(cm.tab20b(np.linspace(0, 1, 4)))
print(np.linspace(0, 1, 4))

condition = str()
for i, cluster in enumerate(individual_clusters):
    df_clusters.select('def_cluster_id==' + str(cluster), name='this_cluster')

    axs[-1].plot(df_clusters.evaluate('Lz', selection='this_cluster') / (10**3), df_clusters.evaluate('En',
                 selection='this_cluster') / (10**4), 'o', ms=3, color=ind_colours[i], zorder=2)
    axs[-1].text(0.03,
                 0.95 - 0.07 * i,
                 r'$\rm \bf Cluster \; \#' + str(cluster) + '$',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=axs[-1].transAxes,
                 fontsize=16,
                 color=ind_colours[i])

print(ind_colours)

# rest of GE, not ind
condition = str()
for cluster in GE_not_ind[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(GE_not_ind[-1]) + ')'
df_clusters.select(condition, name='structure')
#axs[-1].plot(df_clusters.evaluate('Lz', selection='structure')/(10**3), df_clusters.evaluate('En', selection='structure')/(10**4), 'o', ms=2, color='white', alpha=0.1, zorder=2)


plt.tight_layout(w_pad=1, h_pad=1)

plt.show()

# %%
f0, ax = plt.subplots(6, figsize=(15, 9), facecolor='w')  # figsize (x,y)
axs = []
for i in range(2):  # Rows
    for j in range(3):  # Cols
        ax_aux = plt.subplot2grid((2, 3), (i, j))
        axs.append(ax_aux)

clusters = GE_all
condition = str()
for cluster in clusters[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
df_clusters.select(condition, name='all_GE')

axs[0].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('Lperp', selection='all_GE') /
               (10**3), c='silver', s=7, zorder=1)
axs[1].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('Lperp', selection='all_GE') /
               (10**3), c='silver', s=7, zorder=1)
axs[2].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('Lperp', selection='all_GE') /
               (10**3), c='silver', s=7, zorder=1)
axs[3].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('Lperp', selection='all_GE') /
               (10**3), c='silver', s=7, zorder=1)
axs[4].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('Lperp', selection='all_GE') /
               (10**3), c='silver', s=7, zorder=1)
axs[5].scatter(df_clusters.evaluate('Lz', selection='all_GE') /
               (10**3), df_clusters.evaluate('Lperp', selection='all_GE') /
               (10**3), c='silver', s=7, zorder=1)

for j in np.arange(len(names_IoM_plot)):
    clusters = grouped_clusters_IoM[j]

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    axs[j].scatter(df_clusters.evaluate('Lz') /
                   (10**3), df_clusters.evaluate('Lperp') /
                   (10**3), c='grey', s=1.0, alpha=0.1, zorder=0)
    axs[j].scatter(df_clusters.evaluate('Lz', selection='(structure)') /
                   (10**3), df_clusters.evaluate('Lperp', selection='(structure)') /
                   (10**3), c=colours[j], s=7, zorder=2)
    axs[j].set_ylabel(r'$L_\perp \; [10^3 \; kpc \; km/s]$', fontsize=20)
    axs[j].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)

    axs[j].set_title(names_IoM_plot[j], fontsize=20)

    axs[j].tick_params(labelsize=17)

# Independent clusters:
axs[-1].scatter(df_clusters.evaluate('Lz') / (10**3), df_clusters.evaluate('Lperp') /
                (10**3), c='grey', s=1.0, alpha=0.2, zorder=0)
axs[-1].set_ylabel(r'$L_\perp \; [10^3 \; kpc \; km/s]$', fontsize=20)
axs[-1].set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=20)
axs[-1].set_title(r'$\rm Independent \; Clusters$', fontsize=20)
axs[-1].tick_params(labelsize=17)
#mean, width, height, angle, color  = [ 3.9 ,  -2.8], 5.5, 1.2, 85, 'black'
#ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle = angle, lw=2, color=color, fill=False)
# axs[7].add_patch(ell)
# axs[7].autoscale()
axs[-1].set_xlim(-4.5, 4.5)

# Individual clusters:

ind_colours = []
ind_colours.extend(cm.tab20b(np.linspace(0, 1, 4)))

condition = str()
for i, cluster in enumerate(individual_clusters):
    df_clusters.select('def_cluster_id==' + str(cluster), name='this_cluster')

    axs[-1].plot(df_clusters.evaluate('Lz',
                                      selection='this_cluster') / (10**3),
                 df_clusters.evaluate('Lperp',
                                      selection='this_cluster') / (10**3),
                 'o',
                 ms=3,
                 color=ind_colours[i],
                 zorder=2)
    axs[-1].text(0.03,
                 0.95 - 0.07 * i,
                 r'$\rm \bf Cluster \; \#' + str(cluster) + '$',
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=axs[-1].transAxes,
                 fontsize=16,
                 color=ind_colours[i])

# rest of GE, not ind
condition = str()
for cluster in GE_not_ind[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(GE_not_ind[-1]) + ')'
df_clusters.select(condition, name='structure')
#axs[-1].plot(df_clusters.evaluate('Lz', selection='structure')/(10**3), df_clusters.evaluate('Lperp', selection='structure')/(10**3), 'o', ms=2, color='white', alpha=0.1, zorder=2)

plt.tight_layout(w_pad=1, h_pad=1)

plt.show()

# %% [markdown]
# ## F) KS table for the different groups in GE:
#

# %%
# /data/users/ruizlara/halo_clustering/DEF_catalogue_prob_0_2_CDF_3D.hdf5
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'

# %%
systems = ['C1', 'C2', 'C3', 'C4', 'C5', '14', '52', '6']
grouped_clusters_systs = [
    [
        56, 59], [
            10, 7, 13, 20, 19, 43], [
                57, 58, 54, 50, 53, 47, 44, 51, 39, 35, 9, 24, 5, 23, 36, 21, 17], [
                    48, 45, 25, 29, 41, 49, 55], [
                        27, 22, 26], [14], [52], [6]]

compare_MDF_structures(
    df_clusters,
    structures_names=systems,
    structures=grouped_clusters_systs,
    path=path,
    name='test')
df = pd.read_csv(path + 'test_met_table.csv', index_col=0)
cm = sns.light_palette("green", as_cmap=True)
display(df.style.set_caption('perc 80')
        .background_gradient(cmap=cm).applymap(pval_red_or_green))

# %%
name = 'test'
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'

df = pd.read_csv(path + name + '_met_table.csv', index_col=0)
cm = sns.light_palette("green", as_cmap=True)
styled_table = df.style.set_caption('The greener, the more similar the distributions are.')\
    .background_gradient(cmap=cm).applymap(pval_red_or_green)
html = styled_table.render()
print(html)

# %% [markdown]
# ## G) MDF of GE (all together)

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

# %%
clusters_CM = [
    56,
    59,
    10,
    7,
    13,
    20,
    19,
    43,
    57,
    58,
    54,
    50,
    53,
    47,
    44,
    51,
    39,
    35,
    9,
    24,
    5,
    23,
    36,
    21,
    17,
    48,
    45,
    25,
    29,
    41,
    49,
    55,
    27,
    22,
    26,
    14,
    52]  # Including everything, only 6 is missing.

condition = str()
for cluster in clusters_CM[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(clusters_CM[-1]) + ')'
df_clusters.select(condition, name='structure')

f0, (ax1) = plt.subplots(1, figsize=(7, 7), facecolor='w')
ax1 = plt.subplot(111)

Chist, Cbin_edges = np.histogram(df_clusters.evaluate('feh_lamost_lrs'), range=(-2.3, 0.1), bins=20)
Chist = np.append(Chist, 0)
Chist = np.append(0, Chist)
Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
ax1.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='k', where='post', alpha=0.2, lw=0)

ax1.fill_between(np.linspace(np.amin(Cbin_edges),
                             np.amax(Cbin_edges),
                             len(np.r_[np.repeat((Chist / np.float(np.amax(Chist)))[:-1],
                                                 200),
                                       1.03])),
                 0,
                 np.r_[np.repeat((Chist / np.float(np.amax(Chist)))[:-1],
                                 200),
                       1.03],
                 color='k',
                 alpha=0.09,
                 lw=0.0,
                 zorder=3)

Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
    'feh_lamost_lrs', selection='structure'), range=(-2.3, 0.1), bins=20)
error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
ax1.errorbar(
    Cbin_centres,
    Chist /
    np.nanmax(Chist),
    yerr=error /
    np.nanmax(Chist),
    fmt='o',
    ecolor='blue',
    ms=0)
Chist = np.append(Chist, 0)
Chist = np.append(0, Chist)
Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
ax1.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='b', lw=2, where='post')

ax1.text(
    0.97,
    0.95,
    'Gaia-Enceladus',
    horizontalalignment='right',
    verticalalignment='center',
    transform=ax1.transAxes,
    fontsize=16,
    color='blue')
ax1.text(
    0.97,
    0.9,
    'Overall halo',
    horizontalalignment='right',
    verticalalignment='center',
    transform=ax1.transAxes,
    fontsize=16,
    color='gray')

print('average metallicity of GE')
print(np.nanmean(df_clusters.evaluate('feh_lamost_lrs', selection='structure')))
print(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='structure')))

ax1.set_ylim(0.0, 1.03)

ax1.legend(frameon=False, fontsize='x-large')
ax1.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax1.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax1.tick_params(labelsize=20)

plt.show()

print(len(df_clusters.evaluate('feh_lamost_lrs')))
print(len(df_clusters.evaluate('feh_lamost_lrs')[df_clusters.evaluate('feh_lamost_lrs') > -20.0]))

# %%
clusters_CM = [56, 59, 57, 58, 54, 50, 53, 47, 44, 51, 39, 35, 9, 24, 5, 23, 36,
               21, 17, 48, 45, 25, 29, 41, 49, 55, 27, 22, 26, 52]  # Excluding C2, 6 and 14.

condition = str()
for cluster in clusters_CM[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(clusters_CM[-1]) + ')'
df_clusters.select(condition, name='structure')

f0, (ax1) = plt.subplots(1, figsize=(7, 7), facecolor='w')
ax1 = plt.subplot(111)

Chist, Cbin_edges = np.histogram(df_clusters.evaluate('feh_lamost_lrs'), range=(-2.3, 0.1), bins=20)
Chist = np.append(Chist, 0)
Chist = np.append(0, Chist)
Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
ax1.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='k', where='post', alpha=0.2, lw=0)

ax1.fill_between(np.linspace(np.amin(Cbin_edges),
                             np.amax(Cbin_edges),
                             len(np.r_[np.repeat((Chist / np.float(np.amax(Chist)))[:-1],
                                                 200),
                                       1.03])),
                 0,
                 np.r_[np.repeat((Chist / np.float(np.amax(Chist)))[:-1],
                                 200),
                       1.03],
                 color='k',
                 alpha=0.09,
                 lw=0.0,
                 zorder=3)

Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
    'feh_lamost_lrs', selection='structure'), range=(-2.3, 0.1), bins=20)
error, Cbin_centres = np.sqrt(Chist), 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
ax1.errorbar(
    Cbin_centres,
    Chist /
    np.nanmax(Chist),
    yerr=error /
    np.nanmax(Chist),
    fmt='o',
    ecolor='blue',
    ms=0)
Chist = np.append(Chist, 0)
Chist = np.append(0, Chist)
Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
ax1.step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='b', lw=2, where='post')

ax1.text(
    0.97,
    0.95,
    'Gaia-Enceladus',
    horizontalalignment='right',
    verticalalignment='center',
    transform=ax1.transAxes,
    fontsize=16,
    color='blue')
ax1.text(
    0.97,
    0.9,
    'Overall halo',
    horizontalalignment='right',
    verticalalignment='center',
    transform=ax1.transAxes,
    fontsize=16,
    color='gray')


ax1.set_ylim(0.0, 1.03)

ax1.legend(frameon=False, fontsize='x-large')
ax1.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
ax1.set_ylabel(r'$\rm Normalised \; Number \; of \; stars$', fontsize=20)
ax1.tick_params(labelsize=20)

plt.show()

print(len(df_clusters.evaluate('feh_lamost_lrs')))
print(len(df_clusters.evaluate('feh_lamost_lrs')[df_clusters.evaluate('feh_lamost_lrs') > -20.0]))

# %% [markdown]
# ## H) Sequoia plot:

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

# %%
# Sequoia nice plot:

grouped_clusters_seq = [62, 63, 64]
colours = ['blue', 'green', 'red']

f0, (ax1, ax2, ax3) = plt.subplots(3, figsize=(18, 5), facecolor='w')  # figsize (x,y)

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

ax1.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('En') /
            (10**4), c='grey', s=1.0, alpha=0.2)
ax2.scatter(df_clusters.evaluate('vz'), df_clusters.evaluate('vR'), c='grey', s=1.0, alpha=0.2)
ax3.scatter(df_clusters.evaluate('Lz') /
            (10**3), df_clusters.evaluate('Lperp') /
            (10**3), c='grey', s=1.0, alpha=0.2)
'''
axx1 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.65,0.65,0.33,0.33]) #[left, bottom, width, height]
axx1.set_axes_locator(ip)
axx1.scatter(df_clusters.evaluate('Lz')/(10**3), df_clusters.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)
axx1.axes.get_xaxis().set_ticks([])
axx1.axes.get_yaxis().set_ticks([])
axx1.set_xlim(-4., -1.0)
axx1.set_ylim(-12.,-5.)
# Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
mark_inset(ax1, axx1, loc1=2, loc2=4, fc="none", ec='0.0')
#axx1.patch.set_alpha(0.0)
'''
for j, cluster in enumerate(grouped_clusters_seq):
    df_clusters.select('def_cluster_id==' + str(cluster), name='this_cluster')

    ax1.scatter(df_clusters.evaluate('Lz', selection='(this_cluster)') / (10**3),
                df_clusters.evaluate('En', selection='(this_cluster)') / (10**4), c=colours[j], s=4)
    #axx1.scatter(df_clusters.evaluate('Lz', selection='(this_cluster)')/(10**3), df_clusters.evaluate('En', selection='(this_cluster)')/(10**4), c=colours[j],s=3)

    ax2.scatter(
        df_clusters.evaluate(
            'vz',
            selection='(this_cluster)'),
        df_clusters.evaluate(
            'vR',
            selection='(this_cluster)'),
        c=colours[j],
        s=6)

    ax3.scatter(df_clusters.evaluate('Lz', selection='(this_cluster)') / (10**3),
                df_clusters.evaluate('Lperp', selection='(this_cluster)') / (10**3), c=colours[j], s=4)

    ax1.text(1.7, -14.4 - 1.2 * j, r'$\bf Cluster \#' + str(cluster) + '$',
             horizontalalignment='left', verticalalignment='center', fontsize=16, color=colours[j])

ax1.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=21)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=21)

ax2.set_ylabel(r'$v_R \; [km/s]$', fontsize=21)
ax2.set_xlabel(r'$v_z \; [km/s]$', fontsize=21)

ax3.set_ylabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=21)
ax3.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=21)

ax1.tick_params(labelsize=17)
ax2.tick_params(labelsize=17)
ax3.tick_params(labelsize=17)

ax2.set_ylim(-500, 500)
ax2.set_xlim(-500, 500)

plt.tight_layout()

plt.show()

# %% [markdown]
# ## I) Mega-table of the independent structures

# %% [markdown]
# ### I.1) MDFs.

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
systems = ['Helmi', '64', 'Sequoia', '62', 'Thamnos1', 'Thamnos2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters_systs = [[60,
                           61],
                          [64],
                          [63],
                          [62],
                          [37],
                          [8,
                           11],
                          [14,
                           56,
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
                          [38]]

compare_MDF_structures(
    df_clusters,
    structures_names=systems,
    structures=grouped_clusters_systs,
    path=path,
    name='test')
df = pd.read_csv(path + 'test_met_table.csv', index_col=0)
cm = sns.light_palette("green", as_cmap=True)
display(df.style.set_caption('perc 80')
        .background_gradient(cmap=cm).applymap(pval_red_or_green))

# %%
name = 'test'
table_type = '_met_table.csv'
nice_table_from_csv(path, name, table_type)

# %% [markdown]
# ### I.2) CaMDs.
#
# See inspecting_substructures_DEF_3D_CaMD_final.ipynb

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
systems = ['Helmi', '64', 'Sequoia', '62', 'Thamnos1', 'Thamnos2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters_systs = [[60,
                           61],
                          [64],
                          [63],
                          [62],
                          [37],
                          [8,
                           11],
                          [14,
                           56,
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

# %% [markdown]
# ## J) Age-Metallicity plot in IoM and age-metallicity table:
#
# From isochrone fitting, you need to create a table with age and
# metallicity for the different structures
# (iso_fit_results_structures_good_GE1.hdf5).

# %%
# GE as a whole:
names = ['Helmi', '64', 'Sequoia', '62', 'Thamnos1', 'Thamnos2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters = [[60,
                     61],
                    [64],
                    [63],
                    [62],
                    [37],
                    [8,
                     11],
                    [14,
                     56,
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
                    [38]]

for i in np.arange(len(names)):
    print(names[i], grouped_clusters[i])

# %%
df_age_met = vaex.open(
    '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D_80/iso_fit_results_structures_good_GE1.hdf5')
df_age_met

# %%
df_age_met[4:14]

# %%
#del df_clusters
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
# Adding angular momemtum:

Lx_n = df_clusters.evaluate('y') * df_clusters.evaluate('vz') - \
    df_clusters.evaluate('z') * df_clusters.evaluate('vy')
Ly_n = df_clusters.evaluate('z') * df_clusters.evaluate('vx') - \
    df_clusters.evaluate('x') * df_clusters.evaluate('vz')
Lz_n = df_clusters.evaluate('x') * df_clusters.evaluate('vy') - \
    df_clusters.evaluate('y') * df_clusters.evaluate('vx')
Ltotal = np.sqrt(Lx_n**2. + Ly_n**2. + Lz_n**2.)

df_clusters.add_column('Lx_n', Lx_n)
df_clusters.add_column('Ly_n', Ly_n)
df_clusters.add_column('Lz_n', Lz_n)
df_clusters.add_column('Ltotal', Ltotal)

# %%
# Add 2 more columns to df_clusters with age and met.
age_array, met_array = df_clusters.evaluate('ra').copy() * np.nan, df_clusters.evaluate('ra').copy() * np.nan
def_cluster_id_array = df_clusters.evaluate('def_cluster_id').copy()

ages, fehs = df_age_met.evaluate('age').copy(), df_age_met.evaluate('feh').copy()

for i, clusters in enumerate(grouped_clusters):
    name = names[i]
    print(str(name) + '   ' + str(clusters))

    condition = ''
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id_array==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id_array==' + str(clusters[-1]) + ')'
    coords = np.where(eval(condition))

    age_array[coords], met_array[coords] = ages[i], fehs[i]

df_clusters.add_column('Age', age_array)
df_clusters.add_column('Feh', met_array)

# %%
df_clusters.select('(def_cluster_id>0.5)&(def_cluster_id<700.5)', name='all')
df_clusters.select('(def_cluster_id<700.5)', name='all_all')

f0, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(18, 12), facecolor='w')

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

ax1.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('En', selection='all_all') /
         (10**4), 'o', color='gray', ms=1, alpha=0.3, zorder=0)
ax2.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('En', selection='all_all') /
         (10**4), 'o', color='gray', ms=1, alpha=0.3, zorder=0)

scatter1 = ax1.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('En', selection='all') /
                       (10**4), c=df_clusters.evaluate('Age', selection='all'), s=6.0, vmin=7.6, vmax=12.0)
cb = f0.colorbar(scatter1, ax=ax1, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm Age \; [Gyr]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

scatter2 = ax2.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('En', selection='all') /
                       (10**4), c=df_clusters.evaluate('Feh', selection='all'), s=6.0)
cb = f0.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm [Fe/H]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

ax1.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=23)
ax1.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=23)
ax2.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=23)
ax2.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=23)

ax1.tick_params(labelsize=19)
ax2.tick_params(labelsize=19)

ax3.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('Lperp', selection='all_all') /
         (10**3), 'o', color='gray', ms=1, alpha=0.3, zorder=0)
ax4.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('Lperp', selection='all_all') /
         (10**3), 'o', color='gray', ms=1, alpha=0.3, zorder=0)

scatter3 = ax3.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('Lperp', selection='all') /
                       (10**3), c=df_clusters.evaluate('Age', selection='all'), s=6.0, vmin=7.6, vmax=12.0)
cb = f0.colorbar(scatter3, ax=ax3, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm Age \; [Gyr]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

scatter4 = ax4.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('Lperp', selection='all') /
                       (10**3), c=df_clusters.evaluate('Feh', selection='all'), s=6.0)
cb = f0.colorbar(scatter4, ax=ax4, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm [Fe/H]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

ax3.tick_params(labelsize=19)
ax4.tick_params(labelsize=19)

ax3.set_ylabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax3.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=23)
ax4.set_ylabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax4.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=23)

plt.tight_layout(w_pad=3)

plt.show()

# %%
# Adding angular momentum and other things
compute_coordinates_new(df_clusters)
df_clusters

# %%
df_clusters.select('(def_cluster_id>0.5)&(def_cluster_id<700.5)', name='all')
df_clusters.select('(def_cluster_id<700.5)', name='all_all')

f0, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(18, 18), facecolor='w')

ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)

ax1.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('En', selection='all_all') /
         (10**4), 'o', color='gray', ms=1, alpha=0.3, zorder=0)
ax2.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('En', selection='all_all') /
         (10**4), 'o', color='gray', ms=1, alpha=0.3, zorder=0)

scatter1 = ax1.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('En', selection='all') /
                       (10**4), c=df_clusters.evaluate('Age', selection='all'), s=6.0)
cb = f0.colorbar(scatter1, ax=ax1, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm Age \; [Gyr]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

scatter2 = ax2.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('En', selection='all') /
                       (10**4), c=df_clusters.evaluate('Feh', selection='all'), s=6.0)
cb = f0.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm [Fe/H]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

ax1.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=23)
ax1.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax2.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=23)
ax2.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=23)

ax1.tick_params(labelsize=19)
ax2.tick_params(labelsize=19)

ax3.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('Lperp', selection='all_all') /
         (10**3), 'o', color='gray', ms=1, alpha=0.3, zorder=0)
ax4.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('Lperp', selection='all_all') /
         (10**3), 'o', color='gray', ms=1, alpha=0.3, zorder=0)

scatter3 = ax3.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('Lperp', selection='all') /
                       (10**3), c=df_clusters.evaluate('Age', selection='all'), s=6.0)
cb = f0.colorbar(scatter3, ax=ax3, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm Age \; [Gyr]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

scatter4 = ax4.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('Lperp', selection='all') /
                       (10**3), c=df_clusters.evaluate('Feh', selection='all'), s=6.0)
cb = f0.colorbar(scatter4, ax=ax4, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm [Fe/H]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

ax3.tick_params(labelsize=19)
ax4.tick_params(labelsize=19)

ax3.set_ylabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax3.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=23)
ax4.set_ylabel(r'$L_{\perp} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax4.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=23)

# Lx Ly Lz

ax5.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('Lx', selection='all_all') /
         (10**3), 'o', color='gray', ms=1, alpha=0.3, zorder=0)
ax6.plot(df_clusters.evaluate('Lz', selection='all_all') /
         (10**3), df_clusters.evaluate('Ly', selection='all_all') /
         (10**3), 'o', color='gray', ms=1, alpha=0.3, zorder=0)

scatter5 = ax5.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('Lx', selection='all') /
                       (10**3), c=df_clusters.evaluate('Age', selection='all'), s=6.0)
cb = f0.colorbar(scatter5, ax=ax5, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm Age \; [Gyr]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

scatter6 = ax6.scatter(df_clusters.evaluate('Lz', selection='all') /
                       (10**3), df_clusters.evaluate('Ly', selection='all') /
                       (10**3), c=df_clusters.evaluate('Feh', selection='all'), s=6.0)
cb = f0.colorbar(scatter6, ax=ax6, orientation='vertical', pad=0.0)
cb.set_label(label=r'$\rm [Fe/H]$', size='xx-large', weight='bold')
cb.ax.tick_params(labelsize='x-large')

ax5.tick_params(labelsize=19)
ax6.tick_params(labelsize=19)

ax5.set_ylabel(r'$L_{x} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax5.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax6.set_ylabel(r'$L_{y} \; [10^3 \; kpc \; km/s]$', fontsize=23)
ax6.set_xlabel(r'$L_{z} \; [10^3 \; kpc \; km/s]$', fontsize=23)

plt.tight_layout(w_pad=3)

plt.show()

# %% [markdown]
# #### Table!

# %%
df = vaex.open(
    '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D_80/iso_fit_results_structures_good_GE1.hdf5')
df.export_csv(
    '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D_80/iso_fit_results_structures_good_GE1.csv')
df = pd.read_csv(
    '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D_80/iso_fit_results_structures_good_GE1.csv')
df

# %%
print(df.to_latex(index=False))

# %% [markdown]
# ## K) Plot of all MDFs and CMDs

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')  # no segue within quality.
df_clusters.select('(distance<2.5)', name='int_dist')  # intermediate distance, d < 2.5 kpc

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
df_clusters

# %%
# GE as a whole:
names = ['Helmi', '64', 'Sequoia', '62', 'Thamnos 1', 'Thamnos 2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters = [[60,
                     61],
                    [64],
                    [63],
                    [62],
                    [37],
                    [8,
                     11],
                    [14,
                     56,
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
                    [38]]

clusters_GE = [14, 56, 59, 10, 7, 13, 57, 58, 54, 50, 53, 47, 44, 51, 48, 45, 25,
               29, 41, 49, 27, 22, 26, 52, 55, 20, 19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24]

# %%
condition = str()
for cluster in clusters_GE[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(clusters_GE[-1]) + ')'
df_clusters.select(condition, name='GE')

f0, ax = plt.subplots(12, figsize=(21, 16), facecolor='w')  # figsize (x,y)
axs = []
for i in range(3):  # Rows
    for j in range(4):  # Cols
        ax_aux = plt.subplot2grid((3, 4), (i, j))
        axs.append(ax_aux)

for j in np.arange(len(axs)):
    clusters = grouped_clusters[j]
    print(names[j])

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    Chist, Cbin_edges = np.histogram(df_clusters.evaluate('feh_lamost_lrs'), range=(-2.3, 0.1), bins=20)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    axs[j].step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='k', where='post', alpha=0.2, lw=0)
    axs[j].fill_between(
        np.linspace(
            np.amin(Cbin_edges),
            np.amax(Cbin_edges),
            len(
                np.r_[
                    np.repeat(
                        (Chist /
                         np.float(
                             np.amax(Chist)))[
                            :-
                            1],
                        200),
                    1.03])),
        0,
        np.r_[
            np.repeat(
                (Chist /
                 np.float(
                     np.amax(Chist)))[
                    :-
                    1],
                200),
            1.03],
        color='k',
        alpha=0.05,
        lw=0.0,
        zorder=3)

    if names[j] == 'Gaia-Enceladus':
        axs[j].text(0.97,
                    0.95,
                    str(names[j]),
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(0.97,
                    0.88,
                    '(' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                       selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                   selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.97,
            0.81,
            'Overall halo',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='gray')

    elif names[j] in ['6', '3', 'A', '12', '28', '38']:
        Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
            'feh_lamost_lrs', selection='GE'), range=(-2.3, 0.1), bins=20)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)

        axs[j].step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color='k',
            linestyle='dashed',
            lw=1,
            where='post')

        axs[j].text(0.03,
                    0.95,
                    str(names[j]) + ' (' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                                        selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                                    selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.03,
            0.88,
            'Gaia-Enceladus',
            horizontalalignment='left',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='black')
        axs[j].text(
            0.03,
            0.81,
            'Overall halo',
            horizontalalignment='left',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='gray')

    elif names[j] in ['Sequoia', 'Helmi', 'Thamnos 1', 'Thamnos 2', 'GE']:
        Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
            'feh_lamost_lrs', selection='GE'), range=(-2.3, 0.1), bins=20)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)

        axs[j].step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color='k',
            linestyle='dashed',
            lw=1,
            where='post')

        axs[j].text(0.97,
                    0.95,
                    str(names[j]),
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(0.97,
                    0.88,
                    '(' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                       selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                   selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.97,
            0.81,
            'Gaia-Enceladus',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='black')
        axs[j].text(
            0.97,
            0.74,
            'Overall halo',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='gray')

    else:
        Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
            'feh_lamost_lrs', selection='GE'), range=(-2.3, 0.1), bins=20)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
        axs[j].step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color='k',
            linestyle='dashed',
            lw=1,
            where='post')

        axs[j].text(0.97,
                    0.95,
                    str(names[j]) + ' (' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                                        selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                                    selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.97,
            0.88,
            'Gaia-Enceladus',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='black')
        axs[j].text(
            0.97,
            0.81,
            'Overall halo',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='gray')

    Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
        'feh_lamost_lrs', selection='structure'), range=(-2.3, 0.1), bins=20)
    error, Cbin_centres = np.sqrt(Chist) / 1.5, 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
    axs[j].errorbar(
        Cbin_centres,
        Chist /
        np.nanmax(Chist),
        yerr=error /
        np.nanmax(Chist),
        fmt='o',
        ecolor='black',
        ms=0)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    axs[j].step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='k', lw=2, where='post')

    axs[j].set_ylim(0.0, 1.05)
    axs[j].set_xlabel('', fontsize=20)  # 20   r'$\rm [Fe/H]$'
    axs[j].set_ylabel('', fontsize=20)  # r'$\rm Normalised \; Number \; of \; stars$'
    axs[j].tick_params(labelsize=0)  # 20
    #axs[j].set_title(names[j], fontsize=20)
    axs[j].minorticks_on()

# axs[0].set_xticks([])
axs[0].tick_params(labelsize=20)
# axs[4].set_xticks([])
axs[4].tick_params(labelsize=20)
# axs[8].set_xticks([])
axs[8].tick_params(labelsize=20)
axs[9].tick_params(labelsize=20)
axs[10].tick_params(labelsize=20)
axs[11].tick_params(labelsize=20)
# axs[12].set_xticks([])
# axs[12].tick_params(labelsize=20)
# axs[13].tick_params(labelsize=20)
'''
axs[0].minorticks_on()
axs[4].minorticks_on()
axs[8].minorticks_on()
axs[12].minorticks_on()
axs[16].minorticks_on()
'''

f0.text(0.53, 0.02, r'$\rm [Fe/H]$', horizontalalignment='center',
        verticalalignment='center', fontsize=25, color='black')
f0.text(
    0.02,
    0.53,
    r'$\rm Normalised \; Number \; of \; stars$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=25,
    rotation=90,
    color='black')

plt.tight_layout(w_pad=0.8, h_pad=0.3, rect=[0.03, 0.03, 1, 1])

plt.show()

# %%
# Referee report suggestion:

condition = str()
for cluster in clusters_GE[:-1]:
    condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
condition = condition + '(def_cluster_id==' + str(clusters_GE[-1]) + ')'
df_clusters.select(condition, name='GE')

f0, ax = plt.subplots(12, figsize=(21, 16), facecolor='w')  # figsize (x,y)
axs = []
for i in range(3):  # Rows
    for j in range(4):  # Cols
        ax_aux = plt.subplot2grid((3, 4), (i, j))
        axs.append(ax_aux)

for j in np.arange(len(axs)):
    clusters = grouped_clusters[j]
    print(names[j])

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    Chist, Cbin_edges = np.histogram(df_clusters.evaluate('feh_lamost_lrs'), range=(-2.3, 0.1), bins=20)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    axs[j].step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='k', where='post', alpha=0.2, lw=0)
    axs[j].fill_between(
        np.linspace(
            np.amin(Cbin_edges),
            np.amax(Cbin_edges),
            len(
                np.r_[
                    np.repeat(
                        (Chist /
                         np.float(
                             np.amax(Chist)))[
                            :-
                            1],
                        200),
                    1.03])),
        0,
        np.r_[
            np.repeat(
                (Chist /
                 np.float(
                     np.amax(Chist)))[
                    :-
                    1],
                200),
            1.03],
        color='b',
        alpha=0.08,
        lw=0.0,
        zorder=3)

    if names[j] == 'Gaia-Enceladus':
        axs[j].text(0.97,
                    0.95,
                    str(names[j]),
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(0.97,
                    0.88,
                    '(' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                       selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                   selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.97,
            0.81,
            'Overall halo',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='blue',
            alpha=0.4)

    elif names[j] in ['6', '3', 'A', '12', '28', '38']:
        Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
            'feh_lamost_lrs', selection='GE'), range=(-2.3, 0.1), bins=20)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)

        axs[j].step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color='k',
            linestyle='dashed',
            lw=1,
            where='post')

        axs[j].text(0.03,
                    0.95,
                    str(names[j]) + ' (' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                                        selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                                    selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.03,
            0.88,
            'Gaia-Enceladus',
            horizontalalignment='left',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='black')
        axs[j].text(
            0.03,
            0.81,
            'Overall halo',
            horizontalalignment='left',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='blue',
            alpha=0.4)

    elif names[j] in ['Sequoia', 'Helmi', 'Thamnos 1', 'Thamnos 2', 'GE']:
        Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
            'feh_lamost_lrs', selection='GE'), range=(-2.3, 0.1), bins=20)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)

        axs[j].step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color='k',
            linestyle='dashed',
            lw=1,
            where='post')

        axs[j].text(0.97,
                    0.95,
                    str(names[j]),
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(0.97,
                    0.88,
                    '(' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                       selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                   selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.97,
            0.81,
            'Gaia-Enceladus',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='black')
        axs[j].text(
            0.97,
            0.74,
            'Overall halo',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='blue',
            alpha=0.4)

    else:
        Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
            'feh_lamost_lrs', selection='GE'), range=(-2.3, 0.1), bins=20)
        Chist = np.append(Chist, 0)
        Chist = np.append(0, Chist)
        Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
        axs[j].step(
            Cbin_edges,
            Chist /
            np.float(
                np.amax(Chist)),
            color='k',
            linestyle='dashed',
            lw=1,
            where='post')

        axs[j].text(0.97,
                    0.95,
                    str(names[j]) + ' (' + str(len(df_clusters.evaluate('feh_lamost_lrs',
                                                                        selection='structure')[df_clusters.evaluate('feh_lamost_lrs',
                                                                                                                    selection='structure') > -99.999])) + ' stars)',
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=axs[j].transAxes,
                    fontsize=16,
                    color='black',
                    weight='bold')
        axs[j].text(
            0.97,
            0.88,
            'Gaia-Enceladus',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='black')
        axs[j].text(
            0.97,
            0.81,
            'Overall halo',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='blue',
            alpha=0.4)

    Chist, Cbin_edges = np.histogram(df_clusters.evaluate(
        'feh_lamost_lrs', selection='structure'), range=(-2.3, 0.1), bins=20)
    error, Cbin_centres = np.sqrt(Chist) / 1.5, 0.5 * (Cbin_edges[1:] + Cbin_edges[:-1])
    axs[j].errorbar(
        Cbin_centres,
        Chist /
        np.nanmax(Chist),
        yerr=error /
        np.nanmax(Chist),
        fmt='o',
        ecolor='black',
        ms=0)
    Chist = np.append(Chist, 0)
    Chist = np.append(0, Chist)
    Cbin_edges = np.append(Cbin_edges[0] - (Cbin_edges[1] - Cbin_edges[0]), Cbin_edges)
    axs[j].step(Cbin_edges, Chist / np.float(np.amax(Chist)), color='k', lw=2, where='post')

    axs[j].set_ylim(0.0, 1.05)
    axs[j].set_xlabel('', fontsize=20)  # 20   r'$\rm [Fe/H]$'
    axs[j].set_ylabel('', fontsize=20)  # r'$\rm Normalised \; Number \; of \; stars$'
    axs[j].tick_params(labelsize=0)  # 20
    #axs[j].set_title(names[j], fontsize=20)
    axs[j].minorticks_on()

# axs[0].set_xticks([])
axs[0].tick_params(labelsize=20)
# axs[4].set_xticks([])
axs[4].tick_params(labelsize=20)
# axs[8].set_xticks([])
axs[8].tick_params(labelsize=20)
axs[9].tick_params(labelsize=20)
axs[10].tick_params(labelsize=20)
axs[11].tick_params(labelsize=20)
# axs[12].set_xticks([])
# axs[12].tick_params(labelsize=20)
# axs[13].tick_params(labelsize=20)
'''
axs[0].minorticks_on()
axs[4].minorticks_on()
axs[8].minorticks_on()
axs[12].minorticks_on()
axs[16].minorticks_on()
'''

f0.text(0.53, 0.02, r'$\rm [Fe/H]$', horizontalalignment='center',
        verticalalignment='center', fontsize=25, color='black')
f0.text(
    0.02,
    0.53,
    r'$\rm Normalised \; Number \; of \; stars$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=25,
    rotation=90,
    color='black')

plt.tight_layout(w_pad=0.8, h_pad=0.3, rect=[0.03, 0.03, 1, 1])

plt.show()

# %%
# CaMD plot (iso fit)

names = ['Helmi', '64', 'Sequoia', '62', 'Thamnos 1', 'Thamnos 2', 'GE', '6', '3', 'A', '12', '38']

best_isochrones = [
    '/data/users/ruizlara/isochrones/alpha_enhanced/12200z0010493y248P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/11600z0010493y248P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/12200z0010493y248P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/12400z0012500y249P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/11600z0012500y249P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/12400z0015720y249P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/11400z0023650y250P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/11400z0023650y250P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/10600z0061700y255P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/9000z0096177y257P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/10400z0077300y257P04O1D1E1.isc_gaia-dr3',
    '/data/users/ruizlara/isochrones/alpha_enhanced/10800z0077300y257P04O1D1E1.isc_gaia-dr3']
GE_isochrone = '/data/users/ruizlara/isochrones/alpha_enhanced/11400z0023650y250P04O1D1E1.isc_gaia-dr3'

# Bootstrapping results:
# isochrone for 555 11400z0007251y248P04O1D1E1.isc_gaia-dr3  age = 11.5+-1, met the same
# isochrone for 418 10000z0048116y252P04O1D1E1.isc_gaia-dr3  age = 10.1+-2.0, met very similar
f0, ax = plt.subplots(12, figsize=(19, 17), facecolor='w')  # figsize (x,y)
axs = []
for i in range(3):  # Rows
    for j in range(4):  # Cols
        ax_aux = plt.subplot2grid((3, 4), (i, j))
        axs.append(ax_aux)

for j in np.arange(len(axs)):
    clusters = grouped_clusters[j]
    print(names[j])

    condition = str()
    for cluster in clusters[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters[-1]) + ')'
    df_clusters.select(condition, name='structure')

    axs[j].plot(
        df_clusters.evaluate(
            'COL_bp_rp',
            selection='structure'),
        df_clusters.evaluate(
            'M_G',
            selection='structure'),
        'ko',
        ms=1)

    iso = ascii.read(best_isochrones[j])
    iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
    axs[j].plot(iso_color, iso_mag, '-', color='orange', lw=2)

    if names[j] == 'Gaia-Enceladus':
        axs[j].text(
            0.97,
            0.95,
            'Gaia-Enceladus',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='orange',
            weight='bold')

    else:
        iso = ascii.read(GE_isochrone)
        iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        axs[j].plot(iso_color, iso_mag, '-', color='mediumorchid', lw=2)

        axs[j].text(
            0.97,
            0.95,
            names[j],
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='orange',
            weight='bold')
        axs[j].text(
            0.97,
            0.9,
            'Gaia-Enceladus',
            horizontalalignment='right',
            verticalalignment='center',
            transform=axs[j].transAxes,
            fontsize=16,
            color='mediumorchid')

    #axs[j].set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
    #axs[j].set_ylabel(r'$\rm M_{G}$', fontsize=25)
    axs[j].set_xlim(0.3, 2.3)
    axs[j].set_ylim(7.5, -5.5)
    #axs[j].set_title(names[j], fontsize=20)

    axs[j].set_xlabel('', fontsize=20)  # 20   r'$\rm [Fe/H]$'
    axs[j].set_ylabel('', fontsize=20)  # r'$\rm Normalised \; Number \; of \; stars$'
    axs[j].tick_params(labelsize=0)  # 20
    #axs[j].set_title(names[j], fontsize=20)
    axs[j].minorticks_on()

# axs[0].set_xticks([])
axs[0].tick_params(labelsize=20)
# axs[4].set_xticks([])
axs[4].tick_params(labelsize=20)
# axs[8].set_xticks([])
axs[8].tick_params(labelsize=20)
# axs[12].set_xticks([])
# axs[12].tick_params(labelsize=20)
# axs[13].tick_params(labelsize=20)
# axs[17].set_yticks([])
axs[9].tick_params(labelsize=20)
# axs[14].set_yticks([])
axs[10].tick_params(labelsize=20)
axs[11].tick_params(labelsize=20)
# axs[15].set_yticks([])

f0.text(
    0.53,
    0.025,
    r'$\rm G_{BP}-G_{RP}$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=25,
    color='black')
f0.text(
    0.03,
    0.52,
    r'$\rm M_{G}$',
    horizontalalignment='center',
    verticalalignment='center',
    fontsize=25,
    rotation=90,
    color='black')

plt.tight_layout(w_pad=0.1, h_pad=0.3, rect=[0.03, 0.03, 1, 1])

plt.show()

# %%
df_clusters.select('def_cluster_id==28', name='prob')

plt.hist(df_clusters.evaluate('distance', selection='prob'), histtype='step')

# %%
plt.hist(df_clusters.evaluate('ra_parallax_corr', selection='prob'), histtype='step')

# %%
plt.hist(df_clusters.evaluate('dec_parallax_corr', selection='prob'), histtype='step')

# %%
plt.hist(df_clusters.evaluate('parallax_pmra_corr', selection='prob'), histtype='step')

# %%
plt.hist(df_clusters.evaluate('parallax_pmdec_corr', selection='prob'), histtype='step')

# %%
plt.hist(df_clusters.evaluate('parallax_error', selection='prob'), histtype='step')

# %% [markdown]
# ## L) PCA of derived substructures:

# %%
# GE as a whole:
names = ['Helmi', '64', 'Sequoia', '62', 'Thamnos 1', 'Thamnos 2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters = [[60,
                     61],
                    [64],
                    [63],
                    [62],
                    [37],
                    [8,
                     11],
                    [14,
                     56,
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
                    [38]]
clusters_GE = [14, 56, 59, 10, 7, 13, 57, 58, 54, 50, 53, 47, 44, 51, 48, 45, 25,
               29, 41, 49, 27, 22, 26, 52, 55, 20, 19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24]

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')


# %%
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

    cmap, norm = colors.from_levels_and_colors(unique_labels, cmaplist, extend='max')

    return cmap, norm

############################################################################


def confidence_ellipse(x, cov, i, j, ax, n_std, colour, **kwargs):
    """
    Draws an ellipse.

    Parameters:
    x(np.array): Mean vector of a multivariate gaussian
    cov(np.ndarray): Full covariance matrix of the multivariate gaussian
    i, j: indices of features to consider

    Returns:
    A patch added to the specified axis.
    """

    pearson = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

    # in some rare cases (probably) rounding issues are causing the pearson coefficient
    # to minimally exceed 1, which should not be possible, so round this down.
    if(pearson > 1):
        pearson = 0.999999

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl visualization data.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=colour, alpha=0.5, lw=1, edgecolor=colour)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[i, i]) * n_std
    mean_x = np.mean(x[i])

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[j, j]) * n_std
    mean_y = np.mean(x[j])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

############################################################################


def confidence_ellipse_seq2(x, cov, i, j, ax, n_std, colour, **kwargs):
    """
    Draws an ellipse.

    Parameters:
    x(np.array): Mean vector of a multivariate gaussian
    cov(np.ndarray): Full covariance matrix of the multivariate gaussian
    i, j: indices of features to consider

    Returns:
    A patch added to the specified axis.
    """

    pearson = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])

    # in some rare cases (probably) rounding issues are causing the pearson coefficient
    # to minimally exceed 1, which should not be possible, so round this down.
    if(pearson > 1):
        pearson = 0.999999

    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl visualization data.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y *
                      2, lw=2, ls='--', color=colour, fill=False, alpha=0.5)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[i, i]) * n_std
    mean_x = np.mean(x[i])

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[j, j]) * n_std
    mean_y = np.mean(x[j])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


# %%
def plot_substructures_En_Lz(df, colours_def, savepath=None):
    '''
    Plots substructures in En-Lz, displaying their 95% extent and covariance in this 2D projection.

    Parameters:
    df(vaex.DataFrame): Our star catalogue containing substructure labels
    savepath(str): Path to save location of plot, if None the plot is only displayed
    '''

    names = [
        'noise',
        'Helmi',
        '64',
        'Sequoia',
        '62',
        'Thamnos 1',
        'Thamnos 2',
        'GE',
        '6',
        '3',
        'A',
        '12',
        '38']
    name_x_offset = [-0.25, -0.25, 0.2, 0.0, -0.25, -0., 0.25, -0.25, -0.15, -0.25, 0.5, -0.25, -0.2]
    name_y_offset = [0.0, 0.0, 1.3, -1.3, 0.3, 0.7, -0.6, -0.2, 0.0, 0.0, 0.4, 0.0, 0.0]

    fig, axs = plt.subplots(figsize=[8, 7.0], facecolor='w')

    axs.scatter(df.Lz.values / 10**3, df.En.values / 10**4, alpha=0.2, c='grey', s=1.0)

    for jj, substructure in enumerate(np.unique(df.labels_substructure.values)):

        if(substructure == 0):
            # let's not plot an ellipse for the background noise
            continue
        elif(substructure > 13):
            continue
        df_substructure = df[df.labels_substructure == substructure]

        mu = np.mean(df_substructure['Lz/10**3', 'En/10**4'].values.T, axis=1)
        cov = np.cov(df_substructure['Lz/10**3', 'En/10**4'].values.T)

        colour = colours_def[jj]

        # , alpha=0.8, edgecolor='black', zorder=2)
        confidence_ellipse(mu, cov, 0, 1, ax=axs, n_std=1.77, colour=colour)

        axs.text(mu[0] + name_x_offset[jj],
                 mu[1] + name_y_offset[jj],
                 r'$\rm ' + str(names[jj]) + '$',
                 horizontalalignment='right',
                 verticalalignment='center',
                 fontsize=16)  # , color=colour)

    axs.set_ylabel(r'$E \; [10^4 \; km^{\rm 2}/s^{\rm 2}]$', fontsize=21)
    axs.set_xlabel(r'$Lz\; [10^3 \; kpc \; km/s]$', fontsize=21)
    axs.tick_params(labelsize=18)

    axs.set_xlim(-4.5, 4.5)
    axs.set_ylim(-17.5, 0.2)

    # Alternative Sequoia:
    df2 = vaex.open(
        '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_substructure_D_80_cut-alt_sequoia.hdf5')
    df2_substructure = df2[df2.labels_substructure == 2]

    mu = np.mean(df2_substructure['Lz/10**3', 'En/10**4'].values.T, axis=1)
    cov = np.cov(df2_substructure['Lz/10**3', 'En/10**4'].values.T)
    axs.text(-2.0, -7.4, r'$\rm Sequoia*$', horizontalalignment='left',
             verticalalignment='center', fontsize=16, alpha=0.7)
    axs.text(-2.0, -8.0, r'$\rm (\#62 \cup \#63 \cup \#64)$', horizontalalignment='left',
             verticalalignment='center', fontsize=10, alpha=0.7)

    confidence_ellipse_seq2(mu, cov, 0, 1, ax=axs, n_std=1.77, colour='black')

    plt.tight_layout()

    plt.ylim(-17.0, -4.0)
    plt.xlim(-4.0, 3.0)

    if(savepath is not None):
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()

#colours = []
#colours.extend(cm.jet(np.linspace(0, 1, 8)))
#colours_def = ['none', 'black', colours[0], colours[1], 'black', colours[2], colours[2], 'black', 'black', colours[3], colours[4], colours[4], colours[4], 'black', 'black', colours[6], colours[7], colours[7], 'black']


n_substructures = 12
cmap = plt.get_cmap('gist_ncar', n_substructures)
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[-1] = 'indigo'
cmaplist[-2] = 'mediumvioletred'
#cmaplist[11] = 'black'
cmaplist[6] = 'darkorange'
cmaplist[7] = (1.0, 0.9411674780592812, 0.0, 1.0)
cmap_aux = ['black']
cmap_aux.extend(cmaplist)
cmap_def = cmap_aux
colours_def = cmap_def
print(colours_def)

df = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_substructure_D_80_cut.hdf5')
plot_substructures_En_Lz(df, colours_def)

# %%
names = ['Helmi', '64', 'Sequoia', '62', 'Thamnos 1', 'Thamnos 2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters = [[60,
                     61],
                    [64],
                    [63],
                    [62],
                    [37],
                    [8,
                     11],
                    [14,
                     56,
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
                    [38]]
colours = [
    (0.0,
     0.0,
     0.502,
     1.0),
    (0.0,
     0.10163235635038903,
     0.6858583178726035,
     1.0),
    (0.0,
     0.8506542602495544,
     1.0,
     1.0),
    (0.0,
     0.9812871179883946,
     0.5098860935329526,
     1.0),
    (0.3428526315789473,
     0.8267768983957219,
     0.0,
     1.0),
    (0.6415879201854611,
     1.0,
     0.1547211811997526,
     1.0),
    'darkorange',
    (1.0,
     0.9411674780592812,
     0.0,
     1.0),
    (1.0,
     0.08477050092764371,
     0.0,
     1.0),
    (0.8210141414141412,
     0.08948158599749975,
     1.0,
     1.0),
    'mediumvioletred',
    'indigo']
for name, col in zip(names, colours):
    print(name, col)

# %%
df = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_substructure_D_80_cut.hdf5')
df

# %%
print(len(df.evaluate('source_id')))
print(len(df.evaluate('derived_labels_substructures')[df.evaluate('derived_labels_substructures') > 0.0]))
print(100.0 * float(len(df.evaluate('derived_labels_substructures')
      [df.evaluate('derived_labels_substructures') > 0.0])) / float(len(df.evaluate('source_id'))))

# %% [markdown]
# ## M) Appendix

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')  # no segue within quality.
df_clusters.select('(distance<2.5)', name='int_dist')  # intermediate distance, d < 2.5 kpc

# %%
# Adding angular momentum and other things
compute_coordinates_new(df_clusters)

# %%
names = ['Helmi', '64', 'Sequoia', '62', 'Thamnos 1', 'Thamnos 2', 'GE', '6', '3', 'A', '12', '38']
grouped_clusters = [[60,
                     61],
                    [64],
                    [63],
                    [62],
                    [37],
                    [8,
                     11],
                    [14,
                     56,
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
                    [38]]

# %%
for i in np.arange(len(names)):
    substr_IoM_paper_GOOD(df_clusters, clusters=grouped_clusters[i], name=names[i], colour='blue')

# %% [markdown]
# ## N) CaMD in detail of substructure A and cluster #12

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')  # no segue within quality.
df_clusters.select('(distance<2.5)', name='int_dist')  # intermediate distance, d < 2.5 kpc
df_clusters.select(
    '((COL_bp_rp > 0.55)|((COL_bp_rp > 0.3)&(COL_bp_rp < 0.55)&(M_G > 3.5)))',
    name='good_CaMD')

df_clusters

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
detailed_CaMD_comparison(df=df_clusters, path=path, main_names=['A'], mains=[
                         [15, 16, 46, 31, 32, 30, 42, 18, 33, 34]], extra_clusters=[12])

# %%
df_clusters.select('(def_cluster_id==12)', name='this')
df_clusters.select(
    '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))',
    name='quality')  # no segue within quality.
df_clusters.select('(def_cluster_id==12)&(feh_lamost_lrs<-1.46)', name='this_low')
df_clusters.select('(def_cluster_id==12)&(feh_lamost_lrs>-1.46)', name='this_high')
df_clusters.select(
    '(def_cluster_id==12)&(feh_lamost_lrs<-1.46)&(feh_lamost_lrs>-1.7)',
    name='this_second_peak')

# %%
plt.figure()
plt.hist(df_clusters.evaluate('feh_lamost_lrs', selection='this'), bins=40)
plt.axvline(-1.46)
plt.axvline(-1.7)
plt.show()

# %%

# %%
print('Cluster12')
print('average [Fe/H]: ' + str(np.nanmean(df_clusters.evaluate('feh_lamost_lrs', selection='(this)&(quality)'))))
print('sigma [Fe/H]: ' + str(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='(this)&(quality)'))))
print('N stars: ' + str(len(df_clusters.evaluate('feh_lamost_lrs', selection='(this)&(quality)'))))
print('high met')
print('average [Fe/H]: ' +
      str(np.nanmean(df_clusters.evaluate('feh_lamost_lrs', selection='(this_high)&(quality)'))))
print('sigma [Fe/H]: ' + str(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='(this_high)&(quality)'))))
print('N stars: ' + str(len(df_clusters.evaluate('feh_lamost_lrs', selection='(this_high)&(quality)'))))
print('low met')
print('average [Fe/H]: ' +
      str(np.nanmean(df_clusters.evaluate('feh_lamost_lrs', selection='(this_low)&(quality)'))))
print('sigma [Fe/H]: ' + str(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='(this_low)&(quality)'))))
print('N stars: ' + str(len(df_clusters.evaluate('feh_lamost_lrs', selection='(this_low)&(quality)'))))
print('low met (peak)')
print('average [Fe/H]: ' + str(np.nanmean(df_clusters.evaluate('feh_lamost_lrs',
      selection='(this_second_peak)&(quality)'))))
print('sigma [Fe/H]: ' +
      str(np.nanstd(df_clusters.evaluate('feh_lamost_lrs', selection='(this_second_peak)&(quality)'))))
print('N stars: ' + str(len(df_clusters.evaluate('feh_lamost_lrs', selection='(this_second_peak)&(quality)'))))


# %%
newfile = open('high.txt', 'w')
newfile.write('# Color, Color_err, abs_mag, abs_mag_err, AG \n')
print('# Color, Color_err, abs_mag, abs_mag_err, AG')
for i in np.arange(len(df_clusters.evaluate('source_id', selection='this_high'))):
    print(str(df_clusters.evaluate('COL_bp_rp',
                                   selection='this_high')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                               selection='this_high')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                           selection='this_high')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                       selection='this_high')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                                   selection='this_high')[i]))
    newfile.write(str(df_clusters.evaluate('COL_bp_rp',
                                           selection='this_high')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                                       selection='this_high')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                                   selection='this_high')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                               selection='this_high')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                                           selection='this_high')[i]) + ' \n')
newfile.close()

# %%
newfile = open('low.txt', 'w')
newfile.write('# Color, Color_err, abs_mag, abs_mag_err, AG \n')
print('# Color, Color_err, abs_mag, abs_mag_err, AG')
for i in np.arange(len(df_clusters.evaluate('source_id', selection='this_low'))):
    print(str(df_clusters.evaluate('COL_bp_rp',
                                   selection='this_low')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                              selection='this_low')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                         selection='this_low')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                    selection='this_low')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                               selection='this_low')[i]))
    newfile.write(str(df_clusters.evaluate('COL_bp_rp',
                                           selection='this_low')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                                      selection='this_low')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                                 selection='this_low')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                            selection='this_low')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                                       selection='this_low')[i]) + ' \n')
newfile.close()

# %%
newfile = open('peak.txt', 'w')
newfile.write('# Color, Color_err, abs_mag, abs_mag_err, AG \n')
print('# Color, Color_err, abs_mag, abs_mag_err, AG')
for i in np.arange(len(df_clusters.evaluate('source_id', selection='this_second_peak'))):
    print(str(df_clusters.evaluate('COL_bp_rp',
                                   selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                                      selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                                         selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                                            selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                                                               selection='this_second_peak')[i]))
    newfile.write(str(df_clusters.evaluate('COL_bp_rp',
                                           selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                                              selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                                                 selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                                                    selection='this_second_peak')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                                                                       selection='this_second_peak')[i]) + ' \n')
newfile.close()

# %%
newfile = open('cluster12.txt', 'w')
newfile.write('# Color, Color_err, abs_mag, abs_mag_err, AG \n')
print('# Color, Color_err, abs_mag, abs_mag_err, AG')
for i in np.arange(len(df_clusters.evaluate('source_id', selection='this'))):
    print(str(df_clusters.evaluate('COL_bp_rp',
                                   selection='this')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                          selection='this')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                 selection='this')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                        selection='this')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                               selection='this')[i]))
    newfile.write(str(df_clusters.evaluate('COL_bp_rp',
                                           selection='this')[i]) + ',' + str(df_clusters.evaluate('err_COL_bp_rp',
                                                                                                  selection='this')[i]) + ',' + str(df_clusters.evaluate('M_G',
                                                                                                                                                         selection='this')[i]) + ',' + str(df_clusters.evaluate('err_MG',
                                                                                                                                                                                                                selection='this')[i]) + ',' + str(df_clusters.evaluate('A_G',
                                                                                                                                                                                                                                                                       selection='this')[i]) + ' \n')
newfile.close()

# %%
f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,
     ax12) = plt.subplots(12, figsize=(32, 6), facecolor='w')

ax1 = plt.subplot(1, 5, 1)
ax2 = plt.subplot(1, 5, 2)
ax3 = plt.subplot(1, 5, 3)
ax4 = plt.subplot(1, 5, 4)
ax5 = plt.subplot(1, 5, 5)

ax1.scatter(df_clusters.evaluate('vR'), df_clusters.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
ax2.scatter(df_clusters.evaluate('vz'), df_clusters.evaluate('vR'), c='grey', s=1.0, alpha=0.2)
ax3.scatter(df_clusters.evaluate('vz'), df_clusters.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
ax4.scatter(df_clusters.evaluate('Lz') /
            10**3, df_clusters.evaluate('En') /
            10**4, c='grey', s=1.0, alpha=0.2)
ax5.scatter(df_clusters.evaluate('Lz') /
            10**3, df_clusters.evaluate('Lperp') /
            10**3, c='grey', s=1.0, alpha=0.2)

ax1.scatter(
    df_clusters.evaluate(
        'vR',
        selection='this_high'),
    df_clusters.evaluate(
        'vphi',
        selection='this_high'),
    c='blue',
    s=10.0,
    alpha=0.8,
    label='high metallicity')
ax2.scatter(
    df_clusters.evaluate(
        'vz',
        selection='this_high'),
    df_clusters.evaluate(
        'vR',
        selection='this_high'),
    c='blue',
    s=10.0,
    alpha=0.8)
ax3.scatter(
    df_clusters.evaluate(
        'vz',
        selection='this_high'),
    df_clusters.evaluate(
        'vphi',
        selection='this_high'),
    c='blue',
    s=10.0,
    alpha=0.8)
ax4.scatter(
    df_clusters.evaluate(
        'Lz',
        selection='this_high') /
    10**3,
    df_clusters.evaluate(
        'En',
        selection='this_high') /
    10**4,
    c='blue',
    s=10.0,
    alpha=0.8)
ax5.scatter(
    df_clusters.evaluate(
        'Lz',
        selection='this_high') /
    10**3,
    df_clusters.evaluate(
        'Lperp',
        selection='this_high') /
    10**3,
    c='blue',
    s=10.0,
    alpha=0.8)

ax1.scatter(df_clusters.evaluate('vR', selection='this_low'), df_clusters.evaluate(
    'vphi', selection='this_low'), c='red', s=10.0, alpha=0.8, label='low metallicity')
ax2.scatter(
    df_clusters.evaluate(
        'vz',
        selection='this_low'),
    df_clusters.evaluate(
        'vR',
        selection='this_low'),
    c='red',
    s=10.0,
    alpha=0.8)
ax3.scatter(
    df_clusters.evaluate(
        'vz',
        selection='this_low'),
    df_clusters.evaluate(
        'vphi',
        selection='this_low'),
    c='red',
    s=10.0,
    alpha=0.8)
ax4.scatter(
    df_clusters.evaluate(
        'Lz',
        selection='this_low') /
    10**3,
    df_clusters.evaluate(
        'En',
        selection='this_low') /
    10**4,
    c='red',
    s=10.0,
    alpha=0.8)
ax5.scatter(
    df_clusters.evaluate(
        'Lz',
        selection='this_low') /
    10**3,
    df_clusters.evaluate(
        'Lperp',
        selection='this_low') /
    10**3,
    c='red',
    s=10.0,
    alpha=0.8)

ax1.scatter(
    df_clusters.evaluate(
        'vR',
        selection='this_second_peak'),
    df_clusters.evaluate(
        'vphi',
        selection='this_second_peak'),
    c='yellow',
    s=10.0,
    alpha=0.8,
    label='low metallicity (restrictive)')
ax2.scatter(
    df_clusters.evaluate(
        'vz',
        selection='this_second_peak'),
    df_clusters.evaluate(
        'vR',
        selection='this_second_peak'),
    c='yellow',
    s=10.0,
    alpha=0.8)
ax3.scatter(
    df_clusters.evaluate(
        'vz',
        selection='this_second_peak'),
    df_clusters.evaluate(
        'vphi',
        selection='this_second_peak'),
    c='yellow',
    s=10.0,
    alpha=0.8)
ax4.scatter(
    df_clusters.evaluate(
        'Lz',
        selection='this_second_peak') /
    10**3,
    df_clusters.evaluate(
        'En',
        selection='this_second_peak') /
    10**4,
    c='yellow',
    s=10.0,
    alpha=0.8)
ax5.scatter(
    df_clusters.evaluate(
        'Lz',
        selection='this_second_peak') /
    10**3,
    df_clusters.evaluate(
        'Lperp',
        selection='this_second_peak') /
    10**3,
    c='yellow',
    s=10.0,
    alpha=0.8)

ax1.set_xlabel('vR', fontsize=25)
ax1.set_ylabel('vphi', fontsize=25)
ax2.set_xlabel('vz', fontsize=25)
ax2.set_ylabel('vR', fontsize=25)
ax3.set_xlabel('vz', fontsize=25)
ax3.set_ylabel('vphi', fontsize=25)
ax4.set_xlabel('Lz', fontsize=25)
ax4.set_ylabel('E', fontsize=25)
ax5.set_xlabel('Lz', fontsize=25)
ax5.set_ylabel('Lperp', fontsize=25)

ax1.legend()

plt.show()

# %% [markdown]
# ## O) Quick check

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
df_subs = vaex.open(
    '/net/gaia2/data/users/dodd/Clustering_EDR3/Clustering_results_3D/results/df_labels_3D_substructure_D_80_cut.hdf5')
df_subs

# %%
plt.figure()
plt.plot(df_clusters.evaluate('def_cluster_id'), df_subs.evaluate('derived_labels'))
plt.show()

# %% [markdown]
# # Amina's doubts

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
df_clusters.select('def_cluster_id==2.0', name='oio')
df_clusters.select('(def_cluster_id==2.0)&(feh_lamost_lrs<-0.4)&(feh_lamost_lrs>-0.8)', name='oio2')

# %%
print(len(df_clusters.evaluate('source_id', selection='oio')))

# %%
plt.figure()
plt.hist(df_clusters.evaluate('Lz', selection='oio') / 10**3)
plt.show()

# %%
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
df_clusters

# %%
make_IoM_vels_plot_def_cluster_id(df_clusters, clusters=[2], alpha=0.5, name='test')

# %%
f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,
     ax12) = plt.subplots(12, figsize=(29, 8), facecolor='w')

ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)

ax1.scatter(df_clusters.evaluate('vR'), df_clusters.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
ax2.scatter(df_clusters.evaluate('vz'), df_clusters.evaluate('vR'), c='grey', s=1.0, alpha=0.2)
ax3.scatter(df_clusters.evaluate('vz'), df_clusters.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)

ax1.scatter(
    df_clusters.evaluate(
        'vR',
        selection='oio'),
    df_clusters.evaluate(
        'vphi',
        selection='oio'),
    c='blue',
    s=10.0,
    alpha=0.8)
ax2.scatter(
    df_clusters.evaluate(
        'vz',
        selection='oio'),
    df_clusters.evaluate(
        'vR',
        selection='oio'),
    c='blue',
    s=10.0,
    alpha=0.8)
ax3.scatter(
    df_clusters.evaluate(
        'vz',
        selection='oio'),
    df_clusters.evaluate(
        'vphi',
        selection='oio'),
    c='blue',
    s=10.0,
    alpha=0.8)


ax1.scatter(
    df_clusters.evaluate(
        'vR',
        selection='oio2'),
    df_clusters.evaluate(
        'vphi',
        selection='oio2'),
    c='red',
    s=10.0,
    alpha=0.8)
ax2.scatter(
    df_clusters.evaluate(
        'vz',
        selection='oio2'),
    df_clusters.evaluate(
        'vR',
        selection='oio2'),
    c='red',
    s=10.0,
    alpha=0.8)
ax3.scatter(
    df_clusters.evaluate(
        'vz',
        selection='oio2'),
    df_clusters.evaluate(
        'vphi',
        selection='oio2'),
    c='red',
    s=10.0,
    alpha=0.8)

ax1.set_xlabel('vR', fontsize=25)
ax1.set_ylabel('vphi', fontsize=25)
ax2.set_xlabel('vz', fontsize=25)
ax2.set_ylabel('vR', fontsize=25)
ax3.set_xlabel('vz', fontsize=25)
ax3.set_ylabel('vphi', fontsize=25)

plt.show()

# %%

# %%
df_clusters.select('(def_cluster_id==12)&(feh_lamost_lrs<-1.4)', name='oio12')

# %%
print(len(df_clusters.evaluate('source_id', selection='oio2')))

# %% [markdown]
# # Figure 2 without printed information

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
iso_dict = {}
ages_iso, mets_iso = [], []
names_isos = glob.glob('/data/users/ruizlara/isochrones/alpha_enhanced/*.isc_gaia-dr3')
for i, name_iso in enumerate(names_isos):
    lines_iso = open(names_isos[i], 'r').readlines()
    iso_age, iso_mh = lines_iso[4].split('=')[5].split()[0], lines_iso[4].split('=')[2].split()[0]
    ages_iso.append(iso_age)
    mets_iso.append(float(iso_mh))
    iso_dict['age' + str(iso_age) + '-met' + str(iso_mh)] = {'name_iso': name_iso}

# %%
apply_CaMD_fitting(
    df_clusters,
    clusters=[
        60,
        61],
    path_CMD=path,
    name='Helmi',
    iso_dict=iso_dict,
    mah_dist_lim=2.13)


# %% [markdown]
# # substructure F

# %%
F1 = np.array([-1.614, -1.675, -1.735, -1.378, -1.852, -2.07, -1.615, -2.124])
F2 = np.array([-1.187, -1.238, -2.194, -1.225, -1.38, -1.432, -1.15, -1.317, -1.103])
both = np.array([-1.614, -1.675, -1.735, -1.378, -1.852, -2.07, -1.615, -2.124, -
                1.187, -1.238, -2.194, -1.225, -1.38, -1.432, -1.15, -1.317, -1.103])

# %%
plt.figure()
plt.hist(F1, range=(-2.1, -1.1), histtype='step', color='red', label='cluster 65')
plt.hist(F2, range=(-2.1, -1.1), histtype='step', color='blue', label='cluster 66')
#plt.hist(both, range=(-2.1, -1.1), histtype='step', color='black', label='F')
plt.legend()
plt.xlabel(r'$[Fe/H]$')
plt.ylabel(r'$N$')
plt.show()

# %% [markdown]
# # Nothing to do really with substructures... more SFH of Helmi (obtain MDF of Helmi and halo)

# %%
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')
df_clusters

# %%
df_clusters.select('feh_lamost_lrs>-20.0', name='lamost_good')

# %%
columns = ['source_id', 'def_cluster_id', 'feh_lamost_lrs', 'feh_error_lamost_lrs']
os.system('rm halo_clusters_mets_lamost.hdf5')
df_clusters[columns].filter('(lamost_good)').extract().export_hdf5(
    'halo_clusters_mets_lamost.hdf5', progress=True)

# %%
df = vaex.open('halo_clusters_mets_lamost.hdf5')
df

# %%
df.export_csv('halo_clusters_mets_lamost.csv')
df = pd.read_csv('halo_clusters_mets_lamost.csv')
df

# %%
