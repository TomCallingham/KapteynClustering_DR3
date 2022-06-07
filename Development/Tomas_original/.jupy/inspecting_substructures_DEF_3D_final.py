# %%
import pandas as pd  
import numpy as np
import sys,glob,os,time
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
# Reddening computation and moving magnitudes from apparent to absolute plane.
# Updated version of reddening_computation to september 2021 (adding errors and 'else')

def reddening_computation(self):

    # Compute ebv from one of the published 3D maps:
    if dust_map == 'l18':
        # We use the dust map from lallement18. It uses dust_maps_3d/lallement18.py and the file lallement18_nside64.fits.gz
        EBV = l18.ebv(self.evaluate('l'), self.evaluate('b'), self.evaluate('distance')*1000.0)
        delta_EBV = EBV*0.0
    elif dust_map == 'p19':
        # We use the dust map from pepita19. It would use dust_maps_3d/pepita19.py and the file pepita19.fits.gz
        EBV = p19.ebv(self.evaluate('l'), self.evaluate('b'), self.evaluate('distance')) 
        delta_EBV = EBV*0.0
    elif dust_map == 'bayestar':       
        bayestar = BayestarQuery(max_samples=1, version='bayestar2019')
        coords = SkyCoord(self.evaluate('l')*units.deg, self.evaluate('b')*units.deg, distance=self.evaluate('distance')*units.kpc, frame='galactic')        
        EBV = bayestar(coords, mode='mean')
        EBV_perc = bayestar(coords, mode='percentile', pct=[16., 84.])
        delta_EBV = EBV_perc[:,1]-EBV_perc[:,0]
        del bayestar, EBV_perc
        # Problems with Green catalogue: 1- Not EBV; 2- zero values.
        # 1: True_EBV = 0.884*EBV, taken from http://argonaut.skymaps.info/usage#units
        EBV, delta_EBV = 0.884*EBV, 0.884*delta_EBV
        # 2: Discretization from Green email, if EBV = 0.0, assume a number in the range 0.0-0.02 (positive I guess, right?)
        coords = np.where(EBV==0.0)
        EBV[coords] = np.random.uniform(0.0, 0.02, len(EBV[coords]))
        coords = np.where(delta_EBV==0.0)
        delta_EBV[coords] = np.random.uniform(0.0, 0.02, len(delta_EBV[coords]))   
        
    # From those E(B-V) compute the AG and E_BP_RP values

    if ext_rec == 'cas18':

        # Colour--Teff relation determined from Gaia DR2 data --> Computed using color_teff_GDR2.ipynb
        poly = np.array([62.55257114, -975.10845442, 4502.86260828, -8808.88745592, 10494.72444183])
        Teff = np.poly1d(poly)(self.evaluate('bp_rp')) / 1e4
        
        # We enter in an iterative process of 10 steps to refine the AG and E_BP_RP computation. 'Cause we do not have intrinsic colour.
        for i in range(10):
            E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * EBV # A_BP - A_RP
            AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * EBV
            Teff = np.poly1d(poly)(self.evaluate('bp_rp') - E_BP_RP) / 1e4
        delta_AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * delta_EBV
        delta_E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * delta_EBV
        
    elif ext_rec == 'bab18':
        ### Babusiaux et al. 2018 --> http://aselfabs.harvard.edu/abs/2018A%26A...616A..10G (eq 1, Table 1)
        
        A0 = 3.1*EBV
        bp_rp_0 = self.evaluate('bp_rp')
        
        # We enter in an iterative process of 10 steps to refine the kBP, kRP, and kG values. 'Cause we do not have intrinsic colour.
        for i in range(3):
            kG  = 0.9761 - 0.1704*bp_rp_0 + 0.0086*bp_rp_0**2 + 0.0011*bp_rp_0**3 - 0.0438*A0 + 0.00130*A0**2 + 0.0099*bp_rp_0*A0
            kBP = 1.1517 - 0.0871*bp_rp_0 - 0.0333*bp_rp_0**2 + 0.0173*bp_rp_0**3 - 0.0230*A0 + 0.00060*A0**2 + 0.0043*bp_rp_0*A0
            kRP = 0.6104 - 0.0170*bp_rp_0 - 0.0026*bp_rp_0**2 - 0.0017*bp_rp_0**3 - 0.0078*A0 + 0.00005*A0**2 + 0.0006*bp_rp_0*A0
            bp_rp_0 = bp_rp_0 - (kBP-kRP)*A0 # 0 -- intrinsic, otherwise observed, Ax = mx - m0x --> m0x = mx - Ax = mx - kx*A0  ==> m0BP - m0RP = mbp - mrp - (kbp - krp)*A0 = bp_rp
          
        AG = kG*A0  
        delta_A0 = 3.1*delta_EBV
        delta_kG = (-0.0438 + 2.0*0.00130*A0 + 0.0099*bp_rp_0)*delta_A0
        delta_AG = np.sqrt((A0*delta_kG)**2 + (kG*delta_A0)**2)
        del delta_kG
        
        E_BP_RP = (kBP-kRP)*A0 # A_BP - A_RP
        delta_kBP_kRP = (-0.0152 + 2.0*0.00055*A0 + 0.0037*bp_rp_0)*delta_A0
        delta_E_BP_RP = np.sqrt((A0*delta_kBP_kRP)**2 + ((kBP-kRP)*delta_A0)**2)
        del A0, bp_rp_0, kG, kBP, kRP, delta_kBP_kRP, delta_A0
         
    elif ext_rec == 'else':
        AG = 0.86117*3.1*EBV
        Abp = 1.06126*3.1*EBV
        Arp = 0.64753*3.1*EBV
        E_BP_RP = Abp - Arp    
        
        delta_AG = 0.86117*3.1*delta_EBV
        delta_BP_RP = np.sqrt((1.06126*3.1*EBV)**2 + (0.64753*3.1*EBV)**2)

    COL_bp_rp = self.evaluate('bp_rp') - E_BP_RP
    MG = self.evaluate('phot_g_mean_mag')-AG+5.-5.*np.log10(1000.0/self.evaluate('new_parallax'))
    
    #return E_BP_RP, delta_E_BP_RP, AG, delta_AG, COL_bp_rp, MG
    return E_BP_RP, A_G, COL_bp_rp, M_G


# %%
def compare_MDFs_CMDs_low_Nstars(df, clusters_CM, labels, colours, zorder, Nstars_lim, NMC):
    df.select('(feh_lamost_lrs>-10000.0)', name = 'lamost')

    # Compare the MDF of two structures:
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = str()
        for cluster in clusters[:-1]:
            condition = condition+'(def_cluster_id=='+str(cluster)+')|'
        condition = condition+'(def_cluster_id=='+str(clusters[-1])+')'
        df.select(condition, name = 'structure')
        distr_dict[str(system)] = {'feh_distr': df.evaluate('feh_lamost_lrs', selection='(structure)&(lamost)'),
                                   'mag': df.evaluate('M_G', selection='(structure)'), 
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)'),
                                   'en': df.evaluate('En', selection='(structure)')/10**4, 
                                   'lz': df.evaluate('Lz', selection='(structure)')/10**3, 
                                   'lperp': df.evaluate('Lperp', selection='(structure)')/10**3}  
#    for system, clusters in zip(labels, clusters_CM):   
#        print(system)
#        print(str(len(distr_dict[system]['feh_distr']))+' good stars with LAMOST LRS [Fe/H]')
        
    main_feh, second_feh = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    main_feh = main_feh[np.isnan(main_feh)==False]
    second_feh = second_feh[np.isnan(second_feh)==False]        
    
    if len(second_feh) < Nstars_lim:
        f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, figsize=(18,15), facecolor='w')
        ax1 = plt.subplot(331)
        ax2 = plt.subplot(332)
        ax3 = plt.subplot(333)
        ax4 = plt.subplot(334)
        ax5 = plt.subplot(335)
        ax6 = plt.subplot(336)    
        ax7 = plt.subplot(337)
        ax8 = plt.subplot(338)         
    else:
        f0, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(18,11), facecolor='w')
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(232)
        ax3 = plt.subplot(233)
        ax4 = plt.subplot(234)
        ax5 = plt.subplot(235)
        ax6 = plt.subplot(236)    
    
    ax4.scatter(df.evaluate('Lz')/(10**3), df.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('Lperp')/(10**3), df.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)    
    ax6.scatter(df.evaluate('Lz')/(10**3), df.evaluate('Lperp')/(10**3), c='grey', s=1.0, alpha=0.2)    
    
    for i in np.arange(len(labels)):
        ax1.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, label=labels[i])
        ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])  
        ax3.plot(distr_dict[labels[i]]['col'], distr_dict[labels[i]]['mag'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax4.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax5.plot(distr_dict[labels[i]]['lperp'], distr_dict[labels[i]]['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax6.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]['lperp'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        
    indexes = np.arange(len(clusters_CM))
    pairs = list(itertools.combinations(indexes,2))
    count=0
    for pair in pairs:
        feh_x, feh_y = distr_dict[labels[pair[0]]]['feh_distr'], distr_dict[labels[pair[1]]]['feh_distr']
        feh_x = feh_x[np.isnan(feh_x)==False]
        feh_y = feh_y[np.isnan(feh_y)==False]
        KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
        if len(second_feh) < Nstars_lim: 
            ax8.axhline(pval, ls='--', color='black')
            orig_pval = pval
        else:
            orig_pval = pval
            comp_percentage = 999.9
            less_percentage = 999.9
        
        ax2.text(0.95, 0.05+0.05*count, str(labels[pair[0]])+'-'+str(labels[pair[1]])+': '+str(round(pval,3)), horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
        count=count+1
        
    ax1.text(0.04, 0.95, 'N = '+str(len(main_feh)), color=colours[0], transform=ax1.transAxes, fontsize=14)
    ax1.text(0.04, 0.90, 'N = '+str(len(second_feh)), color=colours[1], transform=ax1.transAxes, fontsize=14)
    
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
    

    
    if len(second_feh) < Nstars_lim:
                
        ax7.hist(main_feh, histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[0], lw=2, label=labels[0], zorder=1)
        ax7.hist(second_feh, histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[1], lw=2, label=labels[1], zorder=1)
        
        NNs, pvals = [], []
    
        count = 0
        while count < NMC:
            this_second_feh = np.random.choice(main_feh,len(second_feh))
            KS, pval = stats.ks_2samp(main_feh, this_second_feh, mode='exact')
            NNs.append(count)
            pvals.append(pval)
            ax7.hist(this_second_feh, histtype='step', range=(-2.7, 0.7), bins=25, density=True, color='gray', lw=2, zorder=0, alpha=0.01)
            count = count + 1
        
        coords = np.where((np.array(pvals)<orig_pval))
        less_percentage = float(len(np.array(pvals)[coords]))/float(len(np.array(pvals)))
    
        coords = np.where((np.array(pvals)<orig_pval+0.05)&(np.array(pvals)>orig_pval-0.05))
        compatibility = 100.0*float(len(np.array(pvals)[coords]))/float(len(np.array(pvals)))
        ax7.text(0.05, 0.94, 'Compatible pvals '+str(round(compatibility, 2))+' %', fontsize=14, transform=ax7.transAxes)
        
        coords = np.where((np.array(pvals)>0.05))
        same_distr = 100.0*float(len(np.array(pvals)[coords]))/float(len(np.array(pvals)))        
        ax7.text(0.05, 0.88, 'Same distr '+str(round(same_distr, 2))+' %', fontsize=14, transform=ax7.transAxes)
        
        coords = np.where((np.array(pvals)<0.05))
        diff_distr = 100.0*float(len(np.array(pvals)[coords]))/float(len(np.array(pvals)))        
        ax7.text(0.05, 0.82, 'Different distr '+str(round(diff_distr, 2))+' %', fontsize=14, transform=ax7.transAxes)
        
        ax7.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', label='MC')
        
        ax8.plot(NNs, pvals, 'ko')
        ax8.axhline(np.nanmean(pvals), ls='--', alpha=0.4, color='black')
        coords = np.where(np.array(pvals)<0.05)
        ax8.plot(np.array(NNs)[coords], np.array(pvals)[coords], 'ro')
        ax8.axhline(0.05, ls='--', alpha=0.4, color='red')
        ax8.text(0.0, np.nanmean(pvals)-0.05, 'pval = '+str(round(np.nanmean(pvals), 2)), fontsize=14)
        
        ax7.legend()
        ax7.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
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
def pval_red_or_green(val):
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s' % color


# %%
def create_tables_defining(df, path, name, main):
    
    df.select('(feh_lamost_lrs>-10000.0)', name = 'lamost')    

    tablefile_csv = open(str(path)+str(name)+'_definition_table.csv', 'w')
    tablefile_csv_percs_same_dec = open(str(path)+str(name)+'_definition_table_percs_same_dec.csv', 'w')
    tablefile_csv_percs_less = open(str(path)+str(name)+'_definition_table_percs_less.csv', 'w')    
    chain1 = ','
    for cluster in main:
        df.select('(def_cluster_id=='+str(cluster)+')', name = 'this_cluster')
        if len(df.evaluate('source_id', selection='(lamost)&(this_cluster)')) < 20.0:
            chain1 = chain1+str(cluster)+'*, '
        else:
            chain1 = chain1+str(cluster)+', '

    chain1 = chain1[0:-2]+' \n'
    tablefile_csv.write(chain1)
    tablefile_csv_percs_same_dec.write(chain1)
    tablefile_csv_percs_less.write(chain1)

    chain2 = str(name)+', '
    chain3 = str(name)+', '
    chain4 = str(name)+', '
    for cluster in main:
        coords = np.where(np.array(main)!=cluster)
        orig_pval, comp_percentage, less_percentage = compare_MDFs_CMDs_low_Nstars(df,clusters_CM=[np.array(main)[coords], [cluster]], labels=[str(name), str(cluster)], colours=['red', 'lightblue'], zorder=[1,2], Nstars_lim=20, NMC=100)
        chain2 = chain2+str(orig_pval)+', '
        chain3 = chain3+str(comp_percentage)+', '
        chain4 = chain4+str(less_percentage)+', '
    chain2 = chain2[0:-2]+' \n'
    chain3 = chain3[0:-2]+' \n'
    chain4 = chain4[0:-2]+' \n'
    tablefile_csv.write(chain2)
    tablefile_csv_percs_same_dec.write(chain3)
    tablefile_csv_percs_less.write(chain4)
    
    tablefile_csv.close()
    tablefile_csv_percs_same_dec.close()
    tablefile_csv_percs_less.close()
    
    df = pd.read_csv(str(path)+str(name)+'_definition_table.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(name))\
        .background_gradient(cmap=cm).applymap(pval_red_or_green))

    df = pd.read_csv(str(path)+str(name)+'_definition_table_percs_same_dec.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(name)))
    
    df = pd.read_csv(str(path)+str(name)+'_definition_table_percs_less.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(name)))    


# %%
def create_tables_adding_extra_members(df, path, tab_name, main_names, mains, extra_clusters):

    df.select('(feh_lamost_lrs>-10000.0)', name = 'lamost')
    
    tablefile_csv = open(str(path)+str(tab_name)+'_addition_table.csv', 'w')
    tablefile_csv_percs_same_dec = open(str(path)+str(tab_name)+'_addition_table_percs_same_dec.csv', 'w')
    tablefile_csv_percs_less = open(str(path)+str(tab_name)+'_addition_table_percs_less.csv', 'w')    
    chain1 = ','
    for cluster in extra_clusters:
        df.select('(def_cluster_id=='+str(cluster)+')', name = 'this_cluster')
        if len(df.evaluate('source_id', selection='(lamost)&(this_cluster)')) < 20.0:
            chain1 = chain1+str(cluster)+'*, '
        else:
            chain1 = chain1+str(cluster)+', '
            
    chain1 = chain1[0:-2]+' \n'
    tablefile_csv.write(chain1)
    tablefile_csv_percs_same_dec.write(chain1)
    tablefile_csv_percs_less.write(chain1)
    
    for main_name, main in zip(main_names, mains):

        chain2 = str(main_name)+', '
        chain3 = str(main_name)+', '
        chain4 = str(main_name)+', '
        for cluster in extra_clusters:
            coords = np.where(np.array(main)!=cluster)
            orig_pval, comp_percentage, less_percentage = compare_MDFs_CMDs_low_Nstars(df,clusters_CM=[np.array(main)[coords], [cluster]], labels=[str(main_name), str(cluster)], colours=['red', 'lightblue'], zorder=[1,2], Nstars_lim=20, NMC=100)
            chain2 = chain2+str(orig_pval)+', '
            chain3 = chain3+str(comp_percentage)+', '
            chain4 = chain4+str(less_percentage)+', '
        chain2 = chain2[0:-2]+' \n'
        chain3 = chain3[0:-2]+' \n'
        chain4 = chain4[0:-2]+' \n'
        tablefile_csv.write(chain2)
        tablefile_csv_percs_same_dec.write(chain3)
        tablefile_csv_percs_less.write(chain4)

    tablefile_csv.close()
    tablefile_csv_percs_same_dec.close()
    tablefile_csv_percs_less.close()

    df = pd.read_csv(str(path)+str(tab_name)+'_addition_table.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(tab_name))\
        .background_gradient(cmap=cm).applymap(pval_red_or_green))

    df = pd.read_csv(str(path)+str(tab_name)+'_addition_table_percs_same_dec.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(tab_name)))
    
    df = pd.read_csv(str(path)+str(tab_name)+'_addition_table_percs_less.csv', index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    display(df.style.set_caption(str(tab_name)))    


# %%
def compare_MDFs_CMDs(df, clusters_CM, labels, colours, P, zorder):
    df.select('(feh_lamost_lrs>-10000.0)', name = 'lamost')

    # Compare the MDF of two structures:
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = str()
        for cluster in clusters[:-1]:
            condition = condition+'(def_cluster_id=='+str(cluster)+')|'
        condition = condition+'(def_cluster_id=='+str(clusters[-1])+')'
        df.select(condition, name = 'structure')
        distr_dict[str(system)] = {'feh_distr': df.evaluate('feh_lamost_lrs', selection='(structure)&(lamost)'),
                                   'mag': df.evaluate('M_G', selection='(structure)'), 
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)'),
                                   'en': df.evaluate('En', selection='(structure)')/10**4, 
                                   'lz': df.evaluate('Lz', selection='(structure)')/10**3, 
                                   'lperp': df.evaluate('Lperp', selection='(structure)')/10**3}  
        
    for system, clusters in zip(labels, clusters_CM):   
        print(system)
        print(str(len(distr_dict[system]['feh_distr']))+' good stars with LAMOST LRS [Fe/H]')
        
    f0, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(18,11), facecolor='w')
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    ax6 = plt.subplot(236)    
    
    ax4.scatter(df.evaluate('Lz')/(10**3), df.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('Lperp')/(10**3), df.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)    
    ax6.scatter(df.evaluate('Lz')/(10**3), df.evaluate('Lperp')/(10**3), c='grey', s=1.0, alpha=0.2)    
    
    for i in np.arange(len(labels)):
        ax1.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, label=labels[i])
        ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, cumulative=True, label=labels[i])
        #ax2.hist(distr_dict[labels[i]]['feh_distr'], histtype='step', range=(-2.7, 0.7), bins=25, density=True, color=colours[i], lw=2, ls='--', cumulative=-1, label=labels[i])  
        ax3.plot(distr_dict[labels[i]]['col'], distr_dict[labels[i]]['mag'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax4.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax5.plot(distr_dict[labels[i]]['lperp'], distr_dict[labels[i]]['en'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        ax6.plot(distr_dict[labels[i]]['lz'], distr_dict[labels[i]]['lperp'], 'o', color=colours[i], zorder=zorder[i], ms=3)
        
    indexes = np.arange(len(clusters_CM))
    pairs = list(itertools.combinations(indexes,2))
    count=0
    for pair in pairs:
        feh_x, feh_y = distr_dict[labels[pair[0]]]['feh_distr'], distr_dict[labels[pair[1]]]['feh_distr']
        feh_x = feh_x[np.isnan(feh_x)==False]
        feh_y = feh_y[np.isnan(feh_y)==False]
        if ((len(feh_x)>10)&(len(feh_y)>10)):
            KS, pval = stats.ks_2samp(feh_x, feh_y, mode='exact')
        else:
            KS, pval = 999, 999
        
        ax2.text(0.95, 0.05+0.05*count, str(labels[pair[0]])+'-'+str(labels[pair[1]])+': '+str(round(pval,3)), horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, fontsize=20)
        count=count+1
        
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
      
    plt.suptitle(' perc'+str(P), fontsize=20)
    
    plt.tight_layout()
    
    plt.show()


# %%
def nice_table_from_csv(path, name, table_type):
    
    df = vaex.open(str(path)+str(name)+str(table_type))
    
    columns = df.column_names
    Ncols = len(columns)
    
    header = ''
    for col in columns[:-1]:
        header = header+'{\\bf '+str(col).replace('Unnamed: 0', '')+'} & '
    header = header+'{\\bf '+str(columns[-1])+'} \\\\ \n'
    #print(header)
    
    caption = name
    
    f = open(str(path)+str(name)+str(table_type), 'r')
    lines=f.readlines()
    Nlines = len(lines)    
    f.close()    
    #print(Nlines)
    
    rows = []
    
    for i in np.arange(1, Nlines):
        this_row = ''
        this_line = lines[i]
        #print(this_line)
        elements = this_line.split(',')
        for k, element in enumerate(elements):
            if k == 0:
                this_row = this_row+'{\\bf '+str(element)+'} & '
            else:
                if float(element) < 0.05:
                    this_row = this_row+'\cellcolor[HTML]{'+str(matplotlib.colors.to_hex(plt.cm.Greens(float(element)), keep_alpha=True)).strip('#').replace('ff', 'f')+'}\color{red}'+str(round(float(element),2))+' & '
                elif float(element) > 0.75:
                    this_row = this_row+'\cellcolor[HTML]{'+str(matplotlib.colors.to_hex(plt.cm.Greens(float(element)), keep_alpha=True)).strip('#').replace('ff', 'f')+'}\color{white}'+str(round(float(element),2))+' & '                         
                else:
                    this_row = this_row+'\cellcolor[HTML]{'+str(matplotlib.colors.to_hex(plt.cm.Greens(float(element)), keep_alpha=True)).strip('#').replace('ff', 'f')+'}\color{black}'+str(round(float(element),2))+' & '     
        rows.append(this_row)   
    
    # Start writing on the .tex file:

    latex_table = open(str(path)+'temp.tex', 'w')

    tab_type = ''
    for i in np.arange(Ncols):
        if i == 0:
            tab_type = tab_type+'l'
        else:
            tab_type = tab_type+'r'

    chain = '\\begin{table} \n'
    latex_table.write(chain)
    chain = '{\centering \n'
    latex_table.write(chain)
    chain = '\small \n'
    latex_table.write(chain)
    chain = '\\begin{tabular}{'+str(tab_type)+'} \n'
    latex_table.write(chain)
    chain = '\hline \n'
    latex_table.write(chain)

    # Writing header:
    chain = header
    latex_table.write(chain)

    # Now the rest of rows with the correct format
    for row in rows:
        chain = row[0:-2]+' \\\\ '
        latex_table.write(chain)

    chain = '\hline \hline \n' 
    latex_table.write(chain)
    chain = '\end{tabular} \n'
    latex_table.write(chain)
    chain = '\caption{'+str(caption)+'} \n' 
    latex_table.write(chain)
    chain = '} \n'
    latex_table.write(chain)
    chain = '\end{table} \n'
    latex_table.write(chain)
    latex_table.close()
    
    h = open(str(path)+'temp.tex', "r")
    file_contents = h.read()
    print(file_contents)
    h.close() 
    
    os.system('rm '+str(path)+'temp.tex')


# %%
def comp_2str_IoM(df, clusters1, clusters2,alpha):
    
    condition = str()
    for cluster in clusters1[:-1]:
        condition = condition+'(def_cluster_id=='+str(cluster)+')|'
    condition = condition+'(def_cluster_id=='+str(clusters1[-1])+')'
    df.select(condition, name = 'structure1')
    
    condition = str()
    for cluster in clusters2[:-1]:
        condition = condition+'(def_cluster_id=='+str(cluster)+')|'
    condition = condition+'(def_cluster_id=='+str(clusters2[-1])+')'
    df.select(condition, name = 'structure2')

    f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = plt.subplots(12, figsize=(30,30))

    ax1 = plt.subplot(4,3,1)
    ax2 = plt.subplot(4,3,2)
    ax3 = plt.subplot(4,3,3)
    ax4 = plt.subplot(4,3,4)
    ax5 = plt.subplot(4,3,5)
    ax6 = plt.subplot(4,3,6)
    ax7 = plt.subplot(4,3,7)
    ax8 = plt.subplot(4,3,8)
    ax9 = plt.subplot(4,3,9)
    ax10 = plt.subplot(4,3,10)
    ax11 = plt.subplot(4,3,11)
    ax12 = plt.subplot(4,3,12)    
    
    ax1.scatter(df.evaluate('Lz')/(10**3), df.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)
    ax2.scatter(df.evaluate('Lz')/(10**3), df.evaluate('Lperp')/(10**3), c='grey', s=1.0, alpha=0.2)
    ax3.scatter(df.evaluate('vR'), df.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
    ax4.scatter(df.evaluate('vz'), df.evaluate('vR'), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('vz'), df.evaluate('vphi'), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(df.evaluate('vphi'), np.sqrt(df.evaluate('vR')**2 + df.evaluate('vz')**2), c='grey', s=1.0, alpha=0.2)
    ax7.scatter(df.evaluate('Lperp')/(10**3), df.evaluate('En')/(10**4), c='grey', s=1.0, alpha=0.2)
    #ax8.scatter(df.evaluate('Lz')/(10**3), df.evaluate('Ltotal')/(10**3), c='grey', s=1.0, alpha=0.2)
    ax9.scatter(df.evaluate('En')/(10**4), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax10.scatter(df.evaluate('Lz')/(10**3), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax11.scatter(df.evaluate('Ltotal')/(10**3), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)    
    ax12.scatter(df.evaluate('l'), df.evaluate('b'), c='grey', s=1.0, alpha=0.2)

    scatter1 = ax1.scatter(df.evaluate('Lz', selection='structure1')/(10**3), df.evaluate('En', selection='structure1')/(10**4), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax2.scatter(df.evaluate('Lz', selection='structure1')/(10**3), df.evaluate('Lperp', selection='structure1')/(10**3), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax3.scatter(df.evaluate('vR', selection='structure1'), df.evaluate('vphi', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax4.scatter(df.evaluate('vz', selection='structure1'), df.evaluate('vR', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax5.scatter(df.evaluate('vz', selection='structure1'), df.evaluate('vphi', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax6.scatter(df.evaluate('vphi', selection='structure1'), np.sqrt(df.evaluate('vR', selection='structure1')**2 + df.evaluate('vz', selection='structure1')**2), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax7.scatter(df.evaluate('Lperp', selection='structure1')/(10**3), df.evaluate('En', selection='structure1')/(10**4), c='blue', s=18.0, alpha=alpha)
    #scatter1 = ax8.scatter(df.evaluate('Lz', selection='structure1')/(10**3), df.evaluate('Ltotal', selection='structure1')/(10**3), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax9.scatter(df.evaluate('En', selection='structure1')/(10**4), df.evaluate('circ', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax10.scatter(df.evaluate('Lz', selection='structure1')/(10**3), df.evaluate('circ', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax11.scatter(df.evaluate('Ltotal', selection='structure1')/(10**3), df.evaluate('circ', selection='structure1'), c='blue', s=18.0, alpha=alpha)    
    scatter1 = ax12.scatter(df.evaluate('l', selection='structure1'), df.evaluate('b', selection='structure1'), c='blue', s=32.0, alpha=0.3)
    

    scatter1 = ax1.scatter(df.evaluate('Lz', selection='structure2')/(10**3), df.evaluate('En', selection='structure2')/(10**4), c='red', s=18.0, alpha=alpha)
    scatter1 = ax2.scatter(df.evaluate('Lz', selection='structure2')/(10**3), df.evaluate('Lperp', selection='structure2')/(10**3), c='red', s=18.0, alpha=alpha)
    scatter1 = ax3.scatter(df.evaluate('vR', selection='structure2'), df.evaluate('vphi', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax4.scatter(df.evaluate('vz', selection='structure2'), df.evaluate('vR', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax5.scatter(df.evaluate('vz', selection='structure2'), df.evaluate('vphi', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax6.scatter(df.evaluate('vphi', selection='structure2'), np.sqrt(df.evaluate('vR', selection='structure2')**2 + df.evaluate('vz', selection='structure2')**2), c='red', s=18.0, alpha=alpha)
    scatter1 = ax7.scatter(df.evaluate('Lperp', selection='structure2')/(10**3), df.evaluate('En', selection='structure2')/(10**4), c='red', s=18.0, alpha=alpha)
    #scatter1 = ax8.scatter(df.evaluate('Lz', selection='structure2')/(10**3), df.evaluate('Ltotal', selection='structure2')/(10**3), c='red', s=18.0, alpha=alpha)
    scatter1 = ax9.scatter(df.evaluate('En', selection='structure2')/(10**4), df.evaluate('circ', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax10.scatter(df.evaluate('Lz', selection='structure2')/(10**3), df.evaluate('circ', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax11.scatter(df.evaluate('Ltotal', selection='structure2')/(10**3), df.evaluate('circ', selection='structure2'), c='red', s=18.0, alpha=alpha)    
    scatter1 = ax12.scatter(df.evaluate('l', selection='structure2'), df.evaluate('b', selection='structure2'), c='red', s=32.0, alpha=0.3)    
    
    
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
    
    #plt.savefig('/data/users/ruizlara/halo_clustering/plots_paper/'+str(name)+'.png')

    plt.show()

# %% [markdown]
# # Preparing catalogue

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
df_clusters = vaex.open('/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5')

dust_map = 'l18'
ext_rec = 'bab18'
df_clusters.add_virtual_column('new_parallax','parallax+0.017') # new_parallax in 'mas', 1/parallax --> distance in kpc.
df_clusters.add_virtual_column('distance','1.0/new_parallax') # new_parallax in 'mas', 1/parallax --> distance in kpc.
E_BP_RP, A_G, COL_bp_rp, M_G = reddening_computation(df_clusters)
df_clusters.add_virtual_columns_cartesian_velocities_to_polar(x='(x-8.20)', y='(y)', vx='(vx)', vy='(vy)', vr_out='_vR', vazimuth_out='_vphi', propagate_uncertainties=False)  
df_clusters.add_virtual_column('vphi','-_vphi')  # flip sign 
df_clusters.add_virtual_column('vR','-_vR')  # flip sign 
df_clusters.add_column('E_BP_RP',E_BP_RP)
df_clusters.add_column('A_G',A_G)
df_clusters.add_column('COL_bp_rp',COL_bp_rp)
df_clusters.add_column('M_G',M_G) 
df_clusters.add_column('err_MG', 2.17/df_clusters.evaluate('parallax_over_error')) 
df_clusters.add_column('err_COL_bp_rp', 1.0857*np.sqrt((df_clusters.evaluate('phot_bp_mean_flux_error')/df_clusters.evaluate('phot_bp_mean_flux'))**2 + (df_clusters.evaluate('phot_rp_mean_flux_error')/df_clusters.evaluate('phot_rp_mean_flux'))**2)) # Coming from error propagation theory (derivative).

df_clusters.select('(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))', name='quality') # no segue within quality.

df_clusters

# %% [markdown]
# # Substructure A:

# %%
create_tables_defining(df=df_clusters, path=path, name='A1', main=[34, 28, 32])
# create_tables_defining(df=df_clusters, path=path, name='A2', main=[31, 15, 16])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='A', main_names=['A1', 'A2'], mains=[[34, 32], [31, 15, 16]], extra_clusters=[28, 33, 12, 40, 42, 46, 38, 18, 30])

# %%
create_tables_defining(df=df_clusters, path=path, name='A', main=[33, 12, 40, 34, 28, 32, 31, 15, 16, 42, 38, 18, 30])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='A_def', main_names=['A'], mains=[[15,16,46,31,32,30,42,18,33,34]], extra_clusters=[12, 38, 28, 40])

# %% [markdown]
# ## Conclusion A:
#
# A1 is compatible with being one structure     
# A2 is compatible with being one structure     
# All clusters are compatible with A1 and A2 except 38    
# A1 is compatible with A2...     
#
# SO: A and 38

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[34, 28, 32], [31, 15, 16]], labels=['A1', 'A2'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[34, 32], [31, 15, 16]], labels=['A1', 'A2'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[34, 28, 32, 31, 15, 16, 33, 12, 40, 42, 46, 38, 18, 30], [38]], labels=['A', '38'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], [38], [12], [40], [28]], labels=['A', '38', '12', '40', '28'], colours=['red', 'lightblue', 'blue', 'green', 'magenta'], P=80, zorder=[1,2,3,4,5]) 

# %%
comp_2str_IoM(df_clusters, clusters1=[40], clusters2=[40],alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[40],alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[12], clusters2=[12],alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[12],alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[28], clusters2=[28],alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[28],alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[38], clusters2=[38],alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[38],alpha=1.0)

# %% [markdown]
# ### Thamnos (substructure B):

# %%
create_tables_defining(df=df_clusters, path=path, name='B', main=[37, 8, 11])

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[37], [8, 11]], labels=['Thamnos1', 'Thamnos2'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %% [markdown]
# ## Conclusion B:
#
# Substructure B, can be divided into B1 and B2, with 37 and 8-11, respectively (Thammnos1 and Thamnos2). 

# %% [markdown]
# # GE (substructure C)

# %%
create_tables_defining(df=df_clusters, path=path, name='C3a', main=[57, 58])
create_tables_defining(df=df_clusters, path=path, name='C3b', main=[50, 53])
create_tables_defining(df=df_clusters, path=path, name='C3c', main=[47, 44, 51])

# %%
create_tables_defining(df=df_clusters, path=path, name='C4a', main=[29, 41, 49])

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[57, 58], [50, 53], [47, 44, 51]], labels=['C3a', 'C3b', 'C3c'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1,2,3]) 

# %%
create_tables_defining(df=df_clusters, path=path, name='C1', main=[56, 59])
create_tables_defining(df=df_clusters, path=path, name='C2', main=[10, 7, 13])
create_tables_defining(df=df_clusters, path=path, name='C3', main=[57, 58, 54, 55, 50, 53, 47, 44, 51])
create_tables_defining(df=df_clusters, path=path, name='C4', main=[48, 45, 25, 29, 41, 49])
create_tables_defining(df=df_clusters, path=path, name='C5', main=[27, 22, 26])
create_tables_defining(df=df_clusters, path=path, name='C6', main=[35, 9, 24])

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[57, 58, 54, 50, 53, 47, 44, 51], [55]], labels=['C3', '55'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[35], [9], [24]], labels=['35', '9', '24'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1,2,3]) 

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='C', main_names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], mains=[[56, 59], [10, 7, 13], [57, 58, 54, 55, 50, 53, 47, 44, 51], [48, 45, 25, 29, 41, 49], [27, 22, 26], [35, 9, 24]], extra_clusters=[6, 52, 14, 20, 19, 17, 43, 39, 36, 23, 5, 21])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='C', main_names=['C1', 'C2', 'C3', 'C4', 'C5'], mains=[[56, 59], [10, 7, 13], [57, 58, 54, 50, 53, 47, 44, 51], [48, 45, 25, 29, 41, 49], [27, 22, 26]], extra_clusters=[6, 52, 55, 14, 20, 19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24])

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[56, 59, 10, 7, 13,  57, 58, 54, 50, 53, 47, 44, 51, 48, 45, 25, 29, 41, 49, 27, 22, 26, 52, 55, 20, 19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24], [6], [14]], labels=['GE', '6', '14'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1,2,3]) 

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[57, 58, 54, 50, 53, 47, 44, 51, 39, 35, 9, 24, 5, 23, 36, 21, 17], [9]], labels=['C3', '9'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %% [markdown]
# # Sequoia (substructure D):

# %%
create_tables_defining(df=df_clusters, path=path, name='D', main=[64, 63, 62])

# %% [markdown]
# ## Conclusion D:
#
# As before, everything seems compatible... with cluster #62 showing some differences. Given position in IoM, as well as velocity space, I would put the three separated.

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[64], [63], [62]], labels=['64', 'Sequoia', '62'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1,2,3]) 

# %% [markdown]
# # Helmi (substructure E)

# %%
create_tables_defining(df=df_clusters, path=path, name='E', main=[60, 61])

# %% [markdown]
# ## Conclusion E:
#
# Both clusters belong together forming the Helmi streams.

# %%
compare_MDFs_CMDs(df_clusters,clusters_CM=[[60], [61]], labels=['60', '61'], colours=['red', 'lightblue'], P=80, zorder=[1,2]) 

# %% [markdown]
# # High-energy blob (F):

# %%
create_tables_defining(df=df_clusters, path=path, name='F', main=[65, 66])

# %% [markdown]
# ## Conclusion F:
#
# Low number of stars make it difficult to say almost nothing... if anything, they are not related. In addition, nothing to do with Sgr... so, pass or some minor acretion depicted as significant because of the position they occupy in IoM (low density of stars).

# %% [markdown]
# # Tables in latex format

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
name = 'C1'
table_type = '_definition_table.csv'
nice_table_from_csv(path, name, table_type)

# %%
path = '/data/users/ruizlara/halo_clustering/plots_paper_NEW_maxsig_3D/'
name = 'A_def'
table_type = '_addition_table.csv'
nice_table_from_csv(path, name, table_type)

# %%

# %% [markdown]
# # Inspecting IoM and velocity distributions of doubtful clusters:

# %%
comp_2str_IoM(df_clusters, clusters1=[28], clusters2=[28],alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[40], clusters2=[40],alpha=1.0)

# %%
matplotlib.colors.to_hex(plt.cm.Greens(1.0), keep_alpha=True)

# %%
matplotlib.colors.to_hex(plt.cm.Greens(0.27), keep_alpha=True)

# %%
matplotlib.colors.to_hex(plt.cm.Greens(0.06), keep_alpha=True)

# %%
