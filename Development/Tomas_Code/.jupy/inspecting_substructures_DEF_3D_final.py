# %%
import pandas as pd
import numpy as np
import os
import vaex
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import seaborn as sns
from IPython.display import display

from KapteynClustering.isochrone_utils.reddening_computation import reddening_computation
np.random.seed(0)


# %% [markdown]
# # Definitions

# %%
path = "/data/users/callingham/data/clustering/KapteynClustering_DR3/Development/Tomas_Code/output/"


def read_df_clusters(tomas_clusters_file):
    df_clusters = vaex.open(tomas_clusters_file)

    # new_parallax in 'mas', 1/parallax --> distance in kpc.
    df_clusters.add_virtual_column('new_parallax', 'parallax+0.017')
    # new_parallax in 'mas', 1/parallax --> distance in kpc.
    df_clusters.add_virtual_column('distance', '1.0/new_parallax')
    E_BP_RP, delta_E_BP_RP, A_G, delta_A_G, COL_bp_rp, M_G = reddening_computation(df_clusters, dust_map="l18", ext_rec="bab18")
    df_clusters.add_virtual_columns_cartesian_velocities_to_polar(
        x='(x-8.20)', y='(y)', vx='(vx)', vy='(vy)', vr_out='_vR', vazimuth_out='_vphi', propagate_uncertainties=False)
    df_clusters.add_virtual_column('vphi', '-_vphi')  # flip sign
    print("WHY FLIP vR?")
    df_clusters.add_virtual_column('vR', '-_vR')  # flip sign
    df_clusters.add_column('E_BP_RP', E_BP_RP)
    df_clusters.add_column('A_G', A_G)
    df_clusters.add_column('COL_bp_rp', COL_bp_rp)
    df_clusters.add_column('M_G', M_G)
    df_clusters.add_column('err_MG', 2.17 / df_clusters.evaluate('parallax_over_error'))
    df_clusters.add_column('err_COL_bp_rp', 1.0857 * np.sqrt((df_clusters.evaluate('phot_bp_mean_flux_error') / df_clusters.evaluate('phot_bp_mean_flux'))**2 + (
        df_clusters.evaluate('phot_rp_mean_flux_error') / df_clusters.evaluate('phot_rp_mean_flux'))**2))  # Coming from error propagation theory (derivative).

    # no segue within quality.
    df_clusters.select(
        '(flag_vlos<5.5)&(ruwe < 1.4)&(0.001+0.039*(bp_rp) < log10(phot_bp_rp_excess_factor))&(log10(phot_bp_rp_excess_factor) < 0.12 + 0.039*(bp_rp))', name='quality')
    return df_clusters


tomas_clusters_file = '/data/users/ruizlara/halo_clustering/DEF_catalogue_dist_perc_80_3D.hdf5'
df_clusters = read_df_clusters(tomas_clusters_file)

# %%
def KS_test_pairs(data_dict, label_pairs, key="feh_distr", min_N=10):
    N_pairs = len(label_pairs)
    KSs, pvals  = np.full((N_pairs),None), np.full((N_pairs),None)
    for i,lpair in enumerate(label_pairs):
        x, y = data_dict[lpair[0]][key], data_dict[lpair[1]][key]
        x, y = x[~np.isnan(x)], y[~np.isnan(y)]
        if ((len(x) >= min_N) & (len(y) >= min_N)):
            KSs[i], pvals[i] = stats.ks_2samp(x, y, mode='exact')
    return KSs, pvals
# %%

def create_tables_defining(df, path, name, main):
    #TODO: Change from csv files
    main = np.array(main)

    results = {}
    for cluster in main:
        clusters_CM = [main[main!=cluster], [cluster]]
        results[cluster]= compare_MDFs_CMDs_low_Nstars(df, clusters_CM=clusters_CM, labels=[name, str(cluster)])

    chain1 = ''
    for cluster in main:
        select = f"(def_cluster_id=={cluster})"
        df.select(select, name='this_cluster')
        N_feh = df.count(selection='(lamost)&(this_cluster)')
        if N_feh< 20.0:
            chain1 += f"{cluster}*, "
        else:
            chain1 += f"{cluster}, "
    chain1 = chain1[:-2] + ' \n'

    chain2, chain3, chain4 = f"{name}, " , f"{name}, " ,f"{name}, "
    for cluster in main:
        # for (chain,p) in zip([chain2,chain3,chain4], ["orig_pval", "comp_per", "less_per"]):
        #     chain += f"{results[cluster][p]}, "
        [orig_pval, comp_per, less_per] = [results[cluster][p] for p in ["orig_pval", "comp_per", "less_per"]]
        chain2 +=  f"{orig_pval}, "
        chain3 +=  f"{comp_per}, "
        chain4 +=  f"{less_per}, "

    # for chain in [chain2, chain3, chain4]:
    #     chain = chain[:-2] + " \n"
    chain2 = chain2[:-2] + ' \n'
    chain3 = chain3[:-2] + ' \n'
    chain4 = chain4[:-2] + ' \n'

    f_def_table = f"{path}{name}_definition_table.csv"
    f_def_table_same = f"{path}{name}_definition_table_percs_same_dec.csv"
    f_def_table_less = f"{path}{name}_definition_table_percs_less.csv"

    with open(f_def_table, 'w') as tablefile_csv:
        tablefile_csv.write(chain1)
        tablefile_csv.write(chain2)

    with open(f_def_table_same, 'w') as tablefile_csv_percs_same_dec:
        tablefile_csv_percs_same_dec.write(chain1)
        tablefile_csv_percs_same_dec.write(chain3)

    with open(f_def_table_less, 'w') as tablefile_csv_percs_less:
        tablefile_csv_percs_less.write(chain1)
        tablefile_csv_percs_less.write(chain4)


    for fname in [f_def_table, f_def_table_same, f_def_table_less]:
        display_table(fname = fname, name=name)

    return

def display_table(fname, name, red_green=True):
    df = pd.read_csv(fname, index_col=0)
    cm = sns.light_palette("green", as_cmap=True)
    if red_green:
        def pval_red_or_green(val):
            color = 'red' if val < 0.05 else 'black'
            return 'color: %s' % color
        display(df.style.set_caption(name).background_gradient(cmap=cm).applymap(pval_red_or_green))
    else:
        display(df.style.set_caption(name))
    return


def create_structure_data_dic(df, labels, clusters_CM):
    df.select('(feh_lamost_lrs > -10000.0)', name='lamost')
    distr_dict = {}
    for system, clusters in zip(labels, clusters_CM):
        condition = ""
        if type(clusters)==int:
            clusters = [clusters]
        for cluster in clusters:
            condition += '(def_cluster_id==' + str(cluster) + ')|'
        df.select(condition[:-1], name='structure')
        distr_dict[str(system)] = {'feh_distr': df.evaluate('feh_lamost_lrs', selection='(structure)&(lamost)'),
                                   'mag': df.evaluate('M_G', selection='(structure)'),
                                   'col': df.evaluate('COL_bp_rp', selection='(structure)'),
                                   'en': df.evaluate('En', selection='(structure)') / 10**4,
                                   'lz': df.evaluate('Lz', selection='(structure)') / 10**3,
                                   'lperp': df.evaluate('Lperp', selection='(structure)') / 10**3}
    return distr_dict

# %%
def compare_MDFs_CMDs_low_Nstars(df, clusters_CM, labels, colours=['red', 'lightblue'], zorder=[1,2], Nstars_lim=20, NMC=100):

    # Compare the MDF of two structures:
    distr_dict = create_structure_data_dic(df, labels,clusters_CM)

    main_feh, second_feh = distr_dict[labels[0]]['feh_distr'], distr_dict[labels[1]]['feh_distr']
    main_feh, second_feh = main_feh[~np.isnan(main_feh)], second_feh[~np.isnan(second_feh)]
    N_main_feh, N_second_feh = len(main_feh), len(second_feh)

    # KS CALC
    label_pairs = list(itertools.combinations(labels, 2))
    pvals = KS_test_pairs(distr_dict, label_pairs, min_N=0)[1]
    orig_pval = pvals[-1]
    # TODO: Check: Original pval in both if statements - always last pair?

    # PLOT
    if N_second_feh < Nstars_lim:
        f0, axs_array = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), facecolor='w')
        [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8] = axs_array.flatten()[:-1]
    else:
        f0, axs_array = plt.subplots(nrows=2, ncols=3, figsize=(18, 11), facecolor='w')
        [ax1, ax2, ax3, ax4, ax5, ax6] = axs_array.flatten()
    for ax in axs_array.flatten():
        ax.tick_params(labelsize=20)

    ax4.scatter(df.evaluate('Lz') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax5.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    ax6.scatter(df.evaluate('Lz') / (10**3), df.evaluate('Lperp') / (10**3), c='grey', s=1.0, alpha=0.2)

    for l,c,z in zip(labels, colours, zorder):
        ax1.hist(distr_dict[l]['feh_distr'], histtype='step', range=(-2.7, 0.7),
                 bins=25, density=True, color=c, lw=2, label=l)
        ax2.hist(distr_dict[l]['feh_distr'], histtype='step', range=(-2.7, 0.7),
                 bins=25, density=True, color=c, lw=2, cumulative=True, label=l)

        ax3.plot(distr_dict[l]['col'], distr_dict[l]['mag'], 'o', color=c, zorder=z, ms=3)
        ax4.plot(distr_dict[l]['lz'], distr_dict[l] ['en'], 'o', color=c, zorder=z, ms=3)
        ax5.plot(distr_dict[l]['lperp'], distr_dict[l]['en'], 'o', color=c, zorder=z, ms=3)
        ax6.plot(distr_dict[l]['lz'], distr_dict[l]['lperp'], 'o', color=c, zorder=z, ms=3)

    for i,(pval,lpair) in enumerate(zip(pvals,label_pairs)):
        text = f"{lpair[0]} - {lpair[1]} : {pval:.3f}"
        ax2.text(0.95, 0.05 + 0.05 * i, text, horizontalalignment='right',
                 verticalalignment='center', transform=ax2.transAxes, fontsize=20)


    ax1.text(0.04, 0.95, f"N = {N_main_feh}", color=colours[0], transform=ax1.transAxes, fontsize=14)
    ax1.text(0.04, 0.90, f"N = {N_second_feh}", color=colours[1], transform=ax1.transAxes, fontsize=14)

    ax1.legend()
    ax1.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
    ax1.set_ylabel(r'$\rm Normalised \; N$', fontsize=20)

    ax2.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
    ax2.set_ylabel(r'$\rm Cumulative \; Distribution$', fontsize=20)

    ax3.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=20)
    ax3.set_ylabel(r'$\rm M_{G}$', fontsize=20)
    ax3.set_xlim(-0.2, 3.0)
    ax3.set_ylim(9.0, -4.0)

    ax4.set_ylabel(r'$En/10^4$', fontsize=20)
    ax4.set_xlabel(r'$Lz/10^3$', fontsize=20)

    ax5.set_ylabel(r'$En/10^4$', fontsize=20)
    ax5.set_xlabel(r'$Lperp/10^3$', fontsize=20)

    ax6.set_ylabel(r'$Lperp/10^3$', fontsize=20)
    ax6.set_xlabel(r'$Lz/10^3$', fontsize=20)



    plt.tight_layout()

    if N_second_feh < Nstars_lim:
        for pval in pvals:
            ax8.axhline(pval, ls='--', color='black')

        ax7.hist(main_feh, histtype='step', range=(-2.7, 0.7), bins=25,
                 density=True, color=colours[0], lw=2, label=labels[0], zorder=1)
        ax7.hist(second_feh, histtype='step', range=(-2.7, 0.7), bins=25,
                 density=True, color=colours[1], lw=2, label=labels[1], zorder=1)

        new_pvals = np.zeros((NMC))
        for i in range(NMC):
            this_second_feh = np.random.choice(main_feh, N_second_feh)
            new_pvals[i] = stats.ks_2samp(main_feh, this_second_feh, mode='exact')[1]
            ax7.hist(this_second_feh, histtype='step', range=(-2.7, 0.7), bins=25,
                     density=True, color='gray', lw=2, zorder=0, alpha=0.01)


        compat_filt = (new_pvals < orig_pval + 0.05) * (new_pvals > orig_pval - 0.05)
        compatibility = 100.0 * (compat_filt.sum() / NMC)
        same_distr = 100.0 * ((new_pvals > 0.05).sum() / NMC)
        diff_distr = 100 - same_distr

        ax7.text(0.05, 0.94, f"Compatible new_pvals {compatibility: 2} %", fontsize=14,
                 transform=ax7.transAxes)
        ax7.text(0.05, 0.88, f"Same distr {same_distr: .2f} %", fontsize=14,
                 transform=ax7.transAxes)
        ax7.text(0.05, 0.82, f"Different distr {diff_distr: .2f} %", fontsize=14,
                 transform=ax7.transAxes)
        ax7.plot([None], [None], 'k-', label='MC')
        ax7.legend()
        ax7.set_xlabel(r'$\rm [Fe/H]$', fontsize=20)
        ax7.set_ylabel(r'$\rm Normalised \; N$', fontsize=20)

        NNs = np.arange(NMC)
        ax8.plot(NNs, new_pvals, 'ko')
        ax8.axhline(np.nanmean(new_pvals), ls='--', alpha=0.4, color='black')
        new_incompat_filt = (new_pvals < 0.05)
        ax8.plot(NNs[new_incompat_filt], new_pvals[new_incompat_filt], 'ro')
        ax8.axhline(0.05, ls='--', alpha=0.4, color='red')
        pval_label =f"pval = {np.nanmean(new_pvals): .2f}"
        ax8.text(0.0, np.nanmean(new_pvals) - 0.05, pval_label, fontsize=14)

        ax8.set_xlabel(r'$\rm N_{boot}$', fontsize=20)
        ax8.set_ylabel(r'$\rm pval$', fontsize=20)


    if N_second_feh < Nstars_lim:
        if orig_pval < 0.05:
            comp_per = diff_distr
        else:
            comp_per = same_distr

        less_per = (new_pvals < orig_pval).sum() / NMC
    else:
        comp_per = 999.9
        less_per = 999.9

    plt.show()
    results = {"orig_pval": orig_pval, "comp_per": comp_per, "less_per": less_per}

    return results


# %%
def create_tables_adding_extra_members(df, path, tab_name, main_names, mains, extra_clusters):

    results = {}
    for main_name, main in zip(main_names, mains):
        results[main_name] = {}
        for cluster in extra_clusters:
            clusters_CM = [main[main!=cluster], [cluster]]
            results[main_name][cluster]= compare_MDFs_CMDs_low_Nstars(df, clusters_CM=clusters_CM, labels=[str(main_name), str(cluster)])

    chain1 = ''
    for cluster in extra_clusters:
        select = f"(def_cluster_id=={cluster})"
        df.select(select, name='this_cluster')
        N_feh = df.count(selection='(lamost)&(this_cluster)')
        if N_feh< 20.0:
            chain1 += f"{cluster}*, "
        else:
            chain1 += f"{cluster}, "
    chain1 = chain1[:-2] + ' \n'



    chain2, chain3, chain4 = "","",""


    for main_name, main in zip(main_names, mains):

        chain2 += f"{main_name}, "
        chain3 += f"{main_name}, "
        chain4 += f"{main_name}, "
        for cluster in extra_clusters:
            [orig_pval, comp_per, less_per] = [results[main_name][cluster][p] for p in ["orig_pval", "comp_per", "less_per"]]

            chain2 +=  f"{orig_pval}, "
            chain3 +=  f"{comp_per}, "
            chain4 +=  f"{less_per}, "

        chain2 = chain2[:-2] + ' \n'
        chain3 = chain3[:-2] + ' \n'
        chain4 = chain4[:-2] + ' \n'

    f_table_file = f"{path}{tab_name}_addition_table.csv"
    with  open(f_table_file, 'w') as tablefile_csv:
        tablefile_csv.write(chain1)
        tablefile_csv.write(chain2)

    f_add_table_same = f"{path}{name}_addition_table_percs_same_dec.csv"
    with open(f_add_table_same, 'w') as tablefile_csv_percs_same_dec:
        tablefile_csv_percs_same_dec.write(chain1)
        tablefile_csv_percs_same_dec.write(chain3)

    f_add_table_less = f"{path}{name}_addition_table_percs_less.csv"
    with open(f_add_table_less, 'w') as tablefile_csv_percs_less:
        tablefile_csv_percs_less.write(chain1)
        tablefile_csv_percs_less.write(chain4)


    for fname in [f_table_file, f_add_table_same, f_add_table_less]:
        display_table(fname = fname, name=name)

    return



# %%
def comp_2str_IoM(df, clusters1, clusters2, alpha):

    condition = str()
    for cluster in clusters1[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters1[-1]) + ')'
    df.select(condition, name='structure1')

    condition = str()
    for cluster in clusters2[:-1]:
        condition = condition + '(def_cluster_id==' + str(cluster) + ')|'
    condition = condition + '(def_cluster_id==' + str(clusters2[-1]) + ')'
    df.select(condition, name='structure2')

    f0, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10,
         ax11, ax12) = plt.subplots(12, figsize=(30, 30))

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
    ax6.scatter(df.evaluate('vphi'), np.sqrt(df.evaluate('vR')**2
                + df.evaluate('vz')**2), c='grey', s=1.0, alpha=0.2)
    ax7.scatter(df.evaluate('Lperp') / (10**3), df.evaluate('En') / (10**4), c='grey', s=1.0, alpha=0.2)
    #ax8.scatter(df.evaluate('Lz')/(10**3), df.evaluate('Ltotal')/(10**3), c='grey', s=1.0, alpha=0.2)
    ax9.scatter(df.evaluate('En') / (10**4), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax10.scatter(df.evaluate('Lz') / (10**3), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax11.scatter(df.evaluate('Ltotal') / (10**3), df.evaluate('circ'), c='grey', s=1.0, alpha=0.2)
    ax12.scatter(df.evaluate('l'), df.evaluate('b'), c='grey', s=1.0, alpha=0.2)

    scatter1 = ax1.scatter(df.evaluate('Lz', selection='structure1') / (10**3),
                           df.evaluate('En', selection='structure1') / (10**4), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax2.scatter(df.evaluate('Lz', selection='structure1') / (10**3),
                           df.evaluate('Lperp', selection='structure1') / (10**3), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax3.scatter(df.evaluate('vR', selection='structure1'), df.evaluate(
        'vphi', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax4.scatter(df.evaluate('vz', selection='structure1'), df.evaluate(
        'vR', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax5.scatter(df.evaluate('vz', selection='structure1'), df.evaluate(
        'vphi', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax6.scatter(df.evaluate('vphi', selection='structure1'), np.sqrt(df.evaluate(
        'vR', selection='structure1')**2 + df.evaluate('vz', selection='structure1')**2), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax7.scatter(df.evaluate('Lperp', selection='structure1') / (10**3),
                           df.evaluate('En', selection='structure1') / (10**4), c='blue', s=18.0, alpha=alpha)
    #scatter1 = ax8.scatter(df.evaluate('Lz', selection='structure1')/(10**3), df.evaluate('Ltotal', selection='structure1')/(10**3), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax9.scatter(df.evaluate('En', selection='structure1') / (10**4),
                           df.evaluate('circ', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax10.scatter(df.evaluate('Lz', selection='structure1') / (10**3),
                            df.evaluate('circ', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax11.scatter(df.evaluate('Ltotal', selection='structure1') / (10**3),
                            df.evaluate('circ', selection='structure1'), c='blue', s=18.0, alpha=alpha)
    scatter1 = ax12.scatter(df.evaluate('l', selection='structure1'), df.evaluate(
        'b', selection='structure1'), c='blue', s=32.0, alpha=0.3)

    scatter1 = ax1.scatter(df.evaluate('Lz', selection='structure2') / (10**3),
                           df.evaluate('En', selection='structure2') / (10**4), c='red', s=18.0, alpha=alpha)
    scatter1 = ax2.scatter(df.evaluate('Lz', selection='structure2') / (10**3),
                           df.evaluate('Lperp', selection='structure2') / (10**3), c='red', s=18.0, alpha=alpha)
    scatter1 = ax3.scatter(df.evaluate('vR', selection='structure2'), df.evaluate(
        'vphi', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax4.scatter(df.evaluate('vz', selection='structure2'), df.evaluate(
        'vR', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax5.scatter(df.evaluate('vz', selection='structure2'), df.evaluate(
        'vphi', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax6.scatter(df.evaluate('vphi', selection='structure2'), np.sqrt(df.evaluate(
        'vR', selection='structure2')**2 + df.evaluate('vz', selection='structure2')**2), c='red', s=18.0, alpha=alpha)
    scatter1 = ax7.scatter(df.evaluate('Lperp', selection='structure2') / (10**3),
                           df.evaluate('En', selection='structure2') / (10**4), c='red', s=18.0, alpha=alpha)
    #scatter1 = ax8.scatter(df.evaluate('Lz', selection='structure2')/(10**3), df.evaluate('Ltotal', selection='structure2')/(10**3), c='red', s=18.0, alpha=alpha)
    scatter1 = ax9.scatter(df.evaluate('En', selection='structure2') / (10**4),
                           df.evaluate('circ', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax10.scatter(df.evaluate('Lz', selection='structure2') / (10**3),
                            df.evaluate('circ', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax11.scatter(df.evaluate('Ltotal', selection='structure2') / (10**3),
                            df.evaluate('circ', selection='structure2'), c='red', s=18.0, alpha=alpha)
    scatter1 = ax12.scatter(df.evaluate('l', selection='structure2'), df.evaluate(
        'b', selection='structure2'), c='red', s=32.0, alpha=0.3)

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


# %% [markdown]
# # Preparing catalogue


# %% [markdown]
# # Substructure A:

# %%
create_tables_defining(df=df_clusters, path=path, name='A1', main=[34, 28, 32])
# create_tables_defining(df=df_clusters, path=path, name='A2', main=[31, 15, 16])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='A', main_names=['A1', 'A2'], mains=[
                                   [34, 32], [31, 15, 16]], extra_clusters=[28, 33, 12, 40, 42, 46, 38, 18, 30])

# %%
create_tables_defining(df=df_clusters, path=path, name='A', main=[
                       33, 12, 40, 34, 28, 32, 31, 15, 16, 42, 38, 18, 30])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='A_def', main_names=['A'], mains=[
                                   [15, 16, 46, 31, 32, 30, 42, 18, 33, 34]], extra_clusters=[12, 38, 28, 40])

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
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[34, 28, 32], [31, 15, 16]], labels=[
                  'A1', 'A2'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[34, 32], [31, 15, 16]], labels=[
                  'A1', 'A2'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[34, 28, 32, 31, 15, 16, 33, 12, 40, 42, 46, 38, 18, 30], [
                  38]], labels=['A', '38'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], [38], [12], [40], [28]], labels=[
                  'A', '38', '12', '40', '28'], colours=['red', 'lightblue', 'blue', 'green', 'magenta'], P=80, zorder=[1, 2, 3, 4, 5])

# %%
comp_2str_IoM(df_clusters, clusters1=[40], clusters2=[40], alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[40], alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[12], clusters2=[12], alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[12], alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[28], clusters2=[28], alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[28], alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[38], clusters2=[38], alpha=1.0)
comp_2str_IoM(df_clusters, clusters1=[34, 32, 31, 15, 16, 33, 42, 46, 18, 30], clusters2=[38], alpha=1.0)

# %% [markdown]
# ### Thamnos (substructure B):

# %%
create_tables_defining(df=df_clusters, path=path, name='B', main=[37, 8, 11])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[37], [8, 11]], labels=[
                  'Thamnos1', 'Thamnos2'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

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
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[57, 58], [50, 53], [47, 44, 51]], labels=[
                  'C3a', 'C3b', 'C3c'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1, 2, 3])

# %%
create_tables_defining(df=df_clusters, path=path, name='C1', main=[56, 59])
create_tables_defining(df=df_clusters, path=path, name='C2', main=[10, 7, 13])
create_tables_defining(df=df_clusters, path=path, name='C3', main=[57, 58, 54, 55, 50, 53, 47, 44, 51])
create_tables_defining(df=df_clusters, path=path, name='C4', main=[48, 45, 25, 29, 41, 49])
create_tables_defining(df=df_clusters, path=path, name='C5', main=[27, 22, 26])
create_tables_defining(df=df_clusters, path=path, name='C6', main=[35, 9, 24])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[57, 58, 54, 50, 53, 47, 44, 51], [55]], labels=[
                  'C3', '55'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[35], [9], [24]], labels=[
                  '35', '9', '24'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1, 2, 3])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='C', main_names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], mains=[[56, 59], [10, 7, 13], [
                                   57, 58, 54, 55, 50, 53, 47, 44, 51], [48, 45, 25, 29, 41, 49], [27, 22, 26], [35, 9, 24]], extra_clusters=[6, 52, 14, 20, 19, 17, 43, 39, 36, 23, 5, 21])

# %%
create_tables_adding_extra_members(df=df_clusters, path=path, tab_name='C', main_names=['C1', 'C2', 'C3', 'C4', 'C5'], mains=[[56, 59], [10, 7, 13], [
                                   57, 58, 54, 50, 53, 47, 44, 51], [48, 45, 25, 29, 41, 49], [27, 22, 26]], extra_clusters=[6, 52, 55, 14, 20, 19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[56, 59, 10, 7, 13, 57, 58, 54, 50, 53, 47, 44, 51, 48, 45, 25, 29, 41, 49, 27, 22, 26, 52, 55, 20,
                  19, 17, 43, 39, 36, 23, 5, 21, 35, 9, 24], [6], [14]], labels=['GE', '6', '14'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1, 2, 3])

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[57, 58, 54, 50, 53, 47, 44, 51, 39, 35, 9, 24, 5, 23, 36, 21, 17], [
                  9]], labels=['C3', '9'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

# %% [markdown]
# # Sequoia (substructure D):

# %%
create_tables_defining(df=df_clusters, path=path, name='D', main=[64, 63, 62])

# %% [markdown]
# ## Conclusion D:
#
# As before, everything seems compatible... with cluster #62 showing some differences. Given position in IoM, as well as velocity space, I would put the three separated.

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[64], [63], [62]], labels=[
                  '64', 'Sequoia', '62'], colours=['red', 'lightblue', 'blue'], P=80, zorder=[1, 2, 3])

# %% [markdown]
# # Helmi (substructure E)

# %%
create_tables_defining(df=df_clusters, path=path, name='E', main=[60, 61])

# %% [markdown]
# ## Conclusion E:
#
# Both clusters belong together forming the Helmi streams.

# %%
compare_MDFs_CMDs_low_Mstars(df_clusters, clusters_CM=[[60], [61]], labels=[
                  '60', '61'], colours=['red', 'lightblue'], P=80, zorder=[1, 2])

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
def nice_table_from_csv(path, name, table_type):
    fname = f"{path}{name}{table_type}"
    df = vaex.open(fname)

    columns = df.column_names
    Ncols = len(columns)

    header = ''
    for col in columns[:-1]:
        header = header + '{\\bf ' + str(col).replace('Unnamed: 0', '') + '} & '
    header = header + '{\\bf ' + str(columns[-1]) + '} \\\\ \n'

    with open(fname, 'r') as f:
        lines = f.readlines()
        Nlines = len(lines)

    rows = []

    for i in np.arange(1, Nlines):
        this_row = ''
        this_line = lines[i]
        elements = this_line.split(',')
        for k, element in enumerate(elements):
            if k == 0:
                this_row += '{\\bf ' + str(element) + '} & '
            else:
                this_row += '\cellcolor[HTML]{' + str(matplotlib.colors.to_hex(plt.cm.Greens(float( element)), keep_alpha=True)).strip('#').replace('ff', 'f')
                if float(element) < 0.05:
                    this_row += '}\color{red}'
                elif float(element) > 0.75:
                    this_row += '}\color{white}'
                else:
                    this_row += '}\color{black}'
                this_row += str(round(float(element), 2)) + ' & '
        rows.append(this_row)

    # Start writing on the .tex file:


    tab_type = 'l'
    for _ in range(Ncols-1):
        tab_type += 'r'

    f_latex_table = f"{path}temp.tex"
    latex_table = open(f_latex_table, 'w')
    chain = '\\begin{table} \n'
    latex_table.write(chain)
    chain = '{\centering \n'
    latex_table.write(chain)
    chain = '\small \n'
    latex_table.write(chain)
    chain = '\\begin{tabular}{' + str(tab_type) + '} \n'
    latex_table.write(chain)
    chain = '\hline \n'
    latex_table.write(chain)

    # Writing header:
    chain = header
    latex_table.write(chain)

    # Now the rest of rows with the correct format
    for row in rows:
        chain = row[0:-2] + ' \\\\ '
        latex_table.write(chain)

    chain = '\hline \hline \n'
    latex_table.write(chain)
    chain = '\end{tabular} \n'
    latex_table.write(chain)
    chain = '\caption{' + str(name) + '} \n'
    latex_table.write(chain)
    chain = '} \n'
    latex_table.write(chain)
    chain = '\end{table} \n'
    latex_table.write(chain)
    latex_table.close()

    with open(f_latex_table, "r") as h:
        file_contents = h.read()
    print(file_contents)
    os.system('rm ' + f_latex_table)


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
comp_2str_IoM(df_clusters, clusters1=[28], clusters2=[28], alpha=1.0)

# %%
comp_2str_IoM(df_clusters, clusters1=[40], clusters2=[40], alpha=1.0)

# %%
