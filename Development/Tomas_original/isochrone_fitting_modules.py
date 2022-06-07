import sys,glob,os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import glob
from scipy.spatial import cKDTree
import itertools
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import warnings
warnings.filterwarnings("ignore")

# kdtree to compute distances from stars to the closest point in the isochrone.
def do_kdtree(observed_data, isochrone, k, error_weights):
    mytree = cKDTree(observed_data)
    dist, indexes = mytree.query(isochrone,k=k)
    #parameter = np.nanmean(dist)/float(len(observed_data))
    parameter = np.sqrt(np.average(dist**2, weights=error_weights)) # Minimisation similar to https://ui.adsabs.harvard.edu/abs/2019ApJS..245...32L/abstract
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

BaSTI_iso_folder = "/net/gaia2/data/users/ruizlara/isochrones/"
PARSEC_iso_folder = "/net/gaia2/data/users/ruizlara/isochrones/parsec/"
# Set of isochrones to consider (BaSTI):
def define_set_of_isochrones_basti(age_range, met_range, alpha):
    names_aux, ages_aux, mets_aux = [], [], []
    if alpha == 0:
        auxnames = glob.glob(BaSTI_iso_folder+'solar_scaled/*.isc_gaia-dr3')
    elif alpha == 1:
        auxnames = glob.glob(BaSTI_iso_folder+'/alpha_enhanced/*.isc_gaia-dr3')
    for i, name in enumerate(auxnames):
        lines = open(auxnames[i], 'r').readlines()
        iso_age, iso_mh = lines[4].split('=')[5].split()[0], lines[4].split('=')[2].split()[0]
        ages_aux.append(float(iso_age)/10**3)
        if alpha == 0:
            mets_aux.append(float(iso_mh))
        elif alpha == 1:
            mets_aux.append(float(iso_mh))
        names_aux.append(name)

    names_aux, ages_aux, mets_aux = np.array(names_aux), np.array(ages_aux), np.array(mets_aux)
    coords = np.where(((age_range[0] < ages_aux)&(ages_aux < age_range[1])&(met_range[0] < mets_aux)&(mets_aux < met_range[1])))
    ages, mets, names = ages_aux[coords], mets_aux[coords], names_aux[coords]
    return np.array(ages), np.array(mets), np.array(names)

# Set of isochrones to consider (PARSEC):
def define_set_of_isochrones_parsec(age_range, met_range, alpha):
    names_aux, ages_aux, mets_aux = [], [], []
    if alpha == 0:
        auxnames = glob.glob(PARSEC_iso_folder+'solar_scaled/parsecSS*.txt')
    elif alpha == 1:
        auxnames = glob.glob(PARSEC_iso_folder+'alpha_enhanced/parsecAE*.txt')
    for i, name in enumerate(auxnames):
        lines = open(auxnames[i], 'r').readlines()
        iso_age, iso_mh = lines[15].split()[2], lines[15].split()[1]
        ages_aux.append(10**float(iso_age)/10**9)
        if alpha == 0:
            mets_aux.append(float(iso_mh))
        elif alpha == 1:
            mets_aux.append(float(iso_mh))
        names_aux.append(name)

    names_aux, ages_aux, mets_aux = np.array(names_aux), np.array(ages_aux), np.array(mets_aux)
    coords = np.where(((age_range[0] < ages_aux)&(ages_aux < age_range[1])&(met_range[0] < mets_aux)&(mets_aux < met_range[1])))
    ages, mets, names = ages_aux[coords], mets_aux[coords], names_aux[coords]
    return np.array(ages), np.array(mets), np.array(names)




# Fitting and determination of age and metallicity:
def iso_fit_prob(name, colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, extinction_array_all, extinction_cut, iso_age_range, iso_met_range, col_range, mag_range, deg_surfx, deg_surfy, percentile_solution, iso_models, alpha, path):

    coords = np.where((colour_array_all > col_range[0])&(colour_array_all < col_range[1])&(magnitude_array_all > mag_range[0])&(magnitude_array_all < mag_range[1])&(extinction_array_all<extinction_cut))
    colour_array, colour_error_array, magnitude_array, magnitude_error_array, AG_array = colour_array_all[coords], colour_error_array_all[coords], magnitude_array_all[coords], magnitude_error_array_all[coords], extinction_array_all[coords]

    if len(colour_array) == 0:
        print('No stars after applying all cuts and conditions')
        Age, errA, MH, errMH, d2_param, d2_perc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        import time
        start_time = time.perf_counter()

        print('###############################')
        print('## iso_fit to '+str(name)+' ##')
        print('###############################')

        # Preparing observed CMD:
        observed_CMD, observed_weights = [], []
        for i in np.arange(len(colour_array)):
            observed_CMD.append([colour_array[i], magnitude_array[i]])
            observed_weights.append(1.0/(colour_error_array[i] + magnitude_error_array[i]))
        observed_CMD, observed_weights = np.array(observed_CMD), np.array(observed_weights)

        # Preparing set of isochrones and fit to observed data:
        if iso_models == 1: # BaSTI
            ages, mets, names = define_set_of_isochrones_basti(iso_age_range, iso_met_range, alpha)
        elif iso_models == 2: # PARSEC
            ages, mets, names = define_set_of_isochrones_parsec(iso_age_range, iso_met_range, alpha)
        if (alpha == 0)&(iso_models == 1):
            print('## Using '+str(len(names))+' isochrones (BaSTI, solar-scaled)')
            print('...')
        elif (alpha == 1)&(iso_models == 1):
            print('## Using '+str(len(names))+' isochrones (BaSTI, alpha-enhanced)')
            print('...')
        if (alpha == 0)&(iso_models == 2):
            print('## Using '+str(len(names))+' isochrones (PARSEC, solar-scaled)')
            print('...')
        elif (alpha == 1)&(iso_models == 2):
            print('## Using '+str(len(names))+' isochrones (PARSEC, alpha-enhanced)')
            print('...')

        ages_fit, mets_fit, parameters_fit = [], [], []
        # isochrones

        # print("My load")
        # from KapteynClustering.isochrone_fitting import iso_cmd_lazy_load
        # isochrone_CMD_dic = iso_cmd_lazy_load(names, {1:"BaSTI",2:"PARSEC"}[iso_models])

        for iso_name, iso_age, iso_met in zip(names, ages, mets):
            iso = ascii.read(iso_name)
            if iso_models == 1: # BaSTI
                iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
            elif iso_models == 2: # PARSEC
                iso_color, iso_mag = iso['G_BPmag'] - iso['G_RPmag'], iso['Gmag']
            isochrone_CMD = []
            for j in np.arange(len(iso_color)):
                isochrone_CMD.append([iso_color[j], iso_mag[j]])
            isochrone_CMD = np.array(isochrone_CMD)

            # isochrone_CMD = isochrone_CMD_dic[iso_name]
            parameter, aux2 = do_kdtree(isochrone_CMD, observed_CMD, 1, observed_weights)
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
        condition = (z<np.percentile(z, percentile_solution))
        XX1, XXerr, YY1, YYerr = np.nanmean(x[condition]), np.nanstd(x[condition]), np.nanmean(y[condition]), np.nanstd(y[condition])
        XX2, YY2 = x_[np.where(z_==np.min(z_))], y_[np.where(z_==np.min(z_))]
        XX, YY = XX1, YY1
        #XX, YY = (XX1+XX2)/2.0, (YY1+YY2)/2.0

        f1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24,7), facecolor='w')
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)

        img = ax1.scatter(ages_fit, mets_fit, c=parameters_fit, cmap='viridis_r')
        ax1.tricontourf(x,y,z_plot, 20, cmap='viridis_r') # choose 20 contour levels, just to show how good its interpolation is
        ax1.errorbar(XX, YY, xerr=XXerr, yerr=YYerr, marker='o', mfc='black', mec='black', ecolor='black', ms=10)
        ax1.scatter(ages_fit, mets_fit, c=np.log10(parameters_fit), cmap='viridis_r')
        ax1.scatter(x[condition], y[condition], c='black', alpha = 0.3, s = 5)
        cbar = f1.colorbar(img, ax=ax1, pad=0.0)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r'$d_2$', size=20)


        ax1.set_xlabel(r'$\rm Age \; [Gyr]$', fontsize=25)
        ax1.set_ylabel(r'$\rm [M/H]$', fontsize=25)
        ax1.set_xlim(np.nanmin(ages_fit), np.nanmax(ages_fit))
        ax1.set_ylim(np.nanmin(mets_fit), np.nanmax(mets_fit))

        ax1.tick_params(labelsize=20)

        #ax2.plot(colour_array, magnitude_array, 'bo', alpha = 0.8)
        ax2.errorbar(colour_array_all, magnitude_array_all, xerr=colour_error_array_all, yerr=magnitude_error_array_all, fmt='none', ecolor='blue', alpha=0.4)
        ax2.errorbar(colour_array, magnitude_array, xerr=colour_error_array, yerr=magnitude_error_array, fmt='none', ecolor='blue')
        scatter2 = ax2.scatter(colour_array, magnitude_array, c=observed_weights, cmap='viridis', s=10, vmin=np.percentile(observed_weights, 10), vmax=np.percentile(observed_weights, 90))
        cbar = f1.colorbar(scatter2, ax=ax2, pad=0.0)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r'$Weights$', size=18)


        ax2.plot([np.nan, np.nan], [np.nan, np.nan], 'bo', label = 'Cluster #'+str(name)+', '+str(len(colour_array))+' stars')

        # isochrones:
        whole_age_dist = []
        for k in np.arange(len(names)):
            whole_age_dist.append([ages[k], mets[k]])
            if ((XX-XXerr < ages[k])&(ages[k] < XX+XXerr)&(YY-YYerr < mets[k])&(mets[k] < YY+YYerr)):
                iso = ascii.read(names[k])
                if iso_models == 1: # BaSTI
                    iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
                elif iso_models == 2: # PARSEC
                    iso_color, iso_mag = iso['G_BPmag'] - iso['G_RPmag'], iso['Gmag']
                ax2.plot(iso_color, iso_mag, 'k-', lw = 1, alpha=0.3)
        ax2.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw = 1, alpha=0.3, label='compatible isochrones')

        # Closest isochrone:
        whole_age_distr_weights = np.ones((len(whole_age_dist)))
        age_dist = [[XX, YY]]
        aux1, distances = do_kdtree(age_dist, whole_age_dist, 1, whole_age_distr_weights)
        min_dist_index = np.where(distances==np.min(distances))[0][0]

        best_isochrone = names[min_dist_index]
        iso = ascii.read(best_isochrone)
        if iso_models == 1: # BaSTI
            iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        elif iso_models == 2: # PARSEC
            iso_color, iso_mag = iso['G_BPmag'] - iso['G_RPmag'], iso['Gmag']
        ax2.plot(iso_color, iso_mag, 'r-', lw = 1, label='Best fit: Age = '+str(XX.round(2))+'; [M/H] = '+str(YY.round(2)))

        ax2.plot([col_range[0], col_range[1], col_range[1], col_range[0], col_range[0]], [mag_range[0], mag_range[0], mag_range[1], mag_range[1], mag_range[0]], 'k--', alpha = 0.2)

        ax2.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax2.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax2.set_xlim(col_range[0]-0.25, col_range[1]+0.25)
        ax2.set_ylim(mag_range[1]+1.0, mag_range[0]-1.0)
        ax2.tick_params(labelsize=20)

        ax3.scatter(colour_array_all, magnitude_array_all, c=extinction_array_all, cmap='viridis', s=10, alpha=0.4, vmin=np.nanmin(extinction_array_all), vmax=np.nanmax(extinction_array_all))
        scatter3 = ax3.scatter(colour_array, magnitude_array, c=AG_array, cmap='viridis', s=10, vmin=np.nanmin(extinction_array_all), vmax=np.nanmax(extinction_array_all))
        cbar = f1.colorbar(scatter3, ax=ax3, pad=0.0)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r'$Extinction$', size=18)

        # isochrones:
        whole_age_dist = []
        for k in np.arange(len(names)):
            whole_age_dist.append([ages[k], mets[k]])
            if ((XX-XXerr < ages[k])&(ages[k] < XX+XXerr)&(YY-YYerr < mets[k])&(mets[k] < YY+YYerr)):
                iso = ascii.read(names[k])
                if iso_models == 1: # BaSTI
                    iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
                elif iso_models == 2: # PARSEC
                    iso_color, iso_mag = iso['G_BPmag'] - iso['G_RPmag'], iso['Gmag']
                ax3.plot(iso_color, iso_mag, 'k-', lw = 1, alpha=0.3)
        ax3.plot([np.nan, np.nan], [np.nan, np.nan], 'k-', lw = 1, alpha=0.3, label='compatible isochrones')

        iso = ascii.read(best_isochrone)
        if iso_models == 1: # BaSTI
            iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        elif iso_models == 2: # PARSEC
            iso_color, iso_mag = iso['G_BPmag'] - iso['G_RPmag'], iso['Gmag']
        ax3.plot(iso_color, iso_mag, 'r-', lw = 1, label='Best fit: Age = '+str(XX.round(2))+'; [M/H] = '+str(YY.round(2)))

        ax3.plot([col_range[0], col_range[1], col_range[1], col_range[0], col_range[0]], [mag_range[0], mag_range[0], mag_range[1], mag_range[1], mag_range[0]], 'k--', alpha = 0.2)

        ax3.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax3.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax3.set_xlim(col_range[0]-0.25, col_range[1]+0.25)
        ax3.set_ylim(mag_range[1]+1.0, mag_range[0]-1.0)
        ax3.tick_params(labelsize=20)

        # Goodness of fit and plot:

        coord = np.where((mets_fit==mets[min_dist_index])&(ages_fit==ages[min_dist_index]))
        fit_param = parameters_fit[coord[0][0]]
        coords = (fit_param<parameters_fit)
        fit_param_perc = 100.0*float(len(coords[coords==True]))/float(len(parameters_fit))

        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        axx0 = plt.axes([0,0,1,1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax1, [0.03,0.05,0.4,0.3]) #[left, bottom, width, height]
        axx0.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
        #mark_inset(ax1, axx0, loc1=2, loc2=4, fc="none", ec='0.5')
        axx0.patch.set_alpha(0.0)
        axx0.hist(parameters_fit, bins = 20, histtype='step', color='white')
        axx0.axvline(fit_param, ls='--', color='white')
        #axx0.axes.get_xaxis().set_ticks([])
        axx0.axes.get_yaxis().set_ticks([])
        axx0.spines['right'].set_visible(False)
        axx0.spines['top'].set_visible(False)
        axx0.spines['bottom'].set_color('white')
        axx0.spines['left'].set_color('white')
        axx0.tick_params(axis='x', colors='white')
        ax1.text(0.45, 0.02, r'$\rm d_2 = '+str(fit_param.round(3))+' \; ('+str(round(fit_param_perc, 2))+'\%)$', transform=ax1.transAxes, color='white', fontsize=15)

        ax3.legend(fontsize='x-large')

        plt.tight_layout()

        if (alpha == 0)&(iso_models == 2):
            plt.savefig(str(path)+'/'+str(name)+'_iso_fit_parsecSS.png')
        elif (alpha == 1)&(iso_models == 2):
            plt.savefig(str(path)+'/'+str(name)+'_iso_fit_parsecAE.png')
        elif (alpha == 0)&(iso_models == 1):
            plt.savefig(str(path)+'/'+str(name)+'_iso_fit_BaSTISS.png')
        elif (alpha == 1)&(iso_models == 1):
            plt.savefig(str(path)+'/'+str(name)+'_iso_fit_BaSTIAE.png')

        #plt.show()

        print('###############################')
        print('## iso_fit results: ##')
        print('## Age = '+str(XX.round(2))+' +/- '+str(XXerr.round(2))+' Gyr ##')
        print('## [M/H] = '+str(YY.round(2))+' +/- '+str(YYerr.round(2))+' ##')
        print('## Closest isochrone is '+str(best_isochrone)+' (age,[M/H]) = ('+str(ages[min_dist_index])+', '+str(mets[min_dist_index])+')')
        print('## Goodness of fit: '+str(fit_param.round(3))+' ('+str(round(fit_param_perc, 2))+'%) ##')
        print('###############################')
        print('Execution time: '+str((time.perf_counter() - start_time))+' seconds')

        Age, errA, MH, errMH, d2_param, d2_perc = XX, XXerr, YY, YYerr, fit_param, fit_param_perc

    return Age, errA, MH, errMH, d2_param, d2_perc, best_isochrone
