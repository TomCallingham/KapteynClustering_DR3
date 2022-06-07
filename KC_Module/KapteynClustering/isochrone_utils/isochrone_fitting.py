import glob
import time
import numpy as np
import matplotlib.pyplot as plt
# from astropy.io import ascii
import glob
from scipy.spatial import cKDTree
import itertools
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import warnings
warnings.filterwarnings("ignore")

BaSTI_iso_folder = "/net/gaia2/data/users/ruizlara/isochrones/"
PARSEC_iso_folder = "/net/gaia2/data/users/ruizlara/isochrones/parsec/"

# kdtree to compute distances from stars to the closest point in the isochrone.


def do_kdtree(observed_data, isochrone, k=1, error_weights=None):
    if error_weights is None:
        error_weights = np.ones((len(isochrone)))
    mytree = cKDTree(observed_data)
    dist = mytree.query(isochrone, k=k)[0]
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


def read_age_met_from_auxname(auxname, iso_model):
    lines = open(auxname, 'r').readlines()
    if iso_model == "BaSTI":
        iso_age, iso_mh = float(lines[4].split('=')[5].split()[0]), float(lines[4].split('=')[2].split()[0])
        iso_age = iso_age / (10**3)

    elif iso_model == "PARSEC":
        iso_age, iso_mh = float(lines[15].split()[2]), float(lines[15].split()[1])
        iso_age = (10**iso_age) / (10**9)
    else:
        raise AttributeError(f"iso_model unkown: {iso_model}")
    return iso_age, iso_mh


def auxnames_read(alpha, iso_model):
    if alpha not in ["SS", "AE"]:
        raise AttributeError(f"Alpha= {alpha} not recognised")
    if iso_model == "BaSTI":
        if alpha == "SS":
            auxnames = glob.glob(BaSTI_iso_folder + 'solar_scaled/*.isc_gaia-dr3')
        else:
            auxnames = glob.glob(BaSTI_iso_folder + '/alpha_enhanced/*.isc_gaia-dr3')

    elif iso_model == "PARSEC":
        if alpha == "SS":
            auxnames = glob.glob(PARSEC_iso_folder + 'solar_scaled/parsecSS*.txt')
        else:
            auxnames = glob.glob(PARSEC_iso_folder + 'alpha_enhanced/parsecAE*.txt')
    else:
        raise AttributeError(f"iso_model {iso_model} not recognised")
    return np.array(auxnames)


def define_set_of_isochrones(age_range, met_range, alpha, iso_model="BaSTI"):
    names_aux = auxnames_read(alpha, iso_model=iso_model)
    ages_aux, mets_aux = np.zeros_like(names_aux, dtype=float), np.zeros_like(names_aux, dtype=float)
    for i, name in enumerate(names_aux):
        ages_aux[i], mets_aux[i] = read_age_met_from_auxname(name, iso_model=iso_model)
    filt = (
        age_range[0] < ages_aux) & (
        ages_aux < age_range[1]) & (
            met_range[0] < mets_aux) & (
                mets_aux < met_range[1])
    ages, mets, names = ages_aux[filt], mets_aux[filt], names_aux[filt]
    return ages, mets, names



from multiprocessing import Pool
from itertools import repeat

def iso_cmd_read_single(iso_name, iso_models):
    if iso_models == "BaSTI":
        # iso = ascii.read(iso_name)
        # iso_color, iso_mag = iso['col6'] - iso['col7'], iso['col5']
        iso = np.loadtxt(iso_name)
        iso_color, iso_mag = iso[:,5] - iso[:,6], iso[:,4]
    elif iso_models == "PARSEC":
        # iso = ascii.read(iso_name)
        # iso_color, iso_mag = iso['G_BPmag'] - iso['G_RPmag'], iso['Gmag']
        iso = np.loadtxt(iso_name)
        iso_color, iso_mag = iso[:,29] - iso[:,30], iso[:,28]
    else:
        raise AttributeError(f"iso model {iso_models} not recognised")
    iso_CMD = np.column_stack((iso_color, iso_mag))
    return iso_CMD

def iso_cmd_read(iso_names, iso_models, N_process=8):
    with Pool(processes=N_process) as pool:
        result = pool.starmap(iso_cmd_read_single, zip(iso_names, repeat(iso_models)))
    iso_CMDs = {n:r for (n,r) in zip(iso_names, result)}
    return iso_CMDs

def fit_isochrone_CMDs(isochrone_CMD_list, observed_CMD, observed_weights, N_process=8):
    with Pool(processes=N_process) as pool:
        result = pool.starmap(do_kdtree,zip(isochrone_CMD_list, repeat(observed_CMD), repeat(1),repeat(observed_weights)))
    N_isochrones = len(isochrone_CMD_list)
    parameters_fit = np.array([ result[i][0] for i in range(N_isochrones)])
    distances = np.array([ result[i][1] for i in range(N_isochrones)])
    return parameters_fit, distances

global_iso_CMDs = {"BaSTI":{}, "PARSEC":{}}
def iso_cmd_lazy_load(iso_names, iso_models):
    global global_iso_CMDs
    current_names = np.array(list(global_iso_CMDs[iso_models].keys()))
    new_names = iso_names[np.isin(iso_names,current_names,invert=True)]
    if len(new_names)>0:
        print(f"loading {len(new_names)} new isochrones")
        new_CMDs =  iso_cmd_read(new_names, iso_models,N_process=8)
        global_iso_CMDs[iso_models] = global_iso_CMDs[iso_models]|new_CMDs
    return global_iso_CMDs[iso_models]

# Fitting and determination of age and metallicity:
def iso_fit_prob(total_name, data_arrays, fitting_params, plot=True, save_path=None):
    # deg_surfx, deg_surfy,
    '''
        data_arrays containing: colour, colour_err, magnitude, extinction
        fitting_params: extinction_cut, percentile_solution, iso_models=PARSEC/BaSTI, alpha=SS,AE, ranges
        ranges containing: colour, magnitude, iso_age, iso_met
    '''
    print('###############################')
    print(f'## iso_fit to {total_name} ##')
    print('###############################')

    [extinction_cut, percentile_solution, iso_models, alpha, ranges] = \
            [fitting_params[p] for p in ["extinction_cut", "percentile_solution", "iso_models", "alpha", "ranges"]]
    [iso_age_range, iso_met_range, col_range, mag_range] = \
            [ranges[p] for p in ["iso_age", "iso_met", "colour", "magnitude"]]

    range_filt = (col_range[0] < data_arrays["colour"] ) * (data_arrays["colour"] < col_range[1]) * (mag_range[0] < data_arrays["magnitude"]) * (data_arrays["magnitude"] < mag_range[1]) * (data_arrays["extinction"] < extinction_cut)
    [colour_array, colour_error_array, magnitude_array, magnitude_error_array] = [data_arrays[p][range_filt]
                                                           for p in ["colour", "colour_err", "magnitude", "magnitude_err"]]

    if len(colour_array) == 0:
        print('No stars after applying all cuts and conditions')
        Age, Age_err, MH, MH_err, d2_param, d2_perc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        print("returning best_isochrone as 'None'. Check if appopriate!")
        best_isochrone = None  # CHECK what best_isochrone should go to
        results = {"Age": Age, "Age_err": Age_err, "MH": MH, "MH_err": MH_err,
                   "d2_param": d2_param, "d2_perc": d2_perc, "best_isochrone": best_isochrone}

        return results

    start_time = time.perf_counter()

    # Preparing observed CMD:
    observed_CMD = np.column_stack((colour_array, magnitude_array))
    observed_weights = 1 / (colour_error_array + magnitude_error_array)

    # Preparing set of isochrones and fit to observed data:
    ages, mets, names = define_set_of_isochrones(iso_age_range, iso_met_range, alpha, iso_model=iso_models)
    N_names = len(names)

    print(f'## Using {N_names} isochrones ({iso_models}, with alpha: {alpha}) \n ...')

    isochrone_CMD = iso_cmd_lazy_load(names, iso_models)
    parameters_fit = fit_isochrone_CMDs([isochrone_CMD[n] for n in names], observed_CMD, observed_weights, N_process=8)[0]

    # Compute age-Metallicity values and their error:
    condition = (parameters_fit < np.percentile(parameters_fit, percentile_solution))
    Age, Age_err, MH, MH_err = np.nanmean( ages[condition]), np.nanstd( ages[condition]), np.nanmean( mets[condition]), np.nanstd( mets[condition])

    #TODO: Check if this is needed? Have as an option?
    # Fitting surface to the points:
    # Generate w
    # w = polyfit2d(ages, mets, parameters_fit, m_1=deg_surfx, m_2=deg_surfy)
    # Generate x', y', z'
    # n_ = 1000
    # x_, y_ = np.meshgrid(np.linspace(ages.min(), ages.max(), n_), np.linspace(mets.min(), mets.max(), n_))
    # z_ = np.zeros((n_, n_))
    # for i in range(n_):
    #     z_[i, :] = polyval2d(x_[i, :], y_[i, :], w, mdeg_surfx, m_2=deg_surfy)
    # i_z_min = np.argmin(z_)
    # ages_min_z, mets_min_z = x_[i_z_min], y_[i_z_min]
    # Age, MH = (Age+ages_min_z)/2, (MH+mets_min_z)/2

    # Closest isochrone:
    whole_age_dist = np.column_stack((ages, mets))

    age_dist = [[Age, MH]]
    distances = do_kdtree(age_dist, whole_age_dist)[1]
    min_dist_index = np.argmin(distances)
    best_isochrone, best_age, best_met = names[min_dist_index], ages[min_dist_index], mets[min_dist_index]

    d2_param = parameters_fit[min_dist_index]
    d2_perc = 100 * ((d2_param < parameters_fit).sum() / len(parameters_fit))

    time_taken = time.perf_counter() - start_time
    print('###############################')
    print('## iso_fit results: ##')
    print(f'## Age = {Age:.2f} +/- {Age_err:.2f} Gyr ##')
    print(f'## [M/H] = {MH:.2f} +/- {MH_err:.2f} ##')
    print(f'## Closest isochrone is {best_isochrone} (age,[M/H]) = ({best_age}, {best_met})')
    print(f'## Goodness of fit: {d2_param :.3f} ({round(d2_perc, 2)}%) ##')
    print('###############################')
    print(f'Execution time: {time_taken} seconds')

    results = {"Age": Age, "Age_err": Age_err, "MH": MH, "MH_err": MH_err,
               "d2_param": d2_param, "d2_perc": d2_perc, "best_isochrone": best_isochrone}
    if plot:
        plotting_data = {
            "ages": ages,
            "mets": mets,
            "names": names,
            "parameters_fit": parameters_fit,
            "observed_weights": observed_weights}
        iso_fit_plot(total_name, data_arrays, ranges, fitting_params, results, plotting_data, save_path)

    return results


def iso_fit_plot(total_name, data_arrays, ranges, fitting_params, results, plotting_data, save_path=None):
    print('## Plotting results \n ...')

    # Load
    [extinction_cut, iso_models, alpha] = [fitting_params[p]
                                           for p in ["extinction_cut", "iso_models", "alpha"]]
    col_range, mag_range = ranges["colour"], ranges["magnitude"]
    range_filt = (col_range[0] < data_arrays["colour"] ) * (data_arrays["colour"] < col_range[1]) * (mag_range[0] < data_arrays["magnitude"]) * (data_arrays["magnitude"] < mag_range[1]) * (data_arrays["extinction"] < extinction_cut)


    [colour_array, colour_error_array, magnitude_array, magnitude_error_array, AG_array] =\
        [data_arrays[p][range_filt]
            for p in ["colour", "colour_err", "magnitude", "magnitude_err", "extinction"]]
    [colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, extinction_array_all] =\
        [data_arrays[p] for p in ["colour", "colour_err", "magnitude", "magnitude_err", "extinction"]]
    [Age, Age_err, MH, MH_err, d2_param, d2_perc, best_isochrone] = [results[p]
                                                                     for p in ["Age", "Age_err", "MH", "MH_err", "d2_param", "d2_perc", "best_isochrone"]]
    [ages, mets, names, parameters_fit, observed_weights] = [plotting_data[p]
                                                             for p in ["ages", "mets", "names", "parameters_fit", "observed_weights"]]
    percentile_solution = fitting_params["percentile_solution"]
    condition = (parameters_fit < np.percentile(parameters_fit, percentile_solution))


    # PLOT
    f1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(24, 7), facecolor='w')
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    img = ax1.scatter(ages, mets, c=parameters_fit, cmap='viridis_r')
    # choose 20 contour levels, just to show how good its interpolation is
    ax1.tricontourf(ages, mets, np.log10(parameters_fit), 20, cmap='viridis_r')
    ax1.errorbar(
        Age,
        MH,
        xerr=Age_err,
        yerr=MH_err,
        marker='o',
        mfc='black',
        mec='black',
        ecolor='black',
        ms=10)
    ax1.scatter(ages, mets, c=np.log10(parameters_fit), cmap='viridis_r')
    ax1.scatter(ages[condition], mets[condition], c='black', alpha=0.3, s=5)
    cbar = f1.colorbar(img, ax=ax1, pad=0.0)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r'$d_2$', size=20)

    ax1.set_xlabel(r'$\rm Age \; [Gyr]$', fontsize=25)
    ax1.set_ylabel(r'$\rm [M/H]$', fontsize=25)
    ax1.set_xlim(np.nanmin(ages), np.nanmax(ages))
    ax1.set_ylim(np.nanmin(mets), np.nanmax(mets))

    ax1.tick_params(labelsize=20)

    #ax2.plot(colour_array, magnitude_array, 'bo', alpha = 0.8)
    ax2.errorbar( colour_array_all, magnitude_array_all, xerr=colour_error_array_all, yerr=magnitude_error_array_all,
        fmt='none', ecolor='blue', alpha=0.4)
    ax2.errorbar( colour_array, magnitude_array, xerr=colour_error_array, yerr=magnitude_error_array, fmt='none', ecolor='blue')
    scatter2 = ax2.scatter(
        colour_array,
        magnitude_array,
        c=observed_weights,
        cmap='viridis',
        s=10,
        vmin=np.percentile(
            observed_weights,
            10),
        vmax=np.percentile(
            observed_weights,
            90))
    cbar = f1.colorbar(scatter2, ax=ax2, pad=0.0)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r'$Weights$', size=18)
    ax2.plot([np.nan, np.nan], [np.nan, np.nan], 'bo', label='Cluster #' +
             str(total_name) + ', ' + str(len(colour_array)) + ' stars')

    ax3.scatter(
        colour_array_all,
        magnitude_array_all,
        c=extinction_array_all,
        cmap='viridis',
        s=10,
        alpha=0.4,
        vmin=np.nanmin(extinction_array_all),
        vmax=np.nanmax(extinction_array_all))
    scatter3 = ax3.scatter(
        colour_array,
        magnitude_array,
        c=AG_array,
        cmap='viridis',
        s=10,
        vmin=np.nanmin(extinction_array_all),
        vmax=np.nanmax(extinction_array_all))
    cbar = f1.colorbar(scatter3, ax=ax3, pad=0.0)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r'$Extinction$', size=18)

    for ax in [ax2, ax3]:
        # isochrones:
        iso_CMDs = iso_cmd_lazy_load(names, iso_models)
        for iso_name, age, met in zip(names, ages, mets):
            if ((Age - Age_err < age) & (age < Age + Age_err) & (MH - MH_err < met) & (met < MH + MH_err)):
                iso_color, iso_mag = iso_CMDs[iso_name][:,0], iso_CMDs[iso_name][:,1]
                ax.plot(iso_color, iso_mag, 'k-', lw=1, alpha=0.3)
        ax.plot([None], [None], 'k-', lw=1, alpha=0.3, label='compatible isochrones')
        # iso_color, iso_mag = iso_color_mag_read(best_isochrone, iso_models)
        iso_color, iso_mag = iso_CMDs[best_isochrone][:,0], iso_CMDs[best_isochrone][:,1]
        ax.plot(iso_color, iso_mag, 'r-', lw=1, label='Best fit: Age = ' +
                str(Age.round(2)) + '; [M/H] = ' + str(MH.round(2)))
        ax.plot([col_range[0], col_range[1], col_range[1], col_range[0], col_range[0]], [
                mag_range[0], mag_range[0], mag_range[1], mag_range[1], mag_range[0]], 'k--', alpha=0.2)
        ax.set_xlabel(r'$\rm G_{BP}-G_{RP}$', fontsize=25)
        ax.set_ylabel(r'$\rm M_{G}$', fontsize=25)
        ax.set_xlim(col_range[0] - 0.25, col_range[1] + 0.25)
        ax.set_ylim(mag_range[1] + 1.0, mag_range[0] - 1.0)
        ax.tick_params(labelsize=20)

    # Create a set of inset Axes: these should fill the bounding box allocated to them.
    axx0 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.03, 0.05, 0.4, 0.3])  # [left, bottom, width, height]
    axx0.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines in grey linking the two axes.
    #mark_inset(ax1, axx0, loc1=2, loc2=4, fc="none", ec='0.5')
    axx0.patch.set_alpha(0.0)
    axx0.hist(parameters_fit, bins=20, histtype='step', color='white')
    axx0.axvline(d2_param, ls='--', color='white')
    # axx0.axes.get_xaxis().set_ticks([])
    axx0.axes.get_yaxis().set_ticks([])
    axx0.spines['right'].set_visible(False)
    axx0.spines['top'].set_visible(False)
    axx0.spines['bottom'].set_color('white')
    axx0.spines['left'].set_color('white')
    axx0.tick_params(axis='x', colors='white')
    ax1.text(0.45, 0.02, r'$\rm d_2 = ' + str(d2_param.round(3)) + ' \\; (' +
             str(round(d2_perc, 2)) + '\\%)$', transform=ax1.transAxes, color='white', fontsize=15)

    ax3.legend(fontsize='x-large')

    plt.tight_layout()

    if save_path is not None:
        fname = f"{iso_models}_iso_fit_parsec{alpha}.png"
        plt.savefig(str(save_path) + "/" + fname)
    # plt.show()

    return
