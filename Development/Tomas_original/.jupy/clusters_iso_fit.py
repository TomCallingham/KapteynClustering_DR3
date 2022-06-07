# %%
import numpy as np
import matplotlib.pyplot as plt
import isochrone_fitting_modules as isofit
import vaex
from astropy.io import fits
from astropy.io import ascii
import glob
import os

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# %% [markdown]
# # BaSTI isochrones in Gaia2
#
# The BaSTI isochrones that we used for the eDR3 papers can be found in:
#
# /net/gaia2/data/users/ruizlara/isochrones (two folders, alpha_enhanced and solar_scaled)... this needs to be hardcoded in isochrone_fitting_modules)

# %%
hdul = fits.open('M67MAG.fits')

# %%
data = hdul[1].data
hdr = hdul[1].header

# %%
plt.figure()
plt.plot(data['col2'], data['col1'], 'ko')
plt.ylim(10.0, -4.0)
plt.show()

# %%
# BaSTI Alpha-enhanced

name = 'example' # Name of the run (it can be just a identifier, this is just to name the outputs)
colour_array_all = data['col2'] # Array with all the colours (corrected by extincition) absolute plane. One element per star.
colour_error_array_all = np.ones(len(data['col2']))*0.03 # error in the colour of all stars. If not available provide the same error for every star.
magnitude_array_all = data['col1'] # Array with all the magnitudes (corrected by extincition) absolute plane. One element per star.
magnitude_error_array_all = np.ones(len(data['col1']))*0.03 # error in the magnitude of all stars. If not available provide the same error for every star.
extinction_array_all = np.zeros(len(data['col1'])) # Array with the extinction for every star. This is just to avoid in the fit stars with high extinction (could be AG, AV, etc), or a dummy array (zeros)
extinction_cut = 0.7 # All stars with an extinction parameter (AG, AV, etc) below this value 
iso_age_range = [0.0, 14.0] # Age range. Isochrones with an age in this range will be considered in the fit.
iso_met_range = [-3.0,0.4] # Metallicity range. Isochrones with a metallicity ([M/H]) in this range will be considered in the fit.
col_range = [0.45, 2.0] # Colour range. Stars with a colour in this range (and magnitude in the magnitude range below) will be considered in the fit.
mag_range = [-2.5, 8.0] # Magnitude range. Stars with a magnitude in this range (and colour in the magnitude range above) will be considered in the fit.
deg_surfx = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
deg_surfy = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
percentile_solution = 4 # Percentile of good isochrones. Grey dots and greay isochrones in the plot. All these isochrones will be considered to compute average age and metallicity.
iso_models = 1 # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
alpha = 1 # 0 --> solar-scaled; 1 --> alpha-enhanced models.
# path = '/home/tomas/Documents/work/IAC/alicia/isochrone_fitting/' # Path to store the figure the code will create.
path = './' # Path to store the figure the code will create.

Age, errA, FeH, errFeH, d2_param, d2_perc, best_isochrone = isofit.iso_fit_prob(name, colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, extinction_array_all, extinction_cut, iso_age_range, iso_met_range, col_range, mag_range, deg_surfx, deg_surfy, percentile_solution, iso_models, alpha, path)

# %%
%load_ext line_profiler

# %%
%lprun -f isofit.iso_fit_prob isofit.iso_fit_prob(name, colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, extinction_array_all, extinction_cut, iso_age_range, iso_met_range, col_range, mag_range, deg_surfx, deg_surfy, percentile_solution, iso_models, alpha, path)

# %%
# BaSTI, Solar scaled

name = 'example' # Name of the run (it can be just a identifier, this is just to name the outputs)
colour_array_all = data['col2'] # Array with all the colours (corrected by extincition) absolute plane. One element per star.
colour_error_array_all = np.ones(len(data['col2']))*0.03 # error in the colour of all stars. If not available provide the same error for every star.
magnitude_array_all = data['col1'] # Array with all the magnitudes (corrected by extincition) absolute plane. One element per star.
magnitude_error_array_all = np.ones(len(data['col1']))*0.03 # error in the magnitude of all stars. If not available provide the same error for every star.
extinction_array_all = np.zeros(len(data['col1'])) # Array with the extinction for every star. This is just to avoid in the fit stars with high extinction (could be AG, AV, etc), or a dummy array (zeros)
extinction_cut = 0.7 # All stars with an extinction parameter (AG, AV, etc) below this value 
iso_age_range = [0.0, 14.0] # Age range. Isochrones with an age in this range will be considered in the fit.
iso_met_range = [-3.0,0.4] # Metallicity range. Isochrones with a metallicity ([M/H]) in this range will be considered in the fit.
col_range = [0.45, 2.0] # Colour range. Stars with a colour in this range (and magnitude in the magnitude range below) will be considered in the fit.
mag_range = [-2.5, 8.0] # Magnitude range. Stars with a magnitude in this range (and colour in the magnitude range above) will be considered in the fit.
deg_surfx = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
deg_surfy = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
percentile_solution = 4 # Percentile of good isochrones. Grey dots and greay isochrones in the plot. All these isochrones will be considered to compute average age and metallicity.
iso_models = 1 # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
alpha = 0 # 0 --> solar-scaled; 1 --> alpha-enhanced models.
# path = '/home/tomas/Documents/work/IAC/alicia/isochrone_fitting/' # Path to store the figure the code will create.
path = './' # Path to store the figure the code will create.

Age, errA, FeH, errFeH, d2_param, d2_perc, best_isochrone = isofit.iso_fit_prob(name, colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, extinction_array_all, extinction_cut, iso_age_range, iso_met_range, col_range, mag_range, deg_surfx, deg_surfy, percentile_solution, iso_models, alpha, path)

# %%
# PARSEC Solar Scaled

name = 'example' # Name of the run (it can be just a identifier, this is just to name the outputs)
colour_array_all = data['col2'] # Array with all the colours (corrected by extincition) absolute plane. One element per star.
colour_error_array_all = np.ones(len(data['col2']))*0.03 # error in the colour of all stars. If not available provide the same error for every star.
magnitude_array_all = data['col1'] # Array with all the magnitudes (corrected by extincition) absolute plane. One element per star.
magnitude_error_array_all = np.ones(len(data['col1']))*0.03 # error in the magnitude of all stars. If not available provide the same error for every star.
extinction_array_all = np.zeros(len(data['col1'])) # Array with the extinction for every star. This is just to avoid in the fit stars with high extinction (could be AG, AV, etc), or a dummy array (zeros)
extinction_cut = 0.7 # All stars with an extinction parameter (AG, AV, etc) below this value 
iso_age_range = [0.0, 14.0] # Age range. Isochrones with an age in this range will be considered in the fit.
iso_met_range = [-3.0,0.4] # Metallicity range. Isochrones with a metallicity ([M/H]) in this range will be considered in the fit.
col_range = [0.45, 2.0] # Colour range. Stars with a colour in this range (and magnitude in the magnitude range below) will be considered in the fit.
mag_range = [-2.5, 8.0] # Magnitude range. Stars with a magnitude in this range (and colour in the magnitude range above) will be considered in the fit.
deg_surfx = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
deg_surfy = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
percentile_solution = 4 # Percentile of good isochrones. Grey dots and greay isochrones in the plot. All these isochrones will be considered to compute average age and metallicity.
iso_models = 2 # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
alpha = 0 # 0 --> solar-scaled; 1 --> alpha-enhanced models.
path = '/home/tomas/Documents/work/IAC/alicia/isochrone_fitting/' # Path to store the figure the code will create.

Age, errA, FeH, errFeH, d2_param, d2_perc, best_isochrone = isofit.iso_fit_prob(name, colour_array_all, colour_error_array_all, magnitude_array_all, magnitude_error_array_all, extinction_array_all, extinction_cut, iso_age_range, iso_met_range, col_range, mag_range, deg_surfx, deg_surfy, percentile_solution, iso_models, alpha, path)

# %% [markdown]
# # Creating PARSEC isochrones (SS)

# %%
files = glob.glob('/home/tomas/Documents/work/IAC/alicia/isochrone_fitting/PARSEC_alicia/*.dat')
print(files)

# %%
for file in files:
    
    print(file)

    indexes = []

    in_file = open(file, 'r')  # Open filename

    for h, line in enumerate(in_file):                 # Check each line
        if line.startswith('# Zini '): # line starting with '# Zini'
            indexes.append(h)
    indexes.append(h) 
    in_file.close()
    
    in_file = open(file, 'r')  # Open filename
    inlines = in_file.readlines()

    for h in np.arange(len(indexes)-1):
        newfile = open('newfile.txt', 'w')
        for index in np.arange(indexes[h], indexes[h+1]):
            newfile.write(inlines[index])
        newfile.close()
        iso = ascii.read('newfile.txt') 
        iso_Zini, iso_MH, iso_logAge = iso['Zini'], iso['MH'], iso['logAge'] 
        iso_Zini, iso_MH, iso_logAge = iso_Zini[0], iso_MH[0], iso_logAge[0]
        del iso
        os.system('mv newfile.txt /home/tomas/Documents/work/IAC/alicia/isochrone_fitting/parsec/solar_scaled/parsecSS_Zini_'+str(iso_Zini)+'_MH_'+str(iso_MH)+'_logAge_'+str(iso_logAge)+'.txt')


# %%
