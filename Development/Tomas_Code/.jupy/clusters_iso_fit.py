# %%
import numpy as np
import matplotlib.pyplot as plt
import KapteynClustering.isochrone_utils.isochrone_fitting as isofit
import vaex
from astropy.io import fits
from astropy.io import ascii
import glob
import os

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
%load_ext line_profiler

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
N_stars = len(data["col2"])

# %%
'''
        data_arrays containing: colour, colour_err, magnitude, extinction
        fitting_params: extinction_cut, percentile_solution, iso_models=PARSEC/BaSTI, alpha=SS,AE, ranges:
        ranges containing: colour, magnitude, iso_age, iso_met
    '''
data_arrays = {"colour": data['col2'],  # Array with all the colours (corrected by extincition) absolute plane. One element per star.
               # error in the colour of all stars. If not available provide the same error for every star.
               "colour_err": np.full((N_stars), 0.03),
               # Array with all the magnitudes (corrected by extincition) absolute plane.
               # One element per star.
               "magnitude": data["col1"],
               # error in the magnitude of all stars. If not available provide the same error for every star.
               "magnitude_err": np.full((N_stars), 0.03),
               # Array with the extinction for every star. This is just to avoid in the
               # fit stars with high extinction (could be AG, AV, etc), or a dummy array
               # (zeros)
               "extinction": np.zeros((N_stars))
               }
ranges = {"colour": [0.45, 2.0],  # Colour range. Stars with a colour in this range (and magnitude in the magnitude range below) will be considered in the fit.
          # Magnitude range. Stars with a magnitude in this range (and colour in the
          # magnitude range above) will be considered in the fit.
          "magnitude": [-2.5, 8.0],
          # Age range. Isochrones with an age in this range will be considered in the fit.
          "iso_age": [0.0, 14.0],
          # Metallicity range. Isochrones with a metallicity ([M/H]) in this range
          # will be considered in the fit.
          "iso_met": [-3.0, 0.4]
          }
fitting_params = {"extinction_cut": 0.7,  # All stars with an extinction parameter (AG, AV, etc) below this value
                  # Percentile of good isochrones. Grey dots and greay isochrones in the
                  # plot. All these isochrones will be considered to compute average age and
                  # metallicity.
                  "percentile_solution": 4,
                  "iso_models": "BaSTI",  # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
                  "alpha": "AE",  # "SS" for solar scaled, AE for alpha enhanced
                  "ranges": ranges}


# deg_surfx = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.
# deg_surfy = 7 # degree of the 2D polinomial fit (x-direction). d2 parameter. Do not change.

# %%
# BaSTI Alpha-enhanced

name = 'example'  # Name of the run (it can be just a identifier, this is just to name the outputs)
# Path to store the figure the code will create.
path = '/home/tomas/Documents/work/IAC/alicia/isochrone_fitting/'


results = isofit.iso_fit_prob(name, data_arrays, fitting_params)

# %%
# BaSTI Alpha-enhanced

name = 'example'  # Name of the run (it can be just a identifier, this is just to name the outputs)
# Path to store the figure the code will create.
path = '/home/tomas/Documents/work/IAC/alicia/isochrone_fitting/'


results = isofit.iso_fit_prob(name, data_arrays, fitting_params)

# %%
%lprun - f  isofit.iso_fit_prob isofit.iso_fit_prob(name, data_arrays, fitting_params)

# %%
# BaSTI, Solar scaled

fitting_params = {"extinction_cut": 0.7,  # All stars with an extinction parameter (AG, AV, etc) below this value
                  # Percentile of good isochrones. Grey dots and greay isochrones in the
                  # plot. All these isochrones will be considered to compute average age and
                  # metallicity.
                  "percentile_solution": 4,
                  "iso_models": "BaSTI",  # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
                  "alpha": "SS",  # "SS" for solar scaled, AE for alpha enhanced
                  "ranges": ranges}


results = isofit.iso_fit_prob(name, data_arrays, fitting_params)

# %%
# BaSTI, Solar scaled

fitting_params = {"extinction_cut": 0.7,  # All stars with an extinction parameter (AG, AV, etc) below this value
                  # Percentile of good isochrones. Grey dots and greay isochrones in the
                  # plot. All these isochrones will be considered to compute average age and
                  # metallicity.
                  "percentile_solution": 4,
                  "iso_models": "BaSTI",  # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
                  "alpha": "SS",  # "SS" for solar scaled, AE for alpha enhanced
                  "ranges": ranges}


results = isofit.iso_fit_prob(name, data_arrays, fitting_params)

# %% [markdown]
# # PARSEC

# %%
# PASEC, Solar scaled

fitting_params = {"extinction_cut": 0.7,  # All stars with an extinction parameter (AG, AV, etc) below this value
                  # Percentile of good isochrones. Grey dots and greay isochrones in the
                  # plot. All these isochrones will be considered to compute average age and
                  # metallicity.
                  "percentile_solution": 4,
                  "iso_models": "PARSEC",  # Models to use in the fit. 1 --> BaSTI: 2 --> PARSEC
                  "alpha": "SS",  # "SS" for solar scaled, AE for alpha enhanced
                  "ranges": ranges}


results = isofit.iso_fit_prob(name, data_arrays, fitting_params)
