# %% [markdown]
# File to create the initial file to be input into the clustering algorithm -  Extended sample within 5kpc with toomre cut of 180 km/s. 

# %%
import vaex 
import numpy as np
import sys,glob,os,time
import matplotlib.pyplot as plt
#import coordinates



# %%
# Path to Gaia vlos file for EDR3 
path = '/net/gaia2/data/users/gaia/gaia-edr3/' 

# %%
df = vaex.open(path+'gaia-edr3-poege5-rv-sample-combined-galah-apogee-rave-lamostMR-segue-lamostLR.hdf5')
print(df.count())

# %%
# make quality cuts 
# Quality cuts: ruwe < 1.4, parallax over error > 5, vlos error < 20 km/s. We consider halo as vtoomre > 210 km/s.

df = df[df.ruwe<1.4]
df = df[df.parallax_over_error>=5.]
#df = df[df.dr2_radial_velocity_error<20] for gaia vlos only
df = df[df.any_vlos_error<20] 
df = df[df.flag_vlos<6] 

# %%
# remove false vlos - -9999 etc 
print(df.min('any_vlos'))

print(df.max('any_vlos'))

print(df.count(selection='any_vlos==-9999.0'))

df = df[df.any_vlos!=-9999.0]
print(df.min('any_vlos'))
print()
print(df.count())

# %%
df = vaex.from_pandas(df.to_pandas_df())

# %%

# %%
import KapteynClusteringyn
