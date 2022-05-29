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
print(df.evaluate('flag_vlos'))

# %%
# make quality cuts 
# Quality cuts: ruwe < 1.4, parallax over error > 5, vlos error < 20 km/s. We consider halo as vtoomre > 210 km/s.

df = df[df.ruwe<1.4]
df = df[df.parallax_over_error>=5.]
#df = df[df.dr2_radial_velocity_error<20] for gaia vlos only
df = df[df.any_vlos_error<20] 
df = df[df.flag_vlos<6] 
print(df.count())

# %%
print(df.evaluate('flag_vlos'))

# %%
#coordinates.compute_coordinates_new(self=df) - using my own coordinate transformations taken from here and with some edits 

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
## Coordinate transformations following Tomas's scripts
#sys.path.append(r'/net/gaia2/data/users/ruizlara/useful_codes/')
#import pot_koppelman2018 as Potential
#import lallement18_distant as l18
import math
pi = math.pi # This is a bug within vaex (it seems), we need to define pi first, this way or that: df.add_variable('pi',math.pi).

df.add_variable('k',4.74057)   ;  k = 4.74057    
df.add_variable('R0',8.20)   ;  R0 = 8.2 
df.add_variable('Z0', 0.0)   ;  Z0 = 0.0     
df.add_variable('vlsr',232.8);  vlsr = 232.8
df.add_variable('_U',11.1)   ;  _U = 11.1
df.add_variable('_V',12.24)  ;  _V = 12.24
df.add_variable('_W',7.25)   ;  _W = 7.25

df.add_column('new_parallax', df.evaluate('parallax')+0.017) # new_parallax in 'mas', 1/parallax --> distance in kpc.
df.add_column('distance', 1.0/df.evaluate('new_parallax')) # new_parallax in 'mas', 1/parallax --> distance in kpc.

# Adding Cartesian coordinates (tilde X, Y, Z), centred at the Sun
df.add_virtual_columns_spherical_to_cartesian(alpha='l', delta='b', distance='distance', xname='tildex', yname='tildey', zname='tildez')

# Adding Cylindrical (polar) coordinates, with the Sun at the centre. --> tilde Rho, tilde Phi
df.add_virtual_columns_cartesian_to_polar(x='(tildex)', y='(tildey)', radius_out='tildeRho', azimuth_out='tildePhi', radians=True)

# Adding Cartesian coordinates centred at the galactic centre. --> xyz
# Need x,y,z to be helicentric
#df.add_column('x', df.evaluate('tildex-R0'))
#df.add_column('y', df.evaluate('tildey'))
#df.add_column('z', df.evaluate('tildez')+Z0)
df.add_column('x', df.evaluate('tildex'))
df.add_column('y', df.evaluate('tildey'))
df.add_column('z', df.evaluate('tildez'))

# Adding Cylindrical (polar) coordinates, with the Galactic centre at the centre. --> Rho, Phi
df.add_virtual_columns_cartesian_to_polar(x='(x-8.2)', y='(y)', radius_out='Rho', azimuth_out='Phi', radians=True)

# Adding Spherical coordinates, with the Galactic centre at the centre. alpha, delta, sphR
df.add_virtual_columns_cartesian_to_spherical(x='(x-8.2)', y='(y)', z='z+0.0', alpha='alpha', delta='delta', distance='sphR', radians=True)

# Equatorial proper motions to galactic proper motions:
df.add_virtual_columns_proper_motion_eq2gal(pm_long='pmra', pm_lat='pmdec', pm_long_out='pml', pm_lat_out='pmb')

# Computing vl and vb (linear)
vl = k*df.evaluate('pml')*df.evaluate('distance')
vb = k*df.evaluate('pmb')*df.evaluate('distance')
df.add_column('vl', vl)  # distance in kpc; mu in mas/yr
df.add_column('vb', vb)



# usual cartesian velcoities calculations - ED: 
df.add_virtual_columns_lbrvr_proper_motion2vcartesian(pm_long='pml', pm_lat='pmb', vr='any_vlos', vx='vx_', vy='vy_', vz='vz_')
# Need x,y,z, vx,vy,vz to be helicentric 
#df.add_virtual_column('vx','vx_+_U')
#df.add_virtual_column('vy','vy_+_V+vlsr')
#df.add_virtual_column('vz','vz_+_W')
df.add_virtual_column('vx','vx_')
df.add_virtual_column('vy','vy_')
df.add_virtual_column('vz','vz_')

df.add_virtual_columns_cartesian_velocities_to_polar(x='(x-8.2)', y='(y)', vx='(vx+_U)', vy='(vy+_V+vlsr)', vr_out='vRho', vazimuth_out='_vPhi', propagate_uncertainties=False)  
df.add_virtual_column('vPhi','-_vPhi')  # flip sign 

# Energies and so on using Helmer's code
#En, Lx, Ly, Lz, Lperp = Potential.En_L_calc(df.evaluate('(x-8.2)'), df.evaluate('y'), df.evaluate('z'), df.evaluate('(vx+_U)'), df.evaluate('(vy+_V+vlsr)'), df.evaluate('(vz+_W)'))
#Lz=-Lz # As always, change of sign according to Koppelman's papers.
#df.add_column('En_99',En)
#df.add_column('Lx',Lx)
#df.add_column('Ly',Ly)
#df.add_column('Lz',Lz) 
#df.add_column('Lperp', Lperp) # sqrt(Lx**2+Ly**2)
#df.add_column('Ltot', np.sqrt(df.evaluate('Lx')**2.+df.evaluate('Ly')**2.+df.evaluate('Lz')**2.))
    
    
# Taken from Helmer and Sofies codes 
### NEW TOOMRE CUT

df.add_virtual_columns_cartesian_to_polar(x='(x-8.2)', radius_out='R', azimuth_out='rPhi',radians=True)
df.add_virtual_columns_cartesian_velocities_to_polar(x='(x-8.2)', vx='(vx+_U)', vy='(vy+_V+vlsr)',
                                                     vr_out='vR', vazimuth_out='_vphi',propagate_uncertainties=True)  
df.add_virtual_column('vphi','-_vphi')  # flip sign

df.add_virtual_column('vtoomre','sqrt((vx+_U - vlsr*sin(rPhi))**2+ ((vy+_V+vlsr+vlsr*cos(rPhi))**2) + ((vz+_W)**2))')
#df.add_virtual_column('vtoomre','sqrt((vx+_U)**2+ ((vy+_V+vlsr -vlsr)**2) + ((vz+_W)**2))')


# %%
df.plot('vphi','sqrt(vR**2.+vz**2.)',f='log10')
df.plot('vphi','sqrt(vR**2.+vz**2.)',f='log10',selection='(vtoomre)<180',colormap='Greys')


# %%
df.count(selection='(vtoomre)<180')

# %%
# Make a distance cut

# stars within 5. kpc 

df.plot('vphi','sqrt(vR**2.+vz**2.)',f='log10',selection='(distance<=5.)')
df.plot('vphi','sqrt(vR**2.+vz**2.)',f='log10',selection='((vtoomre)<180)&(distance<=5.)',colormap='Greys')

df.select('((vtoomre)>180)&(distance<=5.)',name='Sample')

plt.figure()
df.plot('vphi','sqrt(vR**2.+vz**2.)',f='log10',selection='Sample')


# %%
df.plot('x','y')

plt.figure()
df.plot('vx','vy')

# %%
# Need x,y,z, vx,vy,vz to be helicentric 
df.filter('(Sample)').extract().export_hdf5('df_vtoomre180_5kpc.hdf5',progress=True)

# %%
fig,ax = plt.subplots(ncols=3,figsize=(21,6))
plt.sca(ax[0])
df.plot('vx','vy',selection='Sample',f='log10')

plt.sca(ax[1])
df.plot('vx','vz',selection='Sample',f='log10')

plt.sca(ax[2])
df.plot('vz','vy',selection='Sample',f='log10')



fig,ax = plt.subplots(ncols=3,figsize=(21,6))
plt.sca(ax[0])
df.plot('x','y',selection='Sample',f='log10')
#plt.xlim(-8.2-3,-8.2+3)
plt.xlim(-3,3)
plt.ylim(-3,3)

plt.sca(ax[1])
df.plot('x','z',selection='Sample',f='log10')
plt.xlim(-8.2-3,-8.2+3)
plt.xlim(-3,3)
plt.ylim(-3,3)

plt.sca(ax[2])
df.plot('z','y',selection='Sample',f='log10')
plt.xlim(-3,3)
plt.ylim(-3,3)

# %%
df = vaex.open('df_vtoomre180_5kpc.hdf5')

# %%
df

# %%
fig,ax = plt.subplots(ncols=3,figsize=(21,6))
plt.sca(ax[0])
df.plot('x','y',f='log10')
#plt.xlim(-8.2-3,-8.2+3)
plt.xlim(-3,3)
plt.ylim(-3,3)

plt.sca(ax[1])
df.plot('x','z',f='log10')
plt.xlim(-8.2-3,-8.2+3)
plt.xlim(-3,3)
plt.ylim(-3,3)

plt.sca(ax[2])
df.plot('z','y',f='log10')
plt.xlim(-3,3)
plt.ylim(-3,3)

# %%

fig,ax = plt.subplots(ncols=3,figsize=(21,6))
plt.sca(ax[0])
df.plot('x','y',f='log10',selection='(vtoomre)>210')
#plt.xlim(-8.2-3,-8.2+3)
plt.xlim(-3,3)
plt.ylim(-3,3)

plt.sca(ax[1])
df.plot('x','z',f='log10',selection='(vtoomre)>210')
plt.xlim(-8.2-3,-8.2+3)
plt.xlim(-3,3)
plt.ylim(-3,3)

plt.sca(ax[2])
df.plot('z','y',f='log10',selection='(vtoomre)>210')
plt.xlim(-3,3)
plt.ylim(-3,3)

# %%
