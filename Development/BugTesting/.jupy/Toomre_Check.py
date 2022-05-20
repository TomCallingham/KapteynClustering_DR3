# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:py39]
#     language: python
#     name: conda-env-py39-py
# ---

# %% tags=[]
import numpy as np
import matplotlib.pyplot as plt
import vaex

# %% [markdown]
# ### Paramaters
# Consistent across Notebooks

# %%
solar_params={'vlsr': 232.8, 'R0': 8.2, '_U': 11.1, '_V': 12.24, '_W': 7.25}
vlsr = solar_params["vlsr"]

# %% [markdown] tags=[]
# # Basic Dataset
# The basic dataset needs to contain
# - (Pos,Vel) in N x 3 form. These are GALACTIC centric
# - (x,y,z,vx,vy,vz). These are helio centric (following sofie)

# %%
sof_catalogue_path = '/net/gaia2/data/users/lovdal/datasample_vtoomre180.hdf5' #Path to raw data
gaia_catalogue = vaex.open(sof_catalogue_path)

# %%
import KapteynClustering.legacy.H99Potential as Potential
import KapteynClustering.legacy.trattcode as tc
def get_GAIA(ds):
    '''
    Computes integrals of motion for stars in the input catalogue and 
    adds these as columns to the dataframe.
    
    Parameters:
    ds(vaex.DataFrame): A dataframe containing heliocentric xyz-coordinates (kpc)
                        and xyz-velocity components (km/s).
                        
    Returns:
    ds(vaex.DataFrame): The input dataframe with added columns of total energy, 
                        angular momentum, circularity, together with positions and 
                        velocity components in polar coordinates.
    '''

    # Motion of the LSR, and solar position w.r.t. Galactic centre (McMillan (2017))
    (vlsr, R0) = (232.8, 8.2)
    # Solar motion w.r.t. LSR (Schoenrich et al. (2010))
    (_U,_V,_W) = (11.1, 12.24, 7.25)

    # Adding these as variables to the dataframe
    ds.add_variable('vlsr',232.8)
    ds.add_variable('R0',8.20)
    ds.add_variable('_U',11.1)
    ds.add_variable('_V',12.24)
    ds.add_variable('_W',7.25)

    # Compute energy and angular momentum components using the potential used in 
    # Koppelman et al. (2019): Characterization and history of the Helmi streams with Gaia DR2
    En, Ltotal, Lz, Lperp = Potential.En_L_calc(*ds.evaluate(['x','y','z','vx','vy','vz']))

    # Adding the columns to the data frame
    ds.add_column('En',En)
    ds.add_column('Lz',-Lz) ## Note: sign is flipped (just a convention)
    ds.add_column('Lperp',Lperp)
    ds.add_column('Ltotal', Ltotal)
    
    # Compute circularity
    En_circ, Lz_circ = tc.calc_tratt()
    # circ = tc.calc_circ(ds.count(), En, En_circ, Lz, Lz_circ)
    circ = tc.calc_circ(En, Lz)
    ds.add_column('circ', circ)
    
    # Add a column representing the maximum possible Lz for each energy value
    Lz_max = tc.get_Lz_max(ds.En.values, En_circ, Lz_circ)
    ds.add_column('Lz_max', Lz_max)

    # Adding some more columns (Galactocentric Cartesian velocities)
    ds.add_virtual_column('vx_apex',f'vx+{_U}')
    ds.add_virtual_column('vy_apex',f'vy+{_V}+{vlsr}')
    ds.add_virtual_column('vz_apex',f'vz+{_W}')
    
    # Polar coordinates
    ds.add_virtual_columns_cartesian_to_polar(x='(x-8.2)', radius_out='R', azimuth_out='rPhi',radians=True)

    # Velocities in polar coordinates
    ds.add_virtual_columns_cartesian_velocities_to_polar(x='(x-8.2)', vx='vx_apex', vy='vy_apex',vr_out='vR', vazimuth_out='_vphi',propagate_uncertainties=False)  
    ds.add_virtual_column('vphi','-_vphi')  # flip sign for convention
    
    # position depending correction for the LSR
    ds.add_column('_vx', ds.vx.values+11.1-ds.variables['vlsr']*np.sin(ds.evaluate('rPhi')))
    ds.add_column('_vy', ds.vy.values+12.24-ds.variables['vlsr']*np.cos(ds.evaluate('rPhi')))
    ds.add_column('_vz', ds.vz.values+7.25)
        
#     # The selection
#     ds.select('(_vx**2 + (_vy-232)**2 + _vz**2) > 210**2', name='halo')

#     # Toomre velocity: velocity offset from being a disk orbit
    ds.add_virtual_column('vtoomre','sqrt(_vx**2 + (_vy-232)**2 + _vz**2)')
    
    # The potential may compute energies larger than 0 in some edge cases, these are filtered out.
    return ds[(ds.En<0)]


# %%
ds = get_GAIA(gaia_catalogue)

# %%
sof_v_toomre = ds["vtoomre"].values

# %%
import KapteynClustering.data_funcs as dataf
my_gaia = dataf.create_galactic_posvel(ds, solar_params=solar_params)
my_v_toomre = dataf.calc_toomre(my_gaia["vel"], my_gaia["rPhi"], vlsr =solar_params["vlsr"]).values
sof_v_toomre = dataf.sof_calc_toomre(my_gaia["vel"], my_gaia["rPhi"], vlsr =solar_params["vlsr"]).values
simple_toomre = np.linalg.norm( my_gaia["vel"].values - np.array([0,vlsr,0]), axis=1)

# %%
plt.figure()
plt.hist(ds["rPhi"].values,bins=100)
plt.show()

# %%
import KapteynClustering.dynamics_funcs as dynf
my_phi = dynf.cart_to_cylinder(pos= ds["pos"].values)[1]

# %%
plt.figure()
plt.scatter(my_phi, ds["rPhi"].values)
plt.xlabel("my_phi")
plt.ylabel("Rphi")
plt.show()


plt.figure()
plt.scatter(my_phi, ds["y"].values)
plt.xlabel("my_phi")
plt.ylabel("gal y")
plt.show()

# %%
print(np.cos(np.pi))
print(np.sin(np.pi))

# %%
print((my_v_toomre>210).sum())
print((sof_v_toomre>210).sum())

# %%
plt.figure()
plt.scatter(my_v_toomre, ds["vtoomre"].values)
plt.show()

plt.figure()
plt.scatter(my_v_toomre, simple_toomre)
plt.show()

plt.figure()
plt.scatter(simple_toomre, ds["vtoomre"].values)
plt.show()

# %% [markdown]
# # Toomre Plots

# %%
sof_halo = ds[ds["vtoomre"]>210]

# %%
plt.figure()
# plt.scatter(ds["_vy"].values, np.sqrt((ds["_vx"]**2)+(ds["_vz"]**2)).values)
plt.scatter(sof_halo["vy_apex"].values, np.sqrt((sof_halo["vx_apex"]**2)+(sof_halo["vz_apex"]**2)).values,
           c=sof_halo["vtoomre"].values)
plt.colorbar()
plt.show()

# %%
plt.figure()
plt.scatter(my_v_toomre, ds["vtoomre"].values/my_v_toomre)
plt.show()

plt.figure()
plt.scatter(ds["rPhi"].values,my_v_toomre- ds["vtoomre"].values)
plt.show()

# %%
my_halo = ds[my_v_toomre>210]

# %%
plt.figure()
# plt.scatter(ds["_vy"].values, np.sqrt((ds["_vx"]**2)+(ds["_vz"]**2)).values)
plt.scatter(sof_halo["vy_apex"].values, np.sqrt((sof_halo["vx_apex"]**2)+(sof_halo["vz_apex"]**2)).values,
           c=sof_halo["vtoomre"].values)
plt.colorbar()
plt.show()

# %%
plt.figure()
# plt.scatter(ds["_vy"].values, np.sqrt((ds["_vx"]**2)+(ds["_vz"]**2)).values)
plt.scatter(my_halo["vy_apex"].values, np.sqrt((my_halo["vx_apex"]**2)+(my_halo["vz_apex"]**2)).values,
           c=my_v_toomre[my_v_toomre>210])
plt.colorbar()
plt.show()

# %%
plt.figure()
plt.scatter(my_v_toomre, sof_v_toomre)
plt.xlabel("my toomre")
plt.ylabel("sof toomre")
plt.show()

# %%
