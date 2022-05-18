import numpy as np
import matplotlib.pyplot as plt
import H99Potential as pot


def calc_and_plot_tratt(out = False):
    Ec, Lzc = calc_tratt()
    plot_tratt(Ec, Lzc)
    if out:
        return Ec, Lzc

def plot_tratt(En_circ, Lz_circ):
    
    plt.plot(Lz_circ, En_circ,c='k')
    plt.plot(-Lz_circ, En_circ,c='k')
    plt.show()


def calc_tratt(rmin=0.1,rmax=100,N=5000):
    """
    This code calculates the circularity of orbiting objects. 
    The circularity indicates how far the Lz of the orbiting
    object is from the Lz of a circular orbit with the same En.

    First, a bunch of combinations of En,Lz of circular orbits
    are calculated. Then the closest combination for each orbiting
    objects is found and Lz/Lz_circ is saved (i.e. the 'circularity').


    ::Input Parameters::
    rmin: smallest circular orbit to be included
    rmax: largest circular orbit to be included
    N:    grid points (number of points on interval [rmin, rmax])
    
    ::Return Parameters::
    En_circ, Lz_circ
    """

    # Generate a bunch of (En,Lz)_circ  combinations of circular orbits
    r = np.linspace(rmin, rmax,N)
    zeros = np.zeros_like(r)

    vc = pot.vc_full(r,zeros,zeros)
    En_circ = 0.5*vc**2 + pot.potential_full(r,zeros,zeros)
    Lz_circ = vc*r
    
    return En_circ, Lz_circ

#TODO:check this    
def calc_circ(Ncl, En, En_circ, Lz, Lz_circ):
    circularity = np.zeros(Ncl)+np.nan

    # Loop over the En and find the closest En_circ to it,
    # and with it the closest Lz_circ.

    for i in range(Ncl):
        dE = np.abs(En[i]-En_circ) #check in which energy-bin this datapoint fits best
        if En[i]<0:
            circularity[i] = -Lz[i]/Lz_circ[dE.argmin()] #circularity is the relative distance from the edge of the tratt
    
    return circularity    

def get_Lz_max(En, En_circ, Lz_circ):
    '''
    Returns the maximum possible Lz for a given energy -
    Equivalent to the Lz of a circular orbit for a given energy
    '''
    Lz_max = np.zeros(len(En))+np.nan
    for i in range(len(En)):
        dE = np.abs(En[i]-En_circ) #check in which energy-bin this datapoint fits best
        if En[i]<0:
            Lz_max[i] = Lz_circ[dE.argmin()] #return maximum Lz for a given energy
        else:
            Lz_max[i] = 10000 #random placeholder.
            
    return Lz_max
    #print(f'En: {En}')
    #dE = np.abs(En_circ-En)
    #print(f'dE: {dE}')
    #Lz_max = Lz_circ[dE.argmin()]
    #print(f'Lz_max: {Lz_max}')
    #print(Lz_max)
    #return Lz_max

# Old code included as example.

# # Get En, Lz of the stars/orbiting objects
# En,Lz = ds.evaluate('En,Lz')


# This bit can be skipped with the new code above.
# # Generate a bunch of (En,Lz)_circ  combinations of circular orbits
# N = 1000 # number of grid points
# x = np.linspace(0,100,N)
# vc = np.sqrt(np.abs(x*totalPot.force(np.array([x,x*0,x*0]).T)[:,0]))
# En_circ = 0.5*vc**2 + totalPot.potential(np.array([x,x*0,x*0]).T)
# Lz_circ = vc*x


# Ncl = int(ds.count())
# Lzc = np.zeros(Ncl)+np.nan

# # Loop over the En, Lz and find the (En,Lz)_circ closest to it
# for i in tqdm(range(Ncl)):
#   dE = np.abs(En[i]-En_circ)
#   if En[i]<0:
#       Lzc[i] = Lz[i]/Lz_circ[dE.argmin()]

# # Save results (can be skipped)
# np.save('Lzcirc.npy',Lzc)
