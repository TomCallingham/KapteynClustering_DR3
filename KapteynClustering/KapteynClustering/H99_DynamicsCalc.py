from .legacy import H99Potential as H99Potential
import numpy as np


def H99_dyn_calc(pos, vel):
    dyn = {}
    dyn['pos'] = pos
    dyn['vel'] = vel

    dyn['K'] = 0.5 * np.sum(vel**2, axis=1)

    Lvec = np.cross(vel,pos)
    dyn['L'] = np.linalg.norm(Lvec, axis=1)
    dyn['Lz'] = Lvec[:, 2]
    dyn['Lperp'] = np.sqrt((dyn['L']**2) - (dyn['Lz']**2))
    dyn['Lvec'] = Lvec
    dyn["U"] = H99Potential.potential_full(pos[:, 0], pos[:, 1], pos[:, 2])
    dyn['En'] = dyn['U'] + dyn['K']
    dyn["circ"] = calc_circ(dyn["En"], dyn["Lz"])
    return dyn



def calc_circ(En,  Lz):
    Ncl = len(En)
    circularity = np.zeros(Ncl)+np.nan
    En_circ, Lz_circ = calc_tratt()
    E_fit = (np.min(En_circ)<En)*(En<np.max(En_circ))
    circularity[E_fit] = Lz[E_fit]/np.interp(En[E_fit],En_circ,Lz_circ)

    return circularity

def calc_tratt(rmin=0.1,rmax=300,N=5000):
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
    r = np.geomspace(rmin, rmax,N)
    zeros = np.zeros_like(r)
    vc = H99Potential.vc_full(r,zeros,zeros)
    En_circ = 0.5*vc**2 + H99Potential.potential_full(r,zeros,zeros)
    Lz_circ = vc*r

    return En_circ, Lz_circ

# from .legacy import trattcode as tc
# def old_H99_dyn_calc(pos, vel):
#     En, Ltotal, Lz, Lperp = H99Potential.gal_En_L_calc(
#         pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2])
#     dyn = {"En": En, "L": Ltotal, "Lz": Lz, "Lperp": Lperp}
#     dyn["circ"] = tc.calc_circ(En, -Lz)
#     return dyn
