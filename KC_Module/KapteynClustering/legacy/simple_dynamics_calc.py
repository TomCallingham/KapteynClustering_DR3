import numpy as np
from KapteynClustering import dynamics_funcs as dynf

def dynamics_calc(pot_fname,pos, vel, circ=True):
    if pot_fname=="H99":
        from . import H99Potential as potential
    elif pot_fname=="BoxChoc":
        from . import BoxChocPotential as potential
    else:
        raise ValueError(f"Pot name {pot_fname}")

    dyn = dynf.angular_momentum_calc(pos=pos, vel=vel)

    dyn['K'] = 0.5 * np.sum(vel**2, axis=1)
    dyn["U"] = potential.potential(pos)
    dyn['En'] = dyn['U'] + dyn['K']
    if circ:
        dyn["circ"] = calc_circ(potential,dyn["En"], dyn["Lz"])
    return dyn


def calc_circ(potential,En,  Lz, rmin=0.01, rmax=500, N=5000):

    pos = np.zeros((N,3))
    r = np.geomspace(rmin, rmax, N)
    pos[:,2] = r
    vc = potential.vc(pos)
    En_circ = 0.5*vc**2 + potential.potential(pos)
    Lz_circ = vc*r

    Ncl = len(En)
    circularity = np.full(Ncl,np.nan)
    E_fit = (En_circ[0] < En)*(En < En_circ[-1])
    circularity[E_fit] = Lz[E_fit]/np.interp(En[E_fit], En_circ, Lz_circ)
    circularity[En_circ[0]>En] = Lz[En_circ[0]>En]/Lz_circ[0]
    circularity[En_circ[-1]<En] = Lz[En_circ[-1]<En]/Lz_circ[-1]
    return circularity
