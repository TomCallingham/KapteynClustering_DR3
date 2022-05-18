import agama
# units: kpc, km/s, Msun; time unit ~ 1 Gyr
agama.setUnits(length=1, velocity=1, mass=1)
import numpy as np

def agama_dyn_calc(pos, vel, pot=None, J_finder=None, angles=True, circ=True, ex=False):
    dyn = {}
    dyn['pos'] = pos
    dyn['vel'] = vel
    xyz = np.column_stack((pos, vel))


    dyn['K'] = 0.5 * np.sum(vel**2, axis=1)

    Lvec = np.cross(pos, vel)
    dyn['L'] = np.linalg.norm(Lvec, axis=1)
    dyn['Lz'] = Lvec[:, 2]
    dyn['Lperp'] = np.sqrt((dyn['L']**2) - (dyn['Lz']**2))
    dyn['Lvec'] = Lvec
    if pot is not None:
        dyn['U'] = pot.potential(pos)
        dyn['En'] = dyn['U'] + dyn['K']
        if circ:
            print("calc Circ")
            NR = 1000
            Rspace = np.geomspace(0.1, 500, NR)
            vc_space = vcirc(Rspace, pot)
            Lspace = Rspace * vc_space
            cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
            pot_space = pot.potential(cart_space)
            Ecspace = pot_space + ((vc_space**2) / 2)
            dyn['Lcirc_En'] = np.interp(dyn['En'], Ecspace, Lspace)
            dyn['Circ'] = dyn['Lz'] / dyn['Lcirc_En']

        if ex:
            print('Extremums')
            Rmaxs = pot.Rperiapo(np.column_stack((dyn['En'], dyn['L'])))
            dyn['Rmin'] = Rmaxs[:, 0]
            dyn['Rmax'] = Rmaxs[:, 1]
            dyn['e'] = (dyn['Rmax'] - dyn['Rmin']) / \
                (dyn['Rmax'] + dyn['Rmin'])

    if J_finder is not False and pot is not None:
        if J_finder is None:
            J_finder = agama.ActionFinder(pot, interp=True)
        if angles:
            Js, As, Os = J_finder(xyz, angles=True)
            dyn['AR'] = As[:, 0]
            dyn['Az'] = As[:, 1]
            dyn['Aphi'] = As[:, 2]
            dyn['OR'] = Os[:, 0]
            dyn['Oz'] = Os[:, 1]
            dyn['Ophi'] = Os[:, 2]
        else:
            Js = J_finder(xyz, angles=False)

        dyn['JR'] = Js[:, 0]
        dyn['Jz'] = Js[:, 1]
        dyn['Jphi'] = Js[:, 2]

    return dyn

def vcirc(Rspace, pot):
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    force = pot.force(cart_space)
    vcirc = np.sqrt(-cart_space[:, 0] * force[:, 0])
    return vcirc

def load_agama_potential(pot_fname):
    ''' Loads an agama potetial localy, or from the installed data folder'''
    if type(pot_fname)== str:
        try:
            a_pot =  agama.Potential(pot_fname)
        except Exception:
            import os.path
            AGAMA_folder = os.path.dirname(agama.__file__) + "/data/"
            a_pot =  agama.Potential(AGAMA_folder + pot_fname)
        return a_pot
    else:
        a_pots = [load_agama_potential(p) for p in pot_fname]
        a_pot = agama.Potential(a_pots)
        return a_pot

