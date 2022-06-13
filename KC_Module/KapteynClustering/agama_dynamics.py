import numpy as np
import agama
from KapteynClustering import dynamics_funcs as dynf
# units: kpc, km/s, Msun; time unit ~ 1 Gyr
agama.setUnits(length=1, velocity=1, mass=1)
def agama_dyn_calc(pos, vel, pot, actions=False, angles=False, circ=True, ex=False):
    xyz = np.column_stack((pos, vel))

    dyn = dynf.angular_momentum_calc(pos=pos, vel=vel)
    dyn['K'] = 0.5 * np.sum(vel**2, axis=1)
    dyn['U'] = pot.potential(pos)
    dyn['En'] = dyn['U'] + dyn['K']
    if circ:
        dyn["circ"], dyn["Lcirc_En"] = agama_circ_calc(pot,dyn["En"], dyn["Lz"])
    if ex:
        print('Extremums')
        Rmaxs = pot.Rperiapo(np.column_stack((dyn['En'], dyn['L'])))
        dyn['Rmin'] = Rmaxs[:, 0]
        dyn['Rmax'] = Rmaxs[:, 1]
        dyn['e'] = (dyn['Rmax'] - dyn['Rmin']) / \
            (dyn['Rmax'] + dyn['Rmin'])

    if np.logical_or(actions,angles):
        action_dic = agama_action_angle_calc(pot,xyz, J_finder=None, angles=angles)
        dyn = dyn| action_dic

    return dyn

def agama_circ_calc(pot, En, Lz):
    NR = 1000
    Rspace = np.geomspace(0.1, 500, NR)
    vc_space = vcirc(Rspace, pot)
    Lspace = Rspace * vc_space
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    pot_space = pot.potential(cart_space)
    Ecspace = pot_space + ((vc_space**2) / 2)
    Lcirc_En = np.interp(En, Ecspace, Lspace)
    circ = Lz / Lcirc_En
    return circ, Lcirc_En

def agama_action_angle_calc(pot,xyz, J_finder=None, angles=False):
    dyn = {}
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
    ''' Loads an agama potential localy, or from the installed data folder'''
    if type(pot_fname) == str:
        try:
            a_pot = agama.Potential(pot_fname)
        except Exception:
            import os.path
            AGAMA_folder = os.path.dirname(agama.__file__) + "/data/"
            a_pot = agama.Potential(AGAMA_folder + pot_fname)
        return a_pot
    else:
        a_pots = [load_agama_potential(p) for p in pot_fname]
        a_pot = agama.Potential(*a_pots)
        return a_pot
