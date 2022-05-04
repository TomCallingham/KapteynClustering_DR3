import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1)   # units: kpc, km/s, Msun; time unit ~ 1 Gyr

def cart_to_cylinder(pos, vel):
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    vx = vel[:, 0]
    vy = vel[:, 1]
    vz = vel[:, 2]
    R = np.sqrt((x * x) + (y * y))
    phi = np.arctan2(y, x)

    vR = ((np.cos(phi) * vx) + (np.sin(phi) * vy))
    vT = ((np.cos(phi) * vy) - (np.sin(phi) * vx))
    return (R, phi, z, vR, vT, vz)

def vcirc(Rspace, pot):
    cart_space = np.column_stack((Rspace, 0 * Rspace, 0 * Rspace))
    force = pot.force(cart_space)
    vcirc = np.sqrt(-cart_space[:, 0] * force[:, 0])
    return vcirc

def agama_dyn_calc(pos, vel, pot=None, J_finder=None, angles=True, circ=True, ex=False):
    dyn = {}
    dyn['pos'] = pos
    dyn['vel'] = vel
    xyz = np.column_stack((pos, vel))

    dyn['R'], dyn['phi'], dyn['z'], dyn['vR'], dyn['vT'], dyn['vz'] = cart_to_cylinder(
        pos, vel)
    dyn['r'] = np.linalg.norm(pos, axis=1)
    dyn['K'] = 0.5 * np.sum(vel**2, axis=1)

    Lvec = np.cross(pos, vel)
    dyn['L'] = np.linalg.norm(Lvec, axis=1)
    dyn['Lz'] = Lvec[:, 2]
    dyn['Lp'] = np.sqrt((dyn['L']**2) - (dyn['Lz']**2))
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

            # dyn['RcircEn'] = pot.Rcirc(E=dyn['En'])
            # dyn['RcircL'] = pot.Rcirc(L=dyn['Lz'])
            dyn['Lcirc_En'] = np.interp(dyn['En'], Ecspace, Lspace)
            dyn['Circ'] = dyn['Lz'] / dyn['Lcirc_En']

        if ex:
            print('Extremums')
            Rmaxs = pot.Rperiapo(np.column_stack((dyn['En'], dyn['L'])))
            dyn['Rmin'] = Rmaxs[:, 0]
            dyn['Rmax'] = Rmaxs[:, 1]
            dyn['e'] = (dyn['Rmax'] - dyn['Rmin']) / (dyn['Rmax'] + dyn['Rmin'])

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

def add_dynamics(stars, pot=None, J_finder=None, angles=True, circ=True, ex=False):
    try:
        pos = stars["Pos"]
        vel = stars["Vel"]
    except Exception:
        pos = stars["pos"]
        vel = stars["vel"]
    dyn = agama_dyn_calc(pos, vel, pot=pot, J_finder=J_finder, angles=angles, circ=circ, ex=ex)
    stars = stars|dyn
    return stars

def load_a_potential(pot_fname):
    a_pot = agama.Potential(pot_fname)
    return a_pot
# "Lz", "En", "Ltotal", "Lperp", "circ", "Actions"
