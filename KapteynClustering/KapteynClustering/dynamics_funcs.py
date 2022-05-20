import numpy as np


def add_dynamics(stars, pot_fname=None, J_finder=None, angles=True, circ=True, ex=False):
    '''
    stars dic containg pos,vel as Nx3.
    pot_fname either "H99" or file path to agama Potential.
    '''
    try:
        pos = stars["pos"].values
        vel = stars["vel"].values
    except Exception:
        pos = stars["pos"]
        vel = stars["vel"]

    if pot_fname == "H99":
        # from .H99_DynamicsCalc import H99_dyn_calc
        from KapteynClustering.H99_DynamicsCalc import H99_dyn_calc
        dyn = H99_dyn_calc(pos, vel)
    else:
        from .agama_dynamics import load_agama_potential, agama_dyn_calc
        a_pot = load_agama_potential(pot_fname)
        dyn = agama_dyn_calc(pos, vel, pot=a_pot, J_finder=J_finder,
                             angles=angles, circ=circ, ex=ex)
    for prop in dyn.keys():
        try:
            del stars[prop]
        except:
            pass
    try:
        stars = stars | dyn
    except Exception:
        import vaex
        stars = stars.join( vaex.from_dict(dyn), how="right")
    stars = add_cylindrical(stars)
    return stars


# TODO: Make clearer x,y,z and vx,vy,vz differences!
def add_cylindrical(stars):
    try:
        pos = stars["pos"].values
        vel = stars["vel"].values
    except Exception:
        pos = stars["pos"]
        vel = stars["vel"]
    stars['r'] = np.linalg.norm(pos, axis=1)
    stars['R'], stars['phi'], temp_z, stars['vR'], stars['vT'], temp_vz = cart_to_cylinder(
        pos, vel)
    return stars


def cart_to_cylinder(pos, vel=None):
    ''' x = R sin(phi)'''
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    R = np.sqrt((x * x) + (y * y))
    phi = np.arctan2(y, x)

    if vel is  None:
        return (R, phi, z)
    else:
        vx = vel[:, 0]
        vy = vel[:, 1]
        vz = vel[:, 2]
        vR = ((np.cos(phi) * vx) + (np.sin(phi) * vy))
        vT = ((np.cos(phi) * vy) - (np.sin(phi) * vx))
        return (R, phi, z, vR, vT, vz)
