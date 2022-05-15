import numpy as np


def add_dynamics(stars, pot_fname=None, J_finder=None, angles=True, circ=True, ex=False):
    '''
    stars dic containg pos,vel as Nx3.
    pot_fname either "H99" or file path to agama Potential.
    '''
    pos = stars["pos"]
    vel = stars["vel"]
    stars = add_cylindrical(stars)

    if pot_fname == "H99":
        from .H99_DynamicsCalc import H99_dyn_calc
        dyn = H99_dyn_calc(pos,vel)
    else:
        from agama_dynamics import load_agama_potential, agama_dyn_calc
        a_pot = load_agama_potential(pot_fname)
        dyn = agama_dyn_calc(pos, vel, pot=a_pot, J_finder=J_finder,
                             angles=angles, circ=circ, ex=ex)
    stars = stars | dyn
    return stars


# TODO: Make clearer x,y,z and vx,vy,vz differences!
def add_cylindrical(stars):
    pos, vel = stars["pos"], stars["vel"]
    stars['r'] = np.linalg.norm(pos, axis=1)
    stars['R'], stars['phi'], temp_z, stars['vR'], stars['vT'], temp_vz = cart_to_cylinder(
        pos, vel)
    return stars

def cart_to_cylinder(pos, vel):
    ''' x = R sin(phi)'''
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
