from .legacy import trattcode as tc
from .legacy import H99Potential as H99Potential

def H99_dyn_calc(pos, vel):
    En_art, Ltotal_art, Lz_art, Lperp_art = H99Potential.gal_En_L_calc(pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2])
    En_circ, Lz_circ = tc.calc_tratt()
    dyn = {"E": En_art, "L": Ltotal_art, "Lz":Lz_art, "Lp":Lperp_art}
    dyn["circ"] = tc.calc_circ(En_art, En_circ, -Lz_art, Lz_circ)
    return dyn
