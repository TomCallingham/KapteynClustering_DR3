from .legacy import trattcode as tc
from .legacy import H99Potential as H99Potential

def H99_dyn_calc(pos, vel):
    En, Ltotal, Lz, Lperp = H99Potential.gal_En_L_calc(pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2])
    En_circ, Lz_circ = tc.calc_tratt()
    dyn = {"En": En, "L": Ltotal, "Lz":Lz, "Lperp":Lperp}
    dyn["circ"] = tc.calc_circ(En, En_circ, -Lz, Lz_circ)
    return dyn
