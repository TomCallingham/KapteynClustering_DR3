import numpy as np

G = 4.30132*1e-6 #kpc Msun^-1 (km/s)^2   # from wikipedia
# parameters from Box of Chocolates paper Helmi (2017)

M_disc = 6.3e10 # Msun
disc_a = 6.5 # kpc
disc_b = 0.26 # kpc

M_bulge = 2.1e10 # Msun
bulge_c = 0.7 # kpc

v_halo = 173.2 # km/s
halo_d = 12 # kpc

def potential(pos):
    phi_halo  = potential_halo(pos)
    phi_disc  = potential_disc(pos)
    phi_bulge = potential_bulge(pos)
    phi_total = phi_halo+phi_bulge+phi_disc
    return phi_total

def dPotential_dr(pos):
    dphi_dr_halo = dPot_dr_halo(pos)
    dphi_dr_disc = dPot_dr_disc(pos)
    dphi_dr_bulge = dPot_dr_bulge(pos)
    dphi_dr_total  = dphi_dr_halo + dphi_dr_disc + dphi_dr_bulge
    return dphi_dr_total

def vc(pos):
    r = np.linalg.norm(pos,axis=1)
    dphi_dr = dPotential_dr(pos)
    vc_total = np.sqrt(-r*dphi_dr)
    return vc_total

def potential_halo(pos):
    " Halo: Logarithmic "
    r = np.linalg.norm(pos,axis=1)
    phi_halo = v_halo ** 2 * np.log(1+r**2 / halo_d**2)
    phi0 = (v_halo**2)*np.log(halo_d**2)
    # - (v_halo**2)*np.log(Rvir**2)

    return phi_halo

def potential_disc(pos):
    " Disk: Miyamoto-Nagai "
    x,y,z = pos[:,0], pos[:,1], pos[:,2]
    R = np.sqrt((x**2) + (y**2))
    phi_disc = - G*M_disc / np.sqrt(R**2 + (disc_a + np.sqrt(z**2 + disc_b**2))**2)
    return phi_disc

def potential_bulge(pos):
    " Bulge: Hernquist "
    r = np.linalg.norm(pos,axis=1)
    phi_bulge = - G*M_bulge/(r+bulge_c)
    return phi_bulge



def dPot_dr_halo(pos):
    " Halo: Logarithmic "
    r = np.linalg.norm(pos,axis=1)
    dPot_dr = 2*v_halo ** 2 * r / (r**2 + halo_d**2)
    return dPot_dr

def dPot_dr_disc(pos):
    " Disk: Miyamoto-Nagai "
    x,y,z = pos[:,0], pos[:,1], pos[:,2]
    R = np.sqrt((x**2) + (y**2))
    r = np.linalg.norm(pos,axis=1)
    dPhi_discdr1 = G*M_disc / np.power(R**2 + (disc_a + np.sqrt(z**2 + disc_b**2))**2, 1.5)
    dPhi_discdr2 = (R*R/r + (disc_a + np.sqrt(z**2 + disc_b**2))*z**2/(r * np.sqrt(z**2+disc_b**2)))
    dPhi_discdr = dPhi_discdr1*dPhi_discdr2
    return dPhi_discdr

def dPot_dr_bulge(pos):
    " Bulge: Hernquist "
    r = np.linalg.norm(pos,axis=1)
    dPhi_bulge_dr = G*M_bulge/(r+bulge_c)**2
    return dPhi_bulge_dr
