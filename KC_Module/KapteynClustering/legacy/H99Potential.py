import numpy as np

################################################################################
# ORIGINAL FILE BY JOVAN VELJANOSKI
################################################################################

G = 4.301*10**-6   # fixed GMsun [kpc km^2/s^2]

M200 = 1e12
rs = 21.5
Rvir = 258

disc_a = 6.5
disc_b = 0.26
Mdisc = 9.3*10**+10

bulge_c = 0.7
Mbulge = 3.0*10**+10

def potential(pos):
    #pos galactic position, kpc, Nx3
    phi_halo  = potential_halo(pos)
    phi_disc  = potential_disc(pos)
    phi_bulge = potential_bulge(pos)
    phi_total = phi_disc + phi_bulge + phi_halo
    return phi_total

def vc(pos):
    '''
    Calculates the potential energy of a star given (x,y,z) coordinates,
    centred on the Sun, via those Aminas FORTRAN functions that
    I've translated here.

    (x,y,z) are assumed to be in kpc
    '''

    # Compute the potentials due to the different components

    # The disc contribution
    vcf_disc = vc_disc(pos)
    # The bulge contribution
    vcf_bulge = vc_bulge(pos)
    # The halo contribution
    vcf_halo = vc_halo(pos)

    # the sum of all these
    vc_total = np.sqrt(vcf_disc**2 + vcf_bulge**2 + vcf_halo**2)

    # Finally, return the total potential
    return vc_total

################################################################################

def potential_halo(pos):
    c = Rvir/rs
    phi_0 = G*M200/Rvir / (np.log(1+c)-c/(1+c))*c
    r = np.linalg.norm(pos,axis=1)
    phi_halo = - phi_0 * rs/r * np.log(1.0 + r/rs)
    return phi_halo



def potential_disc(pos):
    '''
    Calculates the potential due to the disc
    '''
    # parameters
    GMd = G * Mdisc

    x, y, z = pos[:,0], pos[:,1], pos[:,2]

    sqd = np.sqrt(z**2.0 + disc_b**2.0)
    # square root of the density, probably
    sqden1 = np.sqrt(x**2. + y**2.0 + (disc_a+sqd)**2.0)
    # the potential of the disc
    phi_d = -GMd/sqden1

    return phi_d

################################################################################


def potential_bulge(pos):
    '''
    Calculates the potential contribution due to the bulge
    '''
    # parameters
    GMb = G * Mbulge

    # radial distance
    r = np.linalg.norm(pos,axis=1)

    # the potential due to the bulge
    phi_b = -GMb/(r+bulge_c)

    return phi_b

################################################################################




def vc_halo(pos):
    '''
    Calculates the potential contribution coming fromt the halo.
    '''
    c = Rvir/rs

    # halculate the halo potential
    phi_0 = G*M200/Rvir / (np.log(1+c)-c/(1+c))*c
    r = np.linalg.norm(pos,axis=1)

    return np.sqrt(r*(phi_0 * rs/r * (np.log(1 + r/rs)/r - 1/(rs+r))))


def vc_disc(pos):
    '''
    Calculates the potential due to the disc
    '''
    # parameters
    GMd = G * Mdisc

    x,y,z = pos[:,0], pos[:,1], pos[:,2]
    sqd = np.sqrt(z**2.0 + disc_b**2.0)

    # square root of the density, probably
    sqden1 = np.sqrt(x**2. + y**2.0 + (disc_a+sqd)**2.0)

    # the potential of the disc
    vc_d = np.sqrt(x*x+y*y)*(GMd/sqden1**3)

    return vc_d


def vc_bulge(pos):
    # parameters
    GMb = G * Mbulge

    r = np.linalg.norm(pos,axis=1)

    vc_b = np.sqrt(GMb*r)/(r+bulge_c)

    return vc_b
