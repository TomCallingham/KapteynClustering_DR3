import os
import warnings
from astropy.io import fits
import numpy as np
import healpy


class L18Map():
    """
    Returns E(B-V) at Galactic coordinates l, b and distance d, from the
        Lallement et al. 2018 dust maps. It uses Healpix Nside=64 so has an
        angular resolution of ~0.9 deg, and a distance resolution of 20 pc.
    """
    def __init__(self):
        folder = "/net/gaia2/data/users/ruizlara/useful_codes/"
        fname = 'lallement18_nside64.fits.gz'
        # path = os.path.join(os.path.dirname(__file__), fname)
        path = folder+fname
        hdu = fits.open(path)
        self.maps = np.vstack([np.zeros(49152),
                               [hdu[i+1].data['EBV'] for i in range(100)]])

    def __call__(self, l, b, d):
        l = np.asarray(l)
        b = np.asarray(b)
        d = np.asarray(d)

        if (l.shape != b.shape):
            raise ValueError('The shape of l and b must be the same.')

        if (l.shape != d.shape) & (l.size != 1):
            raise ValueError('The shape of l, b and d must be the same, '
                             'or l and b be scalars.')
        elif (l.shape != d.shape) & (l.size == 1):
            l = np.repeat(l, d.size)
            b = np.repeat(b, d.size)

        # Location on the sky
        px = healpy.ang2pix(64, (90.-b)*np.pi/180., l*np.pi/180.)

        # Distance slice
        sl = np.round(d/20).astype(int)
        too_distant = (d > 2000)
        sl[too_distant] = 99
        ebv = self.maps[sl, px]
        return ebv

# ebv = L18Map()
