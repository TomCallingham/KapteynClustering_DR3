# Reddening computation and moving magnitudes from apparent to absolute plane.
# Updated version of reddening_computation to september 2021 (adding errors and 'else')

def reddening_computation(self):

    # Compute ebv from one of the published 3D maps:
    if dust_map == 'l18':
        # We use the dust map from lallement18. It uses dust_maps_3d/lallement18.py and the file lallement18_nside64.fits.gz
        EBV = l18.ebv(self.evaluate('l'), self.evaluate('b'), self.evaluate('distance')*1000.0)
        delta_EBV = EBV*0.0
    elif dust_map == 'p19':
        # We use the dust map from pepita19. It would use dust_maps_3d/pepita19.py and the file pepita19.fits.gz
        EBV = p19.ebv(self.evaluate('l'), self.evaluate('b'), self.evaluate('distance'))
        delta_EBV = EBV*0.0
    elif dust_map == 'bayestar':
        bayestar = BayestarQuery(max_samples=1, version='bayestar2019')
        coords = SkyCoord(self.evaluate('l')*units.deg, self.evaluate('b')*units.deg, distance=self.evaluate('distance')*units.kpc, frame='galactic')
        EBV = bayestar(coords, mode='mean')
        EBV_perc = bayestar(coords, mode='percentile', pct=[16., 84.])
        delta_EBV = EBV_perc[:,1]-EBV_perc[:,0]
        del bayestar, EBV_perc
        # Problems with Green catalogue: 1- Not EBV; 2- zero values.
        # 1: True_EBV = 0.884*EBV, taken from http://argonaut.skymaps.info/usage#units
        EBV, delta_EBV = 0.884*EBV, 0.884*delta_EBV
        # 2: Discretization from Green email, if EBV = 0.0, assume a number in the range 0.0-0.02 (positive I guess, right?)
        coords = np.where(EBV==0.0)
        EBV[coords] = np.random.uniform(0.0, 0.02, len(EBV[coords]))
        coords = np.where(delta_EBV==0.0)
        delta_EBV[coords] = np.random.uniform(0.0, 0.02, len(delta_EBV[coords]))

    # From those E(B-V) compute the AG and E_BP_RP values

    if ext_rec == 'cas18':

        # Colour--Teff relation determined from Gaia DR2 data --> Computed using color_teff_GDR2.ipynb
        poly = np.array([62.55257114, -975.10845442, 4502.86260828, -8808.88745592, 10494.72444183])
        Teff = np.poly1d(poly)(self.evaluate('bp_rp')) / 1e4

        # We enter in an iterative process of 10 steps to refine the AG and E_BP_RP computation. 'Cause we do not have intrinsic colour.
        for i in range(10):
            E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * EBV # A_BP - A_RP
            AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * EBV
            Teff = np.poly1d(poly)(self.evaluate('bp_rp') - E_BP_RP) / 1e4
        delta_AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * delta_EBV
        delta_E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * delta_EBV

    elif ext_rec == 'bab18':
        ### Babusiaux et al. 2018 --> http://aselfabs.harvard.edu/abs/2018A%26A...616A..10G (eq 1, Table 1)

        A0 = 3.1*EBV
        bp_rp_0 = self.evaluate('bp_rp')

        # We enter in an iterative process of 10 steps to refine the kBP, kRP, and kG values. 'Cause we do not have intrinsic colour.
        for i in range(3):
            kG  = 0.9761 - 0.1704*bp_rp_0 + 0.0086*bp_rp_0**2 + 0.0011*bp_rp_0**3 - 0.0438*A0 + 0.00130*A0**2 + 0.0099*bp_rp_0*A0
            kBP = 1.1517 - 0.0871*bp_rp_0 - 0.0333*bp_rp_0**2 + 0.0173*bp_rp_0**3 - 0.0230*A0 + 0.00060*A0**2 + 0.0043*bp_rp_0*A0
            kRP = 0.6104 - 0.0170*bp_rp_0 - 0.0026*bp_rp_0**2 - 0.0017*bp_rp_0**3 - 0.0078*A0 + 0.00005*A0**2 + 0.0006*bp_rp_0*A0
            bp_rp_0 = bp_rp_0 - (kBP-kRP)*A0 # 0 -- intrinsic, otherwise observed, Ax = mx - m0x --> m0x = mx - Ax = mx - kx*A0  ==> m0BP - m0RP = mbp - mrp - (kbp - krp)*A0 = bp_rp

        AG = kG*A0
        delta_A0 = 3.1*delta_EBV
        delta_kG = (-0.0438 + 2.0*0.00130*A0 + 0.0099*bp_rp_0)*delta_A0
        delta_AG = np.sqrt((A0*delta_kG)**2 + (kG*delta_A0)**2)
        del delta_kG

        E_BP_RP = (kBP-kRP)*A0 # A_BP - A_RP
        delta_kBP_kRP = (-0.0152 + 2.0*0.00055*A0 + 0.0037*bp_rp_0)*delta_A0
        delta_E_BP_RP = np.sqrt((A0*delta_kBP_kRP)**2 + ((kBP-kRP)*delta_A0)**2)
        del A0, bp_rp_0, kG, kBP, kRP, delta_kBP_kRP, delta_A0

    elif ext_rec == 'else':
        AG = 0.86117*3.1*EBV
        Abp = 1.06126*3.1*EBV
        Arp = 0.64753*3.1*EBV
        E_BP_RP = Abp - Arp

        delta_AG = 0.86117*3.1*delta_EBV
        delta_BP_RP = np.sqrt((1.06126*3.1*EBV)**2 + (0.64753*3.1*EBV)**2)
    else:
        raise AttributeError

    COL_bp_rp = self.evaluate('bp_rp') - E_BP_RP
    MG = self.evaluate('phot_g_mean_mag')-AG+5.-5.*np.log10(1000.0/self.evaluate('new_parallax'))
    #return E_BP_RP, delta_E_BP_RP, AG, delta_AG, COL_bp_rp, MG
    return E_BP_RP, A_G, COL_bp_rp, M_G
