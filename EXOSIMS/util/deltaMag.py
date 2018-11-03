# -*- coding: utf-8 -*-
import numpy as np
import astropy.io.ascii as ascii
import scipy.interpolate
import astropy.units as units

def deltaMag(p,Rp,d,Phi):
    """ Calculates delta magnitudes for a set of planets, based on their albedo, 
    radius, and position with respect to host star.
    
    Args:
        p (ndarray):
            Planet albedo
        Rp (astropy Quantity array):
            Planet radius in units of km
        d (astropy Quantity array):
            Planet-star distance in units of AU
        Phi (ndarray):
            Planet phase function
    
    Returns:
        dMag (ndarray):
            Planet delta magnitudes
    
    """
    dMag = -2.5*np.log10(p*(Rp/d).decompose()**2*Phi).value
    
    return dMag


def deltaMagThermal(star_mags, Mp, dists, ages, band='H', entropy=9.0):
    """
    Delta magnitude for thermal emission

    Args:
        star_mags: magnitude of stars in the bands
        Mp: Planet masses in Earth mass
        dists: Distance to planet (pc)
        age: age of the planet in Myr
        band: wavelength band (string)
        entropy: initial entropy of the planet in kB/baryon
    """
    cs_tab = ascii.read("coldstart.csv", header_start=0, delimiter=',')
    cs_masses = cs_tab.columns[0]
    cs_ages = np.log(cs_tab['t'])
    cs_abs_mags = cs_tab['M'+band.upper()]

    coldstart = scipy.interpolate.SmoothBivariateSpline(cs_masses, cs_ages, cs_abs_mags)

    Mp_Mjup = Mp * units.earthMass.to(units.Mjup)

    pl_abs_mags = np.array([coldstart(mass, age) for mass, age in zip(Mp_Mjup, ages)])

    pl_mags = pl_abs_mags + 5*np.log10(dists/10)

    pl_delta_mags = pl_mags - star_mags

    return pl_delta_mags

