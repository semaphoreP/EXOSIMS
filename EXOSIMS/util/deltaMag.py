# -*- coding: utf-8 -*-
import os
import numpy as np
import astropy.io.ascii as ascii
import scipy.interpolate
import astropy.units as units
import astropy.table as table


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


def deltaMagThermal(star_mags, Mp, dists, ages, band='H', entropy=13):
    """
    Delta magnitude for thermal emission

    Args:
        star_mags: magnitude of stars in the bands
        Mp: Planet masses in Earth mass
        dists: Distance to planet (pc)
        age: age of the planet in yr
        band: wavelength band (string)
        entropy: initial entropy of the planet in kB/baryon
    """
    if entropy == 13:
        # Read in Hot Start
        # Baraffe et al. 2003
        filepath = os.path.join(os.path.dirname(__file__), "hotstart.txt")
        with open(filepath) as f:
            # skip first 11 lines
            [f.readline() for _ in range(11)]
            
            # start the table
            t = table.Table(None, names=('t', 'M', 'Teff', 'L', 'g', 'R', 'Mv', 'Mr', 'Mi', 'Mj', 'Mh', 'Mk', 'Ml', 'Mm'))
            
            # 10 steps in time (Gyr)
            for i in range(10):
                # skip first 3 lines
                [f.readline() for _ in range(3)]
                
                timeline = f.readline()
                time = timeline.strip().split("=")[1]
                time = float(time) * 1e9 # in years
                
                mass, lum = [], []
                # skip 3 lines of header
                [f.readline() for _ in range(3)]
                line = f.readline()
                while line[0] != '-':
                    args = line.split()

                    args = [float(arg) for arg in args]
                    #args[0] *= units.Msun/units.Mjup
                    t.add_row([time,] + args)
                    line = f.readline()


        hotstart_table = t

        band_index = 'M' + band.lower()

        hotstart = scipy.interpolate.SmoothBivariateSpline(hotstart_table['M'], np.log(hotstart_table['t']), hotstart_table[band_index])
        
        Mp_Mjup = Mp * units.earthMass.to(units.Msun)

        pl_abs_mags = np.array([hotstart(mass, np.log(age)).ravel()[0] for mass, age in zip(Mp_Mjup, ages)])

    else:
        filepath = os.path.join(os.path.dirname(__file__), "coldstart.csv")
        cs_tab = ascii.read(filepath, header_start=0, delimiter=',')
        cs_masses = cs_tab.columns[0]
        cs_ages = np.log(cs_tab['t'])
        cs_abs_mags = cs_tab['M'+band.upper()]

        coldstart = scipy.interpolate.SmoothBivariateSpline(cs_masses, cs_ages, cs_abs_mags)

        Mp_Mjup = Mp * units.earthMass.to(units.Mjup)

        pl_abs_mags = np.array([coldstart(mass, np.log(age)).ravel()[0] for mass, age in zip(Mp_Mjup, ages)])

    pl_mags = pl_abs_mags + 5*np.log10(dists/10)

    pl_delta_mags = pl_mags - star_mags

    return pl_delta_mags

