from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
from EXOSIMS.util.deltaMag import deltaMagThermal
import astropy.units as u
import numpy as np
import astropy.constants as const

class ThermalGPs(SimulatedUniverse):
    """Simulated Universe module based on SAG13 Planet Population module.
    
    """

    def __init__(self, **specs):
        
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        """Generates the planetary systems' physical properties. 
        
        Populates arrays of the orbital elements, albedos, masses and radii 
        of all planets, and generates indices that map from planet to parent star.
        
        """
        
        PPop = self.PlanetPopulation
        TL = self.TargetList
        
        if(type(self.fixedPlanPerStar) == int):#Must be an integer for fixedPlanPerStar
            #Create array of length TL.nStars each w/ value ppStar
            targetSystems = np.ones(TL.nStars).astype(int)*self.fixedPlanPerStar
        else:
            # treat eta as the rate parameter of a Poisson distribution
            targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)

        
        plan2star = []
        for j,n in enumerate(targetSystems):
            plan2star = np.hstack((plan2star, [j]*n))
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        
        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.gen_M0()                           # initial mean anomaly
        self.Mp = PPop.gen_mass(self.nPlans)    # mass
        self.dists = TL.dist[self.plan2star] # pc
        self.delta_mags = TL.Hmag[self.plan2star] # magnitudes!
        
        # The prototype StarCatalog module is made of one single G star at 1pc. 
        # In that case, the SimulatedUniverse prototype generates one Jupiter 
        # at 5 AU to allow for characterization testing.
        # Also generates at least one Jupiter if no planet was generated.
        if TL.Name[0] == 'Prototype' or self.nPlans == 0:
            self.plan2star = np.array([0], dtype=int)
            self.sInds = np.unique(self.plan2star)
            self.nPlans = len(self.plan2star)
            self.a = np.array([5.])*u.AU
            self.e = np.array([0.])
            self.I = np.array([0.])*u.deg # face-on
            self.O = np.array([0.])*u.deg
            self.w = np.array([0.])*u.deg
            self.gen_M0()
            self.Rp = np.array([10.])*u.earthRad
            self.Mp = np.array([300.])*u.earthMass
            self.p = np.array([0.6])

        def set_planet_phase(self, beta = np.pi/2):
            """Positions all planets at input star-planet-observer phase angle
            where possible. For systems where the input phase angle is not achieved,
            planets are positioned at quadrature (phase angle of 90 deg).
            
            The position found here is not unique. The desired phase angle will be
            achieved at two points on the planet's orbit (for non-face on orbits).
            
            Args:
                beta (float):
                    star-planet-observer phase angle in radians.
            
            """
            
            PPMod = self.PlanetPhysicalModel
            ZL = self.ZodiacalLight
            TL = self.TargetList
            
            a = self.a.to('AU').value               # semi-major axis
            e = self.e                              # eccentricity
            I = self.I.to('rad').value              # inclinations
            O = self.O.to('rad').value              # right ascension of the ascending node
            w = self.w.to('rad').value              # argument of perigee
            Mp = self.Mp                            # planet masses
            
            # make list of betas
            betas = beta*np.ones(w.shape)
            mask = np.cos(betas)/np.sin(I) > 1.
            num = len(np.where(mask == True)[0])
            betas[mask] = np.pi/2.
            mask = np.cos(betas)/np.sin(I) < -1.
            num += len(np.where(mask == True)[0])
            betas[mask] = np.pi/2.
            if num > 0:
                print('***Warning***')
                print('{} planets out of {} could not be set to phase angle {} radians.'.format(num,self.nPlans,beta))
                print('These planets are set to quadrature (phase angle pi/2)')
            
            # solve for true anomaly
            nu = np.arcsin(np.cos(betas)/np.sin(I)) - w
            
            # setup for position and velocity
            a1 = np.cos(O)*np.cos(w) - np.sin(O)*np.cos(I)*np.sin(w)
            a2 = np.sin(O)*np.cos(w) + np.cos(O)*np.cos(I)*np.sin(w)
            a3 = np.sin(I)*np.sin(w)
            A = np.vstack((a1, a2, a3))
            
            b1 = -(np.cos(O)*np.sin(w) + np.sin(O)*np.cos(I)*np.cos(w))
            b2 = (-np.sin(O)*np.sin(w) + np.cos(O)*np.cos(I)*np.cos(w))
            b3 = np.sin(I)*np.cos(w)
            B = np.vstack((b1, b2, b3))
            
            r = a*(1.-e**2)/(1.-e*np.cos(nu))
            mu = const.G*(Mp + TL.MsTrue[self.plan2star])
            v1 = -np.sqrt(mu/(self.a*(1.-self.e**2)))*np.sin(nu)
            v2 = np.sqrt(mu/(self.a*(1.-self.e**2)))*(self.e + np.cos(nu))
            
            self.r = (A*r*np.cos(nu) + B*r*np.sin(nu)).T*u.AU           # position
            self.v = (A*v1 + B*v2).T.to('AU/day')                       # velocity
            self.d = np.linalg.norm(self.r, axis=1)*self.r.unit         # planet-star distance
            self.s = np.linalg.norm(self.r[:,0:2], axis=1)*self.r.unit  # apparent separation
            self.phi = PPMod.calc_Phi(np.arccos(self.r[:,2]/self.d))    # planet phase
            self.fEZ = ZL.fEZ(TL.MV[self.plan2star], self.I, self.d)    # exozodi brightness
            
            self.dMag = deltaMagThermal(self.delta_mags, self.Mp, self.dists, 20*np.ones(self.dists.shape))     # delta magnitude

            self.WA = np.arctan(self.s/TL.dist[self.plan2star]).to('arcsec')# working angle