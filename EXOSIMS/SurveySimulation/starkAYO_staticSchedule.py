from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import scipy
#from scipy.optimize import fmin
import timeit
import csv
import os.path
#import datetime
import hashlib
import inspect
from astropy.coordinates import SkyCoord
try:
    import cPickle as pickle
except:
    import pickle
import csv
from pylab import *
from numpy import nan
from scipy.optimize import curve_fit
from scipy.special import factorial

class starkAYO_staticSchedule(SurveySimulation):
    """starkAYO _static Scheduler
    
    This class implements a Scheduler that creates a list of stars to observe and integration times to observe them. It also selects the best star to observe at any moment in time
    Generates cachedfZ.csv and cachedMaxCbyTtime.csv
    If the above exist but some problem is found, moved_MacCbyTtime.csv and moved_fZAllStars.csv are created

    2nd execution time 2 min 30 sec
    """
    def __init__(self, cacheOptTimes=False, **specs):
        SurveySimulation.__init__(self, **specs)

        assert isinstance(cacheOptTimes, bool), 'cacheOptTimes must be boolean.'
        self._outspec['cacheOptTimes'] = cacheOptTimes

        #Load cached Observation Times
        self.starkt0 = None
        if cacheOptTimes:#Checks if flag exists
            #Generate cache Name########################################################################
            cachefname = self.cachefname + 'starkt0'
            if os.path.isfile(cachefname):#check if file exists
                self.vprint("Loading cached t0 from %s"%cachefname)
                with open(cachefname, 'rb') as f:#load from cache
                    self.starkt0 = pickle.load(f)
                sInds = np.arange(self.TargetList.nStars)

        # bring inherited class objects to top level of Survey Simulation
        SU = self.SimulatedUniverse
        OS = SU.OpticalSystem
        ZL = SU.ZodiacalLight
        self.Completeness = SU.Completeness
        TL = SU.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        
        self.starVisits = np.zeros(TL.nStars,dtype=int)
        self.starRevisit = np.array([])

        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        self.mode = detMode
        
        #Create and start Schedule
        self.schedule = np.arange(TL.nStars)#self.schedule is meant to be editable
        self.schedule_startSaved = np.arange(TL.nStars)#preserves initial list of targets
          
        dMagLim = self.Completeness.dMagLim
        self.dmag_startSaved = np.linspace(1, dMagLim, num=1500,endpoint=True)

        sInds = self.schedule_startSaved
        dmag = self.dmag_startSaved
        WA = OS.WA0
        startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

        tovisit = np.zeros(sInds.shape[0], dtype=bool)

        #Generate fZ
        self.fZ_startSaved = self.generate_fZ(sInds)#

        #Estimate Yearly fZmin###########################################
        fZmin, fZminInds = self.calcfZmin(sInds,self.fZ_startSaved)

        #Estimate Yearly fZmax###########################################
        fZmax, fZmaxInds = self.calcfZmax(Obs,TL,TK,sInds,self.mode,self.fZ_startSaved)

        #CACHE Cb Cp Csp################################################Sept 20, 2017 execution time 10.108 sec
        fZ = fZmin/u.arcsec**2#
        fEZ = ZL.fEZ0
        mode = self.mode#resolve this mode is passed into next_target
        allModes = self.OpticalSystem.observingModes
        det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        Cp = np.zeros([sInds.shape[0],dmag.shape[0]])
        Cb = np.zeros(sInds.shape[0])
        Csp = np.zeros(sInds.shape[0])
        for i in xrange(dmag.shape[0]):
            Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dmag[i], WA, det_mode)
        self.Cb = Cb[:]/u.s#Cb[:,0]/u.s#note all Cb are the same for different dmags. They are just star dependent
        self.Csp = Csp[:]/u.s#Csp[:,0]/u.s#note all Csp are the same for different dmags. They are just star dependent
        #self.Cp = Cp[:,:] #This one is dependent upon dmag and each star
        ################################################################

        # #DELETE THIS
        # #Calculate Integration Times at fZmin##########################
        # intTimes = np.logspace(-5,5,num=2000,base=10.0)

        # Comp00 = np.zeros([intTimes.shape[0],TL.nStars])
        # for i in xrange(dmag.shape[0]):
        #     Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dmag[i], WA, det_mode)
        # self.Cb = Cb[:]/u.s#Cb[:,0]/u.s#note all Cb are the same for different dmags. They are just star dependent
        # self.Csp = Csp[:]/u.s#
        # for i in np.arange(intTimes.shape[0]):
        #     Comp00[i,:] = self.Completeness.comp_per_intTime(intTimes[i]*u.d, TL, sInds, fZ, fEZ, WA, mode, self.Cb, self.Csp)
        #     print(np.double(i)/np.double(intTimes.shape[0]))

        # with open('/home/dean/Documents/exosims/EXOSIMS/Scripts/CvsTfZmin_20', "wb") as fo:
        #     wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
        #     for i in range(sInds.shape[0]):#iterate through all stars
        #         wr.writerow(Comp00[:,i])#Write the fZ to file
        #     fo.close()
        #     print('Finished Saving CvsTfZmin*2 for Each Star to File')

        #Calculate Initial Integration Times###########################################
        maxCbyTtime = self.calcTinit(sInds, TL, fZ, fEZ, WA, mode, self.Cb, self.Csp)
        t_dets = maxCbyTtime[sInds]


        #Sacrifice Stars and then Distribute Excess Mission Time################################################Sept 28, 2017 execution time 19.0 sec
        missionLength = (TK.missionLife.to(u.d)*TK.missionPortion).value*12/12#TK.missionLife.to(u.day).value#mission length in days
        overheadTime = self.Observatory.settlingTime.value + self.OpticalSystem.observingModes[0]['syst']['ohTime'].value#OH time in days
        while((sum(t_dets) + sInds.shape[0]*overheadTime) > missionLength):#the sum of star observation times is still larger than the mission length
            sInds, t_dets, sacrificedStarTime, fZ = self.sacrificeStarCbyT(sInds, t_dets, fZ, fEZ, WA, overheadTime)

        if(sum(t_dets + sInds.shape[0]*overheadTime) > missionLength):#There is some excess time
            sacrificedStarTime = missionLength - (sum(t_dets) + sInds.shape[0]*overheadTime)#The amount of time the list is under the total mission Time
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA)
        ###############################################################################

        #STARK AYO LOOP################################################################
        savedSumComp00 = np.zeros(sInds.shape[0])
        firstIteration = 1#checks if this is the first iteration.
        numits = 0#ensure an infinite loop does not occur. Should be depricated
        lastIterationSumComp  = -10000000 #this is some ludacrisly negative number to ensure sumcomp runs. All sumcomps should be positive
        while numits < 100000 and sInds is not None:
            numits = numits+1#we increment numits each loop iteration

            #Sacrifice Lowest Performing Star################################################Sept 28, 2017 execution time 0.0744 0.032 at smallest list size
            sInds, t_dets, sacrificedStarTime, fZ = self.sacrificeStarCbyT(sInds, t_dets, fZ, fEZ, WA, overheadTime)

            #Distribute Sacrificed Time to new star observations############################# Sept 28, 2017 execution time 0.715, 0.64 at smallest (depends on # stars)
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA)

            #AYO Termination Conditions###############################Sept 28, 2017 execution time 0.033 sec
            Comp00 = self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode, self.Cb, self.Csp)

            #change this to an assert
            if 1 >= len(sInds):#if this is the last ement in the list
                break
            savedSumComp00[numits-1] = sum(Comp00)
            #If the total sum of completeness at this moment is less than the last sum, then exit
            if(sum(Comp00) < lastIterationSumComp):#If sacrificing the additional target reduced performance, then Define Output of AYO Process
                CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)/t_dets#takes 5 seconds to do 1 time for all stars
                sortIndex = np.argsort(CbyT,axis=-1)[::-1]

                #This is the static optimal schedule
                self.schedule = sInds[sortIndex]
                self.t_dets = t_dets[sortIndex]
                self.CbyT = CbyT[sortIndex]
                self.fZ = fZ[sortIndex]
                self.Comp00 = Comp00[sortIndex]
                self.Cb = self.Cb[sortIndex]
                self.Csp = self.Csp[sortIndex]
                break
            else:#else set lastIterationSumComp to current sum Comp00
                lastIterationSumComp = sum(Comp00)
                self.vprint(str(numits) + ' SumComp ' + str(sum(Comp00)) + ' Sum(t_dets) ' + str(sum(t_dets)) + ' sInds ' + str(sInds.shape[0]*float(1)) + ' TimeConservation ' + str(sum(t_dets)+sInds.shape[0]*float(1)))# + ' Avg C/T ' + str(np.average(CbyT)))
        #End While Loop

        #Plot magfZ vs Time for all stars over 1 year#####################################
        dt = 365.25/len(np.arange(1000))
        time = [j*dt for j in range(1000)]#Time since mission start
        #fZ = np.zeros([sInds.shape[0], len(resolution)])
        #dt = 365.25/len(resolution)*u.d
        #time = 365.25/resolution
        tmpfZ = np.zeros(len(self.schedule))
        tmpfZ = self.fZ_startSaved[self.schedule]
        magfZ = -2.5*np.log10(tmpfZ)
        
        fig = plt.figure(9000)
        for i in np.arange(len(self.schedule)):
            plt.plot(time,-2.5*np.log10(tmpfZ[i][:]))
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.ylabel('Zodiacal Light in magfZ',weight='bold')
        plt.xlabel('Time of Year (day)',weight='bold')
        plt.title('magfZ for all stars in starkAYO Schedule',weight='bold')
        #plt.legend(loc='lower right')
        plt.show(block=False)
        fig.savefig('/home/dean/Documents/SIOSlab/figmagfZAllStarsinStarkAYOSchedule'+'.svg')
        plt.close()
        ###########################################################

        #Plot magfZ vs Time for all stars including keepout angles (note that NAN does not plot)#######
        # indices of observable stars
        #i=0
        #kogoodStart = Obs.keepout(TL, sInds, TK.currentTimeAbs+time[i]*u.d, mode)
        #print(saltyburrito)
        kogoodStart = np.zeros([len(time),self.schedule.shape[0]])
        for i in np.arange(len(time)):
            kogoodStart[i,:] = Obs.keepout(TL, self.schedule, TK.currentTimeAbs+time[i]*u.d, mode)
            kogoodStart[i,:] = (np.zeros(kogoodStart[i,:].shape[0])+1)*kogoodStart[i,:]
        kogoodStart[kogoodStart==0] = nan

        figfZKO = plt.figure(9001)
        magfZ2 = np.zeros([len(self.schedule),len(time)])
        minmagfZ2 = np.zeros(len(self.schedule))
        maxmagfZ2 = np.zeros(len(self.schedule))
        for i in np.arange(len(self.schedule)):
            magfZ2[i,:] = magfZ[i,:]*kogoodStart[:,i]
            minmagfZ2[i] = min(magfZ2[i,magfZ2[i,:] > 0])#calculate minimum value for all stars
            maxmagfZ2[i] = max(magfZ2[i,magfZ2[i,:] > 0])#calculate maximum value for all stars
            plt.plot(time,magfZ2[i,:])
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.ylabel('fZ in magfZ',weight='bold')
        plt.xlabel('Time (days)',weight='bold')
        plt.title('magfZ for all stars in starkAYO Schedule without KO',weight='bold')
        #plt.legend(loc='lower right')
        plt.show(block=False)
        figfZKO.savefig('/home/dean/Documents/SIOSlab/magfZAllStarsStarkAYOWithoutKO'+'.svg')
        plt.close()
        ############################################################
        #Plot 4 of the keepout angles
        figfZKO_4 = plt.figure(9002)
        magfZ2_4 = np.zeros([len(self.schedule),len(time)])
        for i in [0,1,2,3]:
            magfZ2_4[i,:] = magfZ[i,:]*kogoodStart[:,i]
            plt.plot(time,magfZ2_4[i,:])
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.xlim([0,365.25])
        plt.ylabel('fZ in magfZ',weight='bold')
        plt.xlabel('Time (days)',weight='bold')
        plt.title('magfZ for '+str(self.schedule[0])+' '+str(self.schedule[1])+' '+str(self.schedule[2])+' '+str(self.schedule[3])+' without KO',weight='bold')
        #plt.legend(loc='lower right')
        plt.show(block=False)
        figfZKO_4.savefig('/home/dean/Documents/SIOSlab/magfZfor'+str(self.schedule[0])+str(self.schedule[1])+str(self.schedule[2])+str(self.schedule[3])+'withoutKO'+'.svg')
        plt.close()
        ############################################################

        #Plot Histogram of Minimum Values minmagfZ2#####################
        figfZminHist = plt.figure(9003)
        out = plt.hist(minmagfZ2,label=r'$magfZ_{min}$',color='b')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of '+r'$magfZ_{min}$',weight='bold')
        plt.xlabel(r'$magfZ$',weight='bold')
        plt.ylabel('# of Targets',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.legend()
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        #plt.title('magfZ_{min} for all stars in starkAYO Schedule without KO',weight='bold')
        plt.show(block=False)
        figfZminHist.savefig('/home/dean/Documents/SIOSlab/figfZminHist'+'.svg')
        plt.close()
        ###########################################################

        #Plot Histogram of Maximum Values maxmagfZ2#####################
        fig4 = plt.figure(9004)
        out = plt.hist(maxmagfZ2,label=r'$magfZ_{max}$',color='r')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of '+r'$magfZ_{max}$',weight='bold')
        plt.xlabel('magfZ',weight='bold')
        plt.ylabel('# of Targets',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        plt.legend()
        plt.show(block=False)
        fig4.savefig('/home/dean/Documents/SIOSlab/figfZmaxHist'+'.svg')
        plt.close()
        ###########################################################

        #Plot Histogram of Minimum and Maximum Values Together############
        figfZminmaxHist = plt.figure(9005)
        out = plt.hist(minmagfZ2,label=r'$magfZ_{min}$',color='b',alpha=0.5,bins=np.arange(min(minmagfZ2), max(minmagfZ2) + 0.1, 0.1))
        out = plt.hist(maxmagfZ2,label=r'$magfZ_{max}$',color='r',alpha=0.5,bins=np.arange(min(maxmagfZ2), max(maxmagfZ2) + 0.1, 0.1))
        magfZ0 = -2.5*np.log10(self.ZodiacalLight.fZ0.value)
        out = plt.plot([magfZ0,magfZ0],[0,25],color='k',label=r'$magfZ0$')
        out = plt.plot([mean(minmagfZ2),mean(minmagfZ2)],[0,25],color='b',label=r'$mean(magfZ_{min})$',linestyle='--')
        out = plt.plot([mean(maxmagfZ2),mean(maxmagfZ2)],[0,25],color='r',label=r'$mean(magfZ_{max})$',linestyle='--')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of '+r'$magfZ_{min}$'+' and '+r'$magfZ_{max}$',weight='bold',fontsize=12)
        plt.xlabel('magfZ',weight='bold',fontsize=12)
        plt.ylabel('# of Targets',weight='bold',fontsize=12)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        plt.legend()
        plt.show(block=False)
        #red_patch = matplotlib.mpatches.Patch(color='red', label=r'$magfZ){max}$')
        #plt.legend(handles=[red_patch])
        figfZminmaxHist.savefig('/home/dean/Documents/SIOSlab/figfZminmaxHist'+'.svg')
        plt.close()
        ##################################################################

        #Find and Plot fZmax-fZmin for each star########################################
        maxminmagfZ2 = maxmagfZ2-minmagfZ2
        figfZdiff = plt.figure(9006)
        out = plt.hist(maxminmagfZ2,color='k')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of '+r'$magfZ_{max} - magfZ_{min}$',weight='bold')
        plt.xlabel('magfZ',weight='bold')
        plt.ylabel('# of Targets',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        plt.show(block=False)
        figfZdiff.savefig('/home/dean/Documents/SIOSlab/figfZdiff'+'.svg')
        plt.close()
        ##################################################################

        #Plot minfZ and maxfZ
        maxmagfZ2 = 10**(-maxmagfZ2/(2.5))#This is where Tau would be minimized
        minmagfZ2 = 10**(-minmagfZ2/(2.5))#Where Tau would be maximized
        #Plot Comp vs Tint for a few stars at fZmin and fZmax
        ptminfZComp00 = self.Completeness.comp_per_intTime(self.t_dets*u.d, TL, self.schedule, maxmagfZ2/u.arcsec**2, fEZ, WA, mode)#, CbfZmin4, CspfZmin4)
        ptmaxfZComp00 = self.Completeness.comp_per_intTime(self.t_dets*u.d, TL, self.schedule, minmagfZ2/u.arcsec**2, fEZ, WA, mode)#, CbfZmax4, CspfZmax4)
        print('SumComp at fZmin '+str(sum(ptminfZComp00)))
        print('SumComp at fZmax '+str(sum(ptmaxfZComp00)))

        intTimes = np.logspace(-5,5,num=500,base=10.0)
        minfZComp00 = np.zeros([len(intTimes),self.schedule.shape[0]])
        maxfZComp00 = np.zeros([len(intTimes),self.schedule.shape[0]])
        for i in np.arange(len(intTimes)):
            minfZComp00[i,:] = self.Completeness.comp_per_intTime(intTimes[i]*u.d, TL, self.schedule, maxmagfZ2/u.arcsec**2, fEZ, WA, mode)#, CbfZmin4, CspfZmin4)
            maxfZComp00[i,:] = self.Completeness.comp_per_intTime(intTimes[i]*u.d, TL, self.schedule, minmagfZ2/u.arcsec**2, fEZ, WA, mode)#, CbfZmax4, CspfZmax4)

        for j in np.arange(30):
            fig7 = plt.figure(9007)
            for i in np.arange(5)+j*5:
                plt.plot(intTimes,minfZComp00[:,i],color='r',label='minfZ ')
                plt.plot(intTimes,maxfZComp00[:,i],color='b',label='maxfZ ',linestyle='--')
                plt.plot(self.t_dets[i],ptminfZComp00[i],color='k',label='minfZ pt '+str(i),marker='o')
                plt.plot(self.t_dets[i],ptmaxfZComp00[i],color='k',label='maxfZ pt '+str(i),marker='x')
            
            # plt.plot(intTimes,minfZComp00[:,10],color='r',label='minfZ ') 
            # plt.plot(intTimes,maxfZComp00[:,10],color='b',label='maxfZ ',linestyle='--')
            # plt.plot(tmpt_dets[10],ptminfZComp00[10],color='b',label='minfZ pt '+str(10),marker='x')
            # plt.plot(intTimes,minfZComp00[:,20],color='r',label='minfZ ')
            # plt.plot(intTimes,maxfZComp00[:,20],color='b',label='maxfZ ',linestyle='--')
            # plt.plot(tmpt_dets[20],ptminfZComp00[20],color='b',label='minfZ pt '+str(20),marker='x')
            # plt.plot(intTimes,minfZComp00[:,30],color='r',label='minfZ ')
            # plt.plot(intTimes,maxfZComp00[:,30],color='b',label='maxfZ ',linestyle='--')
            # plt.plot(tmpt_dets[30],ptminfZComp00[30],color='b',label='minfZ pt '+str(30),marker='x')
            
            #plt.xscale('log')
            xmin = 0
            xmax = max(self.t_dets*1.1)
            plt.xlim((xmin,xmax))
            plt.xlabel('Integration Time (days)',weight='bold')
            plt.ylabel('Completeness',weight='bold')
            plt.show(block=False)
            fig7.savefig('/home/dean/Documents/SIOSlab/CvsTmaxandminfZlinearforsInds'+str(j)+'.svg')
        plt.close() 

        #Linear CvsT Superplot#####################################################
        CvsTSuperPlot = plt.figure(9008)
        for i in np.arange(self.schedule.shape[0]):
            plt.plot(intTimes,minfZComp00[:,i],color='r',label='minfZ ')
            plt.plot(intTimes,maxfZComp00[:,i],color='b',label='maxfZ ',linestyle='--')
            plt.plot(self.t_dets[i],ptminfZComp00[i],color='k',label='minfZ pt '+str(i),marker='o')
            plt.plot(self.t_dets[i],ptmaxfZComp00[i],color='k',label='maxfZ pt '+str(i),marker='x')
        plt.xlim((xmin,xmax))
        plt.xlabel(r'$Integration\ Time\ \tau\ (days)$')
        plt.ylabel('Completeness')
        plt.show(block=False)
        CvsTSuperPlot.savefig('/home/dean/Documents/SIOSlab/CvsTlinearSuperplot'+'.svg')
        plt.close()
        ###########################################################################

        #Linear Comp Vs T Points Superplot#########################################
        sAYOCTSuperPlot = plt.figure(9009)
        for i in np.arange(self.schedule.shape[0]):
            #plt.plot(intTimes,minfZComp00[:,i],color='r',label='minfZ ')
            #plt.plot(intTimes,maxfZComp00[:,i],color='b',label='maxfZ ',linestyle='--')
            plt.plot(self.t_dets[i],ptminfZComp00[i],color='b',label='minfZ pt '+str(i),marker='o')
            plt.plot(self.t_dets[i],ptmaxfZComp00[i],color='r',label='maxfZ pt '+str(i),marker='x')
        plt.xlim((xmin,xmax))
        plt.xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold')
        plt.ylabel('Completeness',weight='bold')
        plt.show(block=False)
        sAYOCTSuperPlot.savefig('/home/dean/Documents/SIOSlab/sAYOCTSuperPlot'+'.svg')
        #plt.close()
        ###########################################################################

        #Linear Comp Vs T Points Superplot COLORED fZmin#########################################
        sAYOCTSuperPlotCOLOR = plt.figure(9010)
        #for i in np.arange(self.schedule.shape[0]):
            #plt.plot(self.t_dets[i],ptminfZComp00[i],color=ptminfZComp00[i]/self.t_dets[i],label='minfZ pt '+str(i),marker='o')
        dcbydtATt = np.zeros(self.schedule.shape[0])
        for i in np.arange(self.schedule.shape[0]):
            dcbydtATt[i] = self.Completeness.dcomp_dt(self.t_dets[i]*u.d, self.TargetList, self.schedule[i], self.fZ[i], fEZ, WA, self.mode, self.Cb[i], self.Csp[i]).to(1/u.d).value
        maxdcbydtATt = max(dcbydtATt)
        mindcbydtATt = min(dcbydtATt)
        mV = TL.starMag(self.schedule_startSaved,self.mode['lam'])
        maxmV = max(mV)
        minmV = min(mV)
        cmap = plt.cm.get_cmap('winter')
        for i in np.arange(self.schedule.shape[0]):
            Fraction = (mV[self.schedule[i]]-minmV)/(maxmV-minmV)
            rgba = cmap(Fraction)
            r=rgba[0]
            g=rgba[1]
            b=rgba[2]
            a=1
            assert r<=1
            assert b<=1
            plt.scatter(self.t_dets[i],ptminfZComp00[i],marker='o',c=(r,g,b,a))#np.log10(ptminfZComp00[:]/self.t_dets[:]))
            #plt.plot(self.t_dets[i],ptmaxfZComp00[i],color='r',label='maxfZ pt '+str(i),marker='x')
        #Plot stars not being observed
        for i in np.arange(self.schedule_startSaved.shape[0]):
            if(self.schedule_startSaved[i] in np.setdiff1d(self.schedule_startSaved,self.schedule,assume_unique=True)):
                Fraction = (mV[i]-minmV)/(maxmV-minmV)
                rgba = cmap(Fraction)
                r=rgba[0]
                g=rgba[1]
                b=rgba[2]
                a=1
                assert r<=1
                assert b<=1
                plt.scatter(self.myt0plotting[i],self.compatt0[i],color=(r,g,b,a),marker='x')
        plt.xlim((xmin,xmax))
        plt.ylim((0,max(ptminfZComp00)*1.1))
        plt.xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold',fontsize=14)
        plt.ylabel('Completeness',weight='bold',fontsize=14)
        plt.scatter([nan,nan],[nan,nan],color='b',marker='o',label='Observed')
        plt.scatter([nan,nan],[nan,nan],color='b',marker='x',label='Not Observed')
        plt.legend(loc=1)
        cmap = plt.cm.get_cmap('winter')
        sc = plt.scatter([nan,nan],[nan,nan],c=[minmV,maxmV],cmap=cmap)
        cbar = plt.colorbar(sc)
        lar = np.round([minmV,minmV+0.2*(maxmV-minmV),minmV+0.4*(maxmV-minmV),minmV+0.6*(maxmV-minmV),minmV+0.8*(maxmV-minmV),maxmV],decimals=4)
        cbar.ax.set_yticklabels([str(lar[0]),str(lar[1]),str(lar[2]),str(lar[3]),str(lar[4]),str(lar[5])])
        cbar.set_label('Apparent Intensity ()',weight='bold',fontsize=14)#removed ax.
        plt.show(block=False)
        sAYOCTSuperPlotCOLOR.savefig('/home/dean/Documents/SIOSlab/sAYOCTSuperPlotCOLOR'+'.svg')
        #plt.close()
        ###########################################################################

        #Plot C/Tau vs mV###################################
        #mVparamFig = plt.figure(90105)
        #plt.close()
        ####################################################

        #Histogram of Completeness#################################################
        CHistminfZ = plt.figure(9011)
        out = plt.hist(ptminfZComp00,label='Completeness',color='b')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of Completeness at '+r'$magfZ_{max}$',weight='bold')
        plt.xlabel('Completeness',weight='bold')
        plt.ylabel('# of Targets',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        #plt.title('magfZ_{min} for all stars in starkAYO Schedule without KO',weight='bold')
        plt.show(block=False)
        CHistminfZ.savefig('/home/dean/Documents/SIOSlab/CHistminfZ'+'.svg')
        plt.close()
        ###########################################################################

        #Histogram of t_dets#######################################################
        CHistmaxfZ = plt.figure(9012)
        out = plt.hist(ptmaxfZComp00,label='Completeness',color='b')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of Completeness at '+r'$magfZ_{min}$',weight='bold')
        plt.xlabel('Completeness',weight='bold')
        plt.ylabel('# of Targets',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        #plt.title('magfZ_{min} for all stars in starkAYO Schedule without KO',weight='bold')
        plt.show(block=False)
        CHistmaxfZ.savefig('/home/dean/Documents/SIOSlab/CHistmaxfZ'+'.svg')
        plt.close()
        ##########################################################################

        #magfZmax occurance Histogram TOY#########################################
        fZmin_occurance = fZminInds[self.schedule]*365.25/1000
        fZmin_occurancefig = plt.figure(9013)
        out = plt.hist(fZmin_occurance,label='Completeness',color='b')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Histogram of Min fZ Occurance',weight='bold')
        plt.xlabel('Time of Year (days)',weight='bold')
        plt.ylabel('# of Targets',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #plt.rc('axes',prop_cycle=(cycler('color',['red','blue','black','purple'])))
        #plt.title('magfZ_{min} for all stars in starkAYO Schedule without KO',weight='bold')
        plt.show(block=False)
        fZmin_occurancefig.savefig('/home/dean/Documents/SIOSlab/fZmin_occurancefig'+'.svg')
        plt.close()
        ##########################################################################

        #Plot dcbydt for all stars###################################
        fig = plt.figure(551)
        #Calculate dcompdt###############
        t0 = self.myt0plotting
        dcompdt = np.zeros([self.schedule_startSaved.shape[0]])
        for i in np.arange(self.schedule_startSaved.shape[0]):
            dcompdt[i] = self.Completeness.dcomp_dt(t0[i]*u.d, TL, self.schedule_startSaved[i], fZmin[i]*(1/u.arcsec**2), fEZ, WA, mode, Cb[i]/u.s, Csp[i]/u.s).to(1/u.d).value
        intTimes = np.logspace(-5,5,num=400,base=10.0)
        
        #Calculate and include t_dets of stars used
        dcbydtATt = np.zeros(self.schedule.shape[0])
        for i in np.arange(self.schedule.shape[0]):#np.arange(self.schedule.shape[0]/10-self.schedule.shape[0]%10):
            dcbydtATt[i] = self.Completeness.dcomp_dt(self.t_dets[i]*u.d, self.TargetList, self.schedule[i], self.fZ[i], fEZ, WA, self.mode, self.Cb[i], self.Csp[i]).to(1/u.d).value
            plt.plot(self.t_dets[i],dcbydtATt[i],color='c',marker='o',zorder=4)

        
        for i in np.arange((self.schedule_startSaved.shape[0]-(self.schedule_startSaved.shape[0]%10))/10):#Plot 10% of dcbydt lines
            plt.plot(intTimes,self.dcbydt[:,i],color=(1,0,0,0.5),zorder=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(t0,dcompdt,color='k',zorder=3)
        plt.scatter(self.maxdcbydttimes,self.maxdcbydt,color='b',zorder=2)
        xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold',fontsize=12)
        ylabel(r'$\frac{dC}{d\tau}$',weight='bold',fontsize=12)
        plt.plot([0,0],[0,0],color='r',label=r'$\frac{dC}{d\tau}\ (d^{-1})$')
        plt.scatter([1e-8,1e-8],[1e-8,1e-8],color='b',marker='o',label=r'$max(\frac{dC}{d\tau})$')
        plt.scatter([1e-8,1e-8],[1e-8,1e-8],color='k',marker='o',label=r'$\frac{dC}{d\tau}(t0)$')
        plt.scatter([1e-8,1e-8],[1e-8,1e-8],color='g',marker='o',label=r'$\frac{dC}{d\tau}(observed)$')
        xmin = min(self.maxdcbydttimes)
        xmax = 30#max(t0)#We wont observe for longer than a month
        ymin = min(self.maxdcbydt)#min(dcompdt)
        ymax = max(self.maxdcbydt)#max(dcompdt)
        plt.xlim([xmin,xmax])
        plt.ylim([0.1*ymin,ymax*5])
        plt.legend(loc=1,prop={'size': 10})
        plt.show(block=False)
        fig.savefig('/home/dean/Documents/SIOSlab/dCbydTandOptimalSelection2'+'.svg')
        #############################################

        print(saltyburrito)

        #Plot dc/dt for each star
        intTimes = np.logspace(-5,5,num=400,base=10.0)
        dcdtFig = plt.figure(9014)
        dcbydt = self.dcbydt#np.zeros([intTimes.shape[0],self.schedule.shape[0]])
        dcbydtATt = np.zeros(self.schedule.shape[0])
        for i in np.arange(self.schedule.shape[0]):
            #for j in np.arange(intTimes.shape[0]):
            #    dcbydt[j,i] = self.Completeness.dcomp_dt(intTimes[j]*u.d, self.TargetList, self.schedule[i], self.fZ[i], fEZ, WA, self.mode, self.Cb[i], self.Csp[i]).to(1/u.d).value
            plt.plot(intTimes,dcbydt[:,i],color='r',label=r'$\frac{dC}{d\tau}$')

        #now plot points that were used.
        for i in np.arange(self.schedule.shape[0]):#np.arange(self.schedule.shape[0]/10-self.schedule.shape[0]%10):
            dcbydtATt[i] = self.Completeness.dcomp_dt(self.t_dets[i]*u.d, self.TargetList, self.schedule[i], self.fZ[i], fEZ, WA, self.mode, self.Cb[i], self.Csp[i]).to(1/u.d).value
            plt.plot(self.t_dets[i],dcbydtATt[i],color='k',label=r'$\frac{dC}{dt}\ at\ t_dets$'+str(i),marker='o')
            #plt.plot(self.t_dets[i],ptmaxfZComp00[i],color='k',label='maxfZ pt '+str(i),marker='x')
        xmin = 0
        xmax = max(self.t_dets*1.1)
        plt.xlim((xmin,xmax))
        ymin = 0
        ymax = max(dcbydtATt*1.1)
        plt.ylim((ymin,ymax))
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('dC by dT vs Integration Time',weight='bold')
        plt.xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold')
        plt.ylabel(r'$\frac{dC}{d\tau}\ (days^{-1})$',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.show(block=False)


        print(saltyburrito)
        #END INIT##################################################################
        
    def choose_next_target(self,old_sInd,sInds,slewTime,intTimes):
        """Generate Next Target to Select based off of AYO at this instant in time
        Args:
            sInds - indicies of stars under consideration
            old_sInd - unused
            slewTime - unused
            intTimes - unused

        Returns:
            DRM - A blank structure
            sInd - the single index of self.schedule_startSaved to observe
            t_det - the time to observe sInd in days (u.d)
        """
        SU = self.SimulatedUniverse
        OS = SU.OpticalSystem
        ZL = SU.ZodiacalLight
        self.Completeness = SU.Completeness
        TL = SU.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        mode = self.mode
        # now, start to look for available targets
        cnt = 0
        while not TK.mission_is_over():
            TK.obsStart = TK.currentTimeNorm.to('day')

            dmag = self.dmag_startSaved
            WA = OS.WA0
            startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

            tovisit = np.zeros(self.schedule_startSaved.shape[0], dtype=bool)
            fZtovisit = np.zeros(self.schedule_startSaved.shape[0], dtype=bool)

            DRM = {}#Create DRM

            startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

            #Estimate Yearly fZmin###########################################
            tmpfZ = np.asarray(self.fZ_startSaved)
            fZ_matrix = tmpfZ[self.schedule,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
            #Find minimum fZ of each star
            fZmintmp = np.zeros(self.schedule.shape[0])
            for i in xrange(self.schedule.shape[0]):
                fZmintmp[i] = min(fZ_matrix[i,:])

            #Find current fZ
            indexFrac = np.interp((self.TimeKeeping.currentTimeAbs-self.TimeKeeping.missionStart).value%365.25,[0,365.25],[0,1000])#This is only good for 1 year missions right now
            fZinterp = np.zeros(self.schedule.shape[0])
            fZinterp[:] = (indexFrac%1)*fZ_matrix[:,int(indexFrac)] + (1-indexFrac%1)*fZ_matrix[:,int(indexFrac%1+1)]#this is the current fZ

            commonsInds = [x for x in self.schedule if x in sInds]#finds indicies in common between sInds and self.schedule
            imat = [self.schedule.tolist().index(x) for x in commonsInds]
            CbyT = self.CbyT[imat]
            t_dets = self.t_dets[imat]
            Comp00 = self.Comp00[imat]
            fZ = fZinterp[imat]
            fZmin = fZmintmp[imat]

            commonsInds2 = [x for x in self.schedule_startSaved if((x in sInds) and (x in self.schedule))]#finds indicies in common between sInds and self.schedule
            imat2 = [self.schedule_startSaved.tolist().index(x) for x in commonsInds2]
            dec = self.TargetList.coords.dec[imat2].value

            currentTime = TK.currentTimeAbs
            r_targ = TL.starprop(imat2,currentTime,False)
            #dec = np.zeros(len(imat2))
            #for i in np.arange(len(imat2)):
            c = SkyCoord(r_targ[:,0],r_targ[:,1],r_targ[:,2],representation='cartesian')
            c.representation = 'spherical'
            dec = c.dec

            
            if len(sInds) > 0:
                # store selected star integration time
                selectInd = np.argmin(abs(fZ-fZmin))
                sInd = sInds[selectInd]#finds index of star to sacrifice
                t_det = t_dets[selectInd]*u.d

                #Create a check to determine if the mission length would be exceeded.
                timeLeft = TK.missionFinishNorm - TK.currentTimeNorm#This is how much time we have left in the mission in u.d
                if(timeLeft > (Obs.settlingTime + mode['syst']['ohTime'])):#There is enough time left for overhead time but not for the full t_det
                    if(timeLeft > (t_det+Obs.settlingTime + mode['syst']['ohTime'])):#If the nominal plan for observation time is greater than what we can do
                        t_det = t_det
                    else:
                        t_det = timeLeft - (Obs.settlingTime + mode['syst']['ohTime'])#We reassign t_det to fill the remaining time
                    break 
                else:#There is insufficient time to cover overhead time
                    TK.allocate_time(timeLeft*u.d)
                    sInd = None
                    t_det = None
                    break

            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.allocate_time(TK.waitTime*TK.waitMultiple**cnt)
                cnt += 1
        else:
            return None#DRM, None, None
        return sInd

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """Finds and Returns Precomputed Observation Time
        Args:
            sInds (integer array):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.  
                must be of the same size as sInds 
            mode (dict):
                Selected observing mode for detection
        Returns:
            intTimes (astropy Quantity array):
                Integration times for detection 
                same dimension as sInds
        """
        commonsInds = [val for val in self.schedule if val in sInds]#finds indicies in common between sInds and self.schedule
        imat = [self.schedule.tolist().index(x) for x in commonsInds]#find indicies of occurence of commonsInds in self.schedule
        intTimes = np.zeros(self.TargetList.nStars)#default observation time is 0 days
        intTimes[commonsInds] = self.t_dets[imat]#
        intTimes = intTimes*u.d#add units of day to intTimes

        return intTimes[sInds]
  
    def distributedt(self, sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA):#distributing the sacrificed time
        """Distributes sacrificedStarTime amoung sInds
        Args:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            sacrificedStarTime - time to distribute in days
            fZ[nStars] - zodiacal light for each target
            fEZ - 0 
        Returns:
            t_dets[nStars] - time to observe each star (in days)
        """
        #Calculate dCbydT for each star at this point in time
        dCbydt = self.Completeness.dcomp_dt(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp).to(1/u.d)#dCbydT[nStars]#Sept 28, 2017 0.12sec
        if(len(t_dets) <= 1):
            return t_dets

        timeToDistribute = sacrificedStarTime
        dt_static = 0.1
        dt = dt_static

        #Now decide where to put dt
        numItsDist = 0
        while(timeToDistribute > 0):
            if(numItsDist > 1000000):#this is an infinite loop check
                break
            else:
                numItsDist = numItsDist + 1

            if(timeToDistribute < dt):#if the timeToDistribute is smaller than dt
                dt = timeToDistribute#dt is now the timeToDistribute
            else:#timeToDistribute >= dt under nominal conditions, this is dt to use
                dt = dt_static#this is the maximum quantity of time to distribute at a time.      

            maxdCbydtIndex = np.argmax(dCbydt)#Find most worthy target

            t_dets[maxdCbydtIndex] = t_dets[maxdCbydtIndex] + dt#Add dt to the most worthy target
            timeToDistribute = timeToDistribute - dt#subtract distributed time dt from the timeToDistribute
            dCbydt[maxdCbydtIndex] = self.Completeness.dcomp_dt(t_dets[maxdCbydtIndex]*u.d, self.TargetList, sInds[maxdCbydtIndex], fZ[maxdCbydtIndex], fEZ, WA, self.mode, self.Cb[maxdCbydtIndex], self.Csp[maxdCbydtIndex]).to(1/u.d)#dCbydT[nStars]#Sept 28, 2017 0.011sec
        #End While Loop
        return t_dets

    def sacrificeStarCbyT(self, sInds, t_dets, fZ, fEZ, WA, overheadTime):
        """Sacrifice the worst performing CbyT star
        Args:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            fZ[nStars] - zodiacal light for each target
            fEZ - 0 
            WA - inner working angle of the instrument
            overheadTime - overheadTime added to each observation
        Return:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            sacrificedStarTime - time to distribute in days
            fZ[nStars] - zodiacal light for each target        
        """
        CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)/t_dets#takes 5 seconds to do 1 time for all stars

        sacrificeIndex = np.argmin(CbyT)#finds index of star to sacrifice

        #Need index of sacrificed star by this point
        sacrificedStarTime = t_dets[sacrificeIndex] + overheadTime#saves time being sacrificed
        sInds = np.delete(sInds,sacrificeIndex)
        t_dets = np.delete(t_dets,sacrificeIndex)
        fZ = np.delete(fZ,sacrificeIndex)
        self.Cb = np.delete(self.Cb,sacrificeIndex)
        self.Csp = np.delete(self.Csp,sacrificeIndex)
        return sInds, t_dets, sacrificedStarTime, fZ

    def calcTinit(self, sInds, TL, fZ, fEZ, WA, mode, Cb, Csp):
        #Generate cache Name########################################################################
        cachefname = self.cachefname + 'maxCbyTt0'
        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached maxCbyTt0 from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                maxCbyTtime = pickle.load(f)
            return maxCbyTtime
        ###########################################################################################
        maxCbyTtime = np.zeros(sInds.shape[0])#This contains the time maxCbyT occurs at
        maxCbyT = np.zeros(sInds.shape[0])#this contains the value of maxCbyT


        #*******************************************
        #Here we are testing the model fit stuff
        #self.generateCvsTforAllStars(sInds, TL, fZ, fEZ, WA, mode, Cb, Csp)#Keeping

        popt_saved = self.calc_gaussianFit_Cbydmag(sInds, TL, fZ, fEZ, WA, mode, Cb, Csp)

        Tint0 = self.calc_gaussianTaylorFit_Cbydmag(sInds, TL, fZ, fEZ, WA, mode, Cb, Csp, popt_saved)
        #*******************************************
        return Tint0

        print(saltyburrito)
        
        #Solve Initial Integration Times###############################################
        def CbyTfunc(t_dets, self, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp):
            CbyT = -self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)/t_dets*u.d
            return CbyT.value

        #Calculate Maximum C/T
        for i in xrange(sInds.shape[0]):
            x0 = 0.01
            maxCbyTtime[i] = scipy.optimize.fmin(CbyTfunc, x0, xtol=1e-8, args=(self, TL, sInds[i], fZ[i], fEZ, WA, mode, self.Cb[i], self.Csp[i]), disp=False)
        t_dets = maxCbyTtime
        #Sept 27, Execution time 101 seconds for 651 stars

        with open(cachefname, "wb") as fo:
            wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
            pickle.dump(t_dets,fo)
            self.vprint("Saved cached 1st year Tinit to %s"%cachefname)
        return maxCbyTtime

    def generateCvsTforAllStars(self, sInds, TL, fZ, fEZ, WA, mode, Cb, Csp):
        print('Generate CvsT for All Stars')
        #Generate CvsT for all stars#####
        def CbyTfunc(t_dets, self, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp):#defines CbyT function
            CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)/t_dets*u.d
            return CbyT.value
        intTimes = np.logspace(-6,3,num=400,base=10.0)#define integration times we will evaluate at
        CvsTmat = np.zeros([sInds.shape[0],len(intTimes)])
        for i in np.arange(intTimes.shape[0]):#iterate through stars generating CvsT for all integration times we define
            CvsTmat[:,i] = CbyTfunc(intTimes[i],self,TL,sInds,fZ,fEZ,WA,mode,Cb,Csp)

        fig1 = plt.figure(9015)
        for j in np.arange(sInds.shape[0]):
            plt.plot(intTimes,CvsTmat[j,:])#plots CvsT by T for all stars
        plt.yscale('log')
        plt.xscale('log')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold')

        maxCbyTtimeInd = np.zeros(sInds.shape[0])
        maxCbyTtimeVal = np.zeros(sInds.shape[0])
        for j in np.arange(sInds.shape[0]):
            maxCbyTtimeInd[j] = np.argmax(CvsTmat[j,:])
            maxCbyTtimeVal[j] = CvsTmat[j,int(maxCbyTtimeInd[j])]
            plt.plot(intTimes[int(maxCbyTtimeInd[j])],maxCbyTtimeVal[j],color='k',marker='o')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.ylabel(r'$C/\tau$',weight='bold')
        plt.xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold')
        ymin = 1e-10
        ymax = 2*1e3
        plt.ylim((ymin,ymax))
        plt.show(block=False)
        fig1.savefig('/home/dean/Documents/SIOSlab/CbyTplotAll'+'.svg')
        plt.close()
        ###############################################################

    def generatedCbydTvsTforAllStars(self, sInds, TL, fZ, fEZ, WA, mode, Cb, Csp):
        #Calculates dcbydt for all stars over the range 1e-5 to 1e5
        #Calculated maximum(dcbydt) for all stars
        print('Generate dCbydTvsT for All Stars')
        
        intTimes = np.logspace(-5,5,num=400,base=10.0)
        dcbydt = np.zeros([intTimes.shape[0],sInds.shape[0]])
        #Generate cache Name########################################################################
        cachefname = self.cachefname + 'dcbydt'
        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached dcbydt from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                dcbydt = pickle.load(f)
        else:
            for i in np.arange(sInds.shape[0]):#np.arange(sInds.shape[0]/10-sInds.shape[0]%10):
                self.vprint('Calculating dcbydt completion '+str(float(i)/sInds.shape[0]))
                for j in np.arange(intTimes.shape[0]):
                    dcbydt[j,i] = self.Completeness.dcomp_dt(intTimes[j]*u.d, TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i]).to(1/u.d).value
            with open(cachefname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                pickle.dump(dcbydt,fo)
                self.vprint("Saved constant dcbydt value to %s"%cachefname)
        self.maxdcbydtinds = np.zeros(sInds.shape[0]).astype(int)
        self.maxdcbydt = np.zeros(sInds.shape[0])
        self.maxdcbydttimes = np.zeros(sInds.shape[0])
        for i in np.arange(sInds.shape[0]):
            self.maxdcbydtinds[i] = int(np.argmax(dcbydt[:,i]))
            self.maxdcbydt[i] = dcbydt[self.maxdcbydtinds[i],i]#np.amax(dcbydt,0)
            self.maxdcbydttimes[i] = intTimes[self.maxdcbydtinds[i]]
        self.dcbydt = dcbydt

    def gaussianffit(self,dmag,sig,mu,scale):#model we will fit the probability density function of completeness vs dmag
        f = scale*1/np.sqrt(2*pi*sig**2)*np.exp(-(dmag-mu)**2/(2*sig**2))
        return f#we include scale because the area under the whole joint pdf is 1 but this is a single slice so it will be scaled by the total area under this section.

    def calc_gaussianFit_Cbydmag(self, sInds, TL, fZ, fEZ, WA, mode, Cb, Csp):
        #Generate cache Name########################################################################
        cachefname = self.cachefname + 'gaussianCbydmagParams'

        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached Cbydmag Params from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                popt_saved = pickle.load(f)
            return popt_saved
        ###########################################################################################
        self.vprint("Calculating gaussianCbydmagParams")

        #Model fit for PDF of C vs dMag for each star
        #Declare model we will Fit to
        #def gaussianffit(dmag,sig,mu,scale):
        #    f = scale*1/np.sqrt(2*pi*sig**2)*np.exp(-(dmag-mu)**2/(2*sig**2))
        #    return f#we include scale because the area under the whole joint pdf is 1 but this is a single slice so it will be scaled by the total area under this section.

        #Set the values of dmag we will fit over
        space = np.linspace(12,40,num=200)
        dmag = np.zeros([sInds.shape[0],space.shape[0]])
        for i in np.arange(sInds.shape[0]):
            dmag[i,:] = space

        fsize = 12

        #Compute Completeness PDF and Fit Gaussians###########################################
        print('Computing PDF')
        IWA = self.mode['IWA']
        OWA = self.mode['OWA']
        f = np.zeros([sInds.shape[0],space.shape[0]])
        popt_saved = np.zeros([sInds.shape[0],3])
        pcov_saved = np.zeros([sInds.shape[0],9])
        residuals = np.zeros([sInds.shape[0],space.shape[0]])
        ss_res = np.zeros(sInds.shape[0])
        ss_tot = np.zeros(sInds.shape[0])
        r_squared = np.zeros(sInds.shape[0])
        tmpscale = 0
        UB = np.zeros(sInds.shape[0])
        UBerror = np.zeros(sInds.shape[0])
        tmpscale = self.Completeness.comp_per_intTime(10**3*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
        for i in np.arange(sInds.shape[0]):
            dist = TL.dist[sInds[i]]#distance of this star from point of observation
            smin = (np.tan(IWA)*dist).to('AU').value*np.ones(dmag.shape[1])#instrument limited minimum separation of detectable exoplanet
            smax = (np.tan(OWA)*dist).to('AU').value*np.ones(dmag.shape[1])#instrument limited maximum separation of detecable exoplanet
            f[i,:] = self.Completeness.calc_fdmag(dmag[i,:],smin,smax)#Actual fdmag-vs-dmag
            UB[i] = self.Completeness.comp_calc(smin[0],smax[0],40)#Evaluate at arbitrary dMag=40 so we get the maximum completeness. UB is used for scale to ensure the area under each gaussian is the same as the actual            
            popt, pcov = curve_fit(self.gaussianffit, dmag[i,:],f[i,:],p0=[4,26,UB[i]],bounds=([0.1,20,UB[i]],[15,30,UB[i]+10**(-6)]))#[0.1,15],[20,30],[-100,100]))
            popt_saved[i,:] = popt#save these parameters
            pcov_saved[i,:] = [pcov[0,0],pcov[0,1],pcov[0,2],pcov[1,0],pcov[1,1],pcov[1,2],pcov[2,0],pcov[2,1],pcov[2,2]]

            fig = plt.figure(999)
            if(i==0):
                plt.plot(dmag[i,:],f[i,:],color='k',label='Completeness')
                plt.plot(dmag[i,:],self.gaussianffit(dmag[i,:],*popt),color='b',linestyle='--',label='Gaussian Fit')
            else:
                plt.plot(dmag[i,:],f[i,:],color='k')
                plt.plot(dmag[i,:],self.gaussianffit(dmag[i,:],*popt),color='b',linestyle='--')
            rcParams['axes.linewidth']=2
            rc('font',weight='bold') 
            plt.rc('axes',linewidth=2)
            plt.rc('lines',linewidth=2)
            plt.title('pdf of Completeness marginalized over S (All Stars)',weight='bold',fontsize=fsize)
            plt.ylabel(r'$\frac{dC}{d\Delta mag}\ PDF\ (days^{-1})$',weight='bold',fontsize=fsize)
            plt.xlabel(r'$\Delta mag$',weight='bold')
            plt.xlim([17.5,35])
            plt.ylim([0,0.07])
            #################
            
            #calculate R**2#########
            residuals[i,:] = f[i,:]-self.gaussianffit(dmag[i,:],*popt)
            ss_res[i] = np.sum(residuals[i,:]**2)
            ss_tot[i] = np.sum((f[i,:]-np.mean(f[i,:]))**2)
            r_squared[i] = 1-(ss_res[i]/ss_tot[i])
            ##########

            #Plotting Every 50 or so##############################################
            k = (i-i%50)/50
            fig2 = plt.figure(1000+k)
            if(i%50 == 0):
                plt.plot(dmag[i,:],f[i,:],color='k',label='Completeness')
                plt.plot(dmag[i,:],self.gaussianffit(dmag[i,:],*popt),color='b',linestyle='--',label='Gaussian Fit')
            else:
                plt.plot(dmag[i,:],f[i,:],color='k')
                plt.plot(dmag[i,:],self.gaussianffit(dmag[i,:],*popt),color='b',linestyle='--')
            rcParams['axes.linewidth']=2
            rc('font',weight='bold') 
            plt.rc('axes',linewidth=2)
            plt.rc('lines',linewidth=2)
            plt.title('pdf of Completeness marginalized over S '+str(i-50)+' to '+str(i),weight='bold')
            plt.ylabel(r'$\frac{dC}{d\Delta mag}\ (days^{-1})$',weight='bold',fontsize=fsize)
            plt.xlabel(r'$\Delta mag$',weight='bold',fontsize=fsize)
            plt.legend()
            plt.xlim([17.5,35])
            plt.ylim([0,0.07])
            if(((i%50==49) and i>0) or (i%(sInds.shape[0]-1)==0 and i>0)):
                plt.show(block=False)
                fig2.savefig('/home/dean/Documents/SIOSlab/dCdmagPDFgaussianFit'+str(i-50)+'to'+str(i)+'.svg')
            #############

            #Plotting Specific Gaussian Fits################################################
            if(i in [0,9,36,39,34,37,32,29]):
                fig3 = plt.figure(1020)
                if(i == 0):
                    plt.plot(dmag[i,:],f[i,:],color='k',label='Completeness')
                    plt.plot(dmag[i,:],self.gaussianffit(dmag[i,:],*popt),color='b',linestyle='--',label='Gaussian Fit')
                else:
                    plt.plot(dmag[i,:],f[i,:],color='k')
                    plt.plot(dmag[i,:],self.gaussianffit(dmag[i,:],*popt),color='b',linestyle='--')
                rcParams['axes.linewidth']=2
                rc('font',weight='bold') 
                plt.rc('axes',linewidth=2)
                plt.rc('lines',linewidth=2)
                plt.title('pdf of Completeness marginalized over S '+str(i),weight='bold')
                plt.ylabel(r'$\frac{dC}{d\Delta mag}\ (days^{-1})$',weight='bold',fontsize=fsize)
                plt.xlabel(r'$\Delta mag$',weight='bold',fontsize=fsize)
                plt.legend()
                plt.xlim([17.5,35])
                plt.ylim([0,0.07])
                plt.show(block=False)
                fig2.savefig('/home/dean/Documents/SIOSlab/dCdmagPDFgaussianFitSelective'+str(i)+'.svg')
            ################
        plt.show(block=False)
        fig = plt.figure(999)
        plt.xlim([17.5,35])
        plt.ylim([0,0.07])
        plt.legend()
        fig.savefig('/home/dean/Documents/SIOSlab/dCdmagPDFALL'+'.svg')
        plt.close("All")

        #Plot R_squared Value Histogram##################################
        rSquaredHist = plt.figure(9016)
        out = plt.hist(r_squared,label=r'$R^2$',color='b')
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Gaussian Fit Histogram of '+r'$R^2$',weight='bold')
        plt.xlabel(r'$R^2$',weight='bold',fontsize=fsize)
        plt.ylabel('# of Targets',weight='bold',fontsize=fsize)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.show(block=False)
        rSquaredHist.savefig('/home/dean/Documents/SIOSlab/rSquaredHist'+'.svg')
        plt.close("all")
        ##################################################################

        #Save Gaussian Model Fit Parameters to File
        with open(cachefname, "wb") as fo:
            wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
            pickle.dump(popt_saved,fo)
            self.vprint("Saved gaussian Fit Parameters to %s"%cachefname)
        return popt_saved

    def gaussianTarlorFit(self, x,n,kmax,A,B,C):#Taylor series expansion of the PDF ... note this is not the derivative version
        f = 0
        for k in np.arange(kmax):
            f = f + A*(B**k)*(x-C)**(2*k)/factorial(k)#x is dmag
        return f

    def calc_gaussianTaylorFit_Cbydmag(self, sInds, TL, fZ, fEZ, WA, mode, Cb, Csp, popt_saved):
        #Calculate and Plot C PDFvsdmag from Gaussian Fit at different k##################################
        print('calculate Gaussian Taylor Fit C by dmag')
        #popt_saved
        #scale*1/np.sqrt(2*pi*sig**2)*np.exp(-(dmag-mu)**2/(2*sig**2))
        #sig,mu,scale
        sig = popt_saved[:,0]
        mu = popt_saved[:,1]
        scale = popt_saved[:,2]
        A = scale*1/np.sqrt(2*pi*sig**2)
        B = -1/(2*sig**2)
        C  = mu
        
        #Set the values of dmag we will fit over#COPY AND PASTED FROM calc_gaussianFit_Cbydmag
        space = np.linspace(12,40,num=200)
        dmag = np.zeros([sInds.shape[0],space.shape[0]])
        for i in np.arange(sInds.shape[0]):
            dmag[i,:] = space

        n=0
        kmax=10
        taylorPDF = np.zeros([sInds.shape[0],dmag.shape[1]])
        i=0
        taylorPDF[i,:] = self.gaussianTarlorFit(dmag[i,:],n,kmax,A[i],B[i],C[i])
        kmax=20
        taylorPDF[i+1,:] = self.gaussianTarlorFit(dmag[i,:],n,kmax,A[i],B[i],C[i])
        kmax=50
        taylorPDF[i+2,:] = self.gaussianTarlorFit(dmag[i,:],n,kmax,A[i],B[i],C[i])
        kmax=100
        taylorPDF[i+3,:] = self.gaussianTarlorFit(dmag[i,:],n,kmax,A[i],B[i],C[i])
        kmax=200
        taylorPDF[i+4,:] = self.gaussianTarlorFit(dmag[i,:],n,kmax,A[i],B[i],C[i])
        figGaussian = plt.figure(700)
        plt.plot(dmag[0,:],taylorPDF[0,:],color='r',linestyle='--',label='Taylor k=10')
        plt.plot(dmag[0,:],taylorPDF[1,:],color='g',linestyle='--',label='Taylor k=20')
        plt.plot(dmag[0,:],taylorPDF[2,:],color='c',linestyle='--',label='Taylor k=50')
        plt.plot(dmag[0,:],taylorPDF[3,:],color='k',linestyle='--',label='Taylor k=100')
        plt.plot(dmag[0,:],taylorPDF[4,:],color='m',linestyle='--',label='Taylor k=200')
        plt.plot(dmag[0,:],self.gaussianffit(dmag[i,:],*popt_saved[0,:]),color='b',linestyle='--',label='Gaussian Fit')
        plt.ylim(-0.01,1.1*max(self.gaussianffit(dmag[i,:],*popt_saved[0,:])))
        plt.legend()
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Taylor PDF of Gaussian ',weight='bold')
        plt.xlabel(r'$\Delta$'+'mag',weight='bold')
        plt.ylabel('Completeness',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.show(block=False)
        figGaussian.savefig('/home/dean/Documents/SIOSlab/figGaussian'+'.svg')
        plt.close()
        ###############################
        figGaussian2 = plt.figure(701)
        n=0
        kmax = 100
        for i in np.arange(sInds.shape[0]):
            taylorPDF[i,:] = self.gaussianTarlorFit(dmag[i,:],n,kmax,A[i],B[i],C[i])
            plt.plot(dmag[i,:],taylorPDF[i,:],color='k',linestyle='--',label='Gaussian Fit')
        #find max ylim
        tmpMaxComp = np.zeros(sInds.shape[0])
        for i in np.arange(sInds.shape[0]):
            tmpMaxComp[i] = max(self.gaussianffit(dmag[i,:],*popt_saved[i,:]))
        plt.ylim(-0.005,1.1*max(tmpMaxComp))
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Taylor PDF Gaussian k=100',weight='bold')
        plt.xlabel(r'$\Delta$'+'mag',weight='bold')
        plt.ylabel('Completeness',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.show(block=False)
        figGaussian2.savefig('/home/dean/Documents/SIOSlab/figGaussian2'+'.svg')
        plt.close()
        ################################################################################

        #Plot C vs dmag for several  stars###############################################
        def taylorCDF(x,n,kmax,A,B,C,D=0):#x is dmag
            f=0
            for k in np.arange(kmax):
                f = f + A*(B**k)/factorial(k)*(x-C)**(2*k+1)/(2*k+1)
            f = f + D
            return f
        A = scale*1/np.sqrt(2*pi*sig**2)
        B = -1/(2*sig**2)
        C  = mu
        TaylorCDFofGaussianFig = plt.figure(702)
        n=0
        kmax=100
        taylorCDFn = np.zeros([sInds.shape[0],dmag.shape[1]])
        taylorCDF_saved = np.zeros([sInds.shape[0],dmag.shape[1]])
        taylorCDFn1sig = np.zeros([sInds.shape[0]])
        taylorCDFnmu = np.zeros([sInds.shape[0]])
        taylorCDFn2sig = np.zeros([sInds.shape[0]])
        taylorCDFn3sig = np.zeros([sInds.shape[0]])
        taylorCDFndmagMAX = np.zeros([sInds.shape[0]])
        maxCbydmag2sig = 0
        minCbydmag2sig = 0
        D = np.zeros([sInds.shape[0]])
        cmap = plt.cm.get_cmap('autumn_r')
        for i in np.arange(sInds.shape[0]):
            taylorCDFn[i,:] = taylorCDF(dmag[i,:],n,kmax,A[i],B[i],C[i])
            tmpdmag = np.argmin((dmag[i,:]-15)**2)
            taylorCDF_saved[i,:] = taylorCDFn[i,:]
            D[i] = abs(taylorCDFn[i,tmpdmag])
            taylorCDFn[i,:] = taylorCDFn[i,:] + D[i]
            
            taylorCDFnmu[i] = taylorCDF(mu[i],n,kmax,A[i],B[i],C[i],D[i])
            taylorCDFn1sig[i] = taylorCDF(mu[i]+sig[i],n,kmax,A[i],B[i],C[i],D[i])
            taylorCDFn2sig[i] = taylorCDF(mu[i]+2*sig[i],n,kmax,A[i],B[i],C[i],D[i])
            taylorCDFn3sig[i] = taylorCDF(mu[i]+3*sig[i],n,kmax,A[i],B[i],C[i],D[i])
        for i in np.arange(sInds.shape[0]/10)*10:#[0,9,29,32,34,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]:
            maxCbydmag2sig = max(taylorCDFn2sig/(mu+2*sig))
            minCbydmag2sig = min(taylorCDFn2sig/(mu+2*sig))
            currCbydmag = taylorCDFn2sig[i]/(mu[i]+2*sig[i])
            Fraction = (currCbydmag-minCbydmag2sig)/(maxCbydmag2sig-minCbydmag2sig)
            rgba = cmap(Fraction)
            r=rgba[0]
            g=rgba[1]
            b=rgba[2]
            a=0.5
            assert r<=1
            assert b<=1
            plt.plot(dmag[i,:],taylorCDFn[i,:],color=(r,g,b,a))
        #plot Zoomed Plot
        #Calculate MAXIMUM dmag at some Tint=10**5
        maxDmag = self.OpticalSystem.calc_dMag_per_intTime(np.zeros(sInds.shape[0])*u.d+10**5*u.d, TL, sInds, fZ, fEZ+np.zeros(sInds.shape[0]), WA+np.zeros(sInds.shape[0]), mode, Cb, Csp)
        for i in np.arange(sInds.shape[0]):
            taylorCDFndmagMAX[i] = taylorCDF(maxDmag[i],n,kmax,A[i],B[i],C[i],D[i])
        plt.plot(maxDmag,taylorCDFndmagMAX,color='k',alpha=1,label=r'$\Delta mag_{max}$',linewidth=3.0)#most recently added line!
        plt.rc('lines',linewidth=2)
        plt.xlim(20,23.5)
        plt.ylim(0.0,0.13)
        rcParams['axes.linewidth']=2
        rc('font',weight='bold')
        plt.title('Taylor CDF of Gaussian k=100',weight='bold')
        plt.xlabel(r'$\Delta$'+'mag',weight='bold',fontsize=18)
        plt.ylabel('Completeness',weight='bold',fontsize=18)
        plt.rc('axes',linewidth=2)
        plt.show(block=False)
        TaylorCDFofGaussianFig.savefig('/home/dean/Documents/SIOSlab/TaylorCDFofGaussianFigZoomed'+'.svg')
        #finish zoomed plot continuing with normal plot

        #for i in np.arange(sInds.shape[0]):
        #    plt.plot(mu[i],taylorCDFnmu[i],color='k',marker='o')#most recently added line!
        #    plt.plot(mu[i]+sig[i],taylorCDFn1sig[i],color='k',marker='o')#most recently added line!
        #    plt.plot(mu[i]+2*sig[i],taylorCDFn2sig[i],color='r',marker='o')#most recently added line!
        #    plt.plot(mu[i]+3*sig[i],taylorCDFn3sig[i],color='b',marker='o')#most recently added line!
        cmap = plt.cm.get_cmap('winter')#rmbCMAP = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])
        tmpI = argsort(mu)
        plt.plot(mu[tmpI],taylorCDFnmu[tmpI],color=cmap(0.0),label=r'$50%\ C_{max}$',linewidth=3.0)#most recently added line!,marker='s',alpha=0.5
        plt.plot(mu[tmpI]+sig[tmpI],taylorCDFn1sig[tmpI],color=cmap(0.3333),label=r'$84.1%\ C_{max}$',linewidth=3.0)#most recently added line!,marker='h',alpha=0.5
        plt.plot(mu[tmpI]+2*sig[tmpI],taylorCDFn2sig[tmpI],color=cmap(0.6666),label=r'$97.7%\ C_{max}$',linewidth=3.0)#most recently added line!,marker='D',alpha=0.5
        plt.plot(mu[tmpI]+3*sig[tmpI],taylorCDFn3sig[tmpI],color=cmap(1.0),label=r'$99.8%\ C_{max}$',linewidth=3.0)#most recently added line!,marker='o',alpha=0.5
        cmap = plt.cm.get_cmap('autumn_r')
        sc = plt.scatter([nan,nan],[0,1],c=[maxCbydmag2sig,minCbydmag2sig],cmap=cmap)
        cbar = plt.colorbar(sc)
        lar = np.round([minCbydmag2sig,minCbydmag2sig+0.2*(maxCbydmag2sig-minCbydmag2sig),minCbydmag2sig+0.4*(maxCbydmag2sig-minCbydmag2sig),minCbydmag2sig+0.6*(maxCbydmag2sig-minCbydmag2sig),minCbydmag2sig+0.8*(maxCbydmag2sig-minCbydmag2sig),maxCbydmag2sig],decimals=4)
        cbar.ax.set_yticklabels([str(lar[0]),str(lar[1]),str(lar[2]),str(lar[3]),str(lar[4]),str(lar[5])])
        cbar.set_label(r'$max(\frac{C}{\Delta mag})$',weight='bold',fontsize=18)#removed ax.
        
        plt.legend(loc=4,prop={'size': 9})
        plt.ylim(0.0,0.30)
        plt.xlim(17.5,35)
        rcParams['axes.linewidth']=2
        rc('font',weight='bold')
        plt.title('Taylor CDF of Gaussian k=100',weight='bold')
        plt.xlabel(r'$\Delta$'+'mag',weight='bold',fontsize=18)
        plt.ylabel('Completeness',weight='bold',fontsize=18)
        plt.rc('axes',linewidth=2)
        plt.show(block=False)
        TaylorCDFofGaussianFig.savefig('/home/dean/Documents/SIOSlab/TaylorCDFofGaussianFig'+'.svg')
        #################################


        #Plot vs Gaussian C-vs-intTime
        #intTimesGaussian = np.zeros([sInds.shape[0],dmag.shape[1]])
        #for i in np.arange(dmag.shape[1]):
        #    intTimesGaussian[:,i] = self.OpticalSystem.calc_intTime(TL, sInds, fZ, fEZ, dmag[:,i], WA, mode)
        #    intTimesGaussian[intTimesGaussian[:,i]==0,i] = nan

        #Plot the compPerIntTime actual now.
        TaylorCDFofGaussianFigsTau = plt.figure(703)
        cmap = plt.cm.get_cmap('autumn_r')
        intTimes = np.logspace(-4,3,num=400,base=10.0)#define integration times we will evaluate at
        actualComp = np.zeros([sInds.shape[0],intTimes.shape[0]])
        for j in np.arange(intTimes.shape[0]):
            actualComp[:,j] = self.Completeness.comp_per_intTime((intTimes[j]+np.zeros([sInds.shape[0]]))*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
        #Sort Indicies by maximum valued actual comps
        tmpI = argsort(-np.amax(actualComp,1))

        for l in np.arange(10):#np.arange(sInds.shape[0]):
            l = tmpI[l]
            plt.plot(intTimes,actualComp[l,:],color='k',zorder=1)
        ###############################

        #Plot Gaussian CvsT intTimes####################
        n=0
        kmax=100
        tmpdmag = np.zeros([sInds.shape[0],intTimes.shape[0]])
        taylorCDFvals = np.zeros([sInds.shape[0],intTimes.shape[0]])
        taylorCDFvalsbylogT = np.zeros([sInds.shape[0],intTimes.shape[0]])
        for i in np.arange(intTimes.shape[0]):#Generate dmag corresponding to all reasonable intTimes
            tmpIntTimes = np.zeros([sInds.shape[0]])+intTimes[i]
            tmpdmag[:,i] = self.OpticalSystem.calc_dMag_per_intTime(tmpIntTimes*u.d, TL, sInds, fZ, np.zeros([sInds.shape[0]])+fEZ, np.zeros([sInds.shape[0]])+WA, mode, Cb, Csp)
        for i in np.arange(sInds.shape[0]):
            taylorCDFvals[i,:] = taylorCDF(tmpdmag[i,:],n,kmax,A[i],B[i],C[i],D[i])#Calculate all C from model
            taylorCDFvalsbylogT[i,:] = taylorCDFvals[i,:]/np.log10(intTimes*86400*1000)#intTimes in units of microseconds...#np.log10(intTimes*10**8)#We can't use np.log10(intTimes because the denominator is 0 at 1 day...)
        PerStarMaxtaylorCDFvalsbylogT = np.amax(taylorCDFvalsbylogT,1)#Gives maximum CbyT for each Star with Taylor Fit
        PerStarMaxtaylorCDFinds = np.argmax(taylorCDFvalsbylogT,1)
        AbsMaxtaylorCDFvalsbylogT = max(PerStarMaxtaylorCDFvalsbylogT)#[np.arange(sInds.shape[0])!=np.argmax(PerStarMaxtaylorCDFvalsbylogT)])#Gives the maximum CbyT overall
        PerStarMintaylorCDFvalsbylogT = min(PerStarMaxtaylorCDFvalsbylogT)#Gives minimum of maxCbyT from Taylor Fit for each star
        tmpI = argsort(-PerStarMaxtaylorCDFvalsbylogT)
        #mV = TL.starMag(sInds,self.mode['lam'])#Note these come out really well when applied as coloring
        #maxmV = max(mV)
        #minmV = min(mV)
        #Plot lines ov CvsTau
        for i in np.arange(sInds.shape[0]):#np.arange(10):
            Fraction = (PerStarMaxtaylorCDFvalsbylogT[i]-PerStarMintaylorCDFvalsbylogT)/(AbsMaxtaylorCDFvalsbylogT-PerStarMintaylorCDFvalsbylogT)
            #Fraction = (np.log10(PerStarMaxtaylorCDFvalsbylogT[i])-np.log10(PerStarMintaylorCDFvalsbylogT))/(np.log10(AbsMaxtaylorCDFvalsbylogT)-np.log10(PerStarMintaylorCDFvalsbylogT))
            # Fraction = (mV[i]-minmV)/(maxmV-minmV)#comes out really well when applied as coloring
            rgba = cmap(Fraction)
            r=rgba[0]
            g=rgba[1]
            b=rgba[2]
            a=1-g#0.5 is too light I think. The red doesn't look red enough. maybe assign to r value...
            assert r<=1
            assert b<=1
            plt.plot(intTimes,taylorCDFvals[i,:],color=(r,g,b,a),zorder=2)
        #Plot Max C/log10(Tau) points
        #for i in np.arange(sInds.shape[0]):
        #    plt.scatter(intTimes[PerStarMaxtaylorCDFinds[i]],taylorCDFvals[i,PerStarMaxtaylorCDFinds[i]],color='k',marker='o',zorder=3)
        plt.xscale('log')
        plt.ylim(-0.01,0.15)
        plt.xlim(10e-5,10)
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Taylor CDF of Gaussian k=100',weight='bold')
        plt.xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold',fontsize=14)
        plt.ylabel('Completeness',weight='bold',fontsize=14)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        #Plot Colorbar
        cmap = plt.cm.get_cmap('autumn_r')
        sc = plt.scatter([nan,nan],[0,1],c=[AbsMaxtaylorCDFvalsbylogT,PerStarMintaylorCDFvalsbylogT],cmap=cmap)
        cbar = plt.colorbar(sc)
        lar = np.round([PerStarMintaylorCDFvalsbylogT,PerStarMintaylorCDFvalsbylogT+0.2*(AbsMaxtaylorCDFvalsbylogT-PerStarMintaylorCDFvalsbylogT),PerStarMintaylorCDFvalsbylogT+0.4*(AbsMaxtaylorCDFvalsbylogT-PerStarMintaylorCDFvalsbylogT),PerStarMintaylorCDFvalsbylogT+0.6*(AbsMaxtaylorCDFvalsbylogT-PerStarMintaylorCDFvalsbylogT),PerStarMintaylorCDFvalsbylogT+0.8*(AbsMaxtaylorCDFvalsbylogT-PerStarMintaylorCDFvalsbylogT),AbsMaxtaylorCDFvalsbylogT],decimals=4)
        cbar.ax.set_yticklabels([str(lar[0]),str(lar[1]),str(lar[2]),str(lar[3]),str(lar[4]),str(lar[5])])
        cbar.set_label(r'$max(\frac{C}{log(\tau)})$',weight='bold',fontsize=14)#removed ax.
        plt.show(block=False)
        

        tmpfig = plt.figure(708)
        for i in np.arange(sInds.shape[0]):
            plt.scatter(intTimes[PerStarMaxtaylorCDFinds[i]],taylorCDFvals[i,PerStarMaxtaylorCDFinds[i]],color='k',marker='o')
        plt.xlabel('Integration Time (days)')
        plt.ylabel('Gaussian Derived Completness')
        plt.show(block=False)
        plt.close()
        ####################################

        #Plot of the RMS error##############
        rmsError = np.zeros([sInds.shape[0]])
        for i in np.arange(sInds.shape[0]):
            end = taylorCDFvals.shape[1]
            rmsError[i] = sum(np.sqrt((actualComp[i,70:end-40]-taylorCDFvals[i,70:end-40])**2))
        rmsTaylorHistogram = plt.figure(704)
        plt.hist(rmsError,bins=1000)
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        plt.title('Taylor CDF Error Histogram',weight='bold')
        plt.xlabel('RMS Error',weight='bold')
        plt.ylabel('Completeness',weight='bold')
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.show(block=False)
        rmsTaylorHistogram.savefig('/home/dean/Documents/SIOSlab/rmsTaylorHistogram'+'.svg')
        plt.close()
        #################################################################################

        #Solve Initial Integration Times###############################################
        #This was a failure because most mu and sigma values were above the dmaglim of ~23.2 so they all became nan or 0
        #compute mu, mu+1sigms, mu+2sigma, mu+3sigma integration times.
        #intTmu = self.OpticalSystem.calc_intTime(TL,sInds,fZ,fEZ,mu,WA,mode)
        #intTmu1sig = self.OpticalSystem.calc_intTime(TL,sInds,fZ,fEZ,mu+sig,WA,mode)
        #intTmu2sig = self.OpticalSystem.calc_intTime(TL,sInds,fZ,fEZ,mu+2*sig,WA,mode)
        #intTmu3sig = self.OpticalSystem.calc_intTime(TL,sInds,fZ,fEZ,mu+3*sig,WA,mode)
        #Cmu = self.Completeness.comp_per_intTime(intTmu, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
        #Cmu1sig = self.Completeness.comp_per_intTime(intTmu1sig, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
        #Cmu2sig = self.Completeness.comp_per_intTime(intTmu2sig, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
        #Cmu3sig = self.Completeness.comp_per_intTime(intTmu3sig, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)
        
        #Calculate These Completenesses
        # TaylorCDFofGaussianFigsTau = plt.figure(703)
        # plt.plot(intTmu,Cmu,marker='s')
        # plt.plot(intTmu1sig,Cmu1sig,marker='h')
        # plt.plot(intTmu2sig,Cmu2sig,marker='D')
        # plt.plot(intTmu3sig,Cmu3sig,marker='o')
        # plt.show(block=False)
        ###############################################################################

        #Plot log(Integration Time) vs dmag############################################
        mV = TL.starMag(sInds,self.mode['lam'])
        maxmV = max(mV)
        minmV = min(mV)
        intTimevsdmag = plt.figure(709)
        cmap = plt.cm.get_cmap('winter')
        for i in np.arange(sInds.shape[0]):
            Fraction = (mV[i]-minmV)/(maxmV-minmV)
            rgba = cmap(Fraction)
            r=rgba[0]
            g=rgba[1]
            b=rgba[2]
            a=0.1
            assert r<=1
            assert b<=1
            plt.plot(tmpdmag[i,:],np.log10(intTimes),color=(r,g,b,a))
        plt.ylabel('log(Integration Time) (log(days))')
        plt.xlabel('dmag')
        plt.ylim([1.1*min(np.log10(intTimes)),1.1*max(np.log10(intTimes))])
        plt.xlim([0.9*np.amin(tmpdmag),1.1*np.amax(tmpdmag)])
        #plt.yscale('log')
        cmap = plt.cm.get_cmap('winter')
        sc = plt.scatter([nan,nan],[0,1],c=[maxmV,minmV],cmap=cmap)
        cbar = plt.colorbar(sc)
        lar = np.round([minmV,minmV+0.2*(maxmV-minmV),minmV+0.4*(maxmV-minmV),minmV+0.6*(maxmV-minmV),minmV+0.8*(maxmV-minmV),maxmV],decimals=4)
        cbar.ax.set_yticklabels([str(lar[0]),str(lar[1]),str(lar[2]),str(lar[3]),str(lar[4]),str(lar[5])])
        cbar.set_label(r'$mV$')#removed ax.
        plt.show(block=False)
        intTimevsdmag.savefig('/home/dean/Documents/SIOSlab/IntTimevsdmagShowsmV'+'.svg')
        #################################################################################

        kmax = 40
        dmagInit = np.zeros(sInds.shape[0])
        for i in np.arange(sInds.shape[0]):
            for k in np.arange(kmax-2)+2:#Fixed requirement that k must be greater than 2 (2*k>=n) where here n=3
                dmagInit[i] = dmagInit[i] + C[i]-(factorial(2*k-3)/(B[i]**(2*k)*(k**2)*(2*k-1)*(2*k-2)*((2*k-1)*(2*k-2)-3)))*(factorial(k-1)**2)

        fig = plt.figure(702)
        for i in np.arange(sInds.shape[0]):
            plt.plot(dmagInit[i],taylorCDF(dmagInit[i],n,kmax,A[i],B[i],C[i])+D[i],marker='o',color='k')
        plt.show(block=False)
        fig.savefig('/home/dean/Documents/SIOSlab/TaylorCDFofGaussianFig_2'+'.svg')

        #Generates dCbydt for All Stars######################################
        self.generatedCbydTvsTforAllStars(sInds, TL, fZ, fEZ, WA, mode, Cb, Csp)#calculates or loads self.dcbydt, self.maxdcbydt, and self.maxdcbydtinds
        #Solve the dComp/dt=number (1*10^-7)#####################################################
        print('Calculating Intial Guess t0 such that each star has dCbydT')
        def func(t, TL, sInd, fZ, fEZ, WA, mode, Cb, Csp):
            f = (4*1e-3-self.Completeness.dcomp_dt(t*u.d, TL, sInd, fZ, fEZ, WA, mode, Cb, Csp).to(1/u.d).value[0])**(2)
            return f
        #Generate cache Name########################################################################
        cachefname = self.cachefname + 'constdCbydtt0'
        #Check if file exists#######################################################################
        t0 = np.zeros([sInds.shape[0]])
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached constdCbydtt0 from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                t0 = pickle.load(f)
            #return t0
        else:
            for i in np.arange(sInds.shape[0]):
                self.vprint('sInds Fraction '+str(float(i)/sInds.shape[0]))
                t0[i] = self.maxdcbydttimes[i]#intTimes[self.maxdcbydtinds[i]]
                if(self.maxdcbydt[i] > 4*1e-3+1e-5):

                    #t0[i] = intTimes[self.maxdcbydtinds[i]]
                    #else:
                    #guesst0 = t0[i]#intTimes[self.maxdcbydtinds[i]]#in days
                    #print('Guess '+str(guesst0))
                    #t0[i] = scipy.optimize.fmin(func,guesst0,xtol=1e-5,ftol=1e-3,args=(TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i]),disp=False)#,maxiter=1e10)
                    # c = max(intTimes)#should be largest
                    # a = guesst0*0.9#should be smallest
                    # if(a<1e-10):
                    #     a=1e-1
                    # b = a*1.1
                    t0[i] = scipy.optimize.golden(func,tol=1e-5,brack=None,args=(TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i]),full_output=False)#,maxiter=1e10)
                    #guessVal = self.Completeness.dcomp_dt(guesst0*u.d, TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i]).to(1/u.d).value[0]#func(guesst0,TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i])
                    #t0val = self.Completeness.dcomp_dt(t0[i]*u.d, TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i]).to(1/u.d).value[0]#func(t0[i],TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i])
                    #if(abs(t0val-1e-2) > 1e-4):#If we were not able to optimize to the desired value (abs(max(dC/dT)-const)>tolerance)
                    #    if(t0val < guessVal):#pick largest between guessVal and t0
                    #        t0[i] = guesst0#just checking to be sure the final value is actually less than the actual.
                    #    #else t0[i] = t0[i]#If t0val > guessVal
                if(t0[i]<1e-10):
                    t0[i] = 1e-1
                #print(t0[i])
        #######################################

        #Calculate dcompdt###############
        dcompdt = np.zeros([sInds.shape[0]])
        for i in np.arange(sInds.shape[0]):
            dcompdt[i] = self.Completeness.dcomp_dt(t0[i]*u.d, TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i]).to(1/u.d).value
        intTimes = np.logspace(-5,5,num=400,base=10.0)
        
        #Plot dcbydt for all stars###################################
        fig = plt.figure(550)
        for i in np.arange((sInds.shape[0]-(sInds.shape[0]%10))/10):#Plot 10% of dcbydt lines
            plt.plot(intTimes,self.dcbydt[:,i],color=(1,0,0,0.5),zorder=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(t0,dcompdt,color='k',zorder=3)
        plt.scatter(self.maxdcbydttimes,self.maxdcbydt,color='b',zorder=2)
        xlabel(r'$Integration\ Time\ \tau\ (days)$',weight='bold',fontsize=14)
        ylabel(r'$\frac{dC}{d\tau}$',weight='bold',fontsize=14)
        plt.plot([0,0],[0,0],color='r',label=r'$\frac{dC}{d\tau}$')
        plt.scatter([1e-8,1e-8],[1e-8,1e-8],color='b',marker='o',label=r'$max(\frac{dC}{d\tau})$')
        plt.scatter([1e-8,1e-8],[1e-8,1e-8],color='k',marker='o',label=r'$\frac{dC}{d\tau}(t0)$')
        xmin = min(self.maxdcbydttimes)
        xmax = 30#max(t0)#We wont observe for longer than a month
        ymin = min(self.maxdcbydt)#min(dcompdt)
        ymax = max(self.maxdcbydt)#max(dcompdt)
        plt.xlim([xmin,xmax])
        plt.ylim([0.1*ymin,ymax*5])
        plt.legend(loc=1,prop={'size': 10})
        plt.show(block=False)
        fig.savefig('/home/dean/Documents/SIOSlab/dCbydTandOptimalSelection'+'.svg')
        self.myt0plotting=t0
        #############################################

        #Final Touches of C vs t0...
        TaylorCDFofGaussianFigsTau = plt.figure(703)
        compatt0 = np.zeros([sInds.shape[0]])
        for j in np.arange(sInds.shape[0]):
            compatt0[j] = self.Completeness.comp_per_intTime(t0[j]*u.d, TL, sInds[i], fZ[i], fEZ, WA, mode, Cb[i], Csp[i])
        plt.scatter(t0,compatt0,color='k',marker='o',zorder=3,label=r'$C(\tau_{0})$')
        self.compatt0 = compatt0
        plt.plot([1e-5,1e-5],[0,0],color='k',label='Numerical C')
        plt.plot([1e-5,1e-5],[0,0],color='r',label='Gaussian fit C')
        plt.legend(loc=2)
        plt.xlim([1e-4,1e1])
        plt.ylim([0,0.14])
        TaylorCDFofGaussianFigsTau.savefig('/home/dean/Documents/SIOSlab/TaylorCDFofGaussianFigsTau'+'.svg')

        with open(cachefname, "wb") as fo:
            wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
            pickle.dump(t0,fo)
            self.vprint("Saved constant dcbydt value to %s"%cachefname)

        plt.close("all")
        #print(saltyburrito)
        return t0
        #########################################################################################

        #We are returning the intTimes from argmax(C/np.log10(intTimes))
        return t0#intTimes[PerStarMaxtaylorCDFinds]/1000