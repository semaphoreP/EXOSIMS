from EXOSIMS.util.vprint import vprint
import numpy as np
import astropy.units as u
from astropy.time import Time
import math

class TimeKeeping(object):
    """TimeKeeping class template.
    
    This class keeps track of the current mission elapsed time
    for exoplanet mission simulation.  It is initialized with a
    mission duration, and throughout the simulation, it allocates
    temporal intervals for observations.  Eventually, all available
    time has been allocated, and the mission is over.
    Time is allocated in contiguous windows of size "duration".  If a
    requested interval does not fit in the current window, we move to
    the next one.
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        missionStart (astropy Time):
            Mission start time in MJD
        missionLife (astropy Quantity):
            Mission life time in units of year
        extendedLife (astropy Quantity):
            Extended mission time in units of year
        missionPortion (float):
            Portion of mission devoted to planet-finding
        missionFinishNorm (astropy Quantity):
            Mission finish normalized time in units of day
        missionFinishAbs (astropy Time):
            Mission finish absolute time in MJD
        currentTimeNorm (astropy Quantity):
            Current mission time normalized to zero at mission start in units of day
        currentTimeAbs (astropy Time):
            Current absolute mission time in MJD
        OBnumber (integer):
            Index/number associated with the current observing block (OB). Each 
            observing block has a duration, a start time, an end time, and can 
            host one or multiple observations.
        OBduration (astropy Quantity):
            Default allocated duration of observing blocks, in units of day. If 
            no OBduration was specified, a new observing block is created for 
            each new observation in the SurveySimulation module. 
        OBstartTimes (astropy Quantity array):
            Array containing the normalized start times of each observing block 
            throughout the mission, in units of day
        OBendTimes (astropy Quantity array):
            Array containing the normalized end times of each observing block 
            throughout the mission, in units of day
        obsStart (astropy Quantity):
            Normalized start time of the observation currently executed by the 
            Survey Simulation, in units of day
        obsEnd (astropy Quantity):
            Normalized end time of the observation currently executed by the 
            Survey Simulation, in units of day
        waitTime (astropy Quantity):
            Default allocated duration to wait in units of day, when the
            Survey Simulation does not find any observable target
        waitMultiple (float):
            Multiplier applied to the wait time in case of repeated empty lists of 
            observable targets, which makes the wait time grow exponentially. 
            As soon as an observable target is found, the wait time is reinitialized 
            to the default waitTime value.
        
    """

    _modtype = 'TimeKeeping'
    _outspec = {}

    def __init__(self, missionStart=60634, missionLife=0.1, extendedLife=0, 
            missionPortion=1, OBduration=np.inf, waitTime=1, waitMultiple=2, **specs):
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # illegal value checks
        assert missionLife >= 0, "Need missionLife >= 0, got %f"%missionLife
        assert extendedLife >= 0, "Need extendedLife >= 0, got %f"%extendedLife
        # arithmetic on missionPortion fails if it is outside the legal range
        assert missionPortion > 0 and missionPortion <= 1, \
                "Require missionPortion in the interval ]0,1], got %f"%missionPortion
        
        # set up state variables
        # tai scale specified because the default, utc, requires accounting for leap
        # seconds, causing warnings from astropy.time when time-deltas are added
        self.missionStart = Time(float(missionStart), format='mjd', scale='tai')#The mission start in MJD
        self.missionLife = float(missionLife)*u.year#length of the mission from start to finish in years
        #self.extendedLife = float(extendedLife)*u.year#additional mission time beyond missionLife in years
        #self.missionPortion = float(missionPortion)
        
        # set values derived from quantities above
        #self.missionFinishNorm = self.missionLife.to('day') + self.extendedLife.to('day')
        self.missionEnd = self.missionStart + self.missionLife#The end time of the mission in MJD #self.missionFinishAbs = self.missionStart + self.missionLife + self.extendedLife
        
        # initialize values updated by functions
        self.tSinceMissionStart = 0.*u.day#the time elapsed since mission start #self.currentTimeNorm = 0.*u.day#the current time
        self.currentTimeAbs = self.missionStart#the current time in mjd
        
        # initialize observing block times arrays
        self.OBnumber = 0 #number of detection observations made
        #self.OBduration = float(OBduration)*u.day
        #self.OBstartTimes = [0.]*u.day
        #maxOBduration = self.missionFinishNorm*self.missionPortion
        #self.OBendTimes = [min(self.OBduration, maxOBduration).to('day').value]*u.day
        
        # initialize single observation START and END times
        #self.obsStart = 0.*u.day
        #self.obsEnd = 0.*u.day
        
        # initialize wait parameters
        #self.waitTime = float(waitTime)*u.day
        #self.waitMultiple = float(waitMultiple)
        
        # populate outspec
        for att in self.__dict__.keys():
            if att not in ['vprint']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat,(u.Quantity,Time)) else dat

        #initialize the Event Stack
        self.initEventStack(specs)

    def __str__(self):
        r"""String representation of the TimeKeeping object.
        
        When the command 'print' is used on the TimeKeeping object, this 
        method prints the values contained in the object."""
        
        for att in self.__dict__.keys():
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'TimeKeeping instance at %.6f days' % self.currentTimeNorm.to('day').value

    def initEventStack(self,specs):
        """Initialize the EventStack
        """
        EventStack = list()#create EventStack and append the event to it
        #create missionStart Event
        EventStack.append({'inst':'sim','tEstart':self.missionStart,'tEend':self.missionStart,'state':'missionEnd'})#appends event to the event stack
        #create missionEnd Event
        EventStack.append({'inst':'sim','tEstart':self.missionEnd,'tEend':self.missionEnd,'state':'missionEnd'})#appends event to the event stack
        
        #create TK global attribute EventStack for use in createEvent()
        self.EventStack = EventStack
        #create "other" instrument events
        try:
            inst0startTimes = specs['telescopeInUse']['startTimes']#defined relative to missionStart
            inst0endTimes = specs['telescopeInUse']['endTimes']#defined relative to missionStart
            for i in np.arange(0,startTimes.shape[0]):
                createEvent('inst0',startTimes[i]+self.missionStart,endTimes[i]+self.missionStart,'other')
        except:
            pass
        try:
            blockPeriod = specs['telescopeInUse']['blockPeriod']#block repitition period in days
            blockDuration = specs['telescopeInUse']['blockDuration']#block duration in days
            assert blockPeriod >= blockDuration, "blockPeriod MUST be greater than blockDuration"
            for i in np.arange(0,math.floor(blockPeriod/self.missionLife.to('day').value)):#iterate through number of exclusion blocks created
                createEvent('inst0',(i+1)*blockPeriod-blockDuration+self.missionStart,(i+1)*blockPeriod+self.missionStart,'other')
        except:
            pass

    def get_EventStack(self):
        """Returns the current Event Stack
        Return:
            EventStack[{'inst':'instName','tStart':startTime,'tEnd':endTime,'state':'opType'},...,{'inst','tStart','tEnd','state'}]
                - List[Dict] of Events
        """
        return self.EventStack

    def get_tEstarts_tEends(self):
        """get all Event start and end times
        Returns:
            tEstarts[#events] - all Event Start Times
            tEends[#events] - all Event End Times
        """
        tEstarts = [EventStack[i]['tEstart'] for i in np.arange(0,len(EventStack))]
        tEends = [EventStack[i]['tEend'] for i in np.arange(0,len(EventStack))]
        return tEstarts, tEends

    def createEvent(self,inst,tEstartn,tEendn,opType='other'):
        """Creates a new event in the event stack
        Args:
            inst - instrument the event is being scheduled for
            tEstart - starting time of the event in mjd
            tEend - end time of the event in mjd
            opType - spacecraft state during event i.e. 'detecting' 'characterizing' 'other'
        """
        #Check validity of proposed event
        assert isinstance(tEstartn, (int, long, float)), "tEstartn is not a number"
        assert isinstance(tEendn, (int, long, float)), "tEendn is not a number"
        assert isinstance(inst, basestring), "inst is not a string"#Python 3.x version isinstance(s, str)
        assert isinstance(opType, basestring), "inst is not a string"#Python 3.x version isinstance(s, str)
        assert tEstart >= self.currentTimeAbs, "Need tEstart >= %f, got %f"%(self.currentTimeAbs, tEstart)#new event must occur after current time
        assert tEstart <= tEend, "Need tEstart <= %f, got %f"%(tEend, tEstart)#the end of the new event must occur at or after the start of the event
        [tEstarts, tEends] = get_tEstarts_tEends()
        for i in np.arange(0,len(tEends)):
            assert(not (tEstarts[i] <= tEstartn <= tEends[i]),"Need NOT(%f < tEstartn < %f) where i=%f, got %f" % (tEstarts[i], i, tEends[i], tEstartn))#start of new event must occur outside bounds of existing events
            assert(not (tEstarts[i] <= tEendn <= tEends[i]),"Need NOT(%f < tEendn < %f) where i=%f, got %f" % (tEstarts[i], i, tEends[i], tEendn))#end of new event must occur outside bounds of existing events
            assert(not ((tEstartn < tEstarts[i]) and (tEends[i] < tEendn)), "Need NOT((tEstartn < %f) && (%f < tEendn)) where i=%f, got tEstartn=%f and tEendn=%f" % (tEstarts[i], tEends[i], i, tEstartn, tEendn)) #new event cannot span an existing event
        
        #Append the Event to the EventStack
        try:    
            self.EventStack.append({'inst':inst,'tStart':tEstartn,'tEnd':tEendn,'state':opType})#appends event to the event stack
        except:
            self.EventStack = list()#create EventStack and append the event to it
            self.EventStack.append({'inst':inst,'tStart':tEstartn,'tEnd':tEendn,'state':opType})#appends event to the event stack

    def deleteEvent(self,inst,tEstart,tEend,opType):
        """Deletes event with specified tEstart and tEend
        Args:
            inst - instrument name in EventStack
            tEstart - event start time in EventStack
            tEend - event end time in EventStack
            opType - spacecraft state
        """
        #find index in EventStack with tEstart and tEend
        def findEventIndex(EventStack, key, value):
            for i, dic in enumerate(EventStack):
                if dic[key] == value:
                    return i
            return -1
        myIndex = findEventIndex(self.EventStack,'tEstart',tEstart)
        myIndex2 = findEventIndex(self.EventStack,'tEend',tEend)
        myIndex3 = findEventIndex(self.EventStack,'inst',inst)
        myIndex4 = findEventIndex(self.EventStack,'opType',opType)

        assert myIndex == myIndex2 == myIndex3 == myIndex4, "The indicies of these do not match (there may be multiple of the same event)"
        assert not (myIndex == myIndex2 == myIndex3 == myIndex4 == -1), "This event does not exist in the EventStack"
        #delete index in EventStack with index
        self.EventStack.pop([myIndex])

    def get_nextEvent(self):
        """Finds the next Event in the stack and returns all details
        Returns:
            {}
        """
        def findEventIndex(EventStack, key, value):
            for i, dic in enumerate(EventStack):
                if dic[key] == value:
                    return i
            return -1

        #[tEstarts, tEends] = self.get_tEstarts_tEends()#retrieve list of event start and end times
        #technically min(tEstarts) should get the correct time
        eventIndex = findEventIndex(self.EventStack,'tEstart',self.tSinceMissionStart.to('day').value)

        return self.EventStack[eventIndex]

    def mission_is_over(self):
        r"""Is the time allocated for the mission used up?
        
        This supplies an abstraction around the test:
            (currentTimeNorm > missionFinishNorm)
        so that users of the class do not have to perform arithmetic
        on class variables.
        
        Returns:
            is_over (Boolean):
                True if the mission time is used up, else False.
        """
        
        is_over = (self.tSinceMissionStart >= self.missionLife)
        
        return is_over

    # def wait(self):
    #     """Waits a certain time in case no target can be observed at current time.
        
    #     This method is called in the run_sim() method of the SurveySimulation 
    #     class object. In the prototype version, it simply allocate a temporal block 
    #     of 1 day.
        
    #     """
    #     self.allocate_time(self.waitTime)

    # def allocate_time(self, dt):
    #     r"""Allocate a temporal block of width dt, advancing to the next OB if needed.
        
    #     Advance the mission time by dt units. If this requires moving into the next OB,
    #     call the next_observing_block() method of the TimeKeeping class object.
        
    #     Args:
    #         dt (astropy Quantity):
    #             Temporal block allocated in units of day
        
    #     """
        
    #     if dt == 0:
    #         return
            
    #     self.currentTimeNorm += dt
    #     self.currentTimeAbs += dt
        
    #     if not self.mission_is_over() and (self.currentTimeNorm 
    #             >= self.OBendTimes[self.OBnumber]):
    #         self.next_observing_block()

    # def next_observing_block(self, dt=None):
    #     """Defines the next observing block, start and end times.
        
    #     This method is called in the allocate_time() method of the TimeKeeping 
    #     class object, when the allocated time requires moving outside of the current OB.
        
    #     If no OB duration was specified, a new Observing Block is created for 
    #     each observation in the SurveySimulation module. 
        
    #     """
        
    #     # number of blocks to wait
    #     nwait = (1 - self.missionPortion)/self.missionPortion
        
    #     # For the default case called in SurveySimulation, OBendTime is current time
    #     # Note: the next OB must not happen after mission finish
    #     if dt is not None:
    #         self.OBendTimes[self.OBnumber] = self.currentTimeNorm
    #         nextStart = min(self.OBendTimes[self.OBnumber] + nwait*dt, 
    #                 self.missionFinishNorm)
    #         nextEnd = self.missionFinishNorm
    #     # else, the OB duration is a fixed value
    #     else:
    #         dt = self.OBduration
    #         nextStart = min(self.OBendTimes[self.OBnumber] + nwait*dt, 
    #                 self.missionFinishNorm)
    #         maxOBduration = (self.missionFinishNorm - nextStart)*self.missionPortion
    #         nextEnd = nextStart + min(dt, maxOBduration)
        
    #     # update OB arrays
    #     self.OBstartTimes = np.append(self.OBstartTimes.to('day').value, 
    #             nextStart.to('day').value)*u.day
    #     self.OBendTimes = np.append(self.OBendTimes.to('day').value, 
    #             nextEnd.to('day').value)*u.day
    #     self.OBnumber += 1
        
    #     # If mission is not over, move to the next OB, and update observation start time
    #     self.allocate_time(nextStart - self.currentTimeNorm)
    #     if self.mission_is_over():
    #         self.OBstartTimes = self.OBstartTimes[:-1]
    #         self.OBendTimes = self.OBendTimes[:-1]
    #         self.OBnumber -= 1
    #     else:
    #         self.obsStart = nextStart
    #         self.vprint('OB%s: previous block was %s long, advancing %s.'%(self.OBnumber+1, 
    #                 dt.round(2), (nwait*dt).round(2)))
