"""Purpose: Plot Observation Timeline for mission

Written by Dean Keithly on 23 Apr, 2018
"""
"""Example 1
I have 1000 pkl files in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl and
1qty outspec file in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json

To generate timelines for these run the following code from an ipython session
from ipython
%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl' \
'/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
"""
"""Example 2
I have several folders with foldernames /home/dean/Documents/SIOSlab/*fZ*OB*PP*SU*/
each containing ~1000 pkl files and 1 outspec.json file

To plot a random Timeline from each folder, from ipython
%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/' None
"""
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run136726516274.pkl' '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json


#IF GIVEN FILEPATH
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Mission Timeline Figures")
    parser.add_argument('pklfile', nargs=1, type=str, help='Full path to pkl file (string).')
    parser.add_argument('outspecfile', nargs=1, type=str, help='Full path to outspec file (string).')

    args = parser.parse_args()
    pklfile = args.pklfile[0]
    outspecfile = args.outspecfile[0]

    if not os.path.exists(pklfile):
        raise ValueError('%s not found'%pklfile)

    #Given Filepath for pklfile, Plot a pkl from each testrun in subdir
    pklPaths = list()
    pklfname = list()
    outspecPaths = list()
    if(os.path.isdir(pklfile)):
        #Look for all directories in specified path with structured folder name
        fp1 = pklfile
        dirs = [myString for myString in next(os.walk(fp1))[1] if 'SU' in myString \
            and 'PP' in myString \
            and 'OB' in myString \
            and 'fZ' in myString \
            and 'RS' in myString]  # Folders containing Monte Carlo Runs

        for i in np.arange(len(dirs)):
            pklFiles = [myFileName for myFileName in os.listdir(fp1+dirs[i]) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
            pklfname.append(np.random.choice(pklFiles))
            pklPaths.append(fp1 + dirs[i] + '/' + pklfname[i])  # append a random pkl file to path
            outspecPaths.append(fp1 + dirs[i] + '/' + 'outspec.json')
    elif(os.path.isfile(pklfile)):
        dirs = [pklfile.split('/')[-2]]
        pklfname.append(pklfile.split('/')[-1].split('.')[0])
        pklPaths.append(pklfile)#append filepath provided in args
        outspecPaths.append(outspecfile)#append filepath provided in args

    #Iterate over all pkl files
    for cnt in np.arange(len(pklPaths)):
        try:
            with open(pklPaths[cnt], 'rb') as f:#load from cache
                DRM = pickle.load(f)
        except:
            print('Failed to open pklfile %s'%pklPaths[cnt])
            pass
        try:
            with open(outspecPaths[cnt], 'rb') as g:
                outspec = json.load(g)
        except:
            print('Failed to open outspecfile %s'%outspecPaths[cnt])
            pass

        arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
        sumOHTIME = 1
        det_times = [DRM['DRM'][i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM['DRM']))]
        det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM['DRM']))]
        ObsNums = [DRM['DRM'][i]['ObsNum'] for i in np.arange(len(DRM['DRM']))]
        y_vals = np.zeros(len(det_times)).tolist()
        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
        print(sum(det_times))


        #Check if plotting font #########################################################
        tmpfig = plt.figure(figsize=(30,3.5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "Obs#   ,  d", ha='center',va='center',rotation='vertical', fontsize=8)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        Obstxtwidth = bb.width#Width of text
        Obstxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        plt.show(block=False)
        plt.close()
        daysperpixelapprox = max(arrival_times)/FIGwidth#approximate #days per pixel
        if mean(det_times)*0.8/daysperpixelapprox > Obstxtwidth:
            ObstextBool = True
        else:
            ObstextBool = False

        tmpfig = plt.figure(figsize=(30,3.5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "OB#  , dur.=    d", ha='center',va='center',rotation='horizontal', fontsize=12)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        OBtxtwidth = bb.width#Width of text
        OBtxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        plt.show(block=False)
        plt.close()
        if mean(OBdurations)*0.8/daysperpixelapprox > OBtxtwidth:
            OBtextBool = True
        else:
            OBtextBool = False
        #################################################################################



        colors = 'rb'#'rgbwmc'
        patch_handles = []
        fig = plt.figure(figsize=(30,3.5),num=cnt)
        ax = fig.add_subplot(111)

        # Plot All Observations
        ind = 0
        obs = 0
        for (det_time, l) in zip(det_times, ObsNums):
            #print det_time, l
            patch_handles.append(ax.barh(0, det_time, align='center', left=arrival_times[ind],
                color=colors[int(obs) % len(colors)]))
            ind += 1
            obs += 1
            patch = patch_handles[-1][0] 
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            plt.rc('axes',linewidth=2)
            plt.rc('lines',linewidth=2)
            rcParams['axes.linewidth']=2
            rc('font',weight='bold')
            if ObstextBool: 
                ax.text(x, y, "Obs#%d, %dd" % (l,det_time), ha='center',va='center',rotation='vertical', fontsize=8)

        # Plot Observation Blocks
        patch_handles2 = []
        for (OBnum, OBdur, OBstart) in zip(xrange(len(outspec['OBendTimes'])), OBdurations, np.asarray(outspec['OBstartTimes'])):
            patch_handles2.append(ax.barh(1, OBdur, align='center', left=OBstart, hatch='//',linewidth=2.0, edgecolor='black'))
            patch = patch_handles2[-1][0] 
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            if OBtextBool:
                ax.text(x, y, "OB#%d, dur.= %dd" % (OBnum,OBdur), ha='center',va='center',rotation='horizontal',fontsize=12)

        # Plot Asthetics
        y_pos = np.arange(2)#Number of xticks to have
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        rcParams['axes.linewidth']=2
        rc('font',weight='bold') 
        ax.set_yticks(y_pos)
        ax.set_yticklabels(('Obs','OB'),fontsize=12)
        ax.set_xlabel('Current Normalized Time (days)', weight='bold',fontsize=12)
        title('Mission Timeline for runName: ' + dirs[cnt] + '\nand pkl file: ' + pklfname[cnt], weight='bold',fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.png')
        savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.svg')
        savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.eps')
