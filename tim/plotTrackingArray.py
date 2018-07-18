#!/usr/bin/env python

# vim: ai:sw=4:sts=4:expandtab

###########################################################
#
#    Analyze clock synchronization
#

from collections import namedtuple
from datetime import datetime, timedelta
from dateutil import tz
import pytz
import argparse
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np
import pdb
import os

sys.path.insert(0,'../')
from swglib.export import DataManager, get_exporter

ENABLE_CACHING = True #TODO: Enable sample set for caching

LEGEND_LOCATION = 'lower left'
TZ = {
        'CP': 'America/Santiago',
        'MK': 'Pacific/Honolulu'
        }
PLOT_ZONE_FILE = './zones.cfg'
MARKERSIZE = 2.5


TCS_SYSTEM = 'tcs'
TCS_CA = 'tcs:drives:driveMCS.VALA'
TCS_CAI = 'tcs:drives:driveMCS.VALI'
MCS_SYSTEM = 'mcs'
MCS_CA = 'mc:followA.J'

# Modes
EXEC_TIME_MODE = 'execTime'
POS_DIFF_MODE = 'posDiff' #shortcut for rawDiff -cols 4,5
VEL_MODE = 'vel'
FULL_MODE = 'full'
ENG_MODE = 'eng' #engineer mode uses VALI array from TCS

PERIOD_MODE = 'period'
RAW_MODE = 'raw'
RAW_DIFF_MODE = 'rawDiff'
CAMON_MODE = 'camon'
CAMON_DIFF_MODE = 'camonDiff'



FollowArray = namedtuple('FollowArray', 'timestamp now targetTime trackId azPos elPos diff')
ZoneArray = namedtuple('ZoneArray', 'title begin end color')
Limits = namedtuple('Limits', 'lower upper')
DebugArray = namedtuple('DebugArray', 'timestamp now targetTime deltaApplyT period sem1 sem2 dmdCnt dmdCorr dmdLowCorr dmdHighCorr flCnt available azPos elPos')

'''
** Element 0  = Time at which commands is sent to MCS
** Element 1  = traw from Fast Loop
** Element 2  = Tick from Fast Loop
** Element 3  = Time to execute demand since calculation (tick - traw)
** Element 4  = Time between FastLoop executions
** Element 5  = Time it takes to take and release reading semaphore on tcsFastLoop
** Element 6  = Time it takes to take and release writing semaphore on tcsFastLoop
** Element 7  = Demand Number
** Element 8  = Tick Correction Number
** Element 9  = correction under 20[ms]
** Element 10  = correction over 80[ms]
** Element 11  = Fast Loop Fault counter
** Element 12  = Available
** Element 13  = demanded Azimuth
** Element 14  = demanded Elevation
*/

    FLinfo[0] = tnow;
    FLinfo[1] = traw;
    FLinfo[2] = tick;
    FLinfo[3] = applyDeltaT;
    FLinfo[4] = deltaTraw;
    FLinfo[5] = deltaTick;
    FLinfo[6] = dtGetTelRD;
    FLinfo[7] = dmd_cnt;
    FLinfo[8] = corr_cnt;
    FLinfo[9] = corr_02;
    FLinfo[10] = corr_08;
    FLinfo[11] = flt_cnt;
    FLinfo[12] = 0.0;
    FLinfo[13] = azdef * R2D;
    FLinfo[14] = eldef * R2D;

'''

# Site should be either 'cp' or 'mk'
SITE = 'cp'
if SITE == 'cp':
    # directory where the data is located
    root_data_dir = '/archive/tcsmcs/data'
    if not os.path.exists(root_data_dir):
        root_data_dir = '/net/cpostonfs-nv1/tier2/gem/sto/archiveRTEMS/tcsmcs/data'
else:
    raise NotImplementedError("The script hasn't still been adapted for MK")

timezone = pytz.timezone(TZ['CP'])

DEBUG = False

def parse_args():
    yesterday = (datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d")
    enableDebugDate = "debug"

    parser = argparse.ArgumentParser(description='Filter PMAC Error data')
    parser.add_argument('-mk',     '--mk',         dest='maunakea', action='store_true', help='The site is Mauna Kea (affects timezone). Default is Cerro Pachon')
    parser.add_argument('-sys',    '--system',     dest='system',   default='tcs', help='System to be analyzed: tcs, mcs or both')
    parser.add_argument('-date',   '--date',       dest='date',     default=enableDebugDate, help='Date - format YYYY-MM-DD')
    parser.add_argument('-day',    '--eng_mode',   dest='eng_mode', action='store_true', help='If used daytime will be analyzed 6am-6pm')
    parser.add_argument('-scale',  '--scale',      dest='scale',    type=float,  default=1.0, help='Scale to be applied to data')
    parser.add_argument('-cols',   '--columns',    dest='cols',     default='4,5', help='Columns to be plotted')
    parser.add_argument('-axis',   '--axis',       dest='axis',     default='az', help='Axis to be plotted, can be az or el')
    parser.add_argument('-time',   '--time_range', dest='time_rng', default=None, help='Specific time range format: HHMM-HHMM')
    
    parser.add_argument('-mode',   '--plot_mode',  dest='mode',          default=EXEC_TIME_MODE,
                        help='Different ways of representing the data, could be: {0}, {1} or {2}'.format(EXEC_TIME_MODE, POS_DIFF_MODE, PERIOD_MODE))
    parser.add_argument('-cutoff', '--cutoff',     dest='cutoff',   type=float, default=None, help='Cut the plots a this many sigmas (default, None)')

    args = parser.parse_args()

    if ENABLE_CACHING: #Uses swglib
        if args.eng_mode:
            args.time_rng = '0800-2000'
        else:
            if not args.time_rng:
                args.time_rng = '2000-0800'
    else:
        day_str = "day_" if args.eng_mode else ""
        system = MCS_SYSTEM
        PV = MCS_CA
        if args.system == 'tcs':
            system = TCS_SYSTEM
            PV = TCS_CA
    
        #Get directory name
        pvdir = PV[PV.find(':')+1:].replace(':','_')
        
        #Construct the path to the data    
        args.data_path = os.path.join(
                root_data_dir, SITE, system,
                pvdir, 
                'txt', 
                '{0}_{1}_{2}_{3}export.txt'.format(args.date, SITE, PV.replace(':','-'), day_str)
            )
    
        if args.system == 'both':
            #Construct the path to the TCS data
            system = TCS_SYSTEM
            PV = TCS_CA
            pvdir = PV[PV.find(':')+1:].replace(':','_')
            args.tcs_data_path = os.path.join(
                root_data_dir, SITE, system,
                pvdir, 
                'txt', 
                '{0}_{1}_{2}_{3}export.txt'.format(args.date, SITE, PV.replace(':','-'), day_str)
            )
    
    
        if (args.date == enableDebugDate):
            print "DEBUG MODE ENABLED -\_(\")_/-"
            args.data_path = (
                args.data_path.replace(enableDebugDate, "2018-04-08")
                    .replace(".txt", "_test.txt")
                )                  

        print "Reading: {0}".format(args.data_path)
    return args


def fromtimestampTz(ts):
    return datetime.fromtimestamp(ts, timezone)

def strptimeTz(dateStr, US=False):
    if US:
        when = datetime.strptime(dateStr, '%m/%d/%Y %H:%M:%S.%f')
    else:
        when = datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S')
    return timezone.localize(when)
    
def localizeNp64Tz(np64dt):
    """
    Converts from numpy.datetime64 to standard datetime with TZ
    """
    ts = (np64dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return fromtimestampTz(ts)

"""
If part of the zone is inside the range adjust limits
if its not inside the plotted range None is returned
"""
def insideRange(begin, end, zone):
    if begin > zone.end or end < zone.begin:
        return None
    
    retZone = ZoneArray(zone.title, zone.begin, zone.end, zone.color)
    if zone.begin < begin:
        retZone = retZone._replace(begin=begin)
    if zone.end > end:
        retZone = retZone._replace(end=end)
        
    return retZone
    


def getTimes(dateStr, timeRangeStr):
    try:
        beginT, endT = timeRangeStr.split('-')
        beginH, beginM = beginT[:-2], beginT[-2:]
        endH, endM = endT[:-2], endT[-2:]
        begin = strptimeTz(dateStr+' '+beginH+':'+beginM+':00')
        end = strptimeTz(dateStr+' '+endH+':'+endM+':00')
        if end < begin:
            end = end + timedelta(days=1)
        return begin, end
    except Exception as ex:
        print 'Wrong format for timerange, must be: HHMM-HHMM'
        print ex
        raise
        
    

"""
Timestamp
** Element 0  = time now
** Element 1  = target time
** Element 2  = track identifier
** Element 3  = demanded Azimuth
** Element 4  = demanded Elevation
"""

def producer(filename):
    with open(filename) as source:
        reader = csv.reader(source, delimiter='\t')
        # Skip the headers
        next(reader); next(reader); next(reader); next(reader)

        for row in reader:
            if row[0].startswith('#') or row[1].startswith('#'):
                continue
            receptionTime = strptimeTz(row[0][:-3], True)
            generationTime = fromtimestampTz(float(row[1]))
            targetTime = fromtimestampTz(float(row[2]))
            yield FollowArray(receptionTime, generationTime,\
                 targetTime,  fromtimestampTz(float(row[3])), float(row[4]), float(row[5]), targetTime-receptionTime \
                 )
            
def producerH(pv, dateStr, timeRangeStr):
    
    begin, end = getTimes(dateStr, timeRangeStr)
    print 'Times:', begin, end
    dm = DataManager(get_exporter(SITE.upper()), root_dir='/tmp/rcm/') 
    data = dm.getData(TCS_CA, begin, end)

    for val in data:
            receptionTime = localizeNp64Tz(val[0])
            generationTime = fromtimestampTz(val[1])
            targetTime = fromtimestampTz(val[2])          
            yield FollowArray(receptionTime, generationTime, targetTime, val[3], val[4], val[5], targetTime-receptionTime)
            
    #TODO: Return generator
    #TODO: Implement timezone awareness on the swglib side
    
def getProducer():
    if ENABLE_CACHING:
        return producerH(TCS_CA, args.date, args.time_rng)
    else:
        return producer(args.data_path)

#DebugArray = namedtuple('DebugArray', 'timestamp now targetTime deltaApplyT period sem1 sem2 dmdCnt dmdCorr dmdLowCorr dmdHighCorr flCnt available azPos elPos')
def producerE(pv, dateStr, timeRangeStr):
    
    begin, end = getTimes(dateStr, timeRangeStr)
    print 'Times:', begin, end
    dm = DataManager(get_exporter(SITE.upper()), root_dir='/tmp/rcm/') 
    data = dm.getData(TCS_CAI, begin, end)
    for val in data:
            receptionTime = localizeNp64Tz(val[0])
            generationTime = fromtimestampTz(val[1])
            targetTime = fromtimestampTz(val[2]) 
            yield DebugArray(receptionTime, generationTime, targetTime, val[4], val[5], val[6], val[7],val[8],val[9],val[10],val[11],val[12],val[13],val[14],val[15])



def readZonesFromFile(begin, end):
    markedZones = []
    try:
        with open(PLOT_ZONE_FILE) as source:
            reader = csv.reader(source, skipinitialspace=True)
            for row in reader:
                if len(row)>3:
                    za = ZoneArray(row[0].strip(), strptimeTz(row[1].strip()), strptimeTz(row[2].strip()), row[3].strip())  
                    za = insideRange(begin, end, za)
                    if za:
                        markedZones.append(za)
    except Exception as ex:
        print "Exception reading zones file:"
        print ex
        pass
    return markedZones

def addZones(ax, begin, end):
    markedZones = readZonesFromFile(begin, end)    
    
    for ii, za in enumerate(markedZones):
        ax.axvspan(za.begin, za.end, facecolor=za.color, label=za.title, alpha=0.3)
    
    #ax.legend(loc=LEGEND_LOCATION)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

class YLimitCalculator(object):
    def __init__(self, sigmas):
        self.sigmas = sigmas

    def __call__(self, axis, data, offset=None):
        if not self.sigmas:
            return

        data = np.array(data)
        avg, std = data.mean(), data.std()
        smax, smin = avg + (std * self.sigmas), avg - (std * self.sigmas)
        if smax < smin:
            smax, smin = smin, smax
        mn, mx = data.min(), data.max()
        if (smin < mn < smax) and (smin < mx < smax):
            if offset is None:
                return
            smin = mn - abs(mn)*0.15
            smax = mx + abs(mx)*0.15

        if self.sigmas is None:
            amin = min(smin, mn - abs(mn * 0.15))
            amax = max(smax, mx + abs(mn * 0.15))
        else:
            amin = max(smin, mn - abs(mn * 0.15))
            amax = min(smax, mx + abs(mn * 0.15))

        if offset is not None:
            amin += offset
            amax += offset

        axis.set_ylim(amin, amax)

def plotPeriod():
    """
    Here we are measuring periodicity between executions
    and periodicity (or duration) between consecutive targetTime calculations
    """
    flw_producer = getProducer()
    first = flw_producer.next()
    diffNow_lst, diffTarget_lst = list(), list()
    prevNowT, prevTargetT = first.now, first.targetTime
    outliersInPeriod = 0
    periodLimits = Limits(-1,1)
    
    when = strptimeTz("2017-04-26 11:18:00")#TODO implement this as a argument
    
    for dp in flw_producer:
        tBetweenExecutions = (dp.now-prevNowT).total_seconds()
        tBetweenTargets = (dp.targetTime-prevTargetT).total_seconds()
        if(when<dp.timestamp):
            if tBetweenExecutions > periodLimits.upper or tBetweenExecutions < periodLimits.lower:
                print "tBetweenExecutions={0} tBetweenTargets={1} at GEA time: {2}".format(
                        tBetweenExecutions, tBetweenTargets, dp.timestamp)
                outliersInPeriod += 1
            else:
                diffNow_lst.append((dp.timestamp, tBetweenExecutions*1000.0))
                diffTarget_lst.append((dp.timestamp, tBetweenTargets*1000.0))
        prevNowT, prevTargetT = dp.now, dp.targetTime
         
    print "Last read line with date:", dp.timestamp
    periodTime,periodVal=zip(*diffNow_lst)
    
    fig, ax1 = plt.subplots()   
    plt.title("TCS-MCS Communication Analysis: {0} data from {1}".format(args.system, args.date))
    ax1.plot(periodTime, periodVal, "b.", markersize=MARKERSIZE)
    ax1.grid(True)
    ax1.tick_params("y", colors="b")
    ax1.set_ylabel("timeNow - previous timeNow (milliseconds)", color="b")   
    
    #if outliersInPeriod > 0:
        #ax1.set_ylim(*periodLimits)
    #    print "Fixed scaling due to outliers"
    #else:
    #print "Rescaling Period axis due to no outliers in sample set"
    #ax1.set_yticks(np.arange(-0.05, 0.05, step=0.005))
    #ax1.axvspan(when, when+timedelta(minutes=230), facecolor='0.2', alpha=0.3)    
    
    if True: #Plot also diffs in target time
        targetTime, targetVal=zip(*diffTarget_lst)
    
        ax2 = ax1.twinx()
        ax2.plot(targetTime, targetVal, "r.", markersize=MARKERSIZE)
        ax2.tick_params("y", colors="r")
        ax2.set_ylabel("targetTime - previous targetTime (milliseconds)", color="r")
    
    print "Number of detected period outliers: {0}".format(outliersInPeriod)
    
    addZones(ax1, periodTime[0], periodTime[-1])
    plt.gcf().autofmt_xdate()
    plt.show()


def getIndexes(indexStr):
    elements = indexStr.split(",")
    i = int(elements[0])
    if len(elements) > 1 :
        j = int(elements[1])
    else:
        j = 0
    return i,j
    
def plotRaw(diff=False):
    '''
    Plots the data as it is, if diff flag is true it subtracts 2 consecutive
    values. [a b c] diff >> [b-a c-b]
    '''
    flw_producer = getProducer()
    i,j = getIndexes(args.cols)
    
    timeBase, aLst, bLst = list(), list(), list()
    for dp in flw_producer:
        timeBase.append(dp.timestamp)
        aLst.append(dp[i]*args.scale)
        if j > 1:
            bLst.append(dp[j]*args.scale)
            
    if diff:
        aLst = np.diff(np.array(aLst))
        bLst = np.diff(np.array(bLst))
        timeBase = timeBase[1:]

#Plotting starts here
    fig, ax1 = plt.subplots()   
    plt.title("Title to be defined by the user, date:{1}".format(args.system, args.date))
    ax1.plot(timeBase, aLst, "b.", markersize=MARKERSIZE)
    ax1.grid(True)
    ax1.tick_params("y", colors="b")
    ax1.set_ylabel("To be defined by the user", color="b")
    
    if j > 1:
        ax2 = ax1.twinx()
        ax2.plot(timeBase, bLst, "r.-", markersize=MARKERSIZE)
        ax2.tick_params("y", colors="r")
        ax2.set_ylabel("TBD", color="r")
    
    addZones(ax1, timeBase[0], timeBase[-1])
    plt.gcf().autofmt_xdate()
    plt.show()

def plotExecTime():
    '''
    Here we plot the time that is left until execution time (time before target)
    you can chooose between tcs, mcs or both.
    '''
    flw_producer = getProducer()
        
    diff_lst = list()
    for dp in flw_producer:
        #print dp.targetTime-dp.now
        diff_lst.append((dp.timestamp, dp.diff.total_seconds()))
    
    diffTime,diffVal=zip(*diff_lst)
    
    fig, ax1 = plt.subplots()   
    plt.title("TCS-MCS Communication Analysis: {0} data from {1}\n Time before target time".format(args.system, args.date))
    ax1.plot(diffTime, diffVal, "b.", markersize=MARKERSIZE)
    ax1.grid(True)
    #ax1.set_ylim([-0.05,0.05])
    #ax1.set_yticks(np.arange(-0.05, 0.05, step=0.005))
 
    if args.system == 'both':
        flw_producer2 = producer(args.tcs_data_path)
        #TODO: Fix legend issues when plotting both subsystems
        diff_lst2 = list()
        for dp in flw_producer2:
            diff_lst2.append((dp.timestamp, dp.diff.total_seconds()))
    
        diffTime2,diffVal2=zip(*diff_lst2)    
    
        #ax2 = ax1.twinx()
        ax1.plot(diffTime2, diffVal2, "r.", markersize=MARKERSIZE)
        plt.gca().legend((MCS_SYSTEM+' entry', TCS_SYSTEM+' exit'))
    
    addZones(ax1, diffTime[0],diffTime[-1])
    plt.gcf().autofmt_xdate()
    plt.show()


def plotPosDiff():
    flw_producer = getProducer()

    firstVal = flw_producer.next()
    outliersInPeriod = 0
    periodLimits = Limits(-0.1,0.7)
    prevAz = firstVal.azPos
    prevEl = firstVal.elPos
    prevTimestamp = firstVal.targetTime
    az_lst, el_lst = list(), list()
    for dp in flw_producer:
        tBetweenSamples = (dp.targetTime-prevTimestamp).total_seconds()
        az_lst.append((dp.targetTime, (prevAz - dp.azPos)*3600.0))
        el_lst.append((dp.targetTime, (prevEl - dp.elPos)*3600.0))
        prevAz = dp.azPos
        prevEl = dp.elPos
        prevTimestamp = dp.targetTime
        if tBetweenSamples > periodLimits.upper or tBetweenSamples < periodLimits.lower:
            print "Period out of limits: {0} on date: {1}".format(tBetweenSamples, dp.timestamp)
            outliersInPeriod += 1
            
    print "Last read line with date:", dp.timestamp, dp.now, dp.targetTime
    azTime,azVal=zip(*az_lst)
    elTime,elVal=zip(*el_lst)
    
    
    fig, ax1 = plt.subplots()   
    plt.title("TCS-MCS Communication Analysis: {0} data from {1}".format(args.system, args.date))

    #Plot AZ
    ax1.plot(azTime, azVal, "b.-", markersize=MARKERSIZE)
    ax1.grid(True)
    ax1.tick_params("y", colors="b")
    ax1.set_ylabel("Azimuth difference between samples\n(not respecting timestamp - velocity)", color="b")
    conditionallySetYLim(ax1, azVal, offset = abs(max(azVal)) * 0.05)
    
    #Plot EL
    elTime, elVal=zip(*el_lst)
    ax2 = ax1.twinx()
    ax2.plot(elTime, elVal, "r.-", markersize=MARKERSIZE)
    ax2.tick_params("y", colors="r")
    ax2.set_ylabel("Elevation", color="r")
    conditionallySetYLim(ax2, elVal, offset = abs(min(elVal)) * -0.05)

    print "Number of detected period outliers: {0}".format(outliersInPeriod)
    print "Watch out: plot ylim set to -10 , 10 some information may not be shown on the graph" #TODO: At least count samples left out and report
    addZones(ax1, azTime[0],azTime[-1])
    plt.gcf().autofmt_xdate()
    plt.show()

def plotVel():
    '''
    Here the calculated velocity vs the raw position difference is plotted
    in 2 subplots sharing the X axis
    '''
    AZ = 'az'
    plotAz = True if args.axis == AZ else False
       
    flw_producer = getProducer()

    firstVal = flw_producer.next()
    outliersInPeriod = 0
    periodLimits = Limits(-0.1,0.7)
    prevPos = firstVal.azPos if plotAz else firstVal.elPos
    posIndex = 4 if plotAz else 5
    prevTimestamp = firstVal.targetTime
    posDiff_lst, vel_lst, timebase = list(), list(), list()
    for dp in flw_producer:
        tBetweenSamples = (dp.targetTime-prevTimestamp).total_seconds()
        if (tBetweenSamples <= 0.0):
            print "Time diff is zero or below: {0}, ignoring values {1}".format(tBetweenSamples, dp.timestamp)
        else:
            posDiff_lst.append((prevPos - dp[posIndex])*3600.0)
            vel = (prevPos - dp[posIndex])/tBetweenSamples
            vel_lst.append(vel*3600.0)
            timebase.append(dp.targetTime)
        prevPos = dp[posIndex]
        prevTimestamp = dp.targetTime
        if tBetweenSamples > periodLimits.upper or tBetweenSamples < periodLimits.lower:
            print "Period out of limits: {0} on date: {1}".format(tBetweenSamples, dp.timestamp)
            outliersInPeriod += 1
            
    print "Last read line with date:", dp.timestamp
    
    fig, ax1 = plt.subplots() 
    ax1 = plt.subplot(211)
    plt.title("TCS-MCS Pos./Vel. Analysis: {0} data from {1} - axis:{2}".format(args.system, args.date, args.axis))
    ax1.plot(timebase, posDiff_lst, "r.-", markersize=MARKERSIZE)
    ax1.grid(True)
    ax1.tick_params("y", colors="r")
    ax1.set_ylabel("Raw pos. diff [as]", color="r")
    conditionallySetYLim(ax1, posDiff_lst)

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(timebase, vel_lst, "b.-", markersize=MARKERSIZE)
    ax2.grid(True)
    ax2.tick_params("y", colors="b")
    ax2.set_ylabel("Calculated vel. [as/s]", color="b")
    conditionallySetYLim(ax2, vel_lst)
    
    print "Number of detected period outliers: {0}".format(outliersInPeriod)
    print "Watch out: plot ylim set to -10 , 10 some information may not be shown on the graph" #TODO: At least count samples left out and report

    addZones(ax1, timebase[0],timebase[-1])
    addZones(ax2, timebase[0],timebase[-1])

    #ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m %H:%M:%S.%f"))

    plt.gcf().autofmt_xdate()
    #TODO: Special plot settings need to care about timezone
    """plt.setp(ax1.get_xticklabels(), fontsize=9, visible=False)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S.%f"))
    ax2.xaxis.set_minor_locator(ticker.MaxNLocator(200))
    plt.setp(ax2.get_xticklabels(), fontsize=9, rotation=20, ha='right')
"""    
    plt.show() 

def plotFull():
    '''
    Ignacio and his 10.000 plots on one graph haha
    '''   
    flw_producer = getProducer()

    diff_lst = list()
    exec_lst = list()
    dDmd_lst = list()
    prevTimestamp = flw_producer.next().now
    prevTarget = flw_producer.next().targetTime
    prevElPos = flw_producer.next().elPos
    outliersInPeriod = 0
    periodLimits = Limits(-1,1)

    #FollowArray = namedtuple('FollowArray', 'timestamp now targetTime trackId azPos elPos diff')
    # when = datetime.strptime("2018-04-26 11:18", "%Y-%m-%d %H:%M")

    for dp in flw_producer:
        #print dp.targetTime-dp.now
        #print dp.diff.total_seconds()
        # tBetweenSamples = (dp.now-prevTimestamp).total_seconds()
        tBetweenSamples = dp.now-prevTimestamp
        # tBetweenTargets = dp.targetTime-prevTarget
        tBetweenTargets = dp.targetTime - dp.now
        dDemands = dp.elPos - prevElPos
        diff_lst.append((dp.timestamp, tBetweenSamples.total_seconds()*1000.0))
        exec_lst.append((dp.timestamp, tBetweenTargets.total_seconds()*1000.0))
        dDmd_lst.append((dp.timestamp, dDemands*3600))
        prevTimestamp = dp.now
        prevTarget = dp.targetTime
        prevElPos = dp.elPos
        # if(when<dp.timestamp):
            # if (tBetweenSamples > 1.0):
                # print "tBetweenSamples={0} at GEA time: {1}".format(tBetweenSamples, dp.timestamp)
            # else:
                # diff_lst.append((dp.timestamp, tBetweenSamples*1000.0))
            # exec_lst.append((dp.timestamp, dp.azPos*1000.0))
        # prevTimestamp = dp.now
        #if tBetweenSamples > periodLimits.upper or tBetweenSamples < periodLimits.lower:
        #    print "Period out of limits: {0} on date: {1}".format(tBetweenSamples, dp.timestamp)
        #    outliersInPeriod += 1


    print "Last read line with date:", dp.timestamp

    if args.system=='both':
        mcs_lst = list()
        mcs_producer = producerMCS(args.mcs_data_path)
        for dp in mcs_producer:
            mcs_lst.append((dp.timestamp, dp.pmacError*3600))

    # PLOTTING SECTION
    # Plot 1
    diffTime,diffVal=zip(*diff_lst)

    ax1 = plt.subplot(411)
    plt.title("TCS-MCS Communication Analysis: {0} data from {1}".format(args.system, args.date))
    ax1.plot(diffTime, diffVal, "b.")
    ax1.grid(True)
    ax1.tick_params("y", colors="b")
    ax1.set_ylabel("Diff timeNow [ms]", color="b")
    conditionallySetYLim(ax1, diffVal)

    if outliersInPeriod > 0:
        #ax1.set_ylim(*periodLimits)
        print "Fixed scaling due to outliers"
    else:
        print "Rescaling Period axis due to no outliers in sample set"
    #ax1.set_yticks(np.arange(-0.05, 0.05, step=0.005))
    when = datetime.strptime("2018-04-18 09:45", "%Y-%m-%d %H:%M")
    #ax1.axvspan(when, when+timedelta(minutes=230), facecolor='0.2', alpha=0.3)

    # Plot 2
    execTime, execVal=zip(*exec_lst)

    ax2 = plt.subplot(412, sharex=ax1)
    ax2.grid(True)
    ax2.plot(execTime, execVal, "r.")
    ax2.tick_params("y", colors="r")
    ax2.set_ylabel("Tick - tnow [ms]", color="r")
    conditionallySetYLim(ax2, execVal)

    if args.system=='both':
        # Plot 3
        pmacTime, pmacVal=zip(*mcs_lst)

        ax4 = plt.subplot(413, sharex=ax1)
        ax4.grid(True)
        ax4.plot(pmacTime, pmacVal, "r-")
        ax4.tick_params("y", colors="r")
        ax4.set_ylabel("Elevation Pmac Error [arcsec]", color="r")
        conditionallySetYLim(ax4, pmacVal)

    # Plot 4
    dmdTime, diffDmd=zip(*dDmd_lst)

    ax3 = plt.subplot(414, sharex=ax1)
    ax3.plot(dmdTime, diffDmd, "g.-")
    ax3.grid(True)
    ax3.tick_params("y", colors="b")
    ax3.set_ylabel("Diff EL Demand [arcsec]", color="b")
    conditionallySetYLim(ax3, diffDmd)
    """
    plt.setp(ax1.get_xticklabels(), fontsize=9, visible=False)
    plt.setp(ax2.get_xticklabels(), fontsize=9, visible=False)
    if args.system=='both':        
        plt.setp(ax4.get_xticklabels(), fontsize=9, visible=False)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m %H:%M:%S.%f"))
    ax3.xaxis.set_minor_locator(ticker.MaxNLocator(200))
    plt.setp(ax3.get_xticklabels(), fontsize=9, rotation=20, ha='right')
    plt.gca().xaxis_date(TZ)
#    pdb.set_trace()
    print "Number of detected period outliers: {0}".format(outliersInPeriod)

    ax2.xaxis_date(TZ)    
    ax3.xaxis_date(TZ)
    """
    plt.gcf().autofmt_xdate()
    plt.show()


def plotEng():
    '''
    Here we are using VALI
    '''   
    if ENABLE_CACHING:
        flw_producer = producerE(TCS_CAI, args.date, args.time_rng)
    else:
        print "Not implemented, use caching"
        sys.exit()

    timebase = list()
    delta_lst = list()
    period_lst = list()
    for dp in flw_producer:
        timebase.append(dp.timestamp)
        delta_lst.append(dp.deltaApplyT)
        period_lst.append(dp.period)

    fig, ax1 = plt.subplots() 
    ax1 = plt.subplot(211)
    plt.title("TCS-MCS Period/DeltaT Analysis: {0} data from {1}".format(args.system, args.date))
    ax1.plot(timebase, delta_lst, "r.-", markersize=MARKERSIZE)
    ax1.grid(True)
    ax1.tick_params("y", colors="r")
    ax1.set_ylabel("DeltaT[s.]", color="r")


    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(timebase, period_lst, "b.-", markersize=MARKERSIZE)
    ax2.grid(True)
    ax2.tick_params("y", colors="b")
    ax2.set_ylabel("Period[s.]", color="b")
    
    addZones(ax1, timebase[0],timebase[-1])
    addZones(ax2, timebase[0],timebase[-1])

    plt.gcf().autofmt_xdate()
    plt.show() 


# -----------------------------------------------------------------------------
# ------------------------------------ MAIN -----------------------------------
# -----------------------------------------------------------------------------
args = parse_args()

conditionallySetYLim = YLimitCalculator(sigmas = args.cutoff)

if args.maunakea:
    timezone = pytz.timezone(TZ['MK'])

if args.mode == EXEC_TIME_MODE :
    plotExecTime()

elif args.mode == POS_DIFF_MODE:
    plotPosDiff()

elif args.mode == RAW_MODE:
    plotRaw()

elif args.mode == RAW_DIFF_MODE:
    plotRaw(True)

elif args.mode == PERIOD_MODE:
   plotPeriod()

elif args.mode == VEL_MODE:
    plotVel()
   
elif args.mode == FULL_MODE:
    plotFull()

elif args.mode == ENG_MODE:
    plotEng()
   
else:
    print "Mode not found, check spelling"
