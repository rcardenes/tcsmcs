#!/usr/bin/env python

import numpy as np
from numpy import logical_and as land
from numpy import logical_or as lor
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pandas as pd
import sys, os, fnmatch
import pickle
import argparse
import json
from collections import namedtuple
import pdb

Span = namedtuple('Span', 'first last color alpha')

def parseargs():
    parser = argparse.ArgumentParser(description="Plot PV executiong time jitter")
    parser.add_argument("--list", "-l", action="store_true",
            help="List the available files for a certain site a subsystem, and terminate the script")
    parser.add_argument("--force", "-f", action="store_true",
            help="Rewrite exising plots instead of bypassing them")
    parser.add_argument("site", choices=('MKO', 'CPO'),
            help="Site. Choose one of the available options")
    parser.add_argument("subsys",
            help="Subsystem name. Used to build up the directories names")
    parser.add_argument("source", nargs='*',
            help="If specified, plot only these source files (use the raw file name). Otherwise, all files for the subsystem will be considered. The paths are relative to the subsystem. Force (-f) is implied")
    parser.add_argument("--threshold", "-t", type=float,
            help="Cut-off. Jitter over this value is not plotted on the left axis")
    parser.add_argument("--sigmas", type=float,
            help="No threshold, plot is vertically centered on the mean value, and displays only 'n' sigmas above and under")
    parser.add_argument("--no-outliers", "-n", dest='outliers', action='store_false',
            help="DO NOT plot outliers over the threshold. By default, they're plotted on the right axis, if there is a threshold AND 'sigmas' is not used")
    parser.add_argument("--metadata", "-m",
            help="Path to a file that contains plotting definitions")

    return parser.parse_args()

def list_pkl_in_folder(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            matches.append(os.path.join(root, filename))
    for match in sorted(matches, key=os.path.getmtime):
        yield match[:-4]

def load_per_file_config(metadata, site, subsys):
    config = {
        'titles': {}
    }
    for siteSub in metadata.get('per-site', []):
        if not (siteSub['site'] == site and siteSub['subsys'] == subsys):
            continue
        config['titles'] = siteSub.get('titles', {})
        break

    return config

def get_title_pattern(config, filename):
    return config['titles'].get(filename)

class TimeSpans(object):
    def __init__(self):
        self._cache = {}
        self._data = None

    def set_data(self, data):
        self._data = data

    def to_counts(self, start, end):
        if self._data is None:
            raise RuntimeError("Can't calculate spans without data")

        key = hash(start + end)
        try:
            return self._cache[key]
        except KeyError:
            pass
        dfr, dto = (dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S'),
                    dt.datetime.strptime(end,   '%Y-%m-%d %H:%M:%S'))
        try:
            fr, to = (np.where(self._data >= dfr)[0][0],
                      np.where(self._data <= dto)[0][-1])
        except IndexError:
            return None
        self._cache[key] = (fr, to)

        return (fr, to)

    def vertical_span(self, raw_span):
        try:
            # fr, to = self.to_counts(raw_span['from'], raw_span['to'])
            #return Span(first = fr,
            #            last  = to,
            #            color = raw_span.get('color', 0.2),
            #            alpha = float(raw_span.get('alpha', 0.2)))
            return Span(first = dt.datetime.strptime(raw_span['from'], '%Y-%m-%d %H:%M:%S'),
                        last  = dt.datetime.strptime(raw_span['to'], '%Y-%m-%d %H:%M:%S'), 
                        color = raw_span.get('color', 0.2),
                        alpha = float(raw_span.get('alpha', 0.2)))
        except TypeError:
            return Span(-1, -1, 0, 0)

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

timeSpans = TimeSpans()

args = parseargs()
site = args.site
system = args.subsys
source = args.source

outpath = './'+system+'Png'+site
rawFilePath = './'+system+'Binary'+site

if args.list:
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            print filename
    sys.exit(0)

if not source:
    the_pkl = list_pkl_in_folder(rawFilePath)
else:
    the_pkl = [os.path.join(rawFilePath, fname) for fname in source]
    args.force = True

threshold = args.threshold
sigmas = args.sigmas
plot_outliers = args.outliers
metadata = {}
config = {}

if args.metadata is not None:
    try:
        with open(args.metadata) as md:
            metadata = json.load(md)
            config = load_per_file_config(metadata, args.site, args.subsys)
    except IOError:
        print >> sys.stderr, "Not a valid path: %s" % args.metadata

if 'vertical-spans' not in metadata:
    metadata['vertical-spans'] = []

print "Looking for pickles inside %s." % rawFilePath

dates = []
exclude_counts = []
r1counts = []
r2counts = []
r3counts = []

for f in the_pkl:
    outfile = "%s/%s.png" % (outpath, os.path.basename(f))
    print outfile
    # Has pickle file been processed inside the binary folder yet?
    if os.path.isfile(outfile) and not args.force:
        print "Will BYPASS %s..." % (f)
        continue
    else:
        ts = pd.read_pickle(open(f + '.pkl', 'r'))
        timeSpans.set_data(ts)
        #print ts
        #myrange = "%s to %s" % (mydata['Timestamp'][0], mydata['Timestamp'][mydata.size-1] )
        myrange = "%s to %s" % (ts[0], ts[ts.size-1])
        dates.append(ts[0].date())
	diffs = np.diff(ts)
	tt1 = diffs / np.timedelta64(1, 's')

        try:
            for gap in metadata['gaps']:
                try:
                    # fr, to = timeSpans.to_counts(gap['from'], gap['to'])
                    metadata['vertical-spans'].append({
                        'from':  gap['from'],
                        'to':    gap['to'],
                        'color': 'r'
                        })
                except TypeError:
                    pass
        except KeyError:
            pass

	#tt2 = movingaverage(tt1, 2)

        #Basic stats on our data
        mu = np.mean(tt1)
        sigma = np.std(tt1)

        print "Dataset %s has:\n\t%d samples\n\tmean %f\n\tstd %f" % (f, len(tt1), mu, sigma)

        #3-Sigma Rule 68, 95, and 99.7
        s1Left = (mu - sigma)
        s1Right = (mu + sigma)
        s2Left = (mu - 2*sigma)
        s2Right = (mu + 2*sigma)
        s3Left = (mu - 3*sigma)
        s3Right = (mu + 3*sigma)

        if threshold is not None:
            # The masks apply to ranges you don't want, they're MASKED!
            #   the result is an array with the values you do want
            #
            # Include points not exceeding the input threshold
            mx_include = ma.masked_array(tt1, mask = (tt1 > threshold))

            # Exclude points that exceed the input threshold
            mx_exclude = ma.masked_array(tt1, mask = (tt1 < threshold))
            exclude_counts.append(mx_exclude.count())
        else:
            mx_include = tt1
            mx_exclude = ma.masked_array([])
            exclude_counts.append(0)

        r1counts.append(ma.masked_array(tt1, land(tt1 >= s1Left, tt1 <= s1Right)).count())
        r2counts.append(ma.masked_array(tt1, lor(land(tt1 <= s1Left, tt1 >= s2Left), land(tt1 >= s1Right, tt1 <= s2Right))).count())
        r3counts.append(ma.masked_array(tt1, lor(land(tt1 <= s2Left, tt1 >= s3Left), land(tt1 >= s2Right, tt1 <= s3Right))).count())

        if threshold is not None:
            print '\tExtreme counts greater than %f = %d' % (threshold, mx_exclude.count())

        print '\tR1C=%d, R2C=%d, R3C=%d' % (r1counts[-1], r2counts[-1], r3counts[-1])

        my_dpi=96
	fig, ax1 = plt.subplots(figsize=(1920/my_dpi, 1200/my_dpi), dpi=my_dpi, facecolor='w', edgecolor='k')

        xdata = ts[1:].to_pydatetime()

        ax1.plot(xdata, mx_include, "b.", markersize=2)
	#plt.plot(tt2,"r")
        ax1.set_xlim(xdata[0], xdata[-1])
	ax1.grid(True)
        title = get_title_pattern(config, os.path.basename(f))
        if title is not None:
            ax1.set_title(title.format(**{
                'srj': "Sample Rate Jitter",
                'rng': "Range %s" % myrange
                }))
        else:
            ax1.set_title("%s %s Sample Rate Jitter\nRange %s " % (site, system, myrange) )
	ax1.set_xlabel("Sample Number")
	ax1.set_ylabel("Delta T (seconds)")

        if sigmas is not None:
            med = np.median(mx_include)
            std = np.std(mx_include)
            ax1.set_ylim(med - (std * 2), med + (std * 2))
        elif threshold is not None:
            ax1.set_ylim(0, threshold)

        ax1.tick_params('y', colors='b')

        if threshold is not None and not sigmas and plot_outliers and mx_exclude.count() > 0:
            ax2 = ax1.twinx()
            ax2.plot(xdata, mx_exclude,"r.", markersize=10)
            ax2.set_xlim(xdata[0], xdata[-1])
            ax2.set_ylabel("Excluded Delta T values (seconds)", color='r')
            ax2.tick_params('y', colors='r')

        first_element = xdata[0]
        last_element = xdata[-1]
        for span in [timeSpans.vertical_span(vs) for vs in metadata.get('vertical-spans', [])]:
            if span.first >= last_element or span.first < first_element:
                continue
            plt.axvspan(span.first, min(span.last, last_element), facecolor=span.color, alpha=span.alpha)

        fig.autofmt_xdate()
	fig.tight_layout()
	fig.savefig(outfile, dpi=my_dpi)
        plt.close(fig)
        #sys.exit()

print "sorting dates..."

#Save data for Full Range historgram to pickle
df = pd.DataFrame({'dates':dates,
                   'counts':exclude_counts,
                   'r1counts':r1counts,
                   'r2counts':r2counts,
                   'r3counts':r3counts})
picklefile= "hist%s_%s_%s.pkl" % (site, system, threshold)
dfFile = open(picklefile, 'w')
pickle.dump(df, dfFile) 
print "done"



