import numpy as np
from numpy import logical_and as land
from numpy import logical_or as lor
import numpy.ma as ma
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import sys, os, fnmatch
import pickle
import argparse
import json
from collections import namedtuple

Span = namedtuple('Span', 'first last color alpha')

def parseargs():
    parser = argparse.ArgumentParser(description="Plot PV executiong time jitter")
    parser.add_argument("--force", "-f", action="store_true",
            help="Rewrite exising plots instead of bypassing them")
    parser.add_argument("--site", "-i", choices=('MKO', 'CPO'), default='MKO')
    megroup = parser.add_mutually_exclusive_group(required=True)
    megroup.add_argument("--subsys", "-s",
            help="Plots all the available data for the given subsystem")
    megroup.add_argument("--source", "-r",
            help="Plot only these source files (comma separated)")
    parser.add_argument("--threshold", "-t", type=float,
            help="Cut-off. Jitter over this value is not plotted on the left axis")
    parser.add_argument("--sigmas", type=float,
            help="No threshold, plot is vertically centered on the mean value, and displays only 'n' sigmas above and under")
    parser.add_argument("--no-outliers", "-n", dest='outliers', action='store_false',
            help="DO NOT plot outliers over the threshold. By default, they're plotted on the right axis, if there is a threshold AND 'sigmas' is not used")
    parser.add_argument("--metadata", "-m",
            help="Path to a file that contains plotting definitions")

    return parser.parse_args()

def returnold(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            matches.append(os.path.join(root, filename))
    return sorted(matches, key=os.path.getmtime)

def vertical_spans(spans, data):
    for span in spans:
        dfr, dto = (dt.datetime.strptime(span['from'], '%Y-%m-%d %H:%M:%S'),
                    dt.datetime.strptime(span['to'],   '%Y-%m-%d %H:%M:%S'))
        try:
            yield(Span(first = np.where(data >= dfr)[0][0],
                       last  = np.where(data <= dto)[0][-1],
                       color = span.get('color', 0.2),
                       alpha = float(span.get('alpha', 0.2))))
        except IndexError:
            pass

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

args = parseargs()
site = args.site
system = args.subsys
threshold = args.threshold
sigmas = args.sigmas
plot_outliers = args.outliers
metadata = {}

if args.metadata is not None:
    try:
        with open(args.metadata) as md:
            metadata = json.load(md)
    except IOError:
        print >> sys.stderr, "Not a valid path: %s" % args.metadata

outpath = './'+system+'Png'+site
rawFilePath = './'+system+'Binary'+site

print "Looking for pickles inside %s." % rawFilePath

dates = []
exclude_counts = []
r1counts = []
r2counts = []
r3counts = []

for f in returnold(rawFilePath):
    outfile = "%s/%s%s" % (outpath, os.path.splitext(os.path.basename(f))[0], '.png')
    print outfile
    # Has pickle file been processed inside the binary folder yet?
    if os.path.isfile(outfile) and not args.force:
        print "Will BYPASS %s..." % (f)
        continue
    else:
        ts = pd.read_pickle(open(f, 'r'))
        #print ts
        #myrange = "%s to %s" % (mydata['Timestamp'][0], mydata['Timestamp'][mydata.size-1] )
        myrange = "%s to %s" % (ts[0], ts[ts.size-1])
        dates.append(ts[0].date())
	diffs = np.diff(ts)
	tt1 = diffs / np.timedelta64(1, 's')
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
	
	ax1.plot(mx_include,"b.", markersize=2)
	#plt.plot(tt2,"r")
	ax1.grid(True)
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
            ax2.plot(mx_exclude,"r.", markersize=10)
            ax2.set_ylabel("Excluded Delta T values (seconds)", color='r')
            ax2.tick_params('y', colors='r')

        for span in vertical_spans(metadata.get('vertical-spans', []), ts):
            plt.axvspan(span.first, span.last, facecolor=span.color, alpha=span.alpha)

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



