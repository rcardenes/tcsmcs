import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import sys, os, fnmatch
import pickle

class PathBuilder(object):
    def __init__(self, system, site, prefix='.'):
        self.prefix = prefix
        self.system = system
        self.site = site

    def path(self, kind):
        return os.path.join(self.prefix, '{0}{1}{2}'.format(self.system, kind, self.site))

def returnold(folder):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*.txt'):
            matches.append(os.path.join(root, filename))
    return sorted(matches, key=os.path.getmtime)

if (len(sys.argv) < 2):
    print  "Useage: python procRaw.py <site> <system>"
    print  "	    example: $python procRaw.py MKO crcs"
    sys.exit()

site = sys.argv[1]
system = sys.argv[2]
pb = PathBuilder(system, site)
rawFilePath = pb.path('Data')
binoutPath = pb.path('Binary')

print "Looking for RAW Data inside %s." % rawFilePath

dateconv = lambda x: dt.datetime.strptime(x,'%m/%d/%Y %H:%M:%S:.%f')
col_names = ["Timestamp", "data1"]
dtypes = ["object", "float"]

for f in returnold(rawFilePath):
    tsFilename = "%s/%s%s" % (binoutPath, os.path.basename(f), '.pkl')
    # Has pickle file been processed inside the binary folder yet?
    if os.path.isfile(tsFilename):
        print "Will BYPASS %s..." % (f)
        continue
    else:
        print "Will process %s..." % (f)
        try:
            mydata = np.genfromtxt(f, delimiter='\t',names=col_names, dtype=dtypes, converters={"Time": dateconv})
            myrange = "%s to %s" % (mydata['Timestamp'][0], mydata['Timestamp'][mydata.size-1] )
            print mydata['Timestamp']
            print 'Processing %d lines for Range %s' % (mydata.size, myrange)
            ts = pd.to_datetime(mydata['Timestamp'])
            print ts
            tsFile = open(tsFilename, 'w')
            print "Saving %s to disk." % tsFilename
            pickle.dump(ts, tsFile)
            print "done"
        except ValueError, e:
            print "Could not convert data proper format."
            print e
        #sys.exit()

print "done."
