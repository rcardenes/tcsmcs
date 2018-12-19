#!/usr/bin/env python

from datetime import datetime, timedelta
from collections import namedtuple

from swglib.export import get_exporter, DataManager
from swglib.stats import DataRange, DataRangeCollection
from swglib.stats import producer, find_event_ranges
import numpy as np
from numpy import datetime64, timedelta64
import matplotlib
#matplotlib.use('gtkcairo')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PLOT_DPI = 300

DataSet = namedtuple('DataSet', 'load mbufs clusters')
Reboot  = namedtuple('Reboot', 'start end')

#ALTAIR_REBOOTS=(
#    Reboot(
#        datetime64("2018-12-18 11:28:12"),
#        datetime64("2018-12-18 11:30:44")
#        ),
#    Reboot(
#        datetime64("2018-12-18 11:39:12"),
#        datetime64("2018-12-18 11:41:45")
#        )
#)
#
#TCS_REBOOTS=(
#    Reboot(
#        datetime64("2018-12-18 11:19:49"),
#        datetime64("2018-12-18 11:21:03")
#        ),
#    Reboot(
#        datetime64("2018-12-18 11:45:00"),
#        datetime64("2018-12-18 11:46:21")
#        )
#)

ALTAIR_REBOOTS = ()

TCS_REBOOTS=(
    Reboot(
        datetime64("2018-12-18 18:52:00"),
        datetime64("2018-12-18 18:53:11")
        ),
    Reboot(
        datetime64("2018-12-18 19:57:29"),
        datetime64("2018-12-18 19:58:36")
        ),
    Reboot(
        datetime64("2018-12-18 20:33:14"),
        datetime64("2018-12-18 20:34:20")
        ),
    Reboot(
        datetime64("2018-12-18 20:40:29"),
        datetime64("2018-12-18 20:41:36")
        ),
    Reboot(
        datetime64("2018-12-18 21:42:55"),
        datetime64("2018-12-18 21:44:02")
        ),
)

systems = ('tcs', 'mc', 'cr', 'm1', 'm2', 'ec')
colors = 'bgrcmyk'

ut_diff = timedelta64(10, 'h')
ut_diff_d = timedelta(hours=10)
start_time = datetime64("2018-12-18 18:00").astype(datetime) + ut_diff_d
stop_time  = datetime64("2018-12-18 23:00").astype(datetime) + ut_diff_d

data = {}

dm = DataManager(get_exporter('MK'), root_dir='/tmp/rcm')
for system in systems:
    data[system] = DataSet(
        list(dm.getData('{}:iocStats:LOAD'.format(system), start=start_time, end=stop_time)),
        list(dm.getData('{}:iocStats:SYS_CLUST_AVAIL_0'.format(system), start=start_time, end=stop_time)),
        list(dm.getData('{}:iocStats:SYS_CLUST_AVAIL_1'.format(system), start=start_time, end=stop_time))
    )

def plot_data(data_fn, title, filename):
    tcs_dates = [point[0] - ut_diff for point in data_fn(data['tcs'])]
    diffs = [b - a  for a, b in zip(tcs_dates[:-1], tcs_dates[1:])]
    threshold = np.mean(diffs) * 3
    mkdates = np.ma.array(tcs_dates)
    for n, d in enumerate(diffs):
        if d > threshold:
            mkdates[n-1:n+1] = np.ma.masked

    f, axes = plt.subplots(len(systems), sharex=True)
    plt.minorticks_on()
    lines = []
    common = dict(drawstyle='steps', linestyle='-', marker=None, linewidth=0.5)

    lines.append(axes[0].plot_date(mkdates, [x[1] for x in data_fn(data['tcs'])], color=colors[0], **common)[0])
    for system, ax, color in zip(systems[1:], axes[1:], colors[1:]):
        lines.append(ax.plot_date([x[0] - ut_diff for x in data_fn(data[system])], [x[1] for x in data_fn(data[system])], color=color, **common)[0])
    for ax in axes:
        for rb in TCS_REBOOTS:
            ax.axvspan(rb.start.astype(datetime), rb.end.astype(datetime), facecolor='b', alpha=0.2)
        for rb in ALTAIR_REBOOTS:
            ax.axvspan(rb.start.astype(datetime), rb.end.astype(datetime), facecolor='r', alpha=0.2)
    axes[0].set_title(title)
    xlast = axes[-1].xaxis
    xlast.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    xlast.set_minor_formatter(mdates.DateFormatter('%H:%M:%S'))
    f.autofmt_xdate(which='both')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in axes[:-1]], visible=False)
    plt.figlegend(lines, systems, 'right')
#    plt.show()
    f.set_size_inches(6000.0/PLOT_DPI, 1220.0/PLOT_DPI)
    plt.savefig("png/{}.png".format(filename), dpi=PLOT_DPI)

plot_data(lambda x: x.load, 'CPU Load (%)', 'cpuload')
plot_data(lambda x: x.mbufs, 'Free mbufs (%)', 'mbufs')
plot_data(lambda x: x.clusters, 'Free mbuf clusters (%)', 'clusters')
