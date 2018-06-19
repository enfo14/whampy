"""The init.py module in this program contains the objects and
functions required by the wham program.
"""

import os
import re
import sys
import time
import numpy as np
import wham.simdata as sim
from operator import attrgetter
from wham.init import update_progress

class Window:
    def __init__(self, *args):

        if len(args)<3 or len(args)>5:
            raise IndexError("Wrong number of arguments for Window object")

        if len(args)==3 : args += (1, sim.temp)
        if len(args)==4 : args += (sim.temp, )

        [path, loc, spring, correl_time, temp] = args[:5]
        
        self.path = path
        self.loc = float(loc)
        self.spring = float(spring)
        self.correl_time = correl_time
        self.temp = temp
        self.hist = np.zeros([sim.num_bins,3])
        self.p = np.zeros(sim.num_bins)
        self.min = 0
        self.max = sim.num_bins-1
        self.g = 0.0
        self.num_points = 0

    def __repr__(self):
        return "Window({0})".format(self.path)

    def __str__(self):
        return "Window({0})".format(self.path)


#----------------------------------------------------------------------

def startup(metafile):
    start_time = time.time()
    infile = open(metafile, "r")
    lines = infile.read().splitlines()

    simdata = {}
    print("Reading simulation parameters...", end="")
    for line in lines:
        if line.startswith('#'):
            vals = re.split("[ \t]+", line)
            if re.search('[ \t]+-?\d+', line) is not None:
                simdata.update({vals[1]: float(vals[3])})
            else:
                simdata.update({vals[1]: vals[3]})

    for key, val in simdata.items():
        if key == "hist_min" : sim.hist_min = val 
        if key == "hist_max" : sim.hist_max = val 
        if key == "num_bins" : sim.num_bins = int(val) 
        if key == "temp" : sim.temp = val 
        if key == "tol" : sim.tol = val 
        if key == "first" : sim.first = int(val) 
        if key == "last" : sim.last = int(val) 
        if key == "num_mc_runs" : sim.num_mc_runs = int(val) 

    if simdata.get('period'):
        sim.periodic = True
        sim.period = simdata['period']

    sim.bin_width = (sim.hist_max - sim.hist_min)/ sim.num_bins
    sim.kT = sim.k_B * sim.temp

    print("\tDone")

    simdata.update({'bin_width': sim.bin_width, 
                    'kT': sim.kT, 
                    'periodic': sim.periodic})

    for key, val in simdata.items():
        print("\t{0:<12s} {1}".format(key.upper(), val))

    lines = [line for line in lines if line and not line.startswith("#")]
    sim.num_windows = len(lines)
    simdata.update({'num_windows': sim.num_windows})
    print("\t{0:<12s} {1}".format('num_windows'.upper(), sim.num_windows))

    print("\nReading trajectory paths...", end="")
    windows = []
    for line in lines:
        vals = re.split("[ \t]+", line)
        w = Window(*vals)
        windows += [w]

    windows.sort(key=attrgetter("loc"), reverse=True)
    infile.close()
    print("\tDone")
    init_time = time.time() - start_time
    return windows, init_time

def read_data(windows):

    start_time = time.time()
    print("Setting up data structures...", end="")
    data = np.zeros([sim.num_bins, 4])
    data[:,0] = (np.arange(sim.hist_min, sim.hist_max, sim.bin_width)
                + sim.bin_width/2)
    print("\tDone")

    message = "Reading trajectory files"
    update_progress(message, 0)
    sys.stdout.log.write(message+"...\n")

    for i in range(sim.num_windows):
        string = "Reading window {0} of {1}\n"
        sys.stdout.log.write(string.format(i+1,sim.num_windows))
        window = windows[i]
        window.traj = np.loadtxt(window.path)[int(sim.first):,-1]
        string = "\tTrajectory contains {0} points\n"
        sys.stdout.log.write(string.format(len(window.traj)))
        window.hist[:,0] = data[:,0]
        window.hist[:,1] = np.histogram(window.traj,
                                        bins=sim.num_bins,
                                        range=(sim.hist_min, sim.hist_max),
                                        )[0]
        data[:,1] += window.hist[:,1]
        window.num_points = np.sum(window.hist[:,1])
        string = "\t{0} of {1} points inside histogram bounds\n"
        sys.stdout.log.write(string.format(window.num_points, len(window.traj)))
        window.p = window.hist[:,1]/(window.num_points*sim.bin_width)

        sys.stdout.log.write("\tPruning empty bins from histogram\n\n")
        window.min = np.min(np.nonzero(window.hist[:,1]))
        window.max = np.max(np.nonzero(window.hist[:,1]))
        window.hist = window.hist[window.min:window.max+1,:]
        window.hist[:,2] = np.cumsum(window.hist[:,1])/window.num_points

        update_progress(message, (i+1)/sim.num_windows)

    read_time = time.time() - start_time
    return windows, data, read_time
