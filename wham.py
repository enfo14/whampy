"""AUTHOR: efortin
DATE: 16/05/2018 16:06
VERSION: 1.1

This is a Python3 executable script that performs the WHAM analysis
of a set of umbrella sampling simulations, using various methods.
"""

# IMPORTS

import os
import re
import sys
import time
import warnings

import numpy as np
import wham.simdata as sim
from wham.init import Logger, update_progress, parse_command
from wham.setup import startup, read_data
from wham.minim import minimization, calc_free
from wham.errors import mc_error_analysis, split_analysis, consistency_tests
from wham.prints import print_results, print_consistency
from matplotlib import pyplot as plt
from scipy import optimize, constants
from operator import attrgetter

# DECLARATION OF GLOBAL VARIABLES


# PROGRAM STARTUP (COMMAND LINE PARSING)
start_time = time.time()
np.seterr(all='ignore')

metafile, outfile = parse_command(sys.argv)
print("Using {0} as metadata file".format(metafile))
windows, init_time = startup(metafile)

windows, data, read_time = read_data(windows)
g, min_time = minimization(windows, data)
data[:,2], data[:,3], bin_min = calc_free(g, windows, data)

if sim.num_mc_runs:
    P_std, A_std, G_std, mc_time = mc_error_analysis(windows, data)
else:
    P_std, A_std, G_std, split_time = split_analysis(windows, data)

phi, eta, tests_time = consistency_tests(windows, data)

print_results(outfile, data, A_std, P_std)
print_consistency(outfile, windows, G_std, phi, eta)

total_time = time.time() - start_time
print("WHAM calculation complete")
print("--- Runtime: %s seconds ---" % total_time)
