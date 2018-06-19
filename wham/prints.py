"""The init.py module in this program contains the objects and
functions required by the wham program.
"""

import os
import re
import sys
import time
import datetime
import numpy as np
import argparse
from operator import attrgetter

def print_results(filename, data, A_std, P_std):
    freefile = open(filename+".pmf", "w")
    probfile = open(filename+".prob", "w")
    histfile = open(filename+".hist", "w")

    numpoints = np.sum(data[:,1])
    cumsum = np.cumsum(data[:,1])/numpoints

    string_a = "{0:.3f}\t{1:.6f}\t{2:.6f}\n"
    string_b = "{0:.3f}\t{1}\t{2:.6f}\n"
    for i in range(len(data)):
        freefile.write(string_a.format(data[i,0], data[i,3], A_std[i]))
        probfile.write(string_a.format(data[i,0], data[i,2], P_std[i]))
        histfile.write(string_b.format(data[i,0], data[i,1], cumsum[i]))

def print_consistency(filename, windows, G_std, phi, eta):
    freefile = open(filename+".pmf", "a")
    string = "#{0}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\n"
    for i in range(len(windows)):
        freefile.write(string.format(i, windows[i].g,
                                     G_std[i], phi[i], eta[i]))
