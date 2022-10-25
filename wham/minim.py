"""The wham module contains the functions that compute the function A
and its jacobian (or gradient) as a function of the variable g=ln(f).
It also contains the function that computes the probability and the
free energy as a function of g=ln(f).
"""

import re
import sys
import time
import numpy as np
import wham.simdata as sim
from wham.init import update_progress
from scipy import optimize
from matplotlib import pyplot as plt


def calc_bias(spring, loc, coor):
    dx = coor - loc
    if sim.periodic:
        period = sim.period
        dx = np.abs(dx)
        dx = np.where(dx > (period / 2.0), dx - period, dx)
    return 0.5 * spring * np.square(dx)


def function_A(g, *args):

    winlist = args[0]
    data = args[1]
    num_bins = sim.num_bins
    num_windows = sim.num_windows
    bin_width = sim.bin_width
    kT = sim.kT

    loc = np.array([winlist[i].loc for i in range(num_windows)])
    spring = np.array([winlist[i].spring for i in range(num_windows)])
    num_points = np.array([winlist[i].num_points for i in range(num_windows)])

    first = np.dot(num_points, g)
    second = 0.0

    for l in range(num_bins):
        coor = data[l, 0]
        Ml = data[l, 1]
        if Ml > 0:
            bias = calc_bias(spring, loc, coor)
            denom = np.dot(num_points, np.exp(-bias / kT + g))
            second += Ml * np.log(Ml / denom)

    val = (first + second) * -1
    return val


def gradient_A(g, *args):

    winlist = args[0]
    data = args[1]
    num_bins = sim.num_bins
    num_windows = sim.num_windows
    bin_width = sim.bin_width
    kT = sim.kT

    grad_A = np.zeros(num_windows)
    denom = np.zeros(num_bins)

    loc = np.array([winlist[i].loc for i in range(num_windows)])
    spring = np.array([winlist[i].spring for i in range(num_windows)])
    num_points = np.array([winlist[i].num_points for i in range(num_windows)])

    for l in range(num_bins):
        bias = calc_bias(spring, loc, data[l, 0])
        denom[l] = np.dot(num_points, np.exp(-bias / kT + g))

    denom = np.array(denom)
    for i in range(num_windows):
        bias = calc_bias(winlist[i].spring, winlist[i].loc, data[:, 0])
        summat = np.dot(data[:, 1], np.exp(-bias / kT) / denom)
        grad_A[i] = winlist[i].num_points * (np.exp(g[i]) * summat - 1)

    return grad_A


def calc_free(g, winlist, data):

    num_bins = sim.num_bins
    num_windows = sim.num_windows
    bin_width = sim.bin_width
    kT = sim.kT

    string = "Computing probability distribution and free energy..."
    print(string, end="")

    prob = np.zeros(num_bins)
    free = np.zeros(num_bins)

    loc = np.array([winlist[i].loc for i in range(num_windows)])
    spring = np.array([winlist[i].spring for i in range(num_windows)])
    num_points = np.array([winlist[i].num_points for i in range(num_windows)])

    for l in range(num_bins):
        Ml = data[l, 1]
        bias = calc_bias(spring, loc, data[l, 0])
        denom = np.dot(num_points, np.exp(-bias / kT + g))
        prob[l] = Ml / denom
        free[l] = -kT * np.log(prob[l] / bin_width)
    print("\tDone")

    string = "Rescaling probability distribution and free energy..."
    print(string, end="")
    offset = np.nanmin(free)
    bin_min = np.nanargmin(free)
    free -= offset
    prob /= np.sum(prob)
    print("\tDone")
    return prob, free, bin_min


def minimization(windows, data):

    tol = sim.tol

    start_time = time.time()
    g0 = np.zeros(len(windows))

    string = "Minimizing log-likelihood A..."
    print(string, end="")
    converged = False
    while not converged:
        arglist = (windows, data)

        g_minim = optimize.minimize(
            function_A,
            g0,
            args=arglist,
            jac=gradient_A,
            method="BFGS",
            options={"gtol": tol},
        )

        if g_minim.get("success"):
            print("\tDone")
            g = g_minim.get("x")
            g -= g[0]
            converged = True
        elif tol < g_minim.get("fun") / 1e4:
            tol *= 10
        else:
            text = "Minimization failed for all acceptable tolerances." "Stopping..."
            print(text)
            sys.exit()

    for i in range(len(g)):
        windows[i].g = g[i]

    min_time = time.time() - start_time
    return g, min_time
