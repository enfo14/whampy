"""The errors module contains the various methods for error analysis
that can be used along with the WHAM calculation.
"""

import time
import numpy as np
import wham.simdata as sim
from wham.init import update_progress
from wham.setup import Window
from wham.minim import calc_bias, minimization, calc_free
from scipy import optimize
from matplotlib import pyplot as plt


def mc_error_analysis(windows, data):

    num_bins = sim.num_bins
    num_windows = sim.num_windows
    num_mc_runs = sim.num_mc_runs

    start_time = time.time()
    ave_p = np.zeros(num_bins)
    ave_p2 = np.zeros(num_bins)
    ave_pdf = np.zeros(num_bins)
    ave_pdf2 = np.zeros(num_bins)

    ave_g = np.zeros(num_windows)
    ave_g2 = np.zeros(num_windows)
    if not num_mc_runs:
        print("No MC bootstrap error analysis requested")
        return ave_p2, ave_pdf2, ave_g2, time.time()-start_time

    update_progress("MC_error_analysis", 0)
    for i in range(num_mc_runs):
        fakeset = []
        fakedata = np.zeros([num_bins, 4])
        fakedata[:,0] = data[:,0]

        for w in windows:
            fw = Window(w.path, w.loc, w.spring, w.correl_time)
            fw.num_points = w.num_points
            fw.hist[:,0] = data[:,0]

            j=0
            while j < fw.num_points/int(fw.correl_time):
                index = np.argmin(abs(w.hist[:,2]-np.random.random()))+w.min
                fw.hist[index,1] += 1
                j += 1

            fakedata[:,1] += fw.hist[:,1]
            fw.min = np.min(np.nonzero(fw.hist[:,1]))
            fw.max = np.max(np.nonzero(fw.hist[:,1]))
            fw.hist = fw.hist[fw.min:fw.max+1,:]
            fw.hist[:,2] = np.cumsum(fw.hist[:,1])/fw.num_points
            
            fakeset += [fw]

        fake_g0 = np.array([window.g for window in windows])
        fake_g, fake_time = minimization(fakeset, fakedata)
        fakedata[:,2], fakedata[:,3], bm = calc_free(fake_g, fakeset, fakedata)

        ave_p += fakedata[:,2]
        ave_pdf += fakedata[:,3]
        ave_p2 += fakedata[:,2]**2
        ave_pdf2 += fakedata[:,3]**2

        ave_g += fake_g
        ave_g2 += fake_g**2

        update_progress("MC error analysis", (i+1)/num_mc_runs)

    ave_p /= num_mc_runs
    ave_pdf /= num_mc_runs
    ave_p2 /= num_mc_runs
    ave_p2 = np.sqrt(ave_p2 - ave_p**2)
    ave_pdf2 /= num_mc_runs
    ave_pdf2 = np.sqrt(ave_pdf2 - ave_pdf**2)

    ave_g /= num_mc_runs
    ave_g2 /= num_mc_runs
    ave_g2 = np.sqrt(ave_g2 - ave_g**2)

    mc_time = time.time() - start_time
    print(ave_p2[-1], ave_pdf2[-1], ave_g2[-1], mc_time)
    return ave_p2, ave_pdf2, ave_g2, mc_time


def blockAverage(datastream, isplot=False, maxBlockSize=0):
    Nobs = len(datastream)
    minBlockSize = 1
    if not maxBlockSize:
        maxBlockSize = Nobs//4

    NumBlocks = maxBlockSize - minBlockSize
    blockMean = np.zeros(NumBlocks)
    blockVar = np.zeros(NumBlocks)
    blockCtr = 0

    for blockSize in range(minBlockSize, maxBlockSize):
        Nblock = Nobs//blockSize
        resid = Nobs%blockSize
        obsProp = np.mean(np.reshape(datastream[:Nobs-resid],
                                     (Nblock, blockSize)), axis=1)

        blockMean[blockCtr] = np.mean(obsProp)
        blockVar[blockCtr] = np.var(obsProp)/(Nblock-1)
        blockCtr += 1

    v = np.arange(minBlockSize, maxBlockSize)
    if isplot:
        plt.subplot(2,1,1)
        plt.plot(v, np.sqrt(blockVar), "ro-", lw=2)
        plt.xlabel("block size")
        plt.ylabel("std")
        plt.subplot(2,1,2)
        plt.errorbar(v, blockMean, np.sqrt(blockVar))
        plt.xlabel("block size")
        plt.ylabel("<x>")
        plt.tight_layout()
        plt.show()

    return v, blockVar, blockMean


def varfit(x, a, b):
    y = a * np.arctan(b*x)
    return y


def block_analysis(windows, data):

    num_bins = sim.num_bins
    num_windows = sim.num_windows
    dr = 20.0
    kT = sim.kT
    
    start_time = time.time()
    xbar = np.zeros(num_windows)
    varx = np.zeros(num_windows)
    varg = np.zeros(num_windows)

    update_progress("Block averaging", 0)
    for i in range(num_windows):
        v, blockVar, blockMean = blockAverage(windows[i].traj,
                                              maxBlockSize=50)
        xbar[i] = blockMean[-1]
        try:
            [a,b] = optimize.curve_fit(varfit, v, blockVar)[0]
            varx[i] = varfit(1e21, a, b)
        except RuntimeError:
            varx[i] = blockVar[-1]
        update_progress("Block averaging", (i+1)/num_windows)

    for i in range(num_windows):
        spring = windows[i].spring
        if i == 0:
            varg[i] = 0.0
        else:
            varg[i] = ((spring * dr / kT)**2
                       * ((varx[0] + varx[i])/4 + np.sum(varx[1:i])))

    g = np.array([window.g for window in windows])

    prob_error, free_error, bm = calc_free(g+np.sqrt(varg), windows, data)

    errorbars = np.zeros([num_bins, 2])
    errorbars[:,0] = abs(free_error - data[:,3])
    errorbars[:,1] = abs(prob_error - data[:,2])

    block_time = time.time() - start_time
    return errorbars[:,1], errorbars[:,0], np.sqrt(varg), block_time

def split_analysis(windows, data):

    num_bins = sim.num_bins
    num_windows = sim.num_windows
    hist_min = sim.hist_min
    hist_max = sim.hist_max
    kT = sim.kT

    start_time = time.time()
    g = [window.g for window in windows]

    num_split = 3
    A = np.zeros([num_bins, num_split])
    A_ave = np.zeros(num_bins)
    A_std = np.zeros(num_bins)

    P = np.zeros([num_bins, num_split])
    P_ave = np.zeros(num_bins)
    P_std = np.zeros(num_bins)

    G = np.zeros([num_windows, num_split])
    G_ave = np.zeros(num_windows)
    G_std = np.zeros(num_windows)

    for i in range(num_split):
        splitset = []
        splitdata = np.zeros([num_bins, 4])
        splitdata[:,0] = data[:,0]

        j = 0
        for w in windows:
            j += 1
            splitw = Window(w.path, w.loc, w.spring, w.correl_time)
            splitw.hist[:,0] = data[:,0]

            rangemin = np.int(i*(w.num_points//3))
            rangemax = np.int(rangemin+(w.num_points//3))

            splitw.hist[:,1] = np.histogram(w.traj[rangemin:rangemax],
                                            bins=num_bins,
                                            range=(hist_min, hist_max),
                                            )[0]

            splitdata[:,1] += splitw.hist[:,1]
            splitw.num_points = np.sum(splitw.hist[:,1])
            splitw.p = splitw.hist[:,1]/splitw.num_points

            splitw.min = np.min(np.nonzero(splitw.hist[:,1]))
            splitw.max = np.max(np.nonzero(splitw.hist[:,1]))
            splitw.hist = splitw.hist[splitw.min:splitw.max+1,:]
            splitw.hist[:,2] = np.cumsum(splitw.hist[:,1])/splitw.num_points
            
            splitset += [splitw]
        
        split_g, split_time = minimization(splitset, splitdata)
        
        G[:,i] = split_g
        P[:,i], A[:,i], bm = calc_free(split_g, splitset, splitdata)

    for i in range(num_split):
        P_ave += P[:,i]
        A_ave += A[:,i]
        G_ave += G[:,i]
    P_ave /= num_split
    A_ave /= num_split
    G_ave /= num_split

    for i in range(num_split):
        P_std += (P_ave - P[:,i])**2
        A_std += (A_ave - A[:,i])**2
        G_std += (G_ave - G[:,i])**2
    P_std = np.sqrt(P_std/num_split)
    A_std = np.sqrt(A_std/num_split)
    G_std = np.sqrt(G_std/num_split)

    split_time = time.time() - start_time
    return P_std, A_std, G_std, split_time


def consistency_tests(windows, data):

    num_bins = sim.num_bins
    num_windows = sim.num_windows
    kT = sim.kT

    start_time = time.time()
    phi = np.zeros(num_windows)
    eta = np.zeros(num_windows)

    for i in range(1,num_windows):
        r = (windows[i-1].loc + windows[i].loc)/2
        wl = np.array([calc_bias(windows[i-1].spring,
                                 windows[i-1].loc, data[l,0])
                       for l in range(num_bins)])
        wr = np.array([calc_bias(windows[i].spring,
                                 windows[i].loc, data[l,0])
                       for l in range(num_bins)])

        wl_star = np.array([calc_bias(windows[i-1].spring, r, data[l,0])
                            for l in range(num_bins)])
        wr_star = np.array([calc_bias(windows[i].spring, r, data[l,0])
                            for l in range(num_bins)])

        num = np.array([windows[i-1].p[l]*np.exp((wl[l]-wl_star[l])/kT)
                        for l in range(num_bins)])
        denom = sum(windows[i-1].p[l]*np.exp((wl[l]-wl_star[l])/kT)
                    for l in range(num_bins))
        pl_star = num/denom

        num = np.array([windows[i].p[l]*np.exp((wr[l]-wr_star[l])/kT)
                        for l in range(num_bins)])
        denom = sum(windows[i].p[l]*np.exp((wr[l]-wr_star[l])/kT)
                    for l in range(num_bins))
        pr_star = num/denom

        maxval = max(np.sum(abs(pl_star - pr_star)[:i])
                     for i in range(num_bins))
        phi[i] = maxval

    g = [window.g for window in windows]
    for i in range(num_windows):
        bias = np.array([calc_bias(windows[i].spring,
                                   windows[i].loc, data[l,0])
                         for l in range(num_bins)])
        pil = np.exp(g[i] - bias/kT)*data[:,2]
        eta[i] = np.nansum(windows[i].p * np.log(windows[i].p/pil))

    tests_time = time.time() - start_time
    return phi, eta, tests_time
