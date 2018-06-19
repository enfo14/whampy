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

k_B = 0.0019872041

class Logger(object):
    def __init__(self, logfile, quiet=False):
        self.terminal = sys.stdout
        open(logfile,"w").close
        self.log = open(logfile, "a")
        self.quiet = quiet

        if self.quiet:
            self.terminal = open(os.devnull, "w")

    def write(self, message):
        if not self.quiet: 
            self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def update_progress(message, progress):
    """The update_progress function takes in a string and a number
    between 0 and 1, and returns a progress bar that overwrites the
    last line printed in stdout.
    """
    barLength = 30
    status = ""

    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt\r\n"
    if progress >= 1:
        progress = 1
        status = "Done\r\n"

    block = int(round(barLength*progress))
    text = "\r{3}: [{0}] {1:.2f}% {2}".format(
                "#"*block + "-"*(barLength-block),
                progress*100, status, message)
    sys.stdout.terminal.write(text)
    sys.stdout.terminal.flush()


def print_help():
    text = ("-"*72+"\n"
            + " "*31 + "whampy 1.1" + " "*31 + "\n"
            + "-"*72+"\n")
    out = sys.__stdout__.write(text)

    text = ("\nUsage: python3 wham.py [OPTION]... [-I input] [-O output]\n")
    out = sys.__stdout__.write(text)

    text = ("\nOPTIONS:\n"
            "\t-h, --help   \t display this help text and exit\n"
            "\t-v, --version\t display version information and exit\n"
            "\t-s, --silent \t suppress standard output\n")
    
    out = sys.__stdout__.write(text)
    sys.exit()

#----------------------------------------------------------------------

def parse_command(argv):
    argc = len(argv)
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--silent",
                        help="suppress standard output",
                        action="store_true")

    parser.add_argument("-i", "--input", 
                        type=str, default="whampy.in",
                        help="path to input file")

    parser.add_argument("-o", "--output", 
                        type=str, default="whampy",
                        help="prefix of output files")

    args = parser.parse_args()
    if args.silent:
        sys.stdout = Logger("whampy.log", quiet=True)
    else:
        sys.stdout = Logger("whampy.log")

    print("\nInfo: Running whampy v1.1")
    string = "Info: Execution at %y-%m-%d %H:%M:%S"
    print(datetime.datetime.now().strftime(string))
    print("Info: Working in dir {0}".format(os.path.dirname(os.getcwd())))

    metafile = args.input
    outfile = args.output

    print("Info: command-line call: $", *argv)
    print("")

    return metafile, outfile
