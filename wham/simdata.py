#simdata.py contains the global variables for the simulation

# Constants required for use
k_B = 0.0019872041          # in kcal/mol

# Parameters of the histograms
hist_min = -180.0
hist_max = 180.0
num_bins = 180
bin_width = 2.0

# Parameters of the simulation
num_windows = 0
temp = 273.15
kT = k_B * temp
first = 0
last = 0
periodic = False
period = 0.0

# Parameters for the minimization
tol = 1e-5

# Parameters for error analysis
num_mc_runs = 0
