# `whampy`
whampy is a python3 script that performs a WHAM calculation on umbrella sampling datasets in order to obtain a potential of mean force.

## Getting Started

### Prerequisites

__Disclaimer:__ _program in development_

* `python3.x` must be installed.
* The following `python` packages must be installed:
  * `numpy`
  * `scipy`
  * `matplotlib`
* These can be installed with `pip`

## `whampy` instructions
The whampy" program computes the potential of mean force of an umbrella
sampling simulation using a minimization of a log-likelihood function of
the probability distribution in 1D. 

The execution of the program is as follows:

```shell
python3 wham.py [-h] [-s] [-i INPUT] [-o OUTPUT]
```

where the `INPUT` file is a plain text file with the format as specified
in `example/example.in`.  The  trajectory  files sourced from the paths found in 
the input file are assumed to be in the two-column  NAMD .traj format as 
shown in the `example/example.traj` file. 

For more information about the options, the `[-h]` optional flag brings up
the help text.

## Test execution

A simple calculation of the potential of mean force of the rotation of
the C2-C3 dihedral angle of butane is provided for testing purposes.

To execute the test, the following command suffices:

``` shell
python3 wham.py -i example/input/example.in -o example/output/test
```

The results of the PMF will then be stored in `example/output/test.pmf`,
with the reaction coordinate in the first column, the PMF value in the
second column, and the standard error of the PMF value in the third.

Plotting these results with the first column at the x-axis and the second
column at the y-axis will produce results equivalent to those contained
in the provided .png file.

=======

The 'montecarlo' branch includes a modified version of the code that also generates a dataset with Monte Carlo, useful for testing.
