#!/usr/bin/env python
"""
DALEC ensemble Kalman filter (EnKF) ...

Reference:
---------
* Williams et al. (2005) An improved analysis of forest carbon dynamics using
  data assimilation. Global Change Biology 11, 89–105.
* Evensen (2003) The Ensemble Kalman Filter: theoretical formulation and
  practical implementation. Ocean Dynamics, 53, 343-367

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

__author__  = "Martin De Kauwe"
__version__ = "1.0 (07.03.2018)"
__email__   = "mdekauwe@gmail.com"

import numpy as np
import sys

def main():

    # initialise structures
    mp = GenericClass()
    p = GenericClass()
    c = GenericClass()

    setup_initial_conditions(mp, p, c)

    # Setup matrix holding ensemble members (A)
    A = np.zeros((c.ndims, c.nrens))

    # Setup ensemble covariance matrix of the errors (Q)
    Q = np.zeros((c.ndims, c.nrens))

    # model errors
    p_k = np.zeros(c.ndims)

    # Initialise the ensemble (A)
    A = initialise_ensemble(mp, c, A)

    # Initial error covariance matrix Q matrix
    Q = initialise_error_covariance(Q)


class GenericClass:
    pass

def setup_initial_conditions(mp, p, c):

    # dalec model values
    mp.t1 = 4.41E-06
    mp.t2 = 0.473267
    mp.t3 = 0.314951
    mp.t4 = 0.434401
    mp.t5 = 0.00266518
    mp.t6 = 2.06E-06
    mp.t7 = 2.48E-03
    mp.t8 = 2.28E-02
    mp.t9 = 2.65E-06
    mp.cf0 = 57.7049
    mp.cw0 = 769.863
    mp.cr0 = 101.955
    mp.cl0 = 40.4494
    mp.cs0 = 9896.7

    # acm parameterisation
    p.a0 = 2.155;
    p.a1 = 0.0142;
    p.a2 = 217.9;
    p.a3 = 0.980;
    p.a4 = 0.155;
    p.a5 = 2.653;
    p.a6 = 4.309;
    p.a7 = 0.060;
    p.a8 = 1.062;
    p.a9 = 0.0006;

    # location - oregon
    p.lat = 44.4;

    c.nrobs = 0
    c.sla = 111.
    c.ndims = 16
    c.nrens = 200
    c.max_params = 15
    c.seed = 0
    c.alpha = 0.0;
    c.dump_rt = False
    c.dump_gpp = False
    c.dump_nep = False;

    c.SV_POS_RA = 0
    c.SV_POS_AF = 1
    c.SV_POS_AW = 2
    c.SV_POS_AR = 3
    c.SV_POS_LF = 4
    c.SV_POS_LW = 5
    c.SV_POS_LR = 6
    c.SV_POS_CF = 7
    c.SV_POS_CW = 8
    c.SV_POS_CR = 9
    c.SV_POS_RH1 = 10
    c.SV_POS_RH2 = 11
    c.SV_POS_D = 12
    c.SV_POS_CL = 13
    c.SV_POS_CS = 14
    c.SV_POS_GPP = 15

def initialise_ensemble(mp, c, A):

    for j in range(c.nrens):
        A[c.SV_POS_RA, j] = 1.0 * np.random.normal(0.0, 0.1 * 1.0)
        A[c.SV_POS_AF, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_AW, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_AR, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_LF, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_LW, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_LR, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_RH1, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_RH2, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_D, j] = 0.3 * np.random.normal(0.0, 0.1 * 0.3)
        A[c.SV_POS_GPP, j] = 1.0 * np.random.normal(0.0, 0.1 * 1.0)
        A[c.SV_POS_CF, j] = mp.cf0 * np.random.normal(0.0, 0.1 * mp.cf0)
        A[c.SV_POS_CW, j] = mp.cw0 * np.random.normal(0.0, 0.1 * mp.cw0)
        A[c.SV_POS_CR, j] = mp.cr0 * np.random.normal(0.0, 0.1 * mp.cr0)
        A[c.SV_POS_CL, j] = mp.cl0 * np.random.normal(0.0, 0.1 * mp.cl0)
        A[c.SV_POS_CS, j] = mp.cs0 * np.random.normal(0.0, 0.1 * mp.cs0)

    return A

def initialise_error_covariance(c, Q):

    for i in range(c.ndims):
        for j in range(c.nrens):
            Q[i,j] = np.random.normal(0.0, 1.0)

    return Q

if __name__ == "__main__":

    main()
