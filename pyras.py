# coding: utf-8

"""
Fast RAS biproportional matrix balancing for 2D block matrices.

Mutahar Chalmers, 2016.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def hblock(Z, n):
    """Use numpy stride tricks to access blocks. Assumes C order."""

    if np.isfortran(Z):
        print('Warning: input array is Fortran-ordered!')
    shape = (n, Z.shape[0], Z.shape[1]//n)
    strides = np.array([Z.shape[1]//n, Z.shape[1], 1]) * Z.itemsize
    return ast(Z, shape=shape, strides=strides)


def vblock(Z, n):
    """Use numpy stride tricks to access blocks. Assumes C order."""

    if np.isfortran(Z):
        print('Warning: input array is Fortran-ordered!')
    shape = (n, Z.shape[0]//n, Z.shape[1])
    strides = np.array([Z.shape[0]//n*Z.shape[1], Z.shape[1], 1]) * Z.itemsize
    return ast(Z, shape=shape, strides=strides)


def RAS(Z0, U, V, blocks=(1, 1), max_itr=100, convg=1e-3, eps=1e-9, iprn=10):
    """RAS biproportional block matrix balancing."""

    # Validation checks
    if blocks == (1, 1):
        blkdiv_u = Z0.shape[1]
        blkdiv_v = Z0.shape[0]

        if len(U.shape) > 1 or len(V.shape) > 1:
            print('Target margins must be 1D arrays in non-block mode.')
            return None
        else:
            U = U.reshape((Z0.shape[0], 1))
            V = V.reshape((1, Z0.shape[1]))
            # Margin check-sums
            U_chk = np.sum(U)
            V_chk = np.sum(V)
    else:   # Block mode
        blkdiv_u = blocks[1]
        blkdiv_v = blocks[0]
        # Margin check-sums
        U_chk = sum(vblock(U, blkdiv_u))
        V_chk = sum(hblock(V, blkdiv_v))

    # Check for consistency between margins
    if (abs(U_chk - V_chk) > convg).any():
        print('Margins not consistent!\n')
        print('U: ', U_chk)
        print('V: ', V_chk)
        return None

    Z = Z0.copy()
    Z[Z==0] = eps

    # RAS iterations
    for i in range(max_itr):
        # Balance row sums
        U_act = np.sum(hblock(Z, blkdiv_u), axis=0)
        R = U/U_act
        R[np.isnan(R)] = 1
        Z = np.hstack(R*hblock(Z, blkdiv_u))

        # Balance column sums
        V_act = np.sum(vblock(Z, blkdiv_v), axis=0)
        S = V/V_act
        S[np.isnan(S)] = 1
        Z = np.vstack(S*vblock(Z, blkdiv_v))

        # Check convergence
        U_act = np.sum(hblock(Z, blkdiv_u), axis=0)
        V_act = np.sum(vblock(Z, blkdiv_v), axis=0)
        convg_u = np.max(abs(U_act-U))
        convg_v = np.max(abs(V_act-V))

        # Print current iteration and convergence metrics
        if (i % iprn == 0) and (iprn > 0):
            print('{0} of {1}'.format(i, max_itr))
            print('  convg_u {0:.4e}\n  convg_v {1:.4e}'.format(convg_u, convg_v))

        if convg_u < convg and convg_v < convg:
            break
    return Z, {'n_itr': i, 'convg_u': convg_u, 'convg_v': convg_v}
