# coding: utf-8

"""
Disaggregation tool for Input-Output tables, to enable
Multi-Regional Input-Output (MRIO) analysis.

Mutahar Chalmers, 2016.
"""

from os.path import join, realpath, dirname
from json import load
import numpy as np
import pandas as pd
import pyras as ras


def disagg(f_Zagg, f_xagg, fullZ=False):
    """Load IO data in 'standard' aggregate form and disaggregate."""

    pwd = dirname(realpath(__file__))
    with open(join(pwd, 'settings_disagg.json')) as f:
        settings = load(f)

    reg_data = pd.read_csv(join(pwd, 'regions.csv'), index_col='id')
    regs, secs = reg_data.index, reg_data.columns[2:]

    # Index of disaggregated DataFrame
    ix = pd.MultiIndex.from_product([regs, secs])

    # Load aggregate IO tables in standard format
    Z0 = pd.read_csv(join(pwd, 'agg', f_Zagg), index_col=0)
    x0 = pd.read_csv(join(pwd, 'agg', f_xagg), index_col=0)['x0']

    # Initial Location Quotients (LQs) from region data
    LQ0 = reg_data[secs]

    # Estimate output by region using GDP share
    x_r = reg_data['grp'].mul(x0.sum())

    # Estimate output by region by sector using LQs and rebalance with RAS
    X_rs0 = LQ0.mul(x_r, axis=0).mul(x0/x0.sum(), axis=1).values
    X_rs = pd.DataFrame(ras.RAS(X_rs0, x_r, x0, iprn=-1)[0], index=regs,
                        columns=secs)

    # Aggregate technology matrix (scale inputs to unit output)
    A0 = Z0.divide(x0, axis=1)

    zero = np.zeros((secs.size, secs.size))
    Z_r0, Z_R0, U, V =  [], [], [], []

    # Loop through regions and recalculate LQs for r and R ('not r')
    print('\nInitial estimates of intra- and inter-regional matrices...')
    for r in regs:
        # Calculate output by sector for r and R, and recalculate LQs
        X = X_rs.copy()
        X.ix['R'] = X[X.index!=r].sum(axis=0)
        X = X[(X.index==r)|(X.index=='R')]
        LQ = pd.DataFrame(Xrs_LQ(X), index=X.index, columns=secs).clip(upper=1)

        # Generate A_rr (input) and A_Rr (import) coefficient matrices for r
        A_rr = A0.mul(LQ.ix[r], axis=0)
        A_Rr = A0 - A_rr

        # Convert to flows
        Z_rr = A_rr.mul(X.ix[r], axis=1)
        Z_Rr = A_Rr.mul(X.ix[r], axis=1)

        # Generate A_RR (input) and A_rR (import) coefficient matrices for R
        A_RR = A0.mul(LQ.ix['R'], axis=0)
        A_rR = A0 - A_RR

        # Convert to flows
        Z_RR = A_RR.mul(X.ix['R'], axis=1)
        Z_rR = A_rR.mul(X.ix['R'], axis=1)

        # Marginal row and column sums
        U.append(np.array(Z_rR + (Z_rr if fullZ else 0), order='C'))
        V.append(np.array(Z_Rr + (Z_rr if fullZ else 0), order='C'))

        # Generate intra-regional flow matrix
        intrarow = [Z_rr if r==s else zero for s in regs]
        Z_r0.append(intrarow)

        # Generate first estimate of inter-regional flow matrix 
        interrow = [Z_Rr/(regs.size-1) if r!=s else zero for s in regs]
        Z_R0.append(interrow)

    print('Complete.')

    # Intra- and inter-regional flow matrices
    # Explicit C-ordering for RAS functions (pandas DataFrames Fortran-order)
    Z_r0 = np.array(np.bmat(Z_r0), order='C')
    Z_R0 = np.array(np.bmat(Z_R0).T, order='C')

    # Target row and column sums
    U_t, V_t = np.vstack(U), np.hstack(V)

    #=FIXME Temporary hack to get round inconsistent U and V margins ==========
    sc = sum(ras.hblock(V_t, regs.size))/sum(ras.vblock(U_t, regs.size))
    sc[np.isnan(sc)] = 1
    U_t = np.vstack(ras.vblock(U_t, regs.size) * sc)
    #==========================================================================

    # Carry out RAS balancing
    print('\nRAS biproportional balancing...')
    if fullZ:
        result = ras.RAS(Z_r0+Z_R0, U_t, V_t, blocks=(regs.size, regs.size))
        if result is not None:
            print('Complete.\n')
            Z_out, convg = result
            print(pd.Series(convg))
            return pd.DataFrame(Z_out, index=ix, columns=ix), X_rs
    else:
        result = ras.RAS(Z_R0, U_t, V_t, blocks=(regs.size, regs.size))
        if result is not None:
            print('Complete.\n')
            Z_out, convg = result
            print(pd.Series(convg))
            return pd.DataFrame(Z_out + Z_r0, index=ix, columns=ix), X_rs
    print('No result.')


def Xrs_LQ(X_rs):
    """ Convert matrix of outputs by region by sector to LQs."""

    num = X_rs.div(X_rs.sum(axis=1), axis=0)
    denom = (X_rs.sum(axis=0)/X_rs.sum(axis=0).sum())
    return num/denom
