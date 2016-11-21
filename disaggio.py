# coding: utf-8

"""
Disaggregation tool for Input-Output tables, to enable
Multi-Regional Input-Output (MRIO) analysis.

Mutahar Chalmers, 2016.
"""

from json import load
import numpy as np
import pandas as pd
import pyras as ras


def disagg(f_sttgs='sttgs_disagg.json'):
    """Load IO data in 'standard' aggregate form and disaggregate."""

    with open(f_sttgs) as f:
        sttgs = load(f)

    # Load aggregate IO tables in standard format
    Z0 = pd.read_csv(sttgs['Z_filein'], index_col=0)
    M0 = pd.read_csv(sttgs['M_filein'], index_col=0)
    x0 = pd.read_csv(sttgs['x_filein'], index_col=0)['x0']
    e0 = pd.read_csv(sttgs['e_filein'], index_col=0)['e0']
    # Calculate final demand
    f0 = x0 - Z0.sum(axis=1) - e0

    reg_data = pd.read_csv(sttgs['regs_file'], index_col=0)
    regs, secs = reg_data.index, reg_data.columns[2:]
    n_regs = regs.size

    fullZ = sttgs['fullZ']

    # Index of disaggregated DataFrame
    ix = pd.MultiIndex.from_product([regs, secs])

    # Initial Location Quotients (LQs) from region data
    LQ0 = reg_data[secs]

    # Estimate output by region using GDP share
    x_r = reg_data['grp'].mul(x0.sum())

    # Estimate output by region by sector with LQs, rebalance with RAS
    X_rs0 = LQ0.clip(upper=1).mul(x_r, axis=0).mul(x0/x0.sum(), axis=1).values
    X_rs = pd.DataFrame(ras.RAS(X_rs0, x_r.values, x0.values, iprn=-1)[0],
                        index=regs, columns=secs)
    x_rs = X_rs.T.unstack() # Flattened (vector) form of X_rs matrix

    # Aggregate technology matrices (scale inputs to unit output)
    A0 = Z0.divide(x0, axis=1)
    N0 = M0.divide(x0, axis=1)

    zero = np.zeros((secs.size, secs.size))
    Z_r0, Z_R0, U, V, M_r =  [], [], [], [], []

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
        Z_R0.append([Z_Rr/(n_regs-1) if r!=s else zero for s in regs])

        # FIXME Simple disaggregation; ignores LQs
        M_r.append(N0.mul(X.ix[r]))

    print('Complete.')

    # Intra- and inter-regional flow matrices
    # Explicit C-ordering for RAS functions (pandas DataFrames Fortran-order)
    Z_r0 = np.array(np.bmat(Z_r0), order='C')
    Z_R0 = np.array(np.bmat(Z_R0).T, order='C')

    # Target row and column sums
    U_t, V_t = np.vstack(U), np.hstack(V)

    # Convert imports matrix to array
    M_out = pd.DataFrame(np.hstack(M_r), index=secs, columns=ix)

    #=FIXME Temporary hack to get round inconsistent U and V margins ==========
    sc = sum(ras.hblock(V_t, n_regs))/sum(ras.vblock(U_t, n_regs))
    sc[np.isnan(sc)] = 1
    U_t = np.vstack(ras.vblock(U_t, n_regs) * sc)
    #=FIXME Temporary hack to get round inconsistent U and V margins ==========

    # Carry out RAS balancing
    sRAS = sttgs['RAS']
    print('\nRAS biproportional balancing...')
    if fullZ:
        res = ras.RAS(Z_r0+Z_R0, U_t, V_t, blocks=(n_regs, n_regs), **sRAS)
    else:
        res = ras.RAS(Z_R0, U_t, V_t, blocks=(n_regs, n_regs), **sRAS)

    if res is not None:
        Z_out, convg = res
        print('\nConvergence:\n{0}'.format(pd.Series(convg).to_string()))
        if fullZ:
            Z_out = pd.DataFrame(Z_out, index=ix, columns=ix)
        else:
            Z_out = pd.DataFrame(Z_out+Z_r0, index=ix, columns=ix)
        A_out = Z_out.div(x_rs, axis=1)

        # Calculate Total Final Demand and split into Final Demand and exports
        sc = x_rs / A_out.dot(x_rs)
        if sc.min() < 1:
            print('Rescaling technical coefficients...')
            A_out = A_out.mul(sc.clip(upper=1), axis=0)
            Z_out = A_out.mul(x_rs, axis=1)

        tfd = (x_rs - A_out.dot(x_rs)).clip(lower=0)
        f_e_ratio = f0/(f0+e0)
        F_rs = tfd.mul(f_e_ratio, level=1)
        E_rs = (tfd - F_rs).unstack()

        print('\nWriting files...')
        Z_out.to_csv(sttgs['Z_fileout'])
        A_out.to_csv(sttgs['A_fileout'])
        M_out.to_csv(sttgs['M_fileout'])
        X_rs.to_csv(sttgs['x_fileout'])
        E_rs.to_csv(sttgs['e_fileout'])
        print('Complete.\n')
    else:
        print('No result.')


def Xrs_LQ(X_rs):
    """ Convert matrix of outputs by region by sector to LQs."""

    num = X_rs.div(X_rs.sum(axis=1), axis=0)
    denom = (X_rs.sum(axis=0)/X_rs.sum(axis=0).sum())
    return num/denom
