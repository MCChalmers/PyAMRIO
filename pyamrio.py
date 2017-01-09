# coding: utf-8
"""
Python implementation of Adaptive Multi-Regional Input-Output (AMRIO) model.

Mutahar Chalmers, 2016.
"""

from os.path import join, exists
from os import listdir, makedirs
from json import load
import numpy as np
import pandas as pd

__version__ = '0.2.0'

class PyAMRIO():
    """Load settings, regions, sectors and IO information."""

    def __init__(self, f_sttgs):
        with open(f_sttgs) as f:
            self.settings = load(f)
        sttgs_list = str(list(range(len(self.settings))))
        print('Settings available: ', sttgs_list)


    def load(self, n):
        """Process nth settings configuration."""

        sttgs = self.settings[n]
        self.desc = sttgs['desc']
        print('Loading data for analysis: {0}'.format(self.desc))

        rpath = sttgs['root_path']
        self.opath = join(sttgs['out_rpath'], self.desc)
        if not exists(self.opath):
            makedirs(self.opath)

        if len(self.opath) > 0:
            with open(join(self.opath, 'settings.json'), 'w') as f:
                sout = ',\n '.join(['"{0}": "{1}"'.format(k, v)
                                    for k, v in sttgs.items()])
                f.writelines('{\n '+sout+'\n}')

        self.reg_data = pd.read_csv(join(rpath, sttgs['regs_file']), index_col=0)
        self.regs = self.reg_data.index
        self.n_reg = self.regs.size
        sec_data = pd.read_csv(join(rpath, sttgs['secs_file']), index_col=0)
        self.sec_data = pd.concat({r: sec_data for r in self.regs}, keys=self.regs)
        self.secs = self.sec_data.index
        self.n_sec = self.secs.size
        self.dt = sttgs['dt_1']
        self.eps = sttgs['eps']
        self.convg_lim = sttgs['convg_lim']

        # Load disaggregated IO tables in standard format
        self.Z0 = pd.read_csv(join(rpath, sttgs['Z_file']), index_col=[0,1], header=[0,1])
        self.M0 = pd.read_csv(join(rpath, sttgs['M_file']), index_col=[0], header=[0,1])
        self.x0 = pd.read_csv(join(rpath, sttgs['x_file']), index_col=0).T.unstack()
        self.exp0 = pd.read_csv(join(rpath, sttgs['e_file']), index_col=0).T.unstack()

        # Multi-region and sector index - used to set up new DataFrames
        self.ix = self.Z0.index

        # Masks for intra- and inter-regional flows
        self.intra_mask = pd.DataFrame(np.zeros((self.n_sec, self.n_sec)),
                                       index=self.ix, columns=self.ix)
        for reg in self.regs:
            self.intra_mask.ix[reg, reg] = 1
        self.inter_mask = 1 - self.intra_mask
        self.inter2intra_mask = self.intra_mask - self.inter_mask

        # Fill zeros in x0 and exp0 to prevent divide by zero errors
        self.x0.replace({0: 1e-12}, inplace=True)
        self.exp0.replace({0: 1e-12}, inplace=True)

        # Calculate technical coefficients, orders and intermediate consumption
        self.A0 = self.Z0.div(self.x0, axis=1)
        self.o0 = self.A0.dot(self.x0)
        self.i0 = self.Z0.sum(axis=0)
        # Normalise imports by total output
        self.imp0 = self.M0.sum(axis=0).div(self.x0)

        # Value added (total output less intermediate consumption and imports)
        self.va0 = self.x0 - self.i0 - self.imp0*self.x0

        # Final demand (Fill zeros in f0 to prevent divide by zero errors)
        self.f0 = self.x0 - self.o0 - self.exp0
        self.f0.replace({0: 1e-12}, inplace=True)

        # Calculate labour input
        # Proportional contributions of labour and capital to Gross Value Added
        gva_norm = pd.read_csv(join(rpath, sttgs['GVA_file']), index_col=0)

        # Disaggregated labour input (labour:capital regionally invariant)
        self.lab0 = self.va0.mul(gva_norm['LAB'], level=1).div(self.x0)

        # Profits: value added less payments to labour
        self.prof0 = self.va0 - self.lab0*self.x0

        # Initialise price vector
        self.p0 = np.ones(self.n_sec)

        # Time constants for adaptation
        self.tau = self.sec_data[['f_up','f_dn','exp_up','exp_dn',
                                  'A_up','A_dn','a_up','a_dn']]

        # Overproduction factors
        self.a0, self.a_max = self.sec_data['a0'], self.sec_data['a_max']

        # Pre-event balance of supply & demand
        sd_bal0 = np.abs(self.f0 + self.exp0 + self.o0 - self.x0).max()
        if sd_bal0 > sttgs['convg_lim']:
            print('Initial supply-demand imbalance: {0:3.4e}'.format(sd_bal0))

        # Load supply constraint data and assemble matrix
        Cmax = pd.read_csv(join(rpath, sttgs['c_max']), index_col=[0,1])
        self.x_max0 = pd.concat({reg: Cmax.ix[reg].mul(self.x0.ix[reg], axis=0)
                                 for reg in self.regs}).ix[self.regs]
        print('Data loaded successfully.')


    def ration(self, x, A, f, x_max, itr_max=100, eps0=1e-6, eps1=1e-9):
        """Hallegatte's priority + proportional rationing algorithm."""

        for i in range(1, itr_max+1):
            # For checking convergence
            x0 = x.copy()

            # Calculate total demand, and cap x at min(x_max, td)
            td = A.dot(x) + f
            x = np.minimum(x_max, td)

            # Calculate orders, and ration orders and final demand
            o = A.dot(x)
            factor = np.array(x/o).clip(min=0, max=1).min()
            x, f = x*factor, f*factor

            # Check convergence
            if (np.abs(x-x0).max() < eps0) and (x/A.dot(x)).min() >= (1-eps1):
                return x, A.dot(x)
        print('## Not converged: {0:.3e}'.format(np.abs(x-x0).max()))
        print('## Minimum x/o: {0}'.format((x/A.dot(x)).min()))
        return None


    def agg(self, q, q0):
        """Aggregate AMRIO output by region and by sector, and normalise."""

        agg_reg, agg_sec = q.sum(level=0, axis=1), q.sum(level=1, axis=1)
        agg_reg0, agg_sec0 = q0.sum(level=0), q0.sum(level=1)
        return agg_reg/agg_reg0, agg_sec/agg_sec0


    def amrio(self, sttg=0, verbose=False):
        """Run model."""

        # Load analysis settings and data
        self.load(sttg)

        # Model parameters
        dt, eps, sig, tau = self.dt, self.eps, self.sec_data['sig'], self.tau
        Ed, gamma_p = self.sec_data['Ed'], self.sec_data['gamma_p']

        # Initialise object variables for all results
        shp = {'index': self.x_max0.columns, 'columns': self.ix}
        self.x, self.o = pd.DataFrame(**shp), pd.DataFrame(**shp)
        self.f, self.exp = pd.DataFrame(**shp), pd.DataFrame(**shp)
        self.imp, self.i = pd.DataFrame(**shp), pd.DataFrame(**shp)
        self.va = pd.DataFrame(**shp)
        self.p = pd.DataFrame(**shp)

        # Declare temportary variables
        x_t, A_t = self.x0.copy(), self.A0.copy()
        f_t, exp_t = self.f0.copy(), self.exp0.copy()
        imp_t, lab_t = self.imp0.copy(), self.lab0.copy()
        p_t = self.p0.copy()
        a0, a_max, a_t = self.a0.copy(), self.a_max.copy(), self.a0.copy()

        # Time stepping
        print('AMRIO model running...')
        for ix_t, x_max_t in self.x_max0.iteritems():
            # Increase maximum output by overproduction factor
            x_max = x_max_t.mul(a_t, level=1, axis=0)

            # Ration production in case of bottlenecking
            x_t, o_t = self.ration(x_t, A_t, f_t+exp_t, x_max)

            # Recalculate total demand and excess demand factor
            td_t = o_t + f_t + exp_t
            xs_dmd = ((td_t-x_t)/td_t).clip(lower=0)

            # Prices, profits and macroeconomic factors
            p_t = p_t*(1 + gamma_p*dt*((td_t-x_t)/x_t))
            i_t = A_t.mul(p_t, axis=0).mul(x_t, axis=1).sum(axis=0)
            prof_t = p_t*x_t - i_t - lab_t*x_t - imp_t*x_t
            Macro_t = (prof_t + lab_t*x_t).sum(level=0)/self.va0.sum(level=0)

            # Save results in class variables
            self.x.ix[ix_t], self.o.ix[ix_t] = x_t, o_t
            self.f.ix[ix_t], self.exp.ix[ix_t] = f_t, exp_t
            self.imp.ix[ix_t] = imp_t*x_t
            self.i.ix[ix_t] = i_t
            # Calculate value added using increased prices for input and output
            self.va.ix[ix_t] = p_t*x_t - i_t - imp_t*x_t
            self.p.ix[ix_t] = p_t

            # Adaptation - if total demand exceeds supply =====================
            # Region-sectors where total demand exceeds total output
            mask = (td_t-x_t) > self.eps

            # Final demand and exports from the region-sector switch to others
            delta_f = f_t*sig*xs_dmd*dt/tau['f_dn']
            f_t[mask] = f_t[mask] * (1 - sig*xs_dmd*dt/tau['f_dn'])[mask]
            f_t[~mask] = f_t[~mask] * (1 - sig*xs_dmd*dt/tau['f_dn'])[~mask]

            delta_exp = exp_t * sig * xs_dmd
            exp_t[mask] = exp_t[mask]*(1 - sig*xs_dmd*dt/tau['exp_dn'])[mask]
            exp_t[~mask] = exp_t[~mask]*(1 - sig*xs_dmd*dt/tau['exp_dn'])[~mask]

            """
            print(ix_t, '\n', pd.concat([delta_f[mask], delta_f[~mask]], axis=1))
            if ix_t == 'Month-3':
                return f_t, delta_f, mask
            """

            # Rebalance intermediate consumption
            delta = A_t.mul(sig*xs_dmd, axis=0).div(tau['A_dn'], axis=0)

            # Redirect inter-regional flows to within the region
            d_inter2intra = (self.inter2intra_mask*delta).sum(axis=1, level=1)
            for r in self.regs:
                delta.loc[r, r] = d_inter2intra.ix[r].values

            """
            if sum(mask) > 1:
                print(ix_t, sum(mask))
                return(A_t, A_t[mask], delta, A_t[~mask])
            """

            A_t.ix[mask] = A_t.ix[mask] - delta.ix[mask]*dt

            # Imports increase
            imp_t = imp_t + delta.ix[mask].sum(axis=0)*dt

            # Overproduction where possible
            a_t[mask] = a_max[mask] + ((a_max-a_t)*xs_dmd*dt/tau['a_up'])[mask]

            # Otherwise... ----------------------------------------------------
            # Local final demands and exports recover to pre-disaster levels
            f_fac = (eps+f_t/self.f0)*(self.f0-f_t)/tau['f_up']
            f_t[~mask] = f_t[~mask] + (sig*f_fac)[~mask]*dt
            exp_fac = (eps+exp_t/self.exp0)*(self.exp0-exp_t)/tau['exp_up']
            exp_t[~mask] = exp_t[~mask] + (sig*exp_fac)[~mask]*dt

            # Intermediate consumption recovers to pre-disaster levels
            A_fac = (eps+A_t/self.A0).fillna(1)*(self.A0-A_t)/tau['A_up']
            A_t.ix[~mask] = A_t.ix[~mask] + A_fac.ix[~mask]*dt
            imp_t = imp_t - A_fac.ix[~mask].sum(axis=0)*dt

            # Overproduction reverts to previous norm
            a_t[~mask] = a_max[~mask] + ((a0-a_t)/tau['a_dn'])[~mask]*dt
            # /Adaptation =====================================================

            # Apply price elasticity and macroeconomic factors to final demand
            f_t = (f_t * (1 - Ed*(p_t-1)))#.mul(Macro_t, level=0)
            exp_t = (exp_t * (1 - Ed*(p_t-1)))

        # Calculate normalised results
        self.x_norm = self.x/self.x0
        self.f_norm = self.f/self.f0
        self.exp_norm = self.exp/self.exp0
        self.imp_norm = self.imp/(self.imp0*self.x0)
        self.va_norm = self.va/self.va0

        # Check for negative values
        if self.va_norm.values.min() < 0:
            print('## Warning: some value added quantities are negative.')

        # Aggregate over regions and sectors and normalise
        self.x_reg_norm, self.x_sec_norm = self.agg(self.x, self.x0)
        self.f_reg_norm, self.f_sec_norm = self.agg(self.f, self.f0)
        self.exp_reg_norm, self.exp_sec_norm = self.agg(self.exp, self.exp0)
        self.imp_reg_norm, self.imp_sec_norm = self.agg(self.imp*self.x,
                                                        self.imp0*self.x0)
        self.va_reg_norm, self.va_sec_norm = self.agg(self.va, self.va0)

        # Write value added output to file
        if len(self.opath) > 0:
            self.va.to_csv(join(self.opath, 'va.csv'))
            self.va_norm.to_csv(join(self.opath, 'va_norm.csv'))
            self.va_reg_norm.to_csv(join(self.opath, 'va_reg_norm.csv'))
            self.va_sec_norm.to_csv(join(self.opath, 'va_sec_norm.csv'))
        print('Complete.')