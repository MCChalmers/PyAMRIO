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
from scipy.linalg import block_diag


__version__ = '0.1.0'

class PyAMRIO():
    """Load settings, regions, sectors and IO information."""

    def __init__(self, f_sttgs):
        with open(f_sttgs) as f:
            self.settings = load(f)
        sttgs_list = str(list(range(len(self.settings))))
        print('Settings {0} available.'.format(sttgs_list))


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
        self.Ed = sttgs['Ed']
        self.xi = sttgs['xi']

        # Load disaggregated IO tables in standard format
        self.Z0 = pd.read_csv(join(rpath, sttgs['Z_file']), index_col=[0,1], header=[0,1])
        self.M0 = pd.read_csv(join(rpath, sttgs['M_file']), index_col=[0], header=[0,1])
        self.x0 = pd.read_csv(join(rpath, sttgs['x_file']), index_col=0).T.unstack()
        self.exp0 = pd.read_csv(join(rpath, sttgs['e_file']), index_col=0).T.unstack()

        # Calculate local and import technical coefficients and orders
        self.A0 = self.Z0.div(self.x0, axis=1)
        self.o0 = self.A0.dot(self.x0)

        # Value added (total output less intermediate consumption)
        self.va0 = self.x0 - self.Z0.sum(axis=0) - self.M0.sum(axis=0)

        # Local final demand
        self.lfd0 = self.x0 - self.o0 - self.exp0

        # Normalise imports by total output
        self.imp0 = self.M0.sum(axis=0).div(self.x0)

        # Calculate labour input
        # Proportional contributions of labour and capital to Gross Value Added
        gva_norm = pd.read_csv(join(rpath, sttgs['GVA_file']), index_col=0)

        # Disaggregated labour input (labour:capital regionally invariant)
        self.lab0 = self.va0.mul(gva_norm['LAB'], level=1).div(self.x0)

        # Profits
        self.prof0 = self.x0 - self.o0 - (self.lab0 + self.imp0)*self.x0

        # Industries which are subsitutable by imports
        self.sig = self.sec_data['sig']

        # Initialise price vector
        self.p0 = np.ones(self.n_sec)

        # Pre-event balance of supply & demand
        sd_bal0 = np.abs(self.lfd0 + self.exp0 + self.o0 - self.x0).max()
        if sd_bal0 > sttgs['convg_lim']:
            print('Initial supply-demand imbalance: {0:3.4e}'.format(sd_bal0))

        # Multi-region and sector index - used to set up new DataFrames
        self.ix = self.Z0.index

        # Load supply constraint data and assemble matrix
        Cmax = pd.read_csv(join(rpath, sttgs['c_max']), index_col=[0,1])
        self.x_max0 = pd.concat({reg: Cmax.ix[reg].mul(self.x0.ix[reg], axis=0)
                                 for reg in self.regs}).ix[self.regs]
        print('Data loaded successfully.')


    def bottleneck(self, x, A, f, x_max, itr_max=50, eps=1e-6, verbose=False):
        """Hallegatte's priority + proportional rationing algorithm."""

        L = np.linalg.inv(np.eye(A.shape[0])-A)
        td = L.dot(f)
        x = np.minimum(x_max, td)

        for i in range(itr_max):
            # For checking convergence
            x0 = x.copy()

            # Calculate orders and total demand
            o = A.dot(x)
            chk = np.array(x/o).clip(max=1)
            sc = chk.min()
            x[chk==1] = x[chk==1]*sc

            # Recalculate total demand
            td = A.dot(x) + f
            x = np.minimum(np.minimum(x_max, td), x)

            # Check convergence
            if (np.abs(x - x0).max() < eps):
                if verbose:
                    print('Converged in {0} iterations.'.format(i))
                    print('Minimum x/o: {0:.3e}'.format((x/A.dot(x)).min()))
                return x, o, td
        print('## Not converged ##')
        return None


    def amrio(self):
        """Run model."""

        x_t = self.x0.copy()
        lfd_t = self.lfd0.copy()
        exp_t = self.exp0.copy()
        imp_t = self.imp0.copy()
        lab_t = self.lab0.copy()
        dt, eps = self.dt, self.eps

        A_t = self.A0.copy()

        # Number of region-sectors; number of time steps; 
        n, m = self.x_max0.shape
        """
        if self.x_max0.ix[:,-1].min() < 1:
            print('Extending AMRIO model beyond duration of c_max...')
            extension = pd.concat([self.x0]*m, axis=1)
            self.x_max0 = pd.concat([self.x_max0, extension], axis=1)
            m = 2*m
        """

        a_t = pd.Series(np.ones(self.n_sec), index=self.secs, name='a_t')
        a_max = self.sec_data['a_max'].ix[self.secs]
        a0 = self.sec_data['a0'].ix[self.secs]

        x_out = pd.DataFrame(index=self.x_max0.columns, columns=self.ix)
        lfd_out = pd.DataFrame(index=self.x_max0.columns, columns=self.ix)
        exp_out = pd.DataFrame(index=self.x_max0.columns, columns=self.ix)
        imp_out = pd.DataFrame(index=self.x_max0.columns, columns=self.ix)
        va_out = pd.DataFrame(index=self.x_max0.columns, columns=self.ix)

        # Time stepping
        print('AMRIO model running...')
        for ix_t, x_max_t in self.x_max0.iteritems():
            # Increase maximum output by overproduction factor
            x_max = x_max_t.mul(a_t, level=1, axis=0)

            # Production bottlenecking
            x_t, o_t, td_t = self.bottleneck(x_t, A_t, lfd_t+exp_t, x_max)

            # Excess demand factor
            xs_dmd = ((td_t-x_t)/td_t).clip(lower=0)

            # Prices, profits and labour demand
            p_t = self.p0*(1 + self.Ed*((td_t-x_t)/x_t))
            prof_t = p_t*x_t - (A_t.mul(p_t, axis=1).dot(x_t) + (lab_t+imp_t)*x_t)
            Macro_t = (prof_t + lab_t*x_t).sum()/(self.prof0 + self.lab0*self.x0).sum()

            # Adaptation ======================================================
            # If total demand exceeds supply
            mask = td_t - x_t > 1
            sig = self.sig[mask]
            sig_ = self.sig[~mask]
            xs = xs_dmd[mask]
            sec_mask = self.sec_data.ix[mask]
            sec_mask_ = self.sec_data.ix[~mask]

            # Final demand - final demand and exports reduce--------
            lfd_t[mask] = lfd_t[mask]*(1-sig*xs*dt/sec_mask['tau_lfd_down'])
            exp_t[mask] = exp_t[mask]*(1-sig*xs*dt/sec_mask['tau_exp_down'])

            # Intermediate consumption - inter-industry consumption reduces and imports increase
            delta = A_t.ix[mask].mul(xs_dmd[mask]*sig, axis=0).div(sec_mask['tau_A_down'], axis=0)
            A_t.ix[mask] = A_t.ix[mask] - delta*dt
            imp_t[mask] = imp_t[mask] + delta.sum(axis=1)*dt

            # Production - overproduction where possible
            a_t[mask] = a_max[mask] + (a_max[mask]-a_t[mask]) * xs_dmd[mask] *dt/sec_mask['tau_a']

            # Otherwise... ----------------------------------------------------
            # Local final demands and exports recover to pre-disaster levels
            lfd_fac = (eps+lfd_t[~mask]/self.lfd0[~mask])*(self.lfd0[~mask]-lfd_t[~mask]).div(sec_mask_['tau_lfd_up'], axis=0)
            lfd_t[~mask] = lfd_t[~mask] + sig_*lfd_fac[~mask]*dt
            exp_fac = (eps+exp_t[~mask]/self.exp0[~mask])*(self.exp0[~mask]-exp_t[~mask]).div(sec_mask_['tau_exp_up'], axis=0)
            exp_t[~mask] = exp_t[~mask] + sig_*exp_fac[~mask]*dt

            # Intermediate consumption recovers to pre-disaster levels
            A_fac1 = (eps+A_t.ix[~mask]/self.A0.ix[~mask])
            A_fac2 = (self.A0.ix[~mask]-A_t.ix[~mask])
            A_fac = (A_fac1*A_fac2).div(sec_mask_['tau_A_up'], axis=0).fillna(0)*dt
            A_t.ix[~mask] = A_t.ix[~mask] + A_fac
            imp_t[~mask] = imp_t[~mask] - A_fac.sum(axis=1)[~mask]

            # Overproduction reverts to previous norm
            a_t[~mask] = a_max[~mask]+(a0[~mask]-a_t[~mask]).div(sec_mask_['tau_a'], axis=0)*dt
            # /Adaptation =====================================================

            # Apply additional macroeconomic factors
            lfd = Macro_t * lfd_t * (1 - self.xi*(p_t - 1))
            #lfd_t = Macro_t * lfd_t * (1 - self.xi*(p_t - 1)) # FIXME
            exp = exp_t * (1 - self.xi*(p_t - 1))
            #exp_t = exp_t * (1 - self.xi*(p_t - 1))           # FIXME
            x_t = (np.linalg.inv(np.eye(self.n_sec)-A_t)).dot(lfd+exp)
            #x_t = (np.linalg.inv(np.eye(self.n_sec)-A_t)).dot(lfd_t+exp_t)

            x_out.ix[ix_t] = x_t
            lfd_out.ix[ix_t] = lfd
            exp_out.ix[ix_t] = exp
            imp_out.ix[ix_t] = imp_t * x_t
            va_out.ix[ix_t] = x_t-(A_t.mul(x_t, axis=1).sum(axis=0)+imp_t*x_t)

        # Add full results to class variables
        self.x, self.x_norm = x_out, x_out.div(self.x0, axis=1)
        self.lfd, self.lfd_norm = lfd_out, lfd_out.div(self.lfd0, axis=1)
        self.exp, self.exp_norm = exp_out, exp_out.div(self.exp0, axis=1)
        #self.imp, self.imp_norm = imp_out, imp_out.div(self.imp0*self.x0, axis=1)
        self.imp = imp_out
        self.va, self.va_norm = va_out, va_out.div(self.va0, axis=1)

        # Aggregate over regions and sectors and assign to class variables
        self.x_reg = self.x.sum(level=0, axis=1)
        self.x_reg_norm = self.x_reg.div(self.x0.sum(level=0), axis=1)
        self.x_sec = self.x.sum(level=1, axis=1)
        self.x_sec_norm = self.x_sec.div(self.x0.sum(level=1), axis=1)

        self.lfd_reg = self.lfd.sum(level=0, axis=1)
        self.lfd_reg_norm = self.lfd_reg.div(self.lfd0.sum(level=0), axis=1)
        self.lfd_sec = self.lfd.sum(level=1, axis=1)
        self.lfd_sec_norm = self.lfd_sec.div(self.lfd0.sum(level=1), axis=1)

        self.exp_reg = self.exp.sum(level=0, axis=1)
        self.exp_reg_norm = self.exp_reg.div(self.exp0.sum(level=0), axis=1)
        self.exp_sec = self.exp.sum(level=1, axis=1)
        self.exp_sec_norm = self.exp_sec.div(self.exp0.sum(level=1), axis=1)

        self.imp_reg = self.imp.sum(level=0, axis=1)
        #self.imp_reg_norm = self.imp_reg.div((self.imp0*self.x0).sum(level=0), axis=1)
        self.imp_sec = self.imp.sum(level=1, axis=1)
        #self.imp_sec_norm = self.imp_sec.div((self.imp0*self.x0).sum(level=1), axis=1)

        self.va_reg = self.va.sum(level=0, axis=1)
        self.va_reg_norm = self.va_reg.div(self.va0.sum(level=0), axis=1)
        self.va_sec = self.va.sum(level=1, axis=1)
        self.va_sec_norm = self.va_sec.div(self.va0.sum(level=1), axis=1)

        # Write only Value Added to file
        if len(self.opath) > 0:
            self.va.to_csv(join(self.opath, 'va.csv'))
            self.va_norm.to_csv(join(self.opath, 'va_norm.csv'))
            self.va_reg.to_csv(join(self.opath, 'va_reg.csv'))
            self.va_reg_norm.to_csv(join(self.opath, 'va_reg_norm.csv'))
            self.va_sec.to_csv(join(self.opath, 'va_sec.csv'))
            self.va_sec_norm.to_csv(join(self.opath, 'va_sec_norm.csv'))
        print('Complete.')
