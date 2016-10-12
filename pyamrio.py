# coding: utf-8

# Python implementation of Adaptive Regional Input-Output (ARIO) model
# based on Hallegatte (2008).
from os.path import join, realpath, dirname
from os import listdir
from json import load
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.linalg import block_diag

pwd = dirname(realpath(__file__))

class PyAMRIO():
    """Load settings and sectors information."""

    def __init__(self):
        with open(join(pwd, 'settings.json')) as f:
            self.settings = load(f)

        self.reg_data = pd.read_csv(join(pwd, 'regions.csv'), index_col='id')
        self.n_reg = self.reg_data.index.size
        self.sec_data = pd.read_csv(join(pwd, 'sectors.csv'), index_col='id')
        self.n_sec = self.sec_data.index.size
        self.n_ts = 0 

        # Define index of multiregion, multisector technology matrix
        self.ix = ['r{0}_c{1}'.format(r+1, c+1) for r in range(self.n_reg) for c in range(self.n_sec)]


    def disagg(self, fname='USA_NIOT_2011.csv'):
        """Load raw national IO data in 'standard' form and disaggregate."""

        # Inter-industry sectors are endogenous
        sectors_ii = self.sec_data[self.sec_data['exog']==0].index

        # Location quotients from region data
        self.LQ = self.reg_data[sectors_ii]

        # Load national IO table
        niot = pd.read_csv(join(pwd, 'raw', fname))

        locmask = niot['location'] != 'Imports'
        impmask = niot['location'] == 'Imports'

        # Households endogenous
        if self.settings['hhold_sup_id'] in sectors_ii: 
            consump = [dmd for dmd in self.settings['dmd_ids'] if dmd != self.settings['hhold_dmd_id']] 
            niot[self.settings['hhold_sup_id']] = niot[self.settings['hhold_dmd_id']]
            exomask = np.ones_like(niot['id'], dtype=bool)
        # Households exogenous   
        else:
            consump = self.settings['dmd_ids'] 
            exomask = niot['id'].isin(sectors_ii)
 
        # Total output by sector 
        self.y0 = niot[locmask & exomask].set_index('id')['TotalOutput']
        
        # Output by region
        self.yr = self.reg_data['grp'].mul(self.y0.sum())

        # Direct requirements table (domestic industries only)
        self.Z0 = niot[locmask & exomask].set_index('id')[sectors_ii]
        
        # National technology matrix (scale inputs to unit output)
        self.A0 = self.Z0.divide(self.y0, axis=1)       

        # Output by region by sector
        self.yir = self.LQ.mul(self.yr, axis=0).mul(self.y0/self.y0.sum(), axis=1)

        # Loop through regions and apply Location Quotients
        zero = np.zeros((self.A0.shape))
        self.A, self.Z_in, self.Z_out = {}, {}, {}
        self.Zr0 =  []
        for i, r in enumerate(self.LQ.index):
            # Output by region for non-r regions
            ynotr = self.yr[self.yr.index!=r]
            # Output by sector for non-r regions
            yinotr = self.yir[self.yir.index!=r].sum(axis=0)

            # Select LQ for region r and clip to 1
            LQr = self.LQ.ix[r].clip(upper=1)
            # Calculate LQ for region ~r and clip to 1
            LQnotr = ((yinotr/ynotr.sum())/(self.y0/self.y0.sum())).clip(upper=1)

            # Generate Arr
            self.A['{0}{0}'.format(r)] = self.A0.mul(LQr, axis=1)
            self.Z_in['{0}{0}'.format(r)] = self.A['{0}{0}'.format(r)].mul(self.yir.ix[r], axis=1)

            # Generate import coefficients A~rr
            self.A['~{0}{0}'.format(r)] = self.A0 - self.A['{0}{0}'.format(r)]
            self.Z_in['~{0}{0}'.format(r)] = self.A['~{0}{0}'.format(r)].mul(self.yir.ix[r], axis=1)

            # Generate LQs for ~r~r and then A~r~r (all other non-r regions together)
            self.A['~{0}~{0}'.format(r)] = self.A0.mul(LQnotr, axis=1)
            self.Z_in['~{0}~{0}'.format(r)] = self.A['~{0}~{0}'.format(r)].mul(yinotr, axis=1)

            # Generate export coefficients Ar~r
            self.A['{0}~{0}'.format(r)] = self.A0 - self.A['~{0}~{0}'.format(r)]
            self.Z_in['{0}~{0}'.format(r)] = self.A['{0}~{0}'.format(r)].mul(yinotr, axis=1)
        
            self.Zr0.append([])
            for j, s in enumerate(self.LQ.index):
                if r != s:
                    self.Zr0[-1].append((self.Z_in['{0}~{0}'.format(r)]/(self.n_reg-1)).values)
                    self.Z_out['{0}{1}'.format(r, s)] = self.Z_in['{0}~{0}'.format(r)]/(self.n_reg-1)
                else:
                    self.Zr0[-1].append(zero)
                    self.Z_out['{0}{1}'.format(r, s)] = pd.DataFrame(zero, index=sectors_ii, columns=sectors_ii)

        # Initial estimate of inter-regional flow matrix
        Zr = np.bmat(self.Zr0)
        Zr[Zr==0] = 1

        U_target = block_diag(*[self.Z_in['{0}~{0}'.format(r)].values for r in self.reg_data.index]) 
        V_target = block_diag(*[self.Z_in['~{0}{0}'.format(r)].values for r in self.reg_data.index]) 

        # RAS iterations
        for i in range(self.settings['RAS_itrmax']):
            U_actual = block_diag(*np.split(sum(np.split(Zr, self.n_reg, axis=1)), self.n_reg, axis=0))
            V_actual = block_diag(*np.split(sum(np.split(Zr, self.n_reg, axis=0)), self.n_reg, axis=1))
            R = U_target/U_actual
            R[np.isnan(R)|np.isinf(R)] = 0
            S = V_target/V_actual
            S[np.isnan(S)|np.isinf(S)] = 0
            Zr = R*Zr*S
            Zr[Zr==0] = 1

            # Add Zrr along main diagonal
            self.Z = Zr + block_diag(*[self.Z_in['{0}{0}'.format(r)].values for r in self.reg_data.index]) 
            # Sum of regional inter-industry transactions == national inter-industry transactions
            self.Z_check = pd.DataFrame(sum(np.split(sum(np.split(self.Z, self.n_reg, axis=1)), self.n_reg, axis=0)), index=sectors_ii, columns=sectors_ii)
            convg = np.abs(self.Z_check - self.Z0).values.max()
            print(i, convg)
            if convg < self.settings['RAS_convg']:
                break

        # Local final demand
        lfd0 = niot[locmask & exomask].set_index('id')[consump].sum(axis=1)

        # Imports matrix - flatten and scale to unit output (like A)
        imp0 = niot[impmask & exomask].set_index('id')[sectors_ii].sum(axis=0)
        
        # Exports
        exp0 = niot[locmask & exomask].set_index('id')[self.settings['export_id']]

        # Labour
        lab0 = niot[(niot['id']==self.settings['hhold_sup_id'])&~impmask][sectors_ii].sum(axis=0)

        self.impexplab = pd.DataFrame({'imp0': imp0, 'exp0': exp0, 'lab0': lab0})


    def proc(self):
        """Load processed/disaggregated IO data."""

        ix, n_reg, n_sec, n_ts = self.ix, self.n_reg, self.n_sec, self.n_ts
        lfd_dict, ymax_dict, y_dict, A_dict, iel_dict = {}, {}, {}, {}, {}
        for fname_ in listdir(join(pwd, 'data')):
            fname = fname_[:-4].split('_')
            fpath = join(pwd, 'data', fname_)
            if fname[0] == 'lfd':
                lfd_dict[fname[-1]] = pd.read_csv(fpath, index_col='id')
            elif fname[0] == 'y':
                y_dict[fname[-1]] = pd.read_csv(fpath, index_col='id')
            elif fname[0] == 'ymax':
                ymax_dict[fname[-1]] = pd.read_csv(fpath, index_col='t')
                n_ts = max(n_ts, ymax_dict[fname[-1]].shape[0])
            elif fname[0] == 'impexplab':
                iel_dict[fname[-1]] = pd.read_csv(fpath, index_col='id')
            elif fname[0] == 'A':
                A_dict[fname[-2]+fname[-1]] = pd.read_csv(fpath, index_col='id')

        #TODO - insert checks and validation that data loaded correctly

        self.A0 = pd.DataFrame(np.zeros((n_reg*n_sec, n_reg*n_sec)), index=ix, columns=ix)
        self.lfd0 = pd.Series(np.zeros(n_reg*n_sec), index=ix)
        self.y0 = pd.Series(np.zeros(n_reg*n_sec), index=ix)
        self.imp0 = pd.Series(np.zeros(n_reg*n_sec), index=ix)
        self.exp0 = pd.Series(np.zeros(n_reg*n_sec), index=ix)
        self.lab0 = pd.Series(np.zeros(n_reg*n_sec), index=ix)
        self.ymax = pd.DataFrame(np.zeros((n_ts, n_reg*n_sec)), columns=ix)
       
        # Fill Series/DataFrames in correct order
        for r in range(n_reg):
            self.lfd0.ix[r*n_sec:(r+1)*n_sec] = lfd_dict['r{0}'.format(r+1)]['lfd0'].values
            self.y0.ix[r*n_sec:(r+1)*n_sec] = y_dict['r{0}'.format(r+1)]['y0'].values
            self.imp0.ix[r*n_sec:(r+1)*n_sec] = iel_dict['r{0}'.format(r+1)]['imp0'].values.flatten()
            self.exp0.ix[r*n_sec:(r+1)*n_sec] = iel_dict['r{0}'.format(r+1)]['exp0'].values.flatten()
            self.lab0.ix[r*n_sec:(r+1)*n_sec] = iel_dict['r{0}'.format(r+1)]['lab0'].values.flatten()
            self.ymax.ix[:, r*n_sec:(r+1)*n_sec] = ymax_dict['r{0}'.format(r+1)].values
            for s in range(n_reg):
                self.A0.ix[r*n_sec:(r+1)*n_sec, s*n_sec:(s+1)*n_sec] = A_dict['r{0}s{1}'.format(r+1,s+1)].values

        # Inital intermediate consumption (orders)
        self.o0 = self.A0.dot(self.y0)                                        

        # Leontief inverse
        self.L0 = pd.DataFrame(np.linalg.inv(np.eye(self.A0.shape[0])-self.A0), index=ix, columns=ix)

        # Normalise imports and labour by total output (convention)
        self.imp0 = self.imp0/self.y0
        self.lab0 = self.lab0/self.y0

        # Value added (total output less intermediate consumption)
        self.va0 = self.y0 - self.A0.dot(self.y0).sum(axis=0) 

        # Profits
        self.prof0 = self.y0 - self.o0 - (self.lab0 + self.imp0)*self.y0

        # Industries which are subsitutable by imports
        self.sig = self.sec_data['sig']
        
        # Initialise price vector
        self.p0 = np.ones(n_reg*n_sec)

        # Pre-event balance of supply & demand
        self.sd_bal0 = np.abs(self.lfd0 + self.exp0 + self.o0 - self.y0).max()
        if self.sd_bal0 > self.settings['convg_lim']:
            print('Warning: initial supply-demand balance not satisfied: {0:3.4e}'.format(self.sd_bal0))
        # -------------------------------------------------------------------------------------------------------------


    def f_p(self, y_t, td_t, Ep):
        """Calculate sector prices given excess demand factor."""
        return self.p0*(1+Ep*((td_t-y_t)/y_t))


    def calc_va(self, y, A, imports, labour):
        """Calculate Gross Value Added."""
        return y - (A.multiply(y, axis=1).sum(axis=0) + imports*y + labour*y)


    def amrio(self):
        """Run model."""

        y_t, lfd_t, imp_t, exp_t, lab_t = self.y0.copy(), self.lfd0.copy(), self.imp0.copy(), self.exp0.copy(), self.lab0.copy() 
        A_t = self.A0.copy()
        td_t = self.A0.dot(self.y0) + self.lfd0 + self.exp0

        m, n = self.prod_data.shape[0], self.sectors.shape[0]
        dt = self.settings['dt_1']
        eps = self.settings['eps']

        a_t = pd.Series(np.ones(n), index=self.sectors, name='a_t')
        a_max = self.sector_data['a_max'].ix[self.sectors]
        a0 = self.sector_data['a0'].ix[self.sectors]

        y_out = np.zeros((m, n))
        lfd_out = np.zeros((m, n))
        imp_out = np.zeros((m, n))
        exp_out = np.zeros((m, n))
        va_out = np.zeros((m, n))

        # Time stepping
        for ix_t, y_max_t in self.prod_data.iterrows():
            y_max =  self.y0 * y_max_t.values * a_t
            
            # Production bottlenecking - solve LP problem such that orders are satisfied first
            res = linprog(-np.ones(n), A_ub=np.vstack((np.eye(n)-A_t, np.eye(n))),
                          b_ub=np.hstack((lfd_t+exp_t, y_max)), options={'disp': False})
            y_t = pd.Series(res['x'], index=self.sectors)

            # Recalculate orders based on adjusted output
            o_t = A_t.dot(y_t)
            
            # Check Leontief equation is satisfied
            sc = (y_t-o_t)/(self.y0-self.o0)
            sd_bal = np.abs(self.lfd0*sc + self.exp0*sc + o_t - y_t).max()
            if sd_bal > self.settings['convg_lim']:
                print('Supply-demand balance at timestep {0} not satisfied: {1:3.4e}'.format(int(ix_t), sd_bal))
                break

            # Recalculate total demand using actual orders and adapted LFD and exports
            td_t = lfd_t + exp_t + o_t

            # Prices, profits and labour demand
            p_t = self.f_p(y_t, td_t, self.settings['Ed'])
            prof_t = p_t*y_t - (A_t.mul(p_t, axis=1).dot(y_t) + lab_t*y_t + imp_t*y_t)
            M_t = (prof_t + lab_t*y_t).sum()/(self.prof0 + self.lab0*self.y0).sum()

            # Adaptation =======================================================
            # Excess demand factor
            xs_dmd = (td_t-y_t)/td_t

            # If total demand exceeds supply
            mask = td_t - y_t > 1

            # Final demand adaptation - final demand and exports reduce---------
            lfd_t[mask] = lfd_t[mask] - self.sig[mask]*xs_dmd[mask]*lfd_t[mask]*dt/self.sector_data['tau_lfd_down'].ix[self.sectors]
            exp_t[mask] = exp_t[mask] - self.sig[mask]*xs_dmd[mask]*exp_t[mask]*dt/self.sector_data['tau_exp_down'].ix[self.sectors]
            
            # Intermediate consumption adaptation - inter-industry consumption reduces and imports increase
            delta = A_t.ix[mask,:].mul(xs_dmd[mask]*self.sig[mask], axis=0)*dt/self.sector_data['tau_A_down'].ix[self.sectors]
            A_t.ix[mask,:] = A_t.ix[mask,:] - delta
            imp_t = imp_t + delta.sum(axis=0)

            # Production adaptation - overproduction where possible
            a_t[mask] = a_max[mask] + (a_max[mask]-a_t[mask]) * xs_dmd[mask] * dt/self.sector_data['tau_a'].ix[self.sectors]

            # Otherwise... -----------------------------------------------------
            # Local final demands and exports recover to pre-disaster levels
            lfd_fac = (eps+lfd_t[~mask]/self.lfd0[~mask])*(self.lfd0[~mask]-lfd_t[~mask])*dt/self.sector_data['tau_lfd_up'].ix[self.sectors]
            lfd_t[~mask] = lfd_t[~mask] + self.sig[~mask]*lfd_fac[~mask]
            exp_fac = (eps+exp_t[~mask]/self.exp0[~mask])*(self.exp0[~mask]-exp_t[~mask])*dt/self.sector_data['tau_exp_up'].ix[self.sectors]
            exp_t[~mask] = exp_t[~mask] + self.sig[~mask]*exp_fac[~mask]

            # Intermediate consumption recovers to pre-disaster levels
            A_fac = ((eps+A_t/self.A0)*(self.A0-A_t)).fillna(0)
            A_t = A_t + A_fac*dt/self.sector_data['tau_A_up'].ix[self.sectors]
            imp_t = imp_t - A_fac.sum(axis=0)*dt/self.sector_data['tau_A_up'].ix[self.sectors]
            #imp_t[~mask] = imp_t[~mask] - A_fac.sum(axis=0)[~mask]*dt/self.sector_data['tau_A_up'].ix[self.sectors]

            # Overproduction reverts to previous norm
            a_t[~mask] = a_max[~mask] + (a0[~mask]-a_t[~mask]) * dt/self.sector_data['tau_a'].ix[self.sectors]
            # /Adaptation ======================================================
            
            lfd = M_t * lfd_t * (1 - self.settings['xi']*(p_t - 1))
            #lfd = lfd_t * (1 - self.settings['xi']*(p_t - 1))
            exp = exp_t * (1 - self.settings['xi']*(p_t - 1))
            y_adap = (np.linalg.inv(np.eye(n)-A_t)).dot(lfd+exp)
            #print(pd.DataFrame({1: y_max - y_adap, 2: y_max - y_t}))
            y_out[ix_t] = y_t
            lfd_out[ix_t] = lfd
            imp_out[ix_t] = imp_t*y_t
            exp_out[ix_t] = exp
            va_out[ix_t] = self.calc_va(y_t, A_t, imp_t, lab_t)
            
        self.y, self.lfd, self.imp, self.exp, self.va = y_out, lfd_out, imp_out, exp_out, va_out
