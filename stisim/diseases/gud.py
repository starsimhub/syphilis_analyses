"""
Define genital ulcer disease
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['GUD']


class GUD(ss.Infection):

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_inf = ss.lognorm_ex(mean=2/12, stdev=1/36),
            beta=1.0,  # Placeholder
            init_prev=ss.bernoulli(p=0.4),
        )
        self.update_pars(pars, **kwargs)
        self.add_states(
            ss.FloatArr('ti_recovered'),
        )
        return

    def update_pre(self):
        """ Updates prior to interventions """
        recovered = (self.infected & (self.ti_recovered <= self.sim.ti)).uids
        self.infected[recovered] = False
        self.susceptible[recovered] = True
        return

    def set_prognoses(self, uids, source_uids=None):
        """
        Set initial prognoses for adults newly infected with syphilis
        """
        ti = self.sim.ti
        dt = self.sim.dt

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti

        # Set future recovery
        dur_inf = self.pars.dur_inf.rvs(uids)
        self.ti_recovered[uids] = ti + dur_inf / dt

        return
