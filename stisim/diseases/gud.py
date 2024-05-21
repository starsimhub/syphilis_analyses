"""
Define genital ulcer disease
"""

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti

__all__ = ['GUD']


class GUD(ss.Infection):

    def __init__(self, pars=None, init_prev_data=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_inf = ss.lognorm_ex(mean=3/12, stdev=1/12),
            beta=1.0,  # Placeholder
            init_prev=0,  # See make_init_prev_fn
            rel_init_prev=1,
        )
        self.update_pars(pars, **kwargs)

        # Set initial prevalence
        self.init_prev_data = init_prev_data
        if init_prev_data is not None:
            self.pars.init_prev = ss.bernoulli(self.make_init_prev_fn)

        # Add states
        self.add_states(
            ss.FloatArr('ti_recovered'),
        )
        return

    @staticmethod
    def make_init_prev_fn(self, sim, uids):
        return sti.make_init_prev_fn(self, sim, uids)

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
