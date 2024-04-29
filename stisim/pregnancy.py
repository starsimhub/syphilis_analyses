"""
Define pregnancy
"""

import starsim as ss
import sciris as sc
import numpy as np
# import pandas as pd

__all__ = ['Conception']


class Conception(ss.Condition):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_states(
            ss.State('infertile', bool, False),  # Applies to girls and women outside the fertility window
            ss.State('fecund', bool, True),  # Applies to girls and women inside the fertility window
            ss.State('pregnant', bool, False),  # Currently pregnant
            ss.State('postpartum', bool, False),  # Currently post-partum
            ss.State('ti_pregnant', int, ss.INT_NAN),  # Time pregnancy begins
            ss.State('ti_delivery', int, ss.INT_NAN),  # Time of delivery
            ss.State('ti_postpartum', int, ss.INT_NAN),  # Time postpartum ends
            ss.State('ti_dead', int, ss.INT_NAN),  # Maternal mortality
            ss.State('rel_sus', float, 1.0),
            ss.State('rel_trans', float, 1.0),
            ss.State('ti_infected', int, ss.INT_NAN),
        )

        self.rng_target = ss.random(name='target')

        return

    def make_new_cases(self, sim):
        """
        Add new conceptions, through transmission, incidence, etc.
        """
        new_cases = []
        people = sim.people
        betamap = self._check_betas(sim)

        for nkey, net in sim.networks.items():
            if not len(net):
                break

            nbetas = betamap[nkey]
            contacts = net.contacts

            # Do something here
            rel_sus = (people.female & people.alive & self.fecund) * self.rel_sus
            trg = contacts.p2
            beta = nbetas[0]

            # Skip networks with no transmission
            if beta == 0:
                continue

            # Calculate probability of a->b transmission.
            beta_per_dt = net.beta_per_dt(disease_beta=beta, dt=sim.dt)
            p_transmit = rel_sus[trg] * beta_per_dt
            rvs = self.rng_target.rvs(trg)
            new_cases_bool = rvs < p_transmit
            new_cases.append(trg[new_cases_bool])

        # Tidy up
        if len(new_cases):
            new_cases = np.concatenate(new_cases)
            self.set_prognoses(sim, new_cases)
        else:
            new_cases = np.empty(0, dtype=int)

        return new_cases

    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti
        res.prevalence[ti] = res.n_infected[ti] / np.count_nonzero(sim.people.alive)
        res.new_infections[ti] = np.count_nonzero(self.ti_infected == ti)
        res.cum_infections[ti] = np.sum(res['new_infections'][:ti+1])
        return
