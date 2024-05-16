"""
Define syphilis diagnostics
"""

import numpy as np
import sciris as sc
import starsim as ss


# %% Syphilis diagnostics

__all__ = ['SymptomaticTesting']  # , 'ANCTesting', 'NewbornTesting', 'ANC']


class SymptomaticTesting(ss.Intervention):
    """
    Test given to those presenting with active lesions or genital ulcers
    """
    def __init__(self, pars=None, product=None, test_prob_data=None, start=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.default_pars(
            rel_test=1
        )
        self.update_pars(pars, **kwargs)

        # Set testing probabilities
        self.test_prob_data = test_prob_data
        self.test_prob = ss.bernoulli(self.make_test_prob_fn)
        self.start = start
        self.eligibility = eligibility
        self._parse_product(product)

        # States
        self.tested = ss.BoolArr('tested')
        self.tests = ss.FloatArr('tests', default=0)
        self.ti_tested = ss.FloatArr('ti_tested')

        return

    def initialize(self, sim):
        super().initialize(self, sim)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        self.results += [
            ss.Result('syphilis', 'n_tested', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_dx', sim.npts, dtype=int, scale=True),
        ]
        return

    @staticmethod
    def make_test_prob_fn(self, sim, uids):
        """ Process symptomatic testing probabilites over time by sex and risk group """

        if sc.isnumber(self.test_prob_data):
            test_prob = self.test_prob_data

        else:
            available_years = self.test_prob_data.year.unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]

            df = self.test_prob_data.loc[self.test_prob_data.year == nearest_year]
            df_arr = df[val_label].values  # Pull out dataframe values

        # Scale and validate
        test_prob = test_prob * self.pars.rel_test * sim.dt
        test_prob = np.clip(test_prob, a_min=0, a_max=1)

        return test_prob

    def get_testers(self, sim):
        """
        Perform testing by finding who's eligible, finding who accepts, and applying the product.
        """
        accept_uids = ss.uids()
        eligible_uids = self.check_eligibility(sim)  # Apply eligiblity
        if len(eligible_uids):
            accept_uids = self.test_prob.filter(eligible_uids)
        return accept_uids

    def apply(self, sim):
        accept_uids = self.get_testers(sim)
        if len(accept_uids):
            self.outcomes = self.product.administer(sim, accept_uids)
        return




