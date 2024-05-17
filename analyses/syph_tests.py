"""
Define syphilis diagnostics
"""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti


# %% Syphilis diagnostics

def load_syph_dx():
    """
    Create default diagnostic products
    """
    df = sc.dataframe.read_csv('data/syph_dx.csv')
    hierarchy = ['positive', 'inadequate', 'negative']
    dxprods = dict(
        rpr = ss.Dx(df[df.name == 'rpr'], hierarchy=hierarchy),
        rst = ss.Dx(df[df.name == 'rst'], hierarchy=hierarchy),
        syndromic = ss.Dx(df[df.name == 'syndromic'], hierarchy=hierarchy),
        dual = ss.Dx(df[df.name == 'dual'], hierarchy=hierarchy),
    )
    return dxprods


__all__ = ['BaseTest', 'SymptomaticTesting' , 'ANCTesting', 'LinkedNewbornTesting']


class BaseTest(ss.Intervention):
    """ Base class for syphilis tests """
    def __init__(self, pars=None, product=None, test_prob_data=None, start=None, eligibility=None, **kwargs):
        super().__init__()
        self.default_pars(
            rel_test=1
        )
        self.update_pars(pars, **kwargs)
        self.start = start
        self.eligibility = eligibility
        self._parse_product(product)

        # States
        self.tested = ss.BoolArr('tested')
        self.tests = ss.FloatArr('tests', default=0)
        self.ti_tested = ss.FloatArr('ti_tested')

        return

    def _parse_product_str(self, product):
        products = load_syph_dx()
        if product not in products:
            errormsg = f'Could not find diagnostic product {product} in the standard list ({sc.strjoin(products.keys())})'
            raise ValueError(errormsg)
        else:
            return products[product]

    def initialize(self, sim):
        super().initialize(sim)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        self.results += [
            ss.Result('syphilis', 'n_tested', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_dx', sim.npts, dtype=int, scale=True),
        ]
        return

    def get_testers(self, sim):
        """
        Find who tests by applying eligibility and coverage/uptake
        """
        accept_uids = ss.uids()
        eligible_uids = self.check_eligibility(sim)  # Apply eligiblity
        if len(eligible_uids):
            accept_uids = self.test_prob.filter(eligible_uids)
        return accept_uids

    @staticmethod
    def make_test_prob_fn(self, sim, uids):
        """ Process ANC testing probabilites over time """
        year_ind = sc.findnearest(self.years, sim.year)
        test_prob = self.test_prob_data[year_ind]
        test_prob = test_prob * self.pars.rel_test * sim.dt
        test_prob = np.clip(test_prob, a_min=0, a_max=1)
        return test_prob

    def apply(self, sim):
        accept_uids = self.get_testers(sim)
        if len(accept_uids):
            self.outcomes = self.product.administer(sim, accept_uids)
        return


class SymptomaticTesting(BaseTest):
    """
    Test given to those presenting with active lesions or genital ulcers
    """
    def __init__(self,test_prob_data=None,  pars=None, product=None, start=None, eligibility=None, **kwargs):
        super().__init__(pars=pars, product=product, start=start, eligibility=eligibility, **kwargs)
        # Set testing probabilities
        self.test_prob_data = test_prob_data
        self.test_prob = ss.bernoulli(self.make_test_prob_fn)
        return

    def check_eligibility(self, sim):
        conditions = sim.diseases.syphilis.active
        if self.eligibility is not None:
            other_eligible  = sc.promotetoarray(self.eligibility(sim)) 
            conditions = conditions & other_eligible
        return conditions.uids

    @staticmethod
    def make_test_prob_fn(self, sim, uids):
        """ Process symptomatic testing probabilites over time by sex and risk group """

        if sc.isnumber(self.test_prob_data):
            test_prob = self.test_prob_data

        else:
            test_prob = pd.Series(index=uids)

            # Deal with year
            available_years = self.test_prob_data.year.unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]
            df = self.test_prob_data.loc[self.test_prob_data.year == nearest_year]

            n_risk_groups = sim.networks.structuredsexual.pars.n_risk_groups
            for rg in range(n_risk_groups):
                for sex in ['female', 'male']:
                    for sw in [0, 1]:
                        thisdf = df.loc[(df.risk_group==rg) & (df.sex==sex) & (df.sw==sw)]
                        conditions = (sim.people[sex] & (sim.networks.structuredsexual.risk_group==rg))
                        if sw:
                            if sex == 'female': conditions = conditions & sim.networks.structuredsexual.fsw
                            if sex == 'male':   conditions = conditions & sim.networks.structuredsexual.client
                        test_prob[conditions[uids]] = thisdf.symp_test_prob.values[0]

        # Scale and validate
        test_prob = test_prob * self.pars.rel_test * sim.dt
        test_prob = np.clip(test_prob, a_min=0, a_max=1)

        return test_prob


class ANCTesting(BaseTest):
    """
    Test given to pregnant women
    Need to adjust timing using Trivedi (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7138526/)
    """
    def __init__(self, test_prob_data=None, years=None, pars=None, product=None, eligibility=None, **kwargs):
        super().__init__(pars=pars, product=product, eligibility=eligibility, **kwargs)
        self.test_prob = ss.bernoulli(self.make_test_prob_fn)
        self.test_prob_data = test_prob_data
        self.years = years
        self.newborn_queue = sc.objdict()  # For women who test positive, schedule their newborns for testing post birth
        self.newborn_queue['uids'] = []
        self.newborn_queue['ti_births'] = []
        self.newborn_queue['ti_mother_tested'] = []

        return

    def check_eligibility(self, sim):
        conditions = sim.demographics.pregnancy.pregnant
        if self.eligibility is not None:
            other_eligible  = sc.promotetoarray(self.eligibility(sim)) # Apply any other user-defined eligibility
            conditions      = conditions & other_eligible
        return conditions.uids

    def apply(self, sim):
        super().apply(sim)
        positives = self.outcomes['positive']
        if len(positives):
            pos_mother_inds = np.in1d(sim.networks.maternalnet.p1, positives)
            unborn_uids = sim.networks.maternalnet.p2[pos_mother_inds]
            ti_births = sim.networks.maternalnet.contacts.end[pos_mother_inds].astype(int)
            self.newborn_queue['uids'] += unborn_uids.tolist()
            self.newborn_queue['ti_births'] += ti_births.tolist()
            self.newborn_queue['ti_mother_tested'] += [sim.ti]*len(positives)
        return


class LinkedNewbornTesting(BaseTest):
    """
    Test given to newborns if the mother was confirmed to have syphilis at any stage of the pregnancy
    """
    def __init__(self, test_prob_data=None, years=None, pars=None, product=None, eligibility=None, **kwargs):
        super().__init__(
            pars=pars, product=product, eligibility=eligibility,
            requires=[sti.Syphilis, ANCTesting],
            **kwargs
        )
        self.test_prob = ss.bernoulli(self.make_test_prob_fn)
        self.test_prob_data = test_prob_data
        self.years = years
        return

    def apply(self, sim):
        queue = sim.interventions.anctesting.newborn_queue
        time_to_test = sc.findinds(queue.ti_births, sim.ti)

        # Attempt to test newborns
        if len(time_to_test)>0:
            eligible_uids = np.array(queue.uids)[time_to_test]
            accept_uids = self.test_prob.filter(eligible_uids)
            if len(accept_uids):
                self.outcomes = self.product.administer(sim, accept_uids)

            # Remove newborns from the testing queue
            for key in queue.keys():
                new_queue_entry = [val for i,val in enumerate(queue[key]) if i not in time_to_test]
                sim.interventions.anctesting.newborn_queue[key] = new_queue_entry

        return


