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
    df = sc.dataframe.read_csv(sti.data/'syph_dx.csv')
    hierarchy = ['positive', 'inadequate', 'negative']
    dxprods = dict(
        rpr = ss.Dx(df[df.name == 'rpr'], hierarchy=hierarchy),
        rst = ss.Dx(df[df.name == 'rst'], hierarchy=hierarchy),
        syndromic = ss.Dx(df[df.name == 'syndromic'], hierarchy=hierarchy),
        dual = ss.Dx(df[df.name == 'dual'], hierarchy=hierarchy),
        newborn_exam = ss.Dx(df[df.name == 'newborn_exam'], hierarchy=hierarchy),
        symp_test_assigner = ss.Dx(df[df.name == 'symp_test_assigner'], hierarchy=None),
        pos_assigner = ss.Dx(df[df.name == 'pos_assigner'], hierarchy=None),
    )
    return dxprods


__all__ = ['TestProb', 'ANCTesting', 'LinkedNewbornTesting', 'TreatNum']


class TestProb(ss.Intervention):
    """ Base class for syphilis tests """
    def __init__(self, test_prob_data=None, pars=None, product=None, eligibility=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label)
        self.default_pars(
            rel_test=1,
        )
        self.update_pars(pars, **kwargs)

        # Set testing probabilities
        self.test_prob_data = test_prob_data
        self.test_prob = ss.bernoulli(self.make_test_prob_fn)

        # More initialization
        self.eligibility = eligibility  # Store eligibility
        self._parse_product(product)  # Parse product - belongs to ss.Intervention class

        # States (needed?)
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

    @staticmethod
    def make_test_prob_fn(self, sim, uids):
        """ Process symptomatic testing probabilites over time by sex and risk group """

        if sc.isnumber(self.test_prob_data):
            test_prob = self.test_prob_data

        elif sc.checktype(self.test_prob_data, 'arraylike'):
            year_ind = sc.findnearest(self.years, sim.year)
            test_prob = self.test_prob_data[year_ind]
            test_prob = test_prob * self.pars.rel_test * sim.dt
            test_prob = np.clip(test_prob, a_min=0, a_max=1)

        elif isinstance(self.test_prob_data, pd.DataFrame):
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

        else:
            errormsg = 'Format of test_prob_data must be float, array, or dataframe.'
            raise ValueError(errormsg)

        # Scale and validate
        test_prob = test_prob * self.pars.rel_test * sim.dt
        test_prob = np.clip(test_prob, a_min=0, a_max=1)

        return test_prob

    def initialize(self, sim):
        super().initialize(sim)
        self.outcomes = {k: np.array([], dtype=int) for k in self.product.hierarchy}
        self.results += [
            ss.Result('syphilis', 'n_tested', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_diagnosed', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_false_pos', sim.npts, dtype=int, scale=True),
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

    def apply(self, sim):
        accept_uids = self.get_testers(sim)
        if len(accept_uids):
            self.outcomes = self.product.administer(sim, accept_uids)

        # Store results
        self.results['n_tested'][sim.ti] = len(accept_uids)
        if 'positive' in self.product.hierarchy and (len(self.outcomes['positive'])>0):
            self.results['n_diagnosed'][sim.ti] = len(self.outcomes['positive'])
            false_pos = np.count_nonzero(sim.diseases.syphilis.susceptible[self.outcomes['positive']])
            self.results['n_false_pos'][sim.ti] = false_pos

        return


class ANCTesting(TestProb):
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

        # Must be pregnant to be eligible
        if self.eligibility is None:
            self.eligibility = lambda sim: sim.demographics.pregnancy.pregnant

        return

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


class LinkedNewbornTesting(TestProb):
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
        queue = sim.interventions.anc.newborn_queue
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
                sim.interventions.anc.newborn_queue[key] = new_queue_entry

        return


class TreatNum(ss.Intervention):
    """
    Treat a fixed number of people each timestep.
    """
    def __init__(self, pars=None, treat_prob_data=None, max_capacity=None, years=None, eligibility=None, **kwargs):
        super().__init__(eligibility=eligibility)
        self.default_pars(
            rel_treat_prob=1,
            treat_eff=ss.bernoulli(p=0.95)  # Assuming high efficacy
        )
        self.update_pars(pars, **kwargs)
        self.treat_prob_data = treat_prob_data
        self.treat_prob = ss.bernoulli(self.make_treat_prob_fn)

        self.queue = []
        self.max_capacity = max_capacity
        self.years = years
        self.outcomes = sc.objdict(
            successful=[],
            unsuccessful=[],
            unnecessary=[],
        )
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += [
            ss.Result('syphilis', 'n_treated_success', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_treated_failure', sim.npts, dtype=int, scale=True),
            ss.Result('syphilis', 'n_treated_unnecessary', sim.npts, dtype=int, scale=True),
        ]
        return

    @staticmethod
    def make_treat_prob_fn(self, sim, uids):
        """ Process treatment uptake probabilites over time by sex & pregnancy """

        if sc.isnumber(self.treat_prob_data):
            treat_prob = self.treat_prob_data

        else:
            treat_prob = pd.Series(index=uids)

            # Deal with year
            available_years = self.treat_prob_data.year.unique()
            year_ind = sc.findnearest(available_years, sim.year)
            nearest_year = available_years[year_ind]
            df = self.treat_prob_data.loc[self.treat_prob_data.year == nearest_year]

            # Males
            thisdf = df.loc[(df.sex=='male')]
            conditions = (sim.people.male)
            treat_prob[conditions[uids]] = thisdf.treat_prob.values[0]

            # Females by pregnancy status
            for preg in [0, 1]:
                thisdf = df.loc[(df.sex=='female') & (df.pregnant==preg)]
                conditions = (sim.people.female & (sim.demographics.pregnancy.pregnant==preg))
                treat_prob[conditions[uids]] = thisdf.treat_prob.values[0]

        # Scale and validate
        treat_prob = treat_prob * self.pars.rel_treat_prob * sim.dt
        treat_prob = np.clip(treat_prob, a_min=0, a_max=1)

        return treat_prob

    def add_to_queue(self, sim):
        """
        Add people who are willing to accept treatment to the queue
        """
        accept_uids = ss.uids()
        eligible_uids = self.check_eligibility(sim)  # Apply eligiblity - uses base class from ss.Intervention
        if len(eligible_uids):
            accept_uids = self.treat_prob.filter(eligible_uids)
        if len(accept_uids): self.queue += accept_uids.tolist()
        return

    def get_candidates(self, sim):
        """
        Get the indices of people who are candidates for treatment
        """
        treat_candidates = np.array([], dtype=int)

        if len(self.queue):

            if self.max_capacity is None:
                treat_candidates = self.queue[:]

            else:
                if sc.isnumber(self.max_capacity):
                    max_capacity = self.max_capacity
                elif sc.checktype(self.max_capacity, 'arraylike'):
                    year_ind = sc.findnearest(self.years, sim.year)
                    max_capacity = self.max_capacity[year_ind]

                if max_capacity > len(self.queue):
                    treat_candidates = self.queue[:]
                else:
                    treat_candidates = self.queue[:self.max_capacity]

        return ss.uids(treat_candidates)

    def administer(self, sim, uids, return_format='dict'):
        """ Administer treatment, keeping track of unnecessarily treated individuals """

        inf = sim.diseases.syphilis.infected
        sus = sim.diseases.syphilis.susceptible
        inf_uids = uids[inf[uids]]
        sus_uids = uids[sus[uids]]

        successful = self.pars.treat_eff.filter(inf_uids)
        unsuccessful = np.setdiff1d(inf_uids, successful)
        unnecessary = sus_uids

        if return_format == 'dict':
            output = {'successful': successful, 'unsuccessful': unsuccessful, 'unnecessary': unnecessary}
        elif return_format == 'array':
            output = successful

        return output

    def apply(self, sim):
        """
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        queue, and then will treat as many people in the queue as there is capacity for.
        """
        self.add_to_queue(sim)

        # Get indices of who will get treated and check they're still eligible
        treat_candidates = self.get_candidates(sim)
        still_eligible = self.check_eligibility(sim)
        treat_uids = treat_candidates.intersect(still_eligible)

        # Treat people
        if len(treat_uids):
            self.outcomes = self.administer(sim, treat_uids)
        self.queue = [e for e in self.queue if e not in treat_uids] # Recreate the queue, removing treated people

        # Store results
        self.results['n_treated_success'][sim.ti] = len(self.outcomes['successful'])
        self.results['n_treated_failure'][sim.ti] = len(self.outcomes['unsuccessful'])
        self.results['n_treated_unnecessary'][sim.ti] = len(self.outcomes['unnecessary'])

        return treat_uids

