"""
Define interventions for STIsim
"""

import starsim as ss
import numpy as np
import pandas as pd
from collections import defaultdict
import sciris as sc


__all__ = ['HIVTest', 'ART']


class HIVTest(ss.Intervention):
    """
    Base class for HIV testing

    Args:
         prob           (float/arr)     : annual probability of eligible people being tested
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of screening strategy
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, pars=None, test_prob_data=None, years=None, eligibility=None, name=None, label=None, **kwargs):
        super().__init__(name=name, label=label)
        self.default_pars(
            rel_test=1,
        )
        self.update_pars(pars, **kwargs)

        # Set testing probabilities and years
        self.years = years
        self.test_prob_data = test_prob_data
        self.test_prob = ss.bernoulli(self.make_test_prob_fn)

        # Set eligibility and timepoints
        self.eligibility = eligibility
        self.timepoints = []  # The start and end timepoints of the intervention

        # States
        self.tested = ss.BoolArr('tested', default=False)
        self.ti_tested = ss.FloatArr('ti_tested')
        self.diagnosed = ss.BoolArr('diagnosed', default=False)
        self.ti_diagnosed = ss.FloatArr('ti_diagnosed')

    def initialize(self, sim):
        super().initialize(sim)
        self.init_results()
        return

    def init_results(self):
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'new_diagnoses', npts, dtype=float, scale=True),
            ss.Result(self.name, 'new_tests', npts, dtype=int, scale=True)]
        return

    @staticmethod
    def make_test_prob_fn(self, sim, uids):
        """ Testing probabilites over time """

        if sc.isnumber(self.test_prob_data):
            test_prob = self.test_prob_data

        elif sc.checktype(self.test_prob_data, 'arraylike'):
            year_ind = sc.findnearest(self.years, sim.year)
            test_prob = self.test_prob_data[year_ind]
        else:
            errormsg = 'Format of test_prob_data must be float or array.'
            raise ValueError(errormsg)

        # Scale and validate
        test_prob = test_prob * self.pars.rel_test * sim.dt
        test_prob = np.clip(test_prob, a_min=0, a_max=1)

        return test_prob

    def apply(self, sim):
        hiv = sim.diseases.hiv

        # Find who's eligible to test, who gets a test, and who is diagnosed
        eligible_uids = self.check_eligibility(sim)  # Apply eligiblity
        if len(eligible_uids):
            tester_uids = self.test_prob.filter(eligible_uids)
            if len(tester_uids):
                # Add results and states for testers
                self.results['new_tests'][sim.ti] += len(tester_uids)
                self.tested[tester_uids] = True
                self.ti_tested[tester_uids] = sim.ti

                # Add results and states for diagnoses
                pos_uids = tester_uids[hiv.infected[tester_uids]]
                self.results['new_diagnoses'][sim.ti] += len(pos_uids)
                self.diagnosed[pos_uids] = True
                self.ti_diagnosed[pos_uids] = sim.ti
                hiv.diagnosed[pos_uids] = True
                hiv.ti_diagnosed[pos_uids] = sim.ti

        return

    def check_eligibility(self, sim):
        return self.eligibility(sim).uids


class ART(ss.Intervention):
    """
    ART-treatment intervention by Robyn Stuart, Daniel Klein and Cliff Kerr, edited by Alina Muellenmeister
    """

    def __init__(self, pars=None, coverage_data=None, **kwargs):
        super().__init__()
        self.default_pars(
            dur_on_art=ss.normal(loc=18, scale=5),
            dur_post_art=ss.normal(loc=self.dur_post_art_mean, scale=self.dur_post_art_scale),
            dur_post_art_scale_factor=0.1,
            art_cd4_pars=dict(cd4_max=1000, cd4_healthy=500),
            init_prob=ss.bernoulli(p=0.9),  # Probability that a newly diagnosed person will initiate treatment
            future_coverage={'year': 2022, 'prop':0.9},
        )
        self.update_pars(pars, **kwargs)
        self.coverage_data = coverage_data
        self.coverage = None  # Set below
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.coverage = sc.smoothinterp(sim.yearvec, self.coverage_data.index.values, self.coverage_data.n_art.values)
        self.initialized = True
        return

    @staticmethod
    def dur_post_art_fn(module, sim, uids):
        hiv = sim.diseases.hiv
        dur_mean = np.log(hiv.cd4_preart[uids])*hiv.cd4[uids]/hiv.cd4_potential[uids]
        dur_scale = dur_mean * module.pars.dur_post_art_scale_factor
        return dur_mean, dur_scale

    @staticmethod
    def dur_post_art_mean(module, sim, uids):
        mean, _ = module.dur_post_art_fn(module, sim, uids)
        return mean

    @staticmethod
    def dur_post_art_scale(module, sim, uids):
        _, scale = module.dur_post_art_fn(module, sim, uids)
        return scale

    def apply(self, sim):
        """
        Apply the ART intervention at each time step. Put agents on and off ART and adjust based on data.
        """
        hiv = sim.diseases.hiv
        inf_uids = hiv.infected.uids

        # Figure out how many people should be treated
        if sim.year < self.pars.future_coverage['year']:
            n_to_treat = int(self.coverage[sim.ti]/sim.pars.pop_scale)
        else:
            p_cov = self.pars.future_coverage['prop']
            n_to_treat = int(p_cov*len(inf_uids))

        # Firstly, check who is stopping ART
        if hiv.on_art.any():
            stopping = hiv.on_art & (hiv.ti_stop_art <= sim.ti)
            if stopping.any():
                self.stop_art(stopping.uids)

        # Next, see how many people we need to treat vs how many are already being treated
        ART_coverage = ART_coverage_this_year
        inf_uids = hiv.infected.uids
        dx_uids = hiv.diagnosed.uids
        n_to_treat = int(ART_coverage*len(dx_uids))
        on_art = hiv.on_art

        # A proportion of newly diagnosed agents onto ART will be willing to initiate ART
        diagnosed = hiv.ti_diagnosed == sim.ti
        if len(diagnosed.uids):
            dx_to_treat = self.pars.init_prob.filter(diagnosed.uids)

            # Figure out if there are treatment spots available and if so, prioritize newly diagnosed agents
            n_available_spots = n_to_treat - len(on_art.uids)
            if n_available_spots > 0:
                self.prioritize_art(sim, n=n_available_spots, awaiting_art_uids=dx_to_treat)

        # Apply correction to match ART coverage data:
        self.art_coverage_correction(sim, target_coverage=n_to_treat)

        # Adjust rel_sus for protected unborn agents
        if hiv.on_art[sim.people.pregnancy.pregnant].any():
            mother_uids = (hiv.on_art & sim.people.pregnancy.pregnant).uids
            infants = sim.networks.maternalnet.find_contacts(mother_uids)
            hiv.rel_sus[ss.uids(infants)] = 0

        return

    def start_art(self, sim, uids):
        """
        Check who is ready to start ART treatment and put them on ART
        """
        ti = sim.ti
        dt = sim.dt

        hiv = sim.diseases.hiv
        hiv.on_art[uids] = True
        newly_treated = uids[hiv.never_art[uids]]
        hiv.never_art[newly_treated] = False
        hiv.ti_art[uids] = ti
        hiv.cd4_preart[uids] = hiv.cd4[uids]

        # Determine when agents goes off ART
        dur_on_art = self.pars.dur_on_art.rvs(uids)
        hiv.ti_stop_art[uids] = ti + (dur_on_art / dt).astype(int)

        # ART nullifies all states and all future dates in the natural history
        hiv.acute[uids] = False
        hiv.latent[uids] = False
        hiv.falling[uids] = False
        future_latent = uids[hiv.ti_latent[uids] > sim.ti]
        hiv.ti_latent[future_latent] = np.nan
        future_falling = uids[hiv.ti_falling[uids] > sim.ti]
        hiv.ti_falling[future_falling] = np.nan
        future_zero = uids[hiv.ti_zero[uids] > sim.ti]  # NB, if they are scheduled to die on this time step, they will
        hiv.ti_zero[future_zero] = np.nan

        # Set CD4 potential for anyone new to treatment - retreated people have the same potential
        # Extract growth parameters
        if len(newly_treated) > 0:
            cd4_max = self.pars.art_cd4_pars['cd4_max']
            cd4_healthy = self.pars.art_cd4_pars['cd4_healthy']
            cd4_preart = hiv.cd4_preart[newly_treated]

            # Calculate potential CD4 increase - assuming that growth follows the concave part of a logistic function
            # and that the total gain depends on the CD4 count at initiation
            cd4_scale_factor = (cd4_max-cd4_preart)/cd4_healthy*np.log(cd4_max/cd4_preart)
            cd4_total_gain = cd4_preart*cd4_scale_factor
            hiv.cd4_potential[newly_treated] = hiv.cd4_preart[newly_treated] + cd4_total_gain

        return

    def stop_art(self, uids=None):
        """
        Check who is stopping ART treatment and put them off ART
        """
        hiv = self.sim.diseases.hiv
        ti = self.sim.ti
        dt = self.sim.dt

        # Remove agents from ART
        if uids is None: uids = hiv.on_art & (hiv.ti_stop_art <= ti)
        hiv.on_art[uids] = False
        hiv.cd4_postart[uids] = sc.dcp(hiv.cd4[uids])

        # Set decline
        dur_post_art = self.pars.dur_post_art.rvs(uids)
        hiv.ti_zero[uids] = ti + (dur_post_art / dt).astype(int)

        return

    def prioritize_art(self, sim, n=None, awaiting_art_uids=None):
        """
        Prioritize ART to n agents among those awaiting treatment
        """
        hiv = sim.diseases.hiv
        if awaiting_art_uids is None:
            awaiting_art_uids = (hiv.diagnosed & ~hiv.on_art).uids

        # Enough spots for everyone
        if n > len(awaiting_art_uids):
            start_uids = awaiting_art_uids

        # Not enough spots - construct weights based on CD4 count and care seeking
        else:
            cd4_counts = hiv.cd4[awaiting_art_uids]
            care_seeking = hiv.care_seeking[awaiting_art_uids]
            weights = cd4_counts*(1/care_seeking)
            choices = np.argsort(weights)[:n]
            start_uids = awaiting_art_uids[choices]

        self.start_art(sim, start_uids)

        return

    def art_coverage_correction(self, sim, target_coverage=None):
        """
        Adjust ART coverage to match data
        """
        hiv = sim.diseases.hiv
        on_art = hiv.on_art

        # Too many agents on treatment -> remove
        if len(on_art.uids) > target_coverage:

            # Agents with the highest CD4 counts will go off ART:
            n_to_stop = int(len(on_art.uids) - target_coverage)
            on_art_uids = on_art.uids

            # Construct weights and choice distribution
            cd4_counts = hiv.cd4[on_art_uids]
            care_seeking = hiv.care_seeking[on_art_uids]
            weights = cd4_counts/care_seeking
            choices = np.argsort(-weights)[:n_to_stop]
            stop_uids = on_art_uids[choices]

            hiv.ti_stop_art[stop_uids] = sim.ti
            self.stop_art(stop_uids)

        # Not enough agents on treatment -> add
        elif len(on_art.uids) < target_coverage:
            n_to_add = target_coverage - len(on_art.uids)
            awaiting_art_uids = (hiv.diagnosed & ~hiv.on_art).uids
            self.prioritize_art(sim, n=n_to_add, awaiting_art_uids=awaiting_art_uids)


