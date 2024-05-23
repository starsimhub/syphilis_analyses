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
            test_prob = test_prob * self.pars.rel_test * sim.dt
            test_prob = np.clip(test_prob, a_min=0, a_max=1)
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

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.default_pars(
            ART_coverages_df=None,
            ARV_coverages_df=None,
            duration_on_ART=ss.normal(loc=18, scale=5),
            art_efficacy=0.96,
            init_prob=ss.bernoulli(p=0.9),  # Probability that a newly diagnosed person will initiate treatment
            )
        self.update_pars(pars, **kwargs)
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.initialized = True
        return

    def apply(self, sim):
        """
        Apply the ART intervention at each time step. Put agents on and off ART and adjust based on data.
        """
        hiv = sim.diseases.hiv

        # Get the current ART coverage. If year is not available, assume 90%
        if len(self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()) > 0:
            ART_coverage_this_year = self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()[0]
        else:
            ART_coverage_this_year = self.pars.ART_coverages_df.Value.iloc[-1]  # Assume last coverage

        # Firstly, check who is stopping ART
        if hiv.on_art.any():
            self.stop_art(sim)

        # Next, see how many people we need to treat vs how many are already being treated
        ART_coverage = ART_coverage_this_year
        dx_uids = hiv.diagnosed.uids
        n_to_treat = int(ART_coverage*len(dx_uids))

        # Initiate a proportion of newly diagnosed agents onto ART
        diagnosed = hiv.ti_diagnosed == sim.ti
        if len(diagnosed.uids):
            dx_to_treat = self.pars.init_prob.filter(diagnosed.uids)
            self.start_art(sim, dx_to_treat)

        # Apply correction to match ART coverage data:
        self.art_coverage_correction(sim, n_to_treat)

        # Adjust rel_trans for all treated agents, and also rel_sus for protected unborn agents
        hiv.rel_trans[hiv.on_art] = 1 - self.pars.art_efficacy
        if hiv.on_art[sim.people.pregnancy.pregnant].any():
            mother_uids = (hiv.on_art & sim.people.pregnancy.pregnant).uids
            infants = sim.networks.maternalnet.find_contacts(mother_uids)
            hiv.rel_sus[ss.uids(infants)] = 0

        return

    def start_art(self, sim, uids):
        """
        Check who is ready to start ART treatment and put them on ART
        """
        hiv = sim.diseases.hiv
        hiv.on_art[uids] = True
        hiv.ti_art[uids] = sim.ti
        hiv.ti_reset_cd4[uids] = sim.ti
        hiv.cd4_preart[uids] = hiv.cd4[uids]

        # Determine when agents goes off ART:
        hiv.ti_stop_art[uids] = sim.ti + self.pars.duration_on_ART.rvs(uids).astype(int)

        return

    @staticmethod
    def stop_art(sim):
        """
        Check who is stopping ART treatment and put them off ART
        """
        hiv = sim.diseases.hiv
        ti = sim.ti

        # Non-pregnant agents
        stop_uids = hiv.on_art & (hiv.ti_stop_art <= ti)
        hiv.on_art[stop_uids] = False
        hiv.ti_reset_cd4[stop_uids] = ti

        return

    def art_coverage_correction(self, sim, n_to_treat):
        """
        Adjust number of people on treatment to match data
        """
        hiv = sim.diseases.hiv
        diag_treated = hiv.diagnosed & hiv.on_art
        diag_untreated = hiv.diagnosed & ~hiv.on_art

        # Too many agents on treatment -> remove
        if len(diag_treated.uids) > n_to_treat:

            # Agents with the highest CD4 counts will go off ART:
            n_to_stop = int(len(diag_treated.uids) - n_to_treat)
            on_art = diag_treated.uids

            # Construct weights and choice distribution
            cd4_counts = hiv.cd4[on_art]
            care_seeking = hiv.care_seeking[on_art]
            weights = cd4_counts*care_seeking
            choices = np.argsort(-weights)[:n_to_stop]
            stop_uids = on_art[choices]

            hiv.on_art[stop_uids] = False
            hiv.ti_stop_art[stop_uids] = sim.ti

        # Not enough agents on treatment -> add
        elif len(diag_treated.uids) < n_to_treat:

            # Calculate how many agents need to start ART and how many are available
            n_to_start = int(n_to_treat - len(diag_treated.uids))
            available = diag_untreated.uids
            n_available = len(available)

            if n_available > n_to_start:
                # Construct weights based on CD4 count and care seeking
                cd4_counts = hiv.cd4[available]
                care_seeking = hiv.care_seeking[available]
                weights = cd4_counts*(1/care_seeking)
                choices = np.argsort(weights)[:n_to_start]
                start_uids = available[choices]
            else:
                start_uids = available

            self.start_art(sim, start_uids)

        return


# %% Validation and other checks -- TODO, should this be an analyzer?

__all__ += ['validate_ART']


class validate_ART(ss.Intervention):
    """

    """

    def __init__(self, disease, uids, infect_uids_t, stop_ART=False, restart_ART=False, *args, **kwargs):
        """

        """
        super().__init__(**kwargs)

        self.disease = disease
        self.uids = uids
        self.infect_uids_t = infect_uids_t
        self.stop_ART = stop_ART
        self.restart_ART = restart_ART

        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results = ss.ndict()
        npts = self.sim.npts
        for index, uid in enumerate(self.uids):
            self.results += ss.Result(self.name, 'status_' + str(uid), npts, dtype=np.dtype(('U', 10)))
            self.results += ss.Result(self.name, 'ART_status_' + str(uid), npts, dtype=np.dtype(('U', 10)))
            self.results += ss.Result(self.name, 'cd4_count_' + str(uid), npts, dtype=float, scale=False)
            self.results += ss.Result(self.name, 'transmission_' + str(uid), npts, dtype=float, scale=False)

        return

    def save_viral_histories(self, sim):
        """
        Save results to csv if called
        """

        history_df = pd.DataFrame.from_dict(self.results)
        history_df.to_csv("viral_histories.csv")
        return

    def apply(self, sim):
        """
        Use this function to infect agents at the time step provided
        Save CD4 counts and viral load at each time step
        """
        for index, uid in enumerate(self.uids):
            # Check if it's time to infect this agent:
            if sim.ti == self.infect_uids_t[index] and uid in sim.people.alive.uids and sim.diseases[self.disease].infected[ss.uids(uid)] == False:
                sim.diseases[self.disease].infected[ss.uids(uid)] = True
                sim.diseases[self.disease].ti_infected[ss.uids(uid)] = sim.ti
                sim.diseases[self.disease].ti_since_untreated[ss.uids(uid)] = sim.ti
                sim.diseases[self.disease].susceptible[ss.uids(uid)] = False
                #sim.diseases[self.disease].syphilis_inf[ss.uids(uid)] = True
                #sim.diseases[self.disease].ti_syphilis_inf[ss.uids(uid)] = sim.ti

                # if self.stop_ART:
                #    sim.diseases[self.disease].ti_stop_art[uid] = sim.ti + sim.diseases[self.disease].pars.avg_duration_stop_ART

            if uid in sim.people.alive.uids:
                # Check if it's time to restart ART treatment:
                # if sim.diseases[self.disease].on_art[uid] and self.stop_ART and self.restart_ART and sim.ti == sim.diseases[self.disease].ti_stop_art[uid]:
                #    sim.diseases[self.disease].schedule_ART_treatment(np.array([uid]), sim.ti + sim.diseases[self.disease].pars.avg_duration_restart_ART)

                if sim.diseases[self.disease].on_art[ss.uids(uid)]:
                    ART_status = 'on_ART'
                else:
                    ART_status = 'not_on_ART'
                self.results['cd4_count_' + str(uid)][0] = sim.diseases[self.disease].cd4_start[ss.uids(uid)]
                self.results['cd4_count_' + str(uid)][sim.ti] = sim.diseases[self.disease].cd4[ss.uids(uid)]
                self.results['ART_status_' + str(uid)][sim.ti] = ART_status
                self.results['status_' + str(uid)][sim.ti] = 'alive'
                self.results['transmission_' + str(uid)][sim.ti] = sim.diseases[self.disease].rel_trans[ss.uids(uid)] * sim.diseases[self.disease].infected[ss.uids(uid)]

            else:
                self.results['cd4_count_' + str(uid)][sim.ti] = np.nan
                self.results['status_' + str(uid)][sim.ti] = 'dead'

        return
