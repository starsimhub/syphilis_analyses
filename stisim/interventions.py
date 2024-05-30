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
            dur_on_art=ss.normal(loc=18, scale=5),
            dur_post_art=ss.normal(loc=self.dur_post_art_mean, scale=self.dur_post_art_scale),
            dur_post_art_scale_factor=0.1,
            art_cd4_pars=dict(cd4_max=1000, cd4_healthy=500),
            init_prob=ss.bernoulli(p=0.9),  # Probability that a newly diagnosed person will initiate treatment
            )
        self.update_pars(pars, **kwargs)
        return

    def initialize(self, sim):
        super().initialize(sim)
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

        # Get the current ART coverage. If year is not available, assume 90%
        if len(self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()) > 0:
            ART_coverage_this_year = self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()[0]
        else:
            ART_coverage_this_year = self.pars.ART_coverages_df.Value.iloc[-1]  # Assume last coverage

        # Firstly, check who is stopping ART
        if hiv.on_art.any():
            stopping = hiv.on_art & (hiv.ti_stop_art <= sim.ti)
            if stopping.any():
                self.stop_art(stopping.uids)

        # Next, see how many people we need to treat vs how many are already being treated
        ART_coverage = ART_coverage_this_year
        inf_uids = hiv.infected.uids
        # dx_uids = hiv.diagnosed.uids
        n_to_treat = int(ART_coverage*len(inf_uids))
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
        hiv.acute = False
        hiv.latent = False
        hiv.falling = False
        future_latent = uids[hiv.ti_latent[uids] > sim.ti]
        hiv.ti_latent[future_latent] = np.nan
        future_falling = uids[hiv.ti_falling[uids] > sim.ti]
        hiv.ti_falling[future_falling] = np.nan
        future_dead = uids[hiv.ti_dead[uids] > sim.ti]  # NB, if they are scheduled to die on this time step, they will
        hiv.ti_dead[future_dead] = np.nan

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
<<<<<<< HEAD
        elif len(infected_uids_onART.uids) < ART_coverage_this_year:
            # Agents with the lowest CD4 count will get on ART:
            n_agents_to_start_ART = int(ART_coverage_this_year - len(infected_uids_onART.uids))
            cd4_counts_not_onART = sim.diseases[self.disease].cd4[infected_uids_not_onART]
            # Sort
            uids_not_onART = infected_uids_not_onART.uids
            cd4_counts_sort_idx = np.argsort(cd4_counts_not_onART)
            uids_not_onART_sorted = uids_not_onART[cd4_counts_sort_idx]
            probabilities = (cd4_counts_not_onART / np.sum(cd4_counts_not_onART))
            # Probabilities are increasing with CD4 count, therefore flip uid array:
            uids = np.flipud(uids_not_onART_sorted)
            if n_agents_to_start_ART > len(infected_uids_not_onART.uids):
                start_uids = infected_uids_not_onART.uids
            else:
                start_uids = np.random.choice(uids, n_agents_to_start_ART, p=probabilities, replace=False)

            # Put them on ART
            sim.diseases[self.disease].on_art[ss.uids(start_uids)] = True
            sim.diseases[self.disease].ti_art[ss.uids(start_uids)] = sim.ti

        return

    def update_ART_pregnancies(self, sim):
        """
        Start ART for proportion of pregnant women and put them off ART after 9 months
        """
        pregnant_uids = (sim.diseases[self.disease].infected & (sim.people.pregnancy.ti_pregnant == sim.ti)).uids

        # Get the current ARV coverage.
        if len(self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()) > 0:
            ARV_coverage_this_year = self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()[0]
        else:
            ARV_coverage_this_year = self.pars.ART_coverages_df.Value.iloc[-1]  # Assume last coverage

        pregnant_to_start_ART = pregnant_uids[np.random.random(len(pregnant_uids)) < ARV_coverage_this_year]
        sim.diseases[self.disease].on_art[pregnant_to_start_ART] = True
        sim.diseases[self.disease].ti_art[pregnant_to_start_ART] = sim.ti
        # Determine when agents goes off ART:
        sim.diseases[self.disease].ti_stop_art[pregnant_to_start_ART] = sim.ti + 9 # Put them off ART in 9 months

        # Decrease susceptibility for any unborn infants of pregnant women on ART
        pregnant_onART_uids = (sim.people.pregnancy.pregnant & sim.diseases[self.disease].on_art).uids
        infants = sim.networks.maternalnet.find_contacts(pregnant_onART_uids)
        # TODO Update! Susceptibility should increase again later
        sim.diseases['hiv'].rel_sus[ss.uids(infants)] = 0
        return


class DualTest(ss.Intervention):
    """ Dial test for diagnosing HIV and syphilis """
    def __init__(self, pars=None):
        return

    def apply(self):
        return
=======
        elif len(on_art.uids) < target_coverage:
            n_to_add = target_coverage - len(on_art.uids)
            awaiting_art_uids = (hiv.diagnosed & ~hiv.on_art).uids
            self.prioritize_art(sim, n=n_to_add, awaiting_art_uids=awaiting_art_uids)
>>>>>>> 9f2386d42bb5eb78ac3f4d4bb770b27d3fe0fcdf


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
