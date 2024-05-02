"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss
import pandas as pd
from collections import defaultdict

__all__ = ['HIV']


class HIV(ss.Infection):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        # States
        self.add_states(
            ss.State('on_art', bool, False),
            ss.State('art_transmission_reduction', float, np.nan),
            ss.State('ti_art', int, ss.INT_NAN),
            ss.State('ti_stop_art', float, np.nan),
            ss.State('cd4_start', float, np.nan),
            ss.State('cd4', float, np.nan),  #
            ss.State('ti_dead', int, ss.INT_NAN),  # Time of HIV-cause death
            ss.State('diagnosed', bool, False),
            ss.State('dead', bool, False),
            ss.State('tested', bool, False),
            ss.State('ti_tested', float, np.nan),
            ss.State('ti_pos_test', float, np.nan),
            ss.State('ti_diagnosed', float, np.nan),
            ss.State('ti_infectious', float, np.nan),
            ss.State('ti_since_untreated', float, np.nan)
        )
        # Default parameters
        pars = ss.omergeleft(pars,
                             cd4_start_mean=500,
                             cd4_min=100,
                             cd4_max=500,
                             cd4_rate=5,
                             init_prev=0.05,
                             transmission_sd=0.025,
                             primary_acute_inf_dur=1,  # in months
                             art_efficacy=0.96,
                             death_data=None,
                             death_prob=0.05)

        par_dists = ss.omergeleft(par_dists,
                                  init_prev=ss.bernoulli,
                                  death_prob=ss.bernoulli,
                                  )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self.death_prob_data = sc.dcp(self.pars.death_prob)
        self.pars.death_prob = self.make_death_prob
        self._pending_ARTtreatment = defaultdict(list)
        # self.art_transmission_reduction = None

        return

    def initialize(self, sim=None):
        """
        Initialize
        """
        super().initialize(sim)
        self.cd4_start[sim.people.uid] = ss.normal(loc=self.pars.cd4_start_mean, scale=100).initialize().rvs(len(sim.people))
        self.cd4[sim.people.uid] = self.cd4_start[sim.people.uid]
        self.pars.transmission_timecourse = self.get_transmission_timecourse()
        self.pars.viral_timecourse, self.pars.cd4_timecourse = self.get_viral_dynamics_timecourses()
        self.rel_trans[sim.people.uid] = 0
        # self.art_transmission_reduction = self.pars.art_efficacy / 6  # Assumption: 6 months

        return

    @property
    def symptomatic(self):
        return self.infectious

    @staticmethod
    def make_death_prob(self, sim, uids):
        # p = module.pars
        # out = sim.dt * module.death_prob_data / (p.cd4_min - p.cd4_max) ** 2 * (module.cd4[uids] - p.cd4_max) ** 2
        # out = np.array(out)
        # TODO probably find a better place for this
        death_data = [(range(500, int(np.ceil(np.max(self.cd4_start)))), 0.0036 / 12),
                      (range(350, 500), 0.0036 / 12),
                      (range(200, 350), 0.0088 / 12),
                      (range(50, 200), 0.059 / 12),
                      (range(0, 50), 0.323 / 12)]
        death_probs = [probs[1] for probs in death_data for cd4_count in self.cd4[uids].values if int(cd4_count) in probs[0]]
        return death_probs

    def get_transmission_timecourse(self):
        """
        Define transmission time course
        """
        transmission_timecourse_data = [(0, 0),
                                        (1, 6),
                                        (self.pars.primary_acute_inf_dur, 6),
                                        (np.ceil(self.pars.primary_acute_inf_dur), 1)]
        transmission_timecourse = self._interpolate(transmission_timecourse_data,
                                                    np.arange(0, np.ceil(self.pars.primary_acute_inf_dur) + 1))

        return transmission_timecourse

    def get_viral_dynamics_timecourses(self):
        """
        Define viral dynamic time courses for viral load and CD4 counts
        """

        viral_load_timecourse_data = [(0, 0), (0.5, 1), (1, (10 ** 4.5) / 1e6), (4 * 12, (10 ** 4.5) / 1e6),
                                      (8 * 12, 1)]  # in months
        cd4_timecourse_data = [(0, 1), (0.5, 600 / 1000), (1, 700 / 1000), (1 * 12, 700 / 1000),
                               (10 * 12, 0)]  # in months

        viral_load_timecourse = self._interpolate(viral_load_timecourse_data, np.arange(0, 8 * 12))
        cd4_timecourse = self._interpolate(cd4_timecourse_data, np.arange(0, 10 * 12 + 1))

        return viral_load_timecourse, cd4_timecourse

    @staticmethod
    def _interpolate(vals: list, t):
        vals = sorted(vals, key=lambda x: x[0])  # Make sure values are sorted
        assert len({x[0] for x in vals}) == len(vals)  # Make sure time points are unique
        return np.interp(t, [x[0] for x in vals], [x[1] for x in vals], left=vals[0][1], right=vals[-1][1])

    def update_cd4_starts(self, uids):
        """
        Update initial CD4 counts for new agents
        """
        self.cd4_start[uids] = ss.normal(loc=self.pars.cd4_start_mean, scale=1).initialize().rvs(len(uids))
        self.cd4[uids] = self.cd4_start[uids]
        return

    def update_pre(self, sim):
        """
        """
        # Update cd4 start for new agents:
        self.update_cd4_starts(uids=self.cd4_start[pd.isna(self.cd4_start)].uid)  # TODO probably not the best place?

        # Progress exposed -> infectious
        infectious = ss.true(self.infected & (self.ti_infectious <= sim.ti))
        self.infectious[infectious] = True

        # Update cd4 counts:
        self.update_cd4_counts(sim)

        # Update today's transmission
        self.update_transmission(sim)

        # Update today's deaths
        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.pars.death_prob.filter(can_die)

        sim.people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = sim.ti

        # Update today's diagnoses
        diagnosed = ss.true(self.ti_diagnosed == sim.ti)
        self.diagnosed[diagnosed] = True

        return

    def update_cd4_counts(self, sim):
        """
        Update today's CD4 counts
        """
        infected_uids_onART = sim.people.alive & self.infected & self.on_art
        infected_uids_not_onART = sim.people.alive & self.infected & ~self.on_art

        duration_since_untreated = sim.ti - self.ti_since_untreated[infected_uids_not_onART]
        duration_since_untreated = np.minimum(duration_since_untreated, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_onART = sim.ti - self.ti_art[infected_uids_onART]

        # Assumption: Art impact increases linearly over 6 months
        duration_since_onART = np.minimum(duration_since_onART, 3 * 12)

        cd4_count_changes = np.diff(self.pars.cd4_timecourse)
        cd4_count = self.cd4[infected_uids_not_onART] + cd4_count_changes[duration_since_untreated - 1] * \
                    self.cd4_start[infected_uids_not_onART]
        self.cd4[infected_uids_not_onART] = np.maximum(cd4_count, 1)

        # Update cd4 counts for agents on ART
        if sum(infected_uids_onART.tolist()) > 0:
            self.cd4[infected_uids_onART] = np.minimum(self.cd4_start[infected_uids_onART], self.cd4[
                infected_uids_onART] + duration_since_onART * 15.584 - 0.2113 * duration_since_onART ** 2)  # Assumption: back to 1 in 3 months

        return

    def update_transmission(self, sim):
        """
        Update today's transmission
        """
        infected_uids_onART = sim.people.alive & self.infected & self.on_art
        infected_uids_not_onART = sim.people.alive & self.infected & ~self.on_art

        duration_since_infection = sim.ti - self.ti_infected[infected_uids_not_onART]
        duration_since_infection = np.minimum(duration_since_infection, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_infection_transmission = np.minimum(duration_since_infection,
                                                           len(self.pars.transmission_timecourse) - 1).astype(int)
        duration_since_onART = sim.ti - self.ti_art[infected_uids_onART]

        # Assumption: Art impact increases linearly over 6 months
        duration_since_onART = np.minimum(duration_since_onART, 3 * 12)
        duration_since_onART_transmission = np.minimum(duration_since_onART, 6)

        cd4_count_changes = np.diff(self.pars.cd4_timecourse)
        # Update transmission for agents not on ART with a cd4 count above 200:
        infected_uids_not_onART_cd4_above_200 = infected_uids_not_onART & (self.cd4 >= 200)
        transmission_not_onART_cd4_above_200 = self.pars.transmission_timecourse[duration_since_infection_transmission[ss.true(infected_uids_not_onART_cd4_above_200)]]
        # Randomize:
        transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_not_onART_cd4_above_200))
        self.rel_trans[infected_uids_not_onART_cd4_above_200] = transmission_random_factor * transmission_not_onART_cd4_above_200

        # Update transmission for agents on ART
        # When agents start ART, determine the reduction of transmission (linearly decreasing over 6 months)
        self.update_transmission_reduction(duration_since_onART_transmission)
        transmission_onART = np.maximum(1-self.pars.art_efficacy, self.rel_trans[infected_uids_onART] - self.art_transmission_reduction[infected_uids_onART])
        transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_onART))
        self.rel_trans[infected_uids_onART] = transmission_random_factor * transmission_onART

        # Overwrite transmission for agents whose CD4 counts are below 200:
        uids_below_200 = self.cd4 < 200
        if len(ss.true(uids_below_200)) > 0:
            ti_200_to_50 = (150 / (cd4_count_changes[-1] * self.cd4_start[uids_below_200]) * (-1)).astype(int)
            transmission_below_200 = np.minimum(self.rel_trans[uids_below_200] + (6 - 1) / ti_200_to_50, 6)
            transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_below_200))
            self.rel_trans[uids_below_200] = transmission_random_factor * transmission_below_200

        return

    def update_transmission_reduction(self, durs_onART):
        """
        Get transmission reductions (to decrease transmission linearly over 6 months)
        """
        start_onART_uids = ss.true(durs_onART == 1)
        self.art_transmission_reduction[start_onART_uids] = (self.rel_trans[start_onART_uids] - (1-self.pars.art_efficacy)) /6
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cum_deaths', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_diagnoses', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cum_diagnoses', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_agents_on_art', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'cum_agents_on_art', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'n_agents_on_art', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'prevalence_sw', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'prevalence_client', sim.npts, dtype=float)

        # Add FSW and clients to results:
        for risk_group in np.unique(sim.networks.structuredsexual.risk_group):
            self.results += ss.Result(self.name, 'prevalence_risk_group_' + str(risk_group), sim.npts, dtype=float)
            self.results += ss.Result(self.name, 'new_infections_risk_group_' + str(risk_group), sim.npts, dtype=float)

        return

    def update_results(self, sim):
        super().update_results(sim)
        self.results['new_deaths'][sim.ti] = np.count_nonzero(self.ti_dead == sim.ti) * sim.pars["pop_scale"]
        self.results['cum_deaths'][sim.ti] = np.sum(self.results['new_deaths'][:sim.ti])
        self.results['new_diagnoses'][sim.ti] = np.count_nonzero(self.ti_pos_test == sim.ti) * sim.pars["pop_scale"]
        self.results['cum_diagnoses'][sim.ti] = np.sum(self.results['new_diagnoses'][:sim.ti])
        self.results['new_agents_on_art'][sim.ti] = np.count_nonzero(self.ti_art == sim.ti)
        self.results['n_agents_on_art'][sim.ti] = len(ss.true(self.on_art))
        self.results['cum_agents_on_art'][sim.ti] = np.sum(self.results['new_agents_on_art'][:sim.ti])

        # Subset by FSW and client:
        fsw_infected = self.infected[sim.networks.structuredsexual.fsw]
        client_infected = self.infected[sim.networks.structuredsexual.client]
        for risk_group in np.unique(sim.networks.structuredsexual.risk_group):
            risk_group_infected = self.infected[sim.networks.structuredsexual.risk_group == risk_group]
            self.results['prevalence_risk_group_' + str(risk_group)][sim.ti] = sum(risk_group_infected.values) / len(
                risk_group_infected)
            self.results['new_infections_risk_group_' + str(risk_group)][sim.ti] = sum(
                risk_group_infected.values) / len(risk_group_infected)

        # Add FSW and clients to results:
        self.results['prevalence_sw'][sim.ti] = sum(fsw_infected.values) / len(fsw_infected)
        self.results['prevalence_client'][sim.ti] = sum(client_infected.values) / len(client_infected)

        return

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        self.ti_infectious[uids] = sim.ti + 14
        self.ti_since_untreated[uids] = sim.ti

        return

    def set_congenital(self, sim, target_uids, source_uids):
        return self.set_prognoses(sim, target_uids, source_uids)

    def test(self, uids, t, test_sensitivity=1.0, loss_prob=0.0, test_delay=0):
        '''
        Method to test people. Typically not to be called by the user directly;
        see the test_num() and test_prob() interventions.

        Args:
            inds: indices of who to test
            test_sensitivity (float): probability of a true positive
            loss_prob (float): probability of loss to follow-up
            test_delay (int): number of days before test results are ready
        '''

        uids = np.unique(uids)
        self.tested[uids] = True
        self.ti_tested[uids] = t  # Only keep the last time they tested

        is_infectious = uids[self.infectious[uids]]
        pos_test = np.random.random(len(is_infectious)) < test_sensitivity
        is_inf_pos = is_infectious[pos_test]

        not_diagnosed = is_inf_pos[np.isnan(self.ti_diagnosed[is_inf_pos])]
        not_lost = np.random.random(len(not_diagnosed)) < 1.0 - loss_prob
        final_uids = not_diagnosed[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back
        # positive
        self.ti_diagnosed[final_uids] = t + test_delay
        self.ti_pos_test[final_uids] = t

        return final_uids

    def check_uids(self, current, date, t, filter_uids=None):
        '''
        Return indices for which the current state is false and which meet the date criterion
        '''
        if filter_uids is None:
            not_current = ss.false(current)
        else:
            not_current = filter_uids[np.logical_not(current[filter_uids])]
        has_date = not_current[~np.isnan(date[not_current])]
        uids = has_date[t >= date[has_date]]
        return uids