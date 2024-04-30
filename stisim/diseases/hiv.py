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
                             eff_condoms=0.7,
                             ART_coverages_df=None,
                             ART_prob=0.9,
                             duration_on_ART=ss.normal(loc=18, scale=5),
                             # https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-021-10464-x
                             art_efficacy=0.96,
                             death_prob=0.05)

        par_dists = ss.omergeleft(par_dists,
                                  init_prev=ss.bernoulli,
                                  death_prob=ss.bernoulli,
                                  )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self.death_prob_data = sc.dcp(self.pars.death_prob)
        self.pars.death_prob = self.make_death_prob
        self._pending_ARTtreatment = defaultdict(list)
        self.art_transmission_reduction = None

        return

    def initialize(self, sim=None):
        """
        Initialize
        """
        super().initialize(sim)
        self.cd4_start[sim.people.uid] = ss.normal(loc=self.pars.cd4_start_mean, scale=100).initialize().rvs(
            len(sim.people))
        self.cd4[sim.people.uid] = self.cd4_start[sim.people.uid]
        self.pars.transmission_timecourse = self.get_transmission_timecouse()
        self.pars.viral_timecourse, self.pars.cd4_timecourse = self.get_viral_dynamics_timecourses()
        self.art_transmission_reduction = self.pars.art_efficacy / 6  # Assumption: 6 months

        return

    @property
    def symptomatic(self):
        return self.infectious

    @staticmethod
    def make_death_prob(module, sim, uids):
        p = module.pars
        out = sim.dt * module.death_prob_data / (p.cd4_min - p.cd4_max) ** 2 * (module.cd4[uids] - p.cd4_max) ** 2
        out = np.array(out)
        return out

    def get_transmission_timecouse(self):
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
        self.check_start_ART_treatment(sim)
        self.check_stop_ART_treatment(sim)
        # Apply correction of ART coverage:
        self.ART_coverage_correction(sim)

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

        # Schedule ART for diagnosed cases:
        self.schedule_ART_treatment(diagnosed, sim.ti)

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

        # Update viral load and cd4 count:
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
        self.rel_trans[infected_uids_not_onART_cd4_above_200] = self.pars.transmission_timecourse[
            duration_since_infection_transmission[ss.true(infected_uids_not_onART_cd4_above_200)]]
        # Update transmission for agents on ART
        self.rel_trans[infected_uids_onART] = 1 - self.art_transmission_reduction * duration_since_onART_transmission

        # Overwrite tranmissibility for agents whose CD4 counts are below 200:
        for uid in self.cd4[self.cd4 < 200].uid:
            # Time to drop from 200 to 50:
            ti_200_to_50 = int(150 / (cd4_count_changes[-1] * self.cd4_start[uid]) * (-1))
            self.rel_trans[uid] = np.minimum(self.rel_trans[uid] + (6 - 1) / ti_200_to_50, 6)

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

    def schedule_ART_treatment(self, uids, t, start_date=None, period=None):
        """
        Schedule ART treatment. Typically not called by the user directly

        Args:
            inds (int): indices of who to put on ART treatment, specified by check_quar()
            start_date (int): day to begin ART treatment(defaults to the current day, `sim.t`)
            period (int): quarantine duration (defaults to ``pars['quar_period']``)

        """
        start_date = t if start_date is None else int(start_date)
        uids_ART = uids[np.random.random(len(uids)) < self.pars.ART_prob]
        period = 1000  # self.pars['quar_period'] if period is None else int(period)
        for uid in uids_ART:
            self._pending_ARTtreatment[start_date].append((uid, start_date + period))
        return

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

    def check_start_ART_treatment(self, sim):
        """
        Check who is ready to start ART treatment
        """
        for uid, end_day in self._pending_ARTtreatment[sim.ti]:
            if uid in sim.people.alive.uid:
                self.on_art[uid] = True
                self.ti_art[uid] = sim.ti
                # Determine when agents goes off ART:
                self.ti_stop_art[uid] = sim.ti + int(self.pars.duration_on_ART.rvs(1))
        return

    def check_stop_ART_treatment(self, sim):
        """
        Check who is stopping ART treatment
        """
        stop_uids = self.check_uids(~self.on_art, self.ti_stop_art, sim.ti, filter_uids=None)
        self.on_art[stop_uids] = False
        self.ti_art[stop_uids] = np.nan
        self.ti_stop_art[stop_uids] = np.nan
        self.ti_since_untreated[stop_uids] = sim.ti
        return

    def ART_coverage_correction(self, sim):
        """
        Adjust number of people on treatment to match data
        """
        infected_uids_onART = self.infected & self.on_art
        infected_uids_not_onART = self.infected & ~self.on_art

        # Get the current ART coverage. If year is not available, assume 90%
        if len(self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()) > 0:
            ART_coverage_this_year = \
            self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()[0]
        else:
            ART_coverage_this_year = int(0.9 * len(ss.true(self.infected)))

        # Too many agents on treatment -> remove
        if len(ss.true(infected_uids_onART)) > ART_coverage_this_year:
            # Agents with the highest CD4 counts will go off ART:
            n_agents_to_stop_ART = int(len(ss.true(infected_uids_onART)) - ART_coverage_this_year)
            cd4_counts_onART = self.cd4[infected_uids_onART]
            cd4_counts_onART.sort(axis=0)
            # Grab the last n agents with the highest counts
            probabilities = (cd4_counts_onART / np.sum(cd4_counts_onART)).values
            # Probabilities are increasing with CD4 count
            uids = cd4_counts_onART.uid
            stop_uids = np.random.choice(uids, n_agents_to_stop_ART, p=probabilities,
                                         replace=False)  # TODO eventually put this into a distribution module?

            self.on_art[stop_uids] = False
            self.ti_art[stop_uids] = np.nan
            self.ti_stop_art[stop_uids] = np.nan
            self.ti_since_untreated[stop_uids] = sim.ti

        # Not enough agents on treatment -> add
        elif len(ss.true(infected_uids_onART)) < ART_coverage_this_year:
            # Agents with the lowest CD4 count will get on ART:
            n_agents_to_start_ART = int(ART_coverage_this_year - len(ss.true(infected_uids_onART)))
            cd4_counts_not_onART = self.cd4[infected_uids_not_onART]
            cd4_counts_not_onART.sort(axis=0)
            probabilities = (cd4_counts_not_onART / np.sum(cd4_counts_not_onART)).values
            # Probabilities are increasing with CD4 count, therefore flip uid array:
            uids = np.flipud(cd4_counts_not_onART.uid)
            start_uids = np.random.choice(uids, n_agents_to_start_ART, p=probabilities,
                                          replace=False)  # TODO eventually put this into a distribution module?

            # Put them on ART
            self.on_art[start_uids] = True
            self.ti_art[start_uids] = sim.ti
            # Determine when agents go off ART:
            self.ti_stop_art[start_uids] = sim.ti + self.pars.duration_on_ART.rvs(len(start_uids)).astype(int)

        return
