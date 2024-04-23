"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss
from collections import defaultdict

__all__ = ['HIV']


class HIV(ss.Infection):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        # States
        self.add_states(
            ss.State('on_art', bool, False),
            ss.State('ti_art', int, ss.INT_NAN),
            ss.State('ti_stop_art', float, np.nan),
            # ss.State('cd4', float, 500),  # cells/uL
            ss.State('cd4', float, 1),  #
            ss.State('infected_cells', float, 0),  # cells per uL
            ss.State('virus', float, 0),  #
            ss.State('viral_load', float, 0),  # RNA copies/mL
            ss.State('ti_dead', int, ss.INT_NAN),  # Time of HIV-cause death
            ss.State('diagnosed', bool, False),
            ss.State('dead', bool, False),
            # ss.State('infectious', bool, False),
            ss.State('tested', bool, False),
            ss.State('ti_tested', float, np.nan),
            ss.State('ti_pos_test', float, np.nan),
            ss.State('ti_diagnosed', float, np.nan),
            ss.State('ti_infectious', float, np.nan),
            ss.State('ti_since_untreated', float, np.nan)
        )

        pars = ss.omergeleft(pars,
                             cd4_min=100,
                             cd4_max=500,
                             cd4_rate=5,
                             # From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6155466/#imr12698-bib-0019
                             d_T=0.1 / 24,  # 0.1 cells/uL/day
                             lambda_T=100 / 24,  # 100 cells/uL/day
                             d_I=1 / 24,  # 1 cell/uL/day
                             k=250 / 24,  # 250 virus/cell/day
                             c=25 / 24,  # 25 virus/day
                             beta_noART=(3 * 1e-4) / 24,  # 3 * 1e-7 virus/ml/day -> 3 * 1e-4 virus/uL/day
                             init_prev=0.05,
                             eff_condoms=0.7,
                             ART_prob=0.9,
                             stop_ART_prob=0.1,
                             restart_ART_prob=0.9,
                             avg_duration_stop_ART=12, # 12 months for now, TODO draw from a distribution
                             avg_duration_restart_ART=12,
                             art_efficacy=0.96,
                             death_prob=0.05,
                             viral_timecourse=None
                             )

        par_dists = ss.omergeleft(par_dists,
                                  init_prev=ss.bernoulli,
                                  death_prob=ss.bernoulli,
                                  )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self.death_prob_data = sc.dcp(self.pars.death_prob)
        self.pars.death_prob = self.make_death_prob
        self._pending_ARTtreatment = defaultdict(list)
        self.pars.viral_timecourse,  self.pars.cd4_timecourse = self.get_viral_dynamics_timecourses()

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

    def get_viral_dynamics_timecourses(self):

        viral_load_timecourse_data = [(0, 0), (0.5, 1), (1, (10**4.5) /1e6), (4*12, (10**4.5) / 1e6), (8 * 12, 1)] # in months
        cd4_timecourse_data = [(0, 1), (0.5, 600/1000), (1, 700/1000), (1 * 12, 700/1000), (10*12, 0)]  # in months

        viral_load_timecourse = self._interpolate(viral_load_timecourse_data, np.arange(0, 8*12))
        cd4_timecourse = self._interpolate(cd4_timecourse_data, np.arange(0, 10*12))

        return viral_load_timecourse, cd4_timecourse

    def _interpolate(self, vals: list, t):
        vals = sorted(vals, key=lambda x: x[0])  # Make sure values are sorted
        assert len({x[0] for x in vals}) == len(vals)  # Make sure time points are unique
        return np.interp(t, [x[0] for x in vals], [x[1] for x in vals], left=vals[0][1], right=vals[-1][1])


    def update_pre(self, sim):
        """
        """

        infected_uids = sim.people.alive & self.infected
        infected_uids_onART = sim.people.alive & self.infected & self.on_art

        # Progress exposed -> infectious
        infectious = ss.true(self.infected & (self.ti_infectious <= sim.ti))
        self.infectious[infectious] = True

        # Update viral load and cd4 count:
        duration_since_untreated = sim.ti - self.ti_since_untreated[infected_uids]
        duration_since_untreated = np.minimum(duration_since_untreated, len(self.pars.viral_timecourse) - 1).astype(int)
        duration_since_onART = sim.ti - self.ti_art[infected_uids_onART]
        duration_since_onART = np.minimum(duration_since_onART, 3)
        assert not np.any(duration_since_untreated < 0) # Cannot have negative durations, can disable this check for performance if required

        viral_load = self.pars.viral_timecourse[duration_since_untreated]
        cd4_count = self.pars.cd4_timecourse[duration_since_untreated]

        self.cd4[infected_uids] = cd4_count
        self.virus[infected_uids] = viral_load

        # Update viral load and cd4 counts for agents on ART
        self.virus[infected_uids_onART] = self.virus[infected_uids_onART] - duration_since_onART * self.virus[infected_uids_onART]/3 # Assumption: back to 0 in 3 months
        self.cd4[infected_uids_onART] = self.cd4[infected_uids_onART] + duration_since_onART * (1-self.cd4[infected_uids_onART]) / 3 # Assumption: back to 1 in 3 months

        # Update transmission
        self.rel_trans[sim.people.alive & self.infected] = viral_load

        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.pars.death_prob.filter(can_die)

        sim.people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = sim.ti

        # Update today's diagnoses
        diagnosed = ss.true(self.ti_diagnosed <= sim.ti)
        self.diagnosed[diagnosed] = True


        return

    def update_post(self, sim):
        self.check_start_ART_treatment(sim.ti)
        self.check_stop_ART_treatment(sim.ti)

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cum_deaths', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_diagnoses', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'cum_diagnoses', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'new_on_art', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'cum_on_art', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'prevalence_sw', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'prevalence_client', sim.npts, dtype=float)

        # Add FSW and clients to results:
        for risk_group in np.unique(sim.networks.structuredsexual.risk_group):
            self.results += ss.Result(self.name, 'prevalence_risk_group_' + str(risk_group), sim.npts, dtype=float)
            self.results += ss.Result(self.name, 'new_infections_risk_group_' + str(risk_group), sim.npts, dtype=float)

        return

    def update_results(self, sim):
        super().update_results(sim)

        # Check who will start and stop ART treatment
        self.check_start_ART_treatment(sim)
        self.check_stop_ART_treatment(sim)

        self.results['new_deaths'][sim.ti] = np.count_nonzero(self.ti_dead == sim.ti)
        self.results['cum_deaths'][sim.ti] = np.sum(self.results['new_deaths'][:sim.ti])
        self.results['new_diagnoses'][sim.ti] = np.count_nonzero(self.ti_pos_test == sim.ti)
        self.results['cum_diagnoses'][sim.ti] = np.sum(self.results['new_diagnoses'][:sim.ti])
        self.results['new_on_art'][sim.ti] = np.count_nonzero(self.ti_art == sim.ti)
        self.results['cum_on_art'][sim.ti] = np.sum(self.results['new_on_art'][:sim.ti])

        # Subset by FSW and client:
        fsw_infected = self.infected[sim.networks.structuredsexual.fsw]
        client_infected = self.infected[sim.networks.structuredsexual.client]
        for risk_group in np.unique(sim.networks.structuredsexual.risk_group):
            risk_group_infected = self.infected[sim.networks.structuredsexual.risk_group == risk_group]
            self.results['prevalence_risk_group_' + str(risk_group)][sim.ti] = sum(risk_group_infected.values) / len(risk_group_infected)
            self.results['new_infections_risk_group_' + str(risk_group)][sim.ti] = sum(risk_group_infected.values) / len(risk_group_infected)

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
        self.ti_infectious[uids] = sim.ti  + 14
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

        # A small subset will stop ART after a certain time:
        stop_ART_uids = np.random.random(len(self._pending_ARTtreatment[sim.ti])) < self.pars.stop_ART_prob
        self.ti_stop_art[stop_ART_uids] = sim.ti + self.pars.avg_duration_stop_ART

    def check_stop_ART_treatment(self, sim):
        """
        Check who is stopping ART treatment
        """
        stop_uids = self.check_uids(~self.on_art, self.ti_stop_art, sim.ti, filter_uids=None)
        self.on_art[stop_uids] = False
        self.ti_art[stop_uids] = np.nan
        self.ti_stop_art[stop_uids] = np.nan
        self.ti_since_untreated[stop_uids] = sim.ti

        # Add a subset back to the pool of agents pending ART:
        uids_pending_ART = stop_uids[np.random.random(len(stop_uids)) < self.pars.restart_ART_prob] # Will the agents go back to ART?
        self.schedule_ART_treatment(uids_pending_ART, sim.ti + self.pars.avg_duration_restart_ART)


