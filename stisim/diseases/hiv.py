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
            ss.BoolArr('on_art'),
            ss.FloatArr('art_transmission_reduction'),  # Reduction in transmission dependent on initial cd4 count
            ss.FloatArr('ti_art'),
            ss.FloatArr('ti_stop_art'),
            ss.FloatArr('cd4_start'),  # Initial cd4 count for each agent before an infection
            ss.FloatArr('cd4'),  # Current CD4 count
            ss.FloatArr('ti_dead'),  # Time of HIV-cause death
            ss.BoolArr('diagnosed'),
            ss.FloatArr('ti_diagnosed'),
            ss.FloatArr('ti_since_untreated')  # This is needed for agents who start, stop and restart ART
        )
        # Default parameters
        pars = ss.dictmergeleft(pars,
                                cd4_start_mean=500,
                                cd4_min=100,
                                cd4_max=500,
                                cd4_rate=5,
                                init_prev=0.03,
                                init_diagnosed=ss.bernoulli(p=0.01),
                                transmission_sd=0.025,
                                primary_acute_inf_dur=1,  # in months
                                art_efficacy=0.96,
                                death_prob=0.05)

        par_dists = ss.dictmergeleft(par_dists,
                                     death_prob=ss.bernoulli)

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self.death_prob_data = sc.dcp(self.pars.death_prob)
        self.pars.death_prob = self.make_death_prob
        self._pending_ARTtreatment = defaultdict(list)

        return

    def initialize(self, sim=None):
        """
        Initialize
        """
        super().initialize(sim)
        self.pars.transmission_timecourse = self.get_transmission_timecourse()

        return

    def set_initial_states(self, sim):
        alive_uids = sim.people.alive.uids
        initial_cases = self.pars.init_prev.filter(alive_uids)
        initial_cases_diagnosed = self.pars.init_diagnosed.filter(initial_cases)
        self.susceptible[initial_cases] = False
        self.infected[initial_cases] = True
        self.diagnosed[initial_cases_diagnosed] = True

        # Assume initial cases were infected up to 10 years ago
        self.ti_infected[initial_cases] = ss.uniform(low=-10 * 12, high=0).initialize().rvs(len(initial_cases)).astype(int)
        self.ti_since_untreated[initial_cases] = self.ti_infected[initial_cases]

        # Update CD4 counts for initial cases
        self.pars.viral_timecourse, self.pars.cd4_timecourse = self.get_viral_dynamics_timecourses()
        duration_since_untreated = sim.ti - self.ti_since_untreated[initial_cases]
        duration_since_untreated = np.minimum(duration_since_untreated, len(self.pars.cd4_timecourse) - 1).astype(int)
        self.cd4_start[initial_cases] = ss.normal(loc=self.pars.cd4_start_mean, scale=1).initialize().rvs(len(initial_cases)) #TODO update to positive normal distribution
        self.cd4[initial_cases] = self.cd4_start[initial_cases] * self.pars.cd4_timecourse[duration_since_untreated]

        # Update transmission
        # Assumption: Transmission is at 1 for agents with cd4 count >200, else at 6
        self.pars.transmission_timecourse = self.get_transmission_timecourse()
        self.rel_trans[initial_cases] = 1
        duration_since_infection = sim.ti - self.ti_infected[initial_cases]
        duration_since_infection = np.minimum(duration_since_infection, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_infection_transmission = np.minimum(duration_since_infection, len(self.pars.transmission_timecourse) - 1).astype(int)

        # Update transmission for agents with a cd4 count >=200:
        infected_uids_not_onART_cd4_above_200 = self.cd4[initial_cases] >= 200
        infected_uids_not_onART_cd4_above_200_uids = initial_cases[infected_uids_not_onART_cd4_above_200]
        transmission_not_onART_cd4_above_200 = self.pars.transmission_timecourse[duration_since_infection_transmission[infected_uids_not_onART_cd4_above_200]]
        # Randomize:
        transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_not_onART_cd4_above_200))
        self.rel_trans[ss.uids(infected_uids_not_onART_cd4_above_200_uids)] = transmission_random_factor * transmission_not_onART_cd4_above_200
        # Update transmission for agents with a cd4 count <200
        uids_below_200 = initial_cases[self.cd4[initial_cases] < 200]
        # Calculate how many momths the agent has been <200 counts to get the correct transmission:
        cd4_count_changes = np.diff(self.pars.cd4_timecourse)
        if len(uids_below_200) > 0:
            ti_200_to_50 = (150 / (cd4_count_changes[-1] * self.cd4_start[uids_below_200]) * (-1)).astype(int)
            ti_under200 = (200 - self.cd4[uids_below_200]) / (-1 * cd4_count_changes[-1] * self.cd4_start[uids_below_200])
            transmission_below_200 = np.minimum(self.rel_trans[uids_below_200] + ti_under200 * ((6-1) / ti_200_to_50), 6)
            transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_below_200))
            self.rel_trans[uids_below_200] = transmission_random_factor * transmission_below_200

        return

    @property
    def symptomatic(self):
        return self.infectious

    @staticmethod
    def make_death_prob(self, sim, uids):
        """
        Death probabilities dependent on cd4 counts
        """
        death_data = [(range(500, int(np.ceil(np.max(self.cd4_start)))), 0.0 / 12),
                      (range(350, 500), 0.0 / 12),
                      (range(200, 350), 0.0 / 12),
                      (range(50, 200), 0.0 / 12),
                      (range(0, 50), 0.3 / 12)]
        death_probs = [probs[1] for probs in death_data for cd4_count in self.cd4[uids] if int(cd4_count) in probs[0]]
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
        # Viral load increases based on Figure 1B in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6155466/
        # TODO Currently not needed
        viral_load_timecourse_data = [(0, 0), (0.5, 1), (1, (10 ** 4.5) / 1e6), (4 * 12, (10 ** 4.5) / 1e6),
                                      (8 * 12, 1)]  # in months

        # CD4 count decrease based on Figure 1A in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6155466/
        # Assumption
        # 100% at month 0
        # 60% at month 0.5
        # 70% at months 1 - 12
        # Linear decrease from 70% to 0 after 10 years
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

    def update_cd4_starts(self):
        """
        Update initial CD4 counts for new agents
        """
        uids = self.cd4_start.auids[pd.isna(self.cd4_start.values)]
        values = ss.normal(loc=self.pars.cd4_start_mean, scale=1).initialize().rvs(len(uids))
        self.cd4_start.set(ss.uids(uids), values)
        self.cd4.set(ss.uids(uids), values)

        return

    def update_pre(self, sim):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)
        """
        # Update cd4 start for new agents:
        self.update_cd4_starts()

        # Update cd4 counts:
        self.update_cd4_counts(sim)

        # Update today's transmission
        self.update_transmission(sim)

        # Update today's deaths
        can_die = (sim.people.alive & sim.people.hiv.infected).uids
        hiv_deaths = self.pars.death_prob.filter(can_die)

        sim.people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = sim.ti

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
        cd4_count = self.cd4[infected_uids_not_onART] + cd4_count_changes[duration_since_untreated - 1] * self.cd4_start[infected_uids_not_onART]
        self.cd4[infected_uids_not_onART] = np.maximum(cd4_count, 1)

        # Update cd4 counts for agents on ART
        if sum(infected_uids_onART.tolist()) > 0:
            # Assumption: back to 1 in 3 months, from EMOD
            self.cd4[infected_uids_onART] = np.minimum(self.cd4_start[infected_uids_onART],
                                                       self.cd4[infected_uids_onART] + duration_since_onART * 15.584 - 0.2113 * duration_since_onART ** 2)

        return

    def update_transmission(self, sim):
        """
        Update today's transmission
        """
        infected_uids_onART = sim.people.alive & self.infected & self.on_art
        infected_uids_not_onART = sim.people.alive & self.infected & ~self.on_art

        duration_since_infection = sim.ti - self.ti_infected[infected_uids_not_onART]
        duration_since_infection = np.minimum(duration_since_infection, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_infection_transmission = np.minimum(duration_since_infection, len(self.pars.transmission_timecourse) - 1).astype(int)
        duration_since_onART = sim.ti - self.ti_art[infected_uids_onART]

        # Assumption: Art impact increases linearly over 6 months
        duration_since_onART = np.minimum(duration_since_onART, 3 * 12)
        duration_since_onART_transmission = np.minimum(duration_since_onART, 6)

        # Update transmission for agents not on ART with a cd4 count above 200:
        infected_uids_not_onART_cd4_above_200 = self.cd4[infected_uids_not_onART] >= 200
        infected_uids_not_onART_cd4_above_200_uids = infected_uids_not_onART.uids[infected_uids_not_onART_cd4_above_200]
        transmission_not_onART_cd4_above_200 = self.pars.transmission_timecourse[duration_since_infection_transmission[infected_uids_not_onART_cd4_above_200]]
        # Randomize:
        transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_not_onART_cd4_above_200))
        self.rel_trans[ss.uids(infected_uids_not_onART_cd4_above_200_uids)] = transmission_random_factor * transmission_not_onART_cd4_above_200

        # Update transmission for agents on ART
        # When agents start ART, determine the reduction of transmission (linearly decreasing over 6 months)
        self.get_transmission_reduction(duration_since_onART_transmission, infected_uids_onART.uids)
        transmission_onART = np.maximum(1 - self.pars.art_efficacy, self.rel_trans[infected_uids_onART] - self.art_transmission_reduction[infected_uids_onART])
        transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_onART))
        self.rel_trans[infected_uids_onART] = transmission_random_factor * transmission_onART

        # Overwrite transmission for agents whose CD4 counts are below 200:
        uids_below_200 = self.cd4 < 200
        cd4_count_changes = np.diff(self.pars.cd4_timecourse)
        if len(uids_below_200.uids) > 0:
            ti_200_to_50 = (150 / (cd4_count_changes[-1] * self.cd4_start[uids_below_200]) * (-1)).astype(int)
            transmission_below_200 = np.minimum(self.rel_trans[uids_below_200] + (6 - 1) / ti_200_to_50, 6)
            transmission_random_factor = ss.normal(loc=1, scale=self.pars.transmission_sd).initialize().rvs(len(transmission_below_200))
            self.rel_trans[uids_below_200] = transmission_random_factor * transmission_below_200
        return

    def get_transmission_reduction(self, durs_onART, uids_onART):
        """
        Determine the reduction in transmission once an agent starts ART.
        Transmission decreases linearly over 6 months and is dependent on the agent's current transmission.
        """
        start_onART_uids = uids_onART[(durs_onART == 1)]
        self.art_transmission_reduction[start_onART_uids] = (self.rel_trans[start_onART_uids] - (1 - self.pars.art_efficacy)) / 6
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
        self.results += ss.Result(self.name, 'prevalence_sw', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'prevalence_client', sim.npts, dtype=float)

        # Add FSW and clients to results:
        for risk_group in np.unique(sim.networks.structuredsexual.risk_group):
            for sex in ['f', 'm']:
                self.results += ss.Result(self.name, 'prevalence_risk_group_' + str(risk_group) + '_' + sex, sim.npts,
                                          dtype=float)
                self.results += ss.Result(self.name, 'new_infections_risk_group_' + str(risk_group) + '_' + sex,
                                          sim.npts, dtype=float)

        return

    def update_results(self, sim):
        """
        Update results at each time step
        """
        super().update_results(sim)
        self.results['new_deaths'][sim.ti] = np.count_nonzero(self.ti_dead == sim.ti) * sim.pars["pop_scale"]
        self.results['cum_deaths'][sim.ti] = np.sum(self.results['new_deaths'][:sim.ti + 1])
        self.results['new_diagnoses'][sim.ti] = np.count_nonzero(self.ti_diagnosed == sim.ti) * sim.pars["pop_scale"]
        self.results['cum_diagnoses'][sim.ti] = np.sum(self.results['new_diagnoses'][:sim.ti + 1])
        self.results['new_agents_on_art'][sim.ti] = np.count_nonzero(self.ti_art == sim.ti) * sim.pars["pop_scale"]
        self.results['cum_agents_on_art'][sim.ti] = np.sum(self.results['new_agents_on_art'][:sim.ti + 1])

        # Subset by FSW and client:
        fsw_infected = self.infected[sim.networks.structuredsexual.fsw]
        client_infected = self.infected[sim.networks.structuredsexual.client]
        for risk_group in np.unique(sim.networks.structuredsexual.risk_group):
            for sex in ['f', 'm']:
                risk_group_infected = self.infected[(sim.networks.structuredsexual.risk_group == risk_group) & (sim.people[sex])]
                self.results['prevalence_risk_group_' + str(risk_group) + '_' + sex][sim.ti] = sum(risk_group_infected) / len(risk_group_infected)
                self.results['new_infections_risk_group_' + str(risk_group) + '_' + sex][sim.ti] = sum(risk_group_infected) / len(risk_group_infected)

        # Add FSW and clients to results:
        self.results['prevalence_sw'][sim.ti] = sum(fsw_infected) / len(fsw_infected)
        self.results['prevalence_client'][sim.ti] = sum(client_infected) / len(client_infected)

        return

    def make_new_cases(self, sim):
        """
        Add new HIV cases
        """
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        """
        Set prognoses upon infection
        """
        super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        self.ti_since_untreated[uids] = sim.ti

        return

    def set_congenital(self, sim, target_uids, source_uids):
        return self.set_prognoses(sim, target_uids, source_uids)
