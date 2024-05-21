"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss
import pandas as pd
from collections import defaultdict


__all__ = ['HIV']

import stisim as sti


class HIV(ss.Infection):



    def __init__(self, pars=None, **kwargs):
        super().__init__()

        self.requires = sti.StructuredSexual

        # Parameters
        self.default_pars(
            cd4_start_dist=ss.normal(loc=800, scale=10),
            cd4_min=100,
            cd4_max=500,
            cd4_rate=5,
            init_prev=ss.bernoulli(p=0.05),
            init_diagnosed=ss.bernoulli(p=0.01),
            dist_ti_init_infected=ss.uniform(low=-10 * 12, high=0),
            p_death=None, # Probability of death (default is to use HIV.death_prob(), otherwise can pass in a Dist or anything supported by ss.bernoulli)
            transmission_sd=0.025,
            primary_acute_inf_dur=2.9,  # in months
            art_efficacy=0.96,
            maternal_beta_pmtct_df=None,
            pmtct_coverages_df=None,
            beta=1,
        )

        self.update_pars(pars, **kwargs)

        if self.pars.p_death is None:
            self._death_prob = ss.bernoulli(p=self.death_prob)
        elif isinstance(self.pars.p_death, ss.bernoulli):
            self._death_prob = self.pars.p_death
        else:
            self._death_prob = ss.bernoulli(p=self.pars.p_death)

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

        self._pending_ARTtreatment = defaultdict(list)
        self.initial_hiv_maternal_beta = None

        return

    def initialize(self, sim):
        """
        Initialize
        """
        super().initialize(sim)
        self.pars.transmission_timecourse = self.get_transmission_timecourse()
        self.initial_hiv_maternal_beta = self.pars.beta['maternal'][0]

        return

    def init_vals(self):
        ti = self.sim.ti
        alive_uids = self.sim.people.auids
        initial_cases = self.pars.init_prev.filter(alive_uids)
        initial_cases_diagnosed = self.pars.init_diagnosed.filter(initial_cases)
        self.susceptible[initial_cases] = False
        self.infected[initial_cases] = True
        self.diagnosed[initial_cases_diagnosed] = True

        # Assume initial cases were infected up to 10 years ago
        self.ti_infected[initial_cases] = self.pars.dist_ti_init_infected.rvs(len(initial_cases)).astype(int)
        self.ti_since_untreated[initial_cases] = self.ti_infected[initial_cases]

        # Update CD4 counts for initial cases
        self.pars.viral_timecourse, self.pars.cd4_timecourse = self.get_viral_dynamics_timecourses()
        duration_since_untreated = ti - self.ti_since_untreated[initial_cases]
        duration_since_untreated = np.minimum(duration_since_untreated, len(self.pars.cd4_timecourse) - 1).astype(int)
        self.cd4_start[initial_cases] = self.pars.cd4_start_dist.rvs(initial_cases)  #TODO update to positive normal distribution
        self.cd4[initial_cases] = self.cd4_start[initial_cases] * self.pars.cd4_timecourse[duration_since_untreated]

        # Update transmission
        # Assumption: Transmission is at 1 for agents with cd4 count >200, else at 6
        self.pars.transmission_timecourse = self.get_transmission_timecourse()
        self.rel_trans[initial_cases] = 1
        duration_since_infection = ti - self.ti_infected[initial_cases]
        duration_since_infection = np.minimum(duration_since_infection, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_infection_transmission = np.minimum(duration_since_infection, len(self.pars.transmission_timecourse) - 1).astype(int)

        # Update transmission for agents with a cd4 count >=200:
        infected_uids_not_onART_cd4_above_200 = self.cd4[initial_cases] >= 200
        infected_uids_not_onART_cd4_above_200_uids = initial_cases[infected_uids_not_onART_cd4_above_200]
        transmission_not_onART_cd4_above_200 = self.pars.transmission_timecourse[duration_since_infection_transmission[infected_uids_not_onART_cd4_above_200]]
        # Randomize:
        self.rel_trans[ss.uids(infected_uids_not_onART_cd4_above_200_uids)] = transmission_not_onART_cd4_above_200
        # Update transmission for agents with a cd4 count <200
        uids_below_200 = initial_cases[self.cd4[initial_cases] < 200]
        # Calculate how many momths the agent has been <200 counts to get the correct transmission:
        cd4_count_changes = np.diff(self.pars.cd4_timecourse)
        if len(uids_below_200) > 0:
            ti_200_to_50 = (150 / (cd4_count_changes[-1] * self.cd4_start[uids_below_200]) * (-1)).astype(int)
            ti_under200 = (200 - self.cd4[uids_below_200]) / (-1 * cd4_count_changes[-1] * self.cd4_start[uids_below_200])
            transmission_below_200 = np.minimum(self.rel_trans[uids_below_200] + ti_under200 * ((6-1) / ti_200_to_50), 6)
            self.rel_trans[uids_below_200] = transmission_below_200

        return

    @property
    def symptomatic(self):
        return self.infectious

    @staticmethod
    def death_prob(module, sim=None, size=None):
        cd4_bins = np.array([500, 350, 200, 50, 0])
        death_prob = np.array([0, 0, 0, 0, 0.323])  # Values smaller than the first bin edge get assigned to the last bin.
        return death_prob[np.digitize(module.cd4[size], cd4_bins)]

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
        values = self.pars.cd4_start_dist.rvs(len(uids))
        self.cd4_start.set(ss.uids(uids), values)
        self.cd4.set(ss.uids(uids), values)

        return

    def update_pre(self):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)
        """
        super().update_pre()

        # Update cd4 start for new agents:
        self.update_cd4_starts()

        # Update cd4 counts:
        self.update_cd4_counts()

        # Update today's transmission
        self.update_transmission()

        # Update MTCT - not needed anymore because of ART intervention??
        # self.update_mtct(sim)

        # Update today's deaths
        hiv_deaths = self._death_prob.filter(self.infected.uids)

        self.sim.people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = self.sim.ti

        return

    def update_mtct(self, sim):
        """
        Update mother-to-child-transmission according to the coverage of pregnant women who receive ARV for PMTCT
        """
        # Get this timestep's ma
        if round(sim.year, 3) < round(self.pars.maternal_beta_pmtct_df['Years'].min(), 3):
            maternal_beta_pmtct = self.pars.beta['maternal'][0]
        elif round(sim.year, 3) > round(self.pars.maternal_beta_pmtct_df['Years'].max(), 3):
            maternal_beta_pmtct = self.pars.maternal_beta_pmtct_df['Value'].iloc[-1]
        else:
            maternal_beta_pmtct = self.pars.maternal_beta_pmtct_df[round(self.pars.maternal_beta_pmtct_df['Years'], 3) == round(sim.year, 3)]['Value'].tolist()[0] #TODO find a better way for this

        # Update beta layer for maternal network
        self.pars.beta['maternal'][0] = 1-(1-maternal_beta_pmtct) ** (1/9)
        return


    def update_cd4_counts(self):
        """
        Update today's CD4 counts
        """
        sim = self.sim
        infected_uids_onART = sim.people.alive & self.infected & self.on_art
        infected_uids_not_onART = sim.people.alive & self.infected & ~self.on_art

        duration_since_untreated = sim.ti - self.ti_since_untreated[infected_uids_not_onART]
        self.pars.viral_timecourse, self.pars.cd4_timecourse = self.get_viral_dynamics_timecourses()
        duration_since_untreated = np.minimum(duration_since_untreated, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_onART = sim.ti - self.ti_art[infected_uids_onART]

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

    def update_transmission(self):
        """
        Update today's transmission
        """
        sim = self.sim

        # Reset susceptibility and infectiousness
        self.rel_sus[:] = 1
        self.rel_trans[:] = 1

        infected_uids_not_onART = sim.people.alive & self.infected & ~self.on_art
        duration_since_infection = sim.ti - self.ti_infected[infected_uids_not_onART]
        duration_since_infection = np.minimum(duration_since_infection, len(self.pars.cd4_timecourse) - 1).astype(int)
        duration_since_infection_transmission = np.minimum(duration_since_infection, len(self.pars.transmission_timecourse) - 1).astype(int)

        # Update transmission for agents not on ART with a cd4 count above 200:
        infected_uids_not_onART_cd4_above_200 = self.cd4[infected_uids_not_onART] >= 200
        infected_uids_not_onART_cd4_above_200_uids = infected_uids_not_onART.uids[infected_uids_not_onART_cd4_above_200]
        transmission_not_onART_cd4_above_200 = self.pars.transmission_timecourse[duration_since_infection_transmission[infected_uids_not_onART_cd4_above_200]]
        self.rel_trans[ss.uids(infected_uids_not_onART_cd4_above_200_uids)] = transmission_not_onART_cd4_above_200

        # Update transmission for agents on ART
        # When agents start ART, determine the reduction of transmission (linearly decreasing over 6 months)
        infected_uids_onART = sim.people.alive & self.infected & self.on_art
        duration_since_onART = sim.ti - self.ti_art[infected_uids_onART] # Time

        # Assumption: Art impact increases linearly over 6 months
        duration_since_onART = np.minimum(duration_since_onART, 3 * 12)
        duration_since_onART_transmission = np.minimum(duration_since_onART, 6)



        self.get_transmission_reduction(duration_since_onART_transmission, infected_uids_onART.uids)
        transmission_onART = np.maximum(1 - self.pars.art_efficacy, 1- - self.art_transmission_reduction[infected_uids_onART])
        self.rel_trans[infected_uids_onART] = transmission_onART

        # Overwrite transmission for agents whose CD4 counts are below 200:
        uids_below_200 = self.cd4 < 200
        cd4_count_changes = np.diff(self.pars.cd4_timecourse)
        if len(uids_below_200.uids) > 0:
            ti_200_to_50 = (150 / (cd4_count_changes[-1] * self.cd4_start[uids_below_200]) * (-1)).astype(int)
            transmission_below_200 = np.minimum(self.rel_trans[uids_below_200] + (6 - 1) / ti_200_to_50, 6)
            self.rel_trans[uids_below_200] = transmission_below_200

        return

    def get_transmission_reduction(self, durs_onART, uids_onART):
        """
        Determine the reduction in transmission once an agent starts ART.
        Transmission decreases linearly over 6 months and is dependent on the agent's current transmission.
        """
        start_onART_uids = uids_onART[(durs_onART == 1)]
        self.art_transmission_reduction[start_onART_uids] = (self.rel_trans[start_onART_uids] - (1 - self.pars.art_efficacy)) / 6
        return

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        npts = self.sim.npts
        self.results += ss.Result(self.name, 'new_deaths', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'cum_deaths', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'new_diagnoses', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'cum_diagnoses', npts, dtype=int, scale=True)
        self.results += ss.Result(self.name, 'new_agents_on_art', npts, dtype=float, scale=True)
        self.results += ss.Result(self.name, 'cum_agents_on_art', npts, dtype=float, scale=True)
        self.results += ss.Result(self.name, 'prevalence_sw', npts, dtype=float)
        self.results += ss.Result(self.name, 'prevalence_client', npts, dtype=float)
        self.results += ss.Result(self.name, 'n_on_art_pregnant', npts, dtype=float, scale=True)

        # Add FSW and clients to results:
        for risk_group in range(self.sim.networks.structuredsexual.pars.n_risk_groups):
            for sex in ['female', 'male']:
                self.results += ss.Result(self.name, 'prevalence_risk_group_' + str(risk_group) + '_' + sex, npts,
                                          dtype=float)
                self.results += ss.Result(self.name, 'new_infections_risk_group_' + str(risk_group) + '_' + sex,
                                          npts, dtype=float)

        return

    def update_results(self):
        """
        Update results at each time step
        """
        super().update_results()
        ti = self.sim.ti
        self.results['new_deaths'][ti] = np.count_nonzero(self.ti_dead == ti)
        self.results['cum_deaths'][ti] = np.sum(self.results['new_deaths'][:ti + 1])
        self.results['new_diagnoses'][ti] = np.count_nonzero(self.ti_diagnosed == ti)
        self.results['cum_diagnoses'][ti] = np.sum(self.results['new_diagnoses'][:ti + 1])
        self.results['new_agents_on_art'][ti] = np.count_nonzero(self.ti_art == ti)
        self.results['cum_agents_on_art'][ti] = np.sum(self.results['new_agents_on_art'][:ti + 1])
        self.results['n_on_art_pregnant'][ti] = np.count_nonzero(self.on_art & self.sim.people.pregnancy.pregnant)

        # Subset by FSW and client:
        fsw_infected = self.infected[self.sim.networks.structuredsexual.fsw]
        client_infected = self.infected[self.sim.networks.structuredsexual.client]
        for risk_group in np.unique(self.sim.networks.structuredsexual.risk_group).astype(int):
            for sex in ['female', 'male']:
                risk_group_infected = self.infected[(self.sim.networks.structuredsexual.risk_group == risk_group) & (self.sim.people[sex])]
                if len(risk_group_infected) > 0:
                    self.results['prevalence_risk_group_' + str(risk_group) + '_' + sex][ti] = sum(risk_group_infected) / len(risk_group_infected)
                    self.results['new_infections_risk_group_' + str(risk_group) + '_' + sex][ti] = sum(risk_group_infected) / len(risk_group_infected)

        # Add FSW and clients to results:
        if len(fsw_infected) > 0:
            self.results['prevalence_sw'][ti] = sum(fsw_infected) / len(fsw_infected)
        if len(client_infected) > 0:
            self.results['prevalence_client'][ti] = sum(client_infected) / len(client_infected)

        return

    def make_new_cases(self):
        """
        Add new HIV cases
        """
        super().make_new_cases()
        return

    def set_prognoses(self, uids, source_uids=None):
        """
        Set prognoses upon infection
        """
        super().set_prognoses(uids, source_uids)
        ti = self.sim.ti

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = ti
        self.ti_since_untreated[uids] = ti

        return

    def set_congenital(self, target_uids, source_uids):
        return self.set_prognoses(target_uids, source_uids)
