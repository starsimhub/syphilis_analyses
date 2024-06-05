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
            # Natural history
            cd4_start=ss.normal(loc=800, scale=50),
            cd4_latent=ss.normal(loc=500, scale=50),
            dur_acute=ss.lognorm_ex(3/12, 1/12),    # Duration of acute HIV infection
            dur_latent=ss.lognorm_ex(10, 3),        # Duration of latent, untreated HIV infection
            dur_falling=ss.lognorm_ex(3, 1),        # Duration of late-stage HIV when CD4 counts fall
            p_hiv_death=None,  # Probability of death from HIV-related complications - default is to use HIV.death_prob(), otherwise can pass in a Dist or anything supported by ss.bernoulli)
            include_aids_deaths=True,

            # Transmission
            beta=1,  # Placeholder, replaced by network-specific betas
            beta_m2f=None,
            beta_f2m=None,
            beta_m2c=None,
            rel_trans_acute=ss.normal(loc=6, scale=0.5),  # Increase transmissibility during acute HIV infection
            rel_trans_falling=ss.normal(loc=8, scale=0.5),  # Increase transmissibility during late HIV infection

            # Initialization
            init_prev=ss.bernoulli(p=0.05),
            init_diagnosed=ss.bernoulli(p=0.01),
            dist_ti_init_infected=ss.uniform(low=-10 * 12, high=0),

            # Care seeking
            care_seeking=ss.normal(loc=1, scale=0.1),  # Distribution of relative care-seeking behavior
            maternal_care_scale=2,  # Factor for scaling up care-seeking behavior during pregnancy

            # Treatment effects
            art_cd4_growth=0.1,  # How quickly CD4 reconstitutes after starting ART - used in a logistic growth function
            art_efficacy=0.96,  # Efficacy of ART
            time_to_art_efficacy=0.5,  # Time to reach full ART efficacy (in years) - linear increase in efficacy
            art_cd4_pars=dict(cd4_max=1000, cd4_healthy=500),
            dur_on_art=ss.normal(loc=18, scale=5),
            dur_post_art=ss.normal(loc=self.dur_post_art_mean, scale=self.dur_post_art_scale),
            dur_post_art_scale_factor=0.1,
        )

        self.update_pars(pars, **kwargs)

        # Set death probabilities from HIV-related illness. Note that AIDS deaths are captured separately
        if self.pars.p_hiv_death is None:
            self._death_prob = ss.bernoulli(p=self.death_prob)
        elif isinstance(self.pars.p_hiv_death, ss.bernoulli):
            self._death_prob = self.pars.p_hiv_death
        else:
            self._death_prob = ss.bernoulli(p=self.pars.p_hiv_death)

        # States
        self.add_states(
            # Natural history
            ss.FloatArr('ti_acute'),
            ss.BoolArr('acute'),
            ss.FloatArr('ti_latent'),
            ss.BoolArr('latent'),
            ss.FloatArr('ti_falling'),
            ss.BoolArr('falling'),
            ss.BoolArr('post_art'),  # After stopping ART, CD4 falls linearly until death
            ss.FloatArr('ti_zero'),  # Time of zero CD4 count - generally corresponds to AIDS death
            ss.FloatArr('ti_dead'),  # Time of HIV/AIDS death

            # Care and treatment states
            ss.FloatArr('baseline_care_seeking'),
            ss.FloatArr('care_seeking'),
            ss.BoolArr('never_art', default=True),
            ss.BoolArr('on_art'),
            ss.FloatArr('ti_art'),
            ss.FloatArr('ti_stop_art'),

            # CD4 states
            ss.FloatArr('cd4'),             # Current CD4 count
            ss.FloatArr('cd4_start'),       # Initial CD4 count for each agent before an infection
            ss.FloatArr('cd4_preart'),      # CD4 immediately before initiating ART
            ss.FloatArr('cd4_latent'),      # CD4 count during latent infection
            ss.FloatArr('cd4_nadir'),       # Lowest CD4
            ss.FloatArr('cd4_potential'),   # Potential CD4 count if continually treated
            ss.FloatArr('cd4_postart'),     # CD4 after stopping ART

            # Knowledge of HIV status
            ss.BoolArr('diagnosed'),
            ss.FloatArr('ti_diagnosed'),
        )

        self._pending_ARTtreatment = defaultdict(list)
        self.initial_hiv_maternal_beta = None

        return

    @property
    def include_mtct(self): return 'pregnancy' in self.sim.demographics

    def init_pre(self, sim):
        """
        Initialize
        """
        super().init_pre(sim)

        # Optionally scale betas
        if self.pars.beta_m2f is not None:
            self.pars.beta['structuredsexual'][0] *= self.pars.beta_m2f
        if self.pars.beta_f2m is not None:
            self.pars.beta['structuredsexual'][1] *= self.pars.beta_f2m
        if self.pars.beta_m2c is not None:
            self.pars.beta['maternal'][1] *= self.pars.beta_m2c

        return

    def init_post(self):
        """ Set states """
        # Set initial CD4
        self.init_cd4()
        self.init_care_seeking()

        # Make initial cases, some of which may have occured prior to the sim start
        initial_cases = self.pars.init_prev.filter()
        ti_init_cases = self.pars.dist_ti_init_infected.rvs(initial_cases).astype(int)
        self.set_prognoses(initial_cases, ti=ti_init_cases)
        initial_cases_diagnosed = self.pars.init_diagnosed.filter(initial_cases)
        self.diagnosed[initial_cases_diagnosed] = True
        self.ti_diagnosed[initial_cases_diagnosed] = 0

        return

    # CD4 functions
    def acute_decline(self, uids):
        """ Acute decline in CD4 """
        acute_start = self.ti_acute[uids] - self.sim.ti
        acute_end = self.ti_latent[uids]
        acute_dur = acute_end - acute_start
        cd4_start = self.cd4_start[uids]
        cd4_end = self.cd4_latent[uids]
        per_timestep_decline = sc.safedivide(cd4_start-cd4_end, acute_dur)
        cd4 = cd4_start + per_timestep_decline*acute_start
        return cd4

    def falling_decline(self, uids):
        """ Decline in CD4 during late-stage infection, when counts are falling """
        falling_start = self.ti_falling[uids]
        falling_end = self.ti_zero[uids]
        falling_dur = falling_end - falling_start
        time_falling = self.sim.ti - self.ti_falling[uids]
        cd4_start = self.cd4_latent[uids]
        cd4_end = 1  # To avoid divide by zero problems
        per_timestep_decline = sc.safedivide(cd4_start-cd4_end, falling_dur)
        cd4 = np.maximum(0, cd4_start - per_timestep_decline*time_falling)
        return cd4

    def post_art_decline(self, uids):
        """
        Decline in CD4 after going off treatment
        This implementation has the possibly-undesirable feature that a person
        who goes on ART for a year and then off again might have a slightly shorter
        lifespan than if they'd never started treatment.
        """
        ti_stop_art = self.ti_stop_art[uids]
        ti_zero = self.ti_zero[uids]
        post_art_dur = ti_zero - ti_stop_art
        time_post_art = self.sim.ti - ti_stop_art
        cd4_start = self.cd4_postart[uids]
        cd4_end = 1  # To avoid divide by zero problems
        per_timestep_decline = (cd4_start-cd4_end)/post_art_dur
        cd4 = np.maximum(0, cd4_start - per_timestep_decline*time_post_art)
        return cd4

    def cd4_increase(self, uids):
        """
        Increase CD4 counts for people who are receiving treatment.
        Growth curves are calculated to match EMODs CD4 reconstitution equation for people who initiate treatment
        with a CD4 count of 50 (https://docs.idmod.org/projects/emod-hiv/en/latest/hiv-model-healthcare-systems.html)
        However, here we use a logistic growth function and assume that ART CD4 count depends on CD4 at initiation.
        Sources:
            - https://i-base.info/guides/starting/cd4-increase
            - https://www.sciencedirect.com/science/article/pii/S1876034117302022
            - https://bmcinfectdis.biomedcentral.com/articles/10.1186/1471-2334-8-20
        """
        # Calculate time on ART and CD4 prior to starting
        ti_art = self.ti_art[uids]
        cd4_preart = self.cd4_preart[uids]
        dur_art = self.sim.ti - ti_art

        # Extract growth parameters
        growth_rate = self.pars.art_cd4_growth
        cd4_total_gain = self.cd4_potential[uids] - self.cd4_preart[uids]
        cd4_now = 2*cd4_total_gain/(1+np.exp(-dur_art*growth_rate))-cd4_total_gain+cd4_preart  # Concave logistic

        return cd4_now

    @property
    def symptomatic(self):
        return self.infectious

    @staticmethod
    def death_prob(module, sim=None, uids=None):
        cd4_bins = np.array([1000, 500, 350, 200, 50, 0])
        death_prob = sim.dt*np.array([0.003, 0.003, 0.005, 0.001, 0.05, 0.200])  # Values smaller than the first bin edge get assigned to the last bin.
        return death_prob[np.digitize(module.cd4[uids], cd4_bins)]

    @staticmethod
    def _interpolate(vals: list, t):
        vals = sorted(vals, key=lambda x: x[0])  # Make sure values are sorted
        assert len({x[0] for x in vals}) == len(vals)  # Make sure time points are unique
        return np.interp(t, [x[0] for x in vals], [x[1] for x in vals], left=vals[0][1], right=vals[-1][1])

    def init_cd4(self):
        """
        Set CD4 counts
        """
        uids = ss.uids(self.cd4_start.isnan)
        self.cd4_start[uids] = self.pars.cd4_start.rvs(uids)
        self.cd4_nadir[uids] = sc.dcp(self.cd4_start[uids])
        return

    def init_care_seeking(self):
        """
        Set care seeking behavior
        """
        uids = ss.uids(self.care_seeking.isnan)
        self.care_seeking[uids] = self.pars.care_seeking.rvs(uids)
        self.baseline_care_seeking[uids] = sc.dcp(self.care_seeking[uids])  # Copy it so pregnancy can modify it
        return

    def update_pre(self):
        """
        Carry out autonomous updates at the start of the timestep (prior to transmission)
        """
        ti = self.sim.ti

        # Set initial CD4 counts for new agents:
        self.init_cd4()

        # Handle care seeking behavior. First, initialize, then adjust depending on pregnancy:
        # increase care-seeking for pregnant women and decrease again after postpartum.
        # This makes it much less likely that pregnant women will stop treatment
        self.init_care_seeking()
        if self.include_mtct:
            pregnant = self.sim.demographics.pregnancy.pregnant
            self.care_seeking[pregnant] = self.baseline_care_seeking[pregnant] * self.pars.maternal_care_scale
            self.care_seeking[~pregnant] = self.baseline_care_seeking[~pregnant]

        # Adjust CD4 counts for people receiving treatment - logarithmic increase
        if self.on_art.any():
            art_uids = self.on_art.uids
            self.cd4[art_uids] = self.cd4_increase(art_uids)

        # Adjust CD4 counts for people who have gone off treatment - linear decline
        if (~self.on_art & ~self.never_art).any():
            off_art_uids = (~self.on_art & ~self.never_art).uids
            self.cd4[off_art_uids] = self.post_art_decline(off_art_uids)

        # Update states for people who have never been on ART (ART removes these)
        # Acute & not on ART
        acute = self.acute
        self.cd4[acute.uids] = self.acute_decline(acute.uids)

        # Latent & not on ART
        latent = self.acute & (self.ti_latent <= ti)
        self.acute[latent] = False
        self.latent[latent] = True

        untreated_latent = self.latent
        self.cd4[untreated_latent.uids] = self.cd4_latent[untreated_latent.uids]

        # Falling & not on ART
        falling = self.latent & (self.ti_falling <= ti)
        self.latent[falling] = False
        self.falling[falling] = True

        untreated_falling = self.falling 
        if untreated_falling.any():
            self.cd4[untreated_falling.uids] = self.falling_decline(untreated_falling.uids)

        # Update CD4 nadir for anyone not on treatment
        untreated = self.infected & ~self.on_art
        self.cd4_nadir[untreated] = np.minimum(self.cd4_nadir[untreated], self.cd4[untreated])

        # Update transmission
        self.update_transmission()

        # Update deaths. We capture deaths from AIDS (i.e., when CD4 count drops to ~0) as well as deaths from
        # serious HIV-related illnesses, which can occur throughout HIV.
        off_art = (self.infected & ~self.on_art).uids
        hiv_deaths = self._death_prob.filter(off_art)
        if len(hiv_deaths):
            self.ti_dead[hiv_deaths] = ti
            self.sim.people.request_death(hiv_deaths)
        if self.pars.include_aids_deaths:
            aids_deaths = (self.ti_zero <= ti).uids
            if len(aids_deaths):
                self.ti_dead[aids_deaths] = ti
                self.sim.people.request_death(aids_deaths)
        return

    def update_transmission(self):
        """
        Update rel_trans and rel_sus for all agents. These are reset on each timestep then adjusted depending on states.
        Adjustments are made throughout different modules:
           - rel_trans for acute and late-stage untreated infection are adjusted below
           - rel_trans for all people on treatment (including pregnant women) below
           - rel_sus for unborn babies of pregnant WLHIV receiving treatment is adjusted in the ART intervention
        """
        sim = self.sim
        ti = sim.ti

        # Reset susceptibility and infectiousness
        self.rel_sus[:] = 1
        self.rel_trans[:] = 1

        # Update rel_trans to account for acute and late-stage infection
        self.rel_trans[self.acute] *= self.pars.rel_trans_acute.rvs(self.acute.uids)
        aids = self.cd4 < 200
        self.rel_trans[aids] *= self.pars.rel_trans_falling.rvs(aids.uids)

        # Update transmission for agents on ART
        # When agents start ART, determine the reduction of transmission (linearly decreasing over 6 months)
        if self.on_art.any():
            full_eff = self.pars.art_efficacy
            time_to_full_eff = self.pars.time_to_art_efficacy
            art_uids = self.on_art.uids
            dur_art = ti - self.ti_art[art_uids]
            months_on_art = dur_art*sim.dt*12
            new_on_art = months_on_art < (time_to_full_eff/sim.dt)
            efficacy_to_date = np.full_like(art_uids, fill_value=full_eff, dtype=float)
            efficacy_to_date[new_on_art] = months_on_art[new_on_art]*full_eff/(time_to_full_eff/sim.dt)
            self.rel_trans[art_uids] *= 1 - efficacy_to_date

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
        if self.include_mtct:
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
        if self.include_mtct:
            self.results['n_on_art_pregnant'][ti] = np.count_nonzero(self.on_art & self.sim.people.pregnancy.pregnant)

        # Subset by FSW and client:
        fsw_infected = self.infected[self.sim.networks.structuredsexual.fsw]
        client_infected = self.infected[self.sim.networks.structuredsexual.client]
        for risk_group in range(self.sim.networks.structuredsexual.pars.n_risk_groups):
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

    def set_prognoses(self, uids, source_uids=None, ti=None):
        """
        Set prognoses upon infection
        """
        if ti is None:
            ti = self.sim.ti
        else:
            # Check that ti is consistent with uids
            if not (sc.isnumber(ti) or len(ti) == len(uids)):
                errormsg = 'ti for set_prognoses must be int or array of length uids'
                raise ValueError(errormsg)

        dt = self.sim.dt

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.acute[uids] = True

        self.ti_infected[uids] = ti
        self.ti_acute[uids] = ti

        # Set timing and CD4 count of latent infection
        dur_acute = self.pars.dur_acute.rvs(uids)
        self.ti_latent[uids] = self.ti_acute[uids] + (dur_acute / dt).astype(int)
        self.cd4_latent[uids] = self.pars.cd4_latent.rvs(uids)

        # Set time of onset of late-stage CD4 decline
        dur_latent = self.pars.dur_latent.rvs(uids)
        self.ti_falling[uids] = self.ti_latent[uids] + (dur_latent / dt).astype(int)
        dur_falling = self.pars.dur_falling.rvs(uids)
        self.ti_zero[uids] = self.ti_falling[uids] + (dur_falling / dt).astype(int)

        return

    def set_congenital(self, target_uids, source_uids):
        return self.set_prognoses(target_uids, source_uids)

    # Treatment-related changes
    def start_art(self, uids):
        """
        Check who is ready to start ART treatment and put them on ART
        """
        ti = self.sim.ti
        dt = self.sim.dt

        self.on_art[uids] = True
        newly_treated = uids[self.never_art[uids]]
        self.never_art[newly_treated] = False
        self.ti_art[uids] = ti
        self.cd4_preart[uids] = self.cd4[uids]

        # Determine when agents goes off ART
        dur_on_art = self.pars.dur_on_art.rvs(uids)
        self.ti_stop_art[uids] = ti + (dur_on_art / dt).astype(int)

        # ART nullifies all states and all future dates in the natural history
        self.acute[uids] = False
        self.latent[uids] = False
        self.falling[uids] = False
        future_latent = uids[self.ti_latent[uids] > ti]
        self.ti_latent[future_latent] = np.nan
        future_falling = uids[self.ti_falling[uids] > ti]
        self.ti_falling[future_falling] = np.nan
        future_zero = uids[self.ti_zero[uids] > ti]  # NB, if they are scheduled to die on this time step, they will
        self.ti_zero[future_zero] = np.nan

        # Set CD4 potential for anyone new to treatment - retreated people have the same potential
        # Extract growth parameters
        if len(newly_treated) > 0:
            cd4_max = self.pars.art_cd4_pars['cd4_max']
            cd4_healthy = self.pars.art_cd4_pars['cd4_healthy']
            cd4_preart = self.cd4_preart[newly_treated]

            # Calculate potential CD4 increase - assuming that growth follows the concave part of a logistic function
            # and that the total gain depends on the CD4 count at initiation
            cd4_scale_factor = (cd4_max-cd4_preart)/cd4_healthy*np.log(cd4_max/cd4_preart)
            cd4_total_gain = cd4_preart*cd4_scale_factor
            self.cd4_potential[newly_treated] = self.cd4_preart[newly_treated] + cd4_total_gain

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

    def stop_art(self, uids=None):
        """
        Check who is stopping ART treatment and put them off ART
        """
        ti = self.sim.ti
        dt = self.sim.dt

        # Remove agents from ART
        if uids is None: uids = self.on_art & (self.ti_stop_art <= ti)
        self.on_art[uids] = False
        self.ti_stop_art[uids] = ti
        self.cd4_postart[uids] = sc.dcp(self.cd4[uids])

        # Set decline
        dur_post_art = self.pars.dur_post_art.rvs(uids)
        self.ti_zero[uids] = ti + (dur_post_art / dt).astype(int)

        return

