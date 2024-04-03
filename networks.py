"""
Define sexual network for syphilis.

Overview:
- Risk groups: agents are randomly assigned into one of 3 main risk groups:
    - 0 = marry and remain married to a single partner throughout their lifetime
    - 1 = marry and then divorce or who have concurrent partner(s) during their marriage
    - 2 = never marry
- In addition, a proportion of each of the groups above engages in sex work
"""

import starsim as ss
import numpy as np
import pandas as pd


class StructuredSexual(ss.SexualNetwork, ss.DynamicNetwork):
    """
    Structured sexual network
    """

    def __init__(self, pars=None, par_dists=None, key_dict=None, **kwargs):
        pars = ss.omergeleft(
            pars,

            # Settings - generally shouldn't be adjusted
            n_risk_groups=3,
            f_age_group_bins=dict(  # For separating women into age groups: teens, young women, adult women
                teens=(10, 20),
                young=(20, 25),
                adult=(25, 100),
            ),

            # Debut
            debut_f=ss.lognorm_ex(20, 3),
            debut_m=ss.lognorm_ex(21, 3),

            # Risk groups
            risk_groups_f=ss.choice(a=3, p=np.array([0.85, 0.14, 0.01])),
            risk_groups_m=ss.choice(a=3, p=np.array([0.78, 0.21, 0.01])),

            # Age difference preferences
            age_diff_pars=dict(
                teens=[(7, 3), (6, 3), (5, 1)],  # (mu,stdev) for levels 0, 1, 2
                young=[(8, 3), (7, 3), (5, 2)],
                adult=[(8, 3), (7, 3), (5, 2)],
            ),

            # Concurrency preferences - TODO, tidy
            f0_conc=ss.poisson(lam=0.0001),
            f1_conc=ss.poisson(lam=0.01),
            f2_conc=ss.poisson(lam=0.1),
            m0_conc=ss.poisson(lam=0.01),
            m1_conc=ss.poisson(lam=0.2),
            m2_conc=ss.poisson(lam=0.5),

            # Relationship initiation, stability, and duration
            p_pair_form=ss.bernoulli(p=0.5),  # Probability of a pair forming between two matched people
            p_stable0=ss.bernoulli(p=0.9),
            p_stable1=ss.bernoulli(p=0.5),
            p_stable2=ss.bernoulli(p=0),
            stable_dur_pars=dict(
                teens=[(100, 1),  (8, 2), (1e-4, 1e-4)],  # (mu,stdev) for levels 0, 1, 2
                young=[(100, 1), (10, 3), (1e-4, 1e-4)],
                adult=[(100, 1), (12, 3), (1e-4, 1e-4)],
            ),
            casual_dur_pars=dict(
                teens=[(1, 0.25)]*3,  # (mu,stdev) for levels 0, 1, 2
                young=[(1, 0.25)]*3,
                adult=[(1, 0.25)]*3,
            ),

            # Acts
            acts=ss.normal(loc=90, scale=30),  # Annual acts

            # Sex work parameters
            fsw_shares=0.02,
            client_shares=0.12,
            sw_seeking_rate=0.5,  # Annual rate at which clients seek FSWs (0.5 = 1 new SW partner every 2 years)
            sw_seeking_dist=ss.bernoulli(p=0.5),  # Placeholder value replaced by dt-adjusted sw_seeking_rate
            sw_beta=0.5,  # Replace with condom use

            # Distributions derived from parameters above - don't adjust
            age_diffs=ss.normal(loc=self.age_diff_fn_loc, scale=self.age_diff_fn_scale),
            dur_stable=ss.normal(loc=self.stable_loc, scale=self.stable_scale),  # TODO: change to lognorm
            dur_casual=ss.normal(loc=self.casual_loc, scale=self.casual_scale),
        )

        par_dists = ss.omergeleft(
            par_dists,
            fsw_shares=ss.bernoulli,
            client_shares=ss.bernoulli,
        )

        key_dict = ss.omerge({
            'sw': bool,
            'age_p1': float,
            'age_p2': float,
        }, key_dict)

        ss.DynamicNetwork.__init__(self, key_dict=key_dict, **kwargs)
        ss.SexualNetwork.__init__(self, pars, key_dict=key_dict, **kwargs)

        self.par_dists = par_dists

        # Add states
        self.risk_group = ss.State('risk_group', int, default=0)
        self.fsw = ss.State('fsw', bool, default=False)
        self.client = ss.State('client', bool, default=False)
        self.debut = ss.State('debut', float, default=0)
        self.participant = ss.State('participant', bool, default=True)
        self.concurrency = ss.State('concurrency', int, default=1)
        self.partners = ss.State('partners', int, default=0)
        self.lifetime_partners = ss.State('lifetime_partners', int, default=0)

        return

    @staticmethod
    def get_age_risk_pars(module, sim, uids, par):
        abins = module.pars.f_age_group_bins
        loc = pd.Series(0., index=uids)
        scale = pd.Series(1., index=uids)
        for a_label, a_range in abins.items():
            for rg in range(module.pars.n_risk_groups):
                age_conds = (sim.people.age[uids] >= a_range[0]) & (sim.people.age[uids] < a_range[1])
                f_el_bools = age_conds & (module.risk_group[uids] == rg) & sim.people.female[uids]
                f_el_uids = ss.true(f_el_bools)
                loc[f_el_uids] = module.pars[par][a_label][rg][0]
                scale[f_el_uids] = module.pars[par][a_label][rg][1]
        return loc, scale

    @staticmethod
    def age_diff_fn_loc(module, sim, uids, par='age_diff_pars'):
        loc, _ = module.get_age_risk_pars(module, sim, uids, par)
        return loc

    @staticmethod
    def age_diff_fn_scale(module, sim, uids, par='age_diff_pars'):
        _, scale = module.get_age_risk_pars(module, sim, uids, par)
        return scale

    @staticmethod
    def stable_loc(module, sim, uids, par='stable_dur_pars'):
        loc, _ = module.get_age_risk_pars(module, sim, uids, par)
        return loc

    @staticmethod
    def stable_scale(module, sim, uids, par='stable_dur_pars'):
        _, scale = module.get_age_risk_pars(module, sim, uids, par)
        return scale

    @staticmethod
    def casual_loc(module, sim, uids, par='casual_dur_pars'):
        loc, _ = module.get_age_risk_pars(module, sim, uids, par)
        return loc

    @staticmethod
    def casual_scale(module, sim, uids, par='casual_dur_pars'):
        _, scale = module.get_age_risk_pars(module, sim, uids, par)
        return scale

    def initialize(self, sim):
        super().initialize(sim)
        self.set_network_states(sim.people)
        return

    def set_network_states(self, people, upper_age=1000):
        self.set_risk_groups(people, upper_age=upper_age)
        self.set_concurrency(people, upper_age=upper_age)
        self.set_sex_work(people, upper_age=upper_age)
        self.set_debut(people, upper_age=upper_age)
        return

    @staticmethod
    def _get_uids(people, upper_age=None, by_sex=True):
        if upper_age is None: uids = people.uid
        else: uids = people.uid[(people.age < upper_age)]
        if by_sex:
            f_uids = uids[people.female[uids]]
            m_uids = uids[people.male[uids]]
            return f_uids, m_uids
        else:
            return uids

    def set_risk_groups(self, people, upper_age=None):
        """ Assign each person to a risk group """
        f_uids, m_uids = self._get_uids(people, upper_age=upper_age)
        self.risk_group[f_uids] = self.pars.risk_groups_f.rvs(f_uids) - f_uids
        self.risk_group[m_uids] = self.pars.risk_groups_m.rvs(m_uids) - m_uids
        return

    def set_concurrency(self, people, upper_age=1000):
        """ Assign each person a preferred number of simultaneous partners """
        for rg in range(self.pars.n_risk_groups):
            f_conc = self.pars[f'f{rg}_conc']
            m_conc = self.pars[f'm{rg}_conc']
            f_uids = people.female & (self.risk_group == rg) & (people.age < upper_age)
            m_uids = people.male & (self.risk_group == rg) & (people.age < upper_age)
            self.concurrency[f_uids] = f_conc.rvs(f_uids)+1
            self.concurrency[m_uids] = m_conc.rvs(m_uids)+1
        return

    def set_sex_work(self, people, upper_age=None):
        f_uids, m_uids = self._get_uids(people, upper_age=upper_age)
        self.fsw[f_uids] = self.pars.fsw_shares.rvs(f_uids)
        self.client[m_uids] = self.pars.client_shares.rvs(m_uids)
        return

    def set_debut(self, people, upper_age=None):
        f_uids, m_uids = self._get_uids(people, upper_age=upper_age)
        self.debut[f_uids] = self.pars.debut_f.rvs(f_uids)
        self.debut[m_uids] = self.pars.debut_m.rvs(m_uids)
        return

    def match_pairs(self, ppl):
        """
        Match pairs by age
        """

        # Find people eligible for a relationship
        f_active = self.active(ppl) & ppl.female
        m_active = self.active(ppl) & ppl.male
        underpartnered = self.partners < self.concurrency
        f_eligible = f_active & underpartnered
        m_eligible = m_active & underpartnered
        f_looking = self.pars.p_pair_form.filter(ss.true(f_eligible))  # To do: let p vary by age and with dt

        # Get mean age differences and desired ages
        age_gaps = self.pars.age_diffs.rvs(f_looking)   # Sample the age differences
        desired_ages = ppl.age[f_looking] + age_gaps    # Desired ages of the male partners

        # Sort the females according to the desired age of their partners
        desired_age_idx = np.argsort(desired_ages)  # Array positions for sorting the desired ages
        p2 = desired_ages.uid[desired_age_idx]      # Female UIDs sorted by age of their desired partner
        sorted_desired_ages = desired_ages[p2]      # Sorted desired ages

        # Sort the males by age
        m_ages = ppl.age[m_eligible]            # Ages of eligible males
        m_age_sidx = np.argsort(m_ages)         # Array positions for sorting the ages of males
        sorted_m_uids = m_ages.uid[m_age_sidx]  # Male UIDs sorted by age
        sorted_m_ages = m_ages[sorted_m_uids]   # Sort male ages

        # Get matches
        try:
            match_inds = abs(sorted_desired_ages.values[:, None] - sorted_m_ages.values[None, :]).argmin(axis=-1)
        except:
            import traceback; traceback.print_exc(); import pdb; pdb.set_trace()
        p1 = sorted_m_uids[match_inds]

        self.partners[p1] += 1
        self.partners[p2] += 1
        self.lifetime_partners[p1] += 1
        self.lifetime_partners[p2] += 1

        return p1, p2

    def add_pairs(self, ppl, ti=None):
        """ Add pairs """
        p1, p2 = self.match_pairs(ppl)

        # Initialize beta, acts, duration
        beta = pd.Series(1., index=p2)
        dur = pd.Series(ppl.dt, index=p2)  # Default duration is dt, replaced for stable matches
        acts = (self.pars.acts.rvs(p2) * ppl.dt).astype(int)  # Number of acts does not depend on commitment/risk group
        sw = np.full_like(p1, False, dtype=bool)
        age_p1 = ppl.age[p1].values
        age_p2 = ppl.age[p2].values

        # If both partners are in the same risk group, determine the probability they'll commit
        for rg in range(self.pars.n_risk_groups):
            matched_risk = (self.risk_group[p1] == rg) & (self.risk_group[p2] == rg)

            # If there are any matched pairs, check if they commit
            if len(ss.true(matched_risk)) > 0:

                matched_p2 = p2[matched_risk]
                stable_dist = self.pars[f'p_stable{rg}']   # To do: let p vary by age
                stable_bools = stable_dist.rvs(matched_p2)
                casual_bools = ~stable_bools

                if stable_bools.any():
                    stable_p2 = matched_p2[stable_bools]
                    dur[stable_p2] = self.pars.dur_stable.rvs(stable_p2)

                if casual_bools.any():
                    casual_p2 = matched_p2[casual_bools]
                    dur[casual_p2] = self.pars.dur_casual.rvs(casual_p2)

        # Get sex work values
        p1_sw, p2_sw, beta_sw, dur_sw, acts_sw, sw_sw, age_p1_sw, age_p2_sw = self.add_sex_work(ppl)

        self.contacts.p1 = np.concatenate([self.contacts.p1, p1, p1_sw])
        self.contacts.p2 = np.concatenate([self.contacts.p2, p2, p2_sw])
        self.contacts.beta = np.concatenate([self.contacts.beta, beta.values, beta_sw.values])
        self.contacts.dur = np.concatenate([self.contacts.dur, dur.values, dur_sw.values])
        self.contacts.acts = np.concatenate([self.contacts.acts, acts, acts_sw])
        self.contacts.sw = np.concatenate([self.contacts.sw, sw, sw_sw])
        self.contacts.age_p1 = np.concatenate([self.contacts.age_p1, age_p1, age_p1_sw])
        self.contacts.age_p2 = np.concatenate([self.contacts.age_p2, age_p2, age_p2_sw])

    def add_sex_work(self, ppl):
        """ Match sex workers to clients """

        # Find people eligible for a relationship
        active_fsw = self.active(ppl) & ppl.female & self.fsw
        active_clients = self.active(ppl) & ppl.male & self.client

        # Find clients who will seek FSW
        self.pars.sw_seeking_dist.pars.p = self.pars.sw_seeking_rate * ppl.dt
        m_looking = self.pars.sw_seeking_dist.filter(ss.true(active_clients))

        # Replace this with choice
        if len(m_looking) > len(ss.true(active_fsw)):  # Replace this - should assign an FSW to all potential clients
            n_pairs = len(ss.true(active_fsw))
            p2 = ss.true(active_fsw)
            p1 = m_looking[:n_pairs]
        else:
            n_pairs = len(m_looking)
            p2 = ss.true(active_fsw)[:n_pairs]
            p1 = m_looking

        # Beta, acts, duration
        beta = pd.Series(self.pars.sw_beta, index=p2)
        dur = pd.Series(ppl.dt, index=p2)  # Assumed instantaneous
        acts = (self.pars.acts.rvs(p2) * ppl.dt).astype(int)  # Could alternatively set to 1 and adjust beta
        sw = np.full_like(p1, True, dtype=bool)

        self.lifetime_partners[p1] += 1
        self.lifetime_partners[p2] += 1

        return p1, p2, beta, dur, acts, sw, ppl.age[p1].values, ppl.age[p2].values

    def end_pairs(self, people):
        dt = people.dt
        self.contacts.dur = self.contacts.dur - dt

        # Non-alive agents are removed
        alive_bools = people.alive[self.contacts.p1] & people.alive[self.contacts.p2]
        active = (self.contacts.dur > 0) & alive_bools

        # For gen pop contacts that are due to expire, decrement the partner count
        inactive_gp = ~active & (~self.contacts.sw)
        self.partners[self.contacts.p1[inactive_gp]] -= 1
        self.partners[self.contacts.p2[inactive_gp]] -= 1

        # For all contacts that are due to expire, remove them from the contacts list
        if len(active) > 0:
            for k in self.meta_keys():
                self.contacts[k] = self.contacts[k][active]

        return

    def update(self, people, dt=None):
        self.end_pairs(people)
        self.set_network_states(people, upper_age=people.dt)
        self.add_pairs(people)

        return
