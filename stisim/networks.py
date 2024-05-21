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
import sciris as sc
import numpy as np
import pandas as pd

ss_float_ = ss.dtypes.float

# Specify all externally visible functions this file defines; see also more definitions below
__all__ = ['StructuredSexual']

class NoPartnersFound(Exception):
    # Raise this exception if the matching algorithm wasn't able to match any partners
    pass


class StructuredSexual(ss.SexualNetwork):
    """
    Structured sexual network
    """

    def __init__(self, pars=None, key_dict=None, **kwargs):

        key_dict = sc.mergedicts({
            'sw': bool,
            'age_p1': ss_float_,
            'age_p2': ss_float_,
        }, key_dict)

        super().__init__(key_dict=key_dict)

        self.default_pars(
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
            fsw_shares=ss.bernoulli(p=0.02),
            client_shares=ss.bernoulli(p=0.12),
            sw_seeking_rate=0.5,  # Annual rate at which clients seek FSWs (0.5 = 1 new SW partner every 2 years)
            sw_seeking_dist=ss.bernoulli(p=0.5),  # Placeholder value replaced by dt-adjusted sw_seeking_rate
            sw_beta=0.5,  # Replace with condom use

            # Distributions derived from parameters above - don't adjust
            age_diffs=ss.normal(loc=self.age_diff_fn_loc, scale=self.age_diff_fn_scale),
            dur_stable=ss.normal(loc=self.stable_loc, scale=self.stable_scale),  # TODO: change to lognorm
            dur_casual=ss.normal(loc=self.casual_loc, scale=self.casual_scale),
        )

        self.update_pars(pars=pars, **kwargs)

        # Add states
        self.participant = ss.BoolArr('participant', default=True)
        self.risk_group = ss.FloatArr('risk_group')     # Which risk group an agent belongs to
        self.fsw = ss.BoolArr('fsw')                    # Whether an agent is a female sex worker
        self.client = ss.BoolArr('client')              # Whether an agent is a client of sex workers
        self.concurrency = ss.FloatArr('concurrency')   # Preferred number of concurrent partners
        self.partners = ss.FloatArr('partners', default=0)  # Actual number of concurrent partners
        self.lifetime_partners = ss.FloatArr('lifetime_partners', default=0)   # Lifetime total number of partners

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
                f_el_uids = ss.uids(f_el_bools)
                f_el_uids = uids[f_el_bools.nonzero()[0]]  # FIX THIS
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

    def init_vals(self):
        super().init_vals(add_pairs=False)
        self.set_network_states()
        return

    def set_network_states(self, upper_age=None):
        self.set_risk_groups(upper_age=upper_age)
        self.set_concurrency(upper_age=upper_age)
        self.set_sex_work(upper_age=upper_age)
        self.set_debut(upper_age=upper_age)
        return

    def _get_uids(self, upper_age=None, by_sex=True):
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age < upper_age
        if by_sex:
            f_uids = (within_age & people.female).uids
            m_uids = (within_age & people.male).uids
            return f_uids, m_uids
        else:
            uids = within_age.uids
            return uids

    def set_risk_groups(self, upper_age=None):
        """ Assign each person to a risk group """
        f_uids, m_uids = self._get_uids(upper_age=upper_age)
        self.risk_group[f_uids] = self.pars.risk_groups_f.rvs(f_uids)
        self.risk_group[m_uids] = self.pars.risk_groups_m.rvs(m_uids)
        return

    def set_concurrency(self, upper_age=None):
        """ Assign each person a preferred number of simultaneous partners """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        in_age_lim = (people.age < upper_age)
        for rg in range(self.pars.n_risk_groups):
            f_conc = self.pars[f'f{rg}_conc']
            m_conc = self.pars[f'm{rg}_conc']
            in_risk_group = self.risk_group == rg
            in_group = in_risk_group & in_age_lim
            f_uids = (people.female & in_group).uids
            m_uids = (people.male   & in_group).uids
            self.concurrency[f_uids] = f_conc.rvs(f_uids) + 1
            self.concurrency[m_uids] = m_conc.rvs(m_uids) + 1
        return

    def set_sex_work(self, upper_age=None):
        f_uids, m_uids = self._get_uids(upper_age=upper_age)
        self.fsw[f_uids] = self.pars.fsw_shares.rvs(f_uids)
        self.client[m_uids] = self.pars.client_shares.rvs(m_uids)
        return

    def set_debut(self, upper_age=None):
        f_uids, m_uids = self._get_uids(upper_age=upper_age)
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
        f_looking = self.pars.p_pair_form.filter(f_eligible.uids)  # ss.uids of women looking for partners

        if len(f_looking) == 0 or len(m_eligible) == 0:
            raise NoPartnersFound()

        # Get mean age differences and desired ages
        age_gaps = self.pars.age_diffs.rvs(f_looking)   # Sample the age differences
        desired_ages = ppl.age[f_looking] + age_gaps    # Desired ages of the male partners

        # Sort the females according to the desired age of their partners
        desired_age_idx = np.argsort(desired_ages)  # Array positions for sorting the desired ages
        p2 = f_looking[desired_age_idx]      # Female UIDs sorted by age of their desired partner
        sorted_desired_ages = desired_ages[desired_age_idx]      # Sorted desired ages

        # Sort the males by age
        m_ages = ppl.age[m_eligible]            # Ages of eligible males
        m_age_sidx = np.argsort(m_ages)         # Array positions for sorting the ages of males
        sorted_m_uids = ss.uids(m_eligible.uids[m_age_sidx])  # Male UIDs sorted by age
        sorted_m_ages = m_ages[m_age_sidx]   # Sort male ages

        # Get matches
        match_inds = abs(sorted_desired_ages[:, None] - sorted_m_ages[None, :]).argmin(axis=-1)
        p1 = sorted_m_uids[match_inds]

        self.partners[p1] += 1
        self.partners[p2] += 1
        self.lifetime_partners[p1] += 1
        self.lifetime_partners[p2] += 1

        return p1, p2

    def add_pairs(self, ti=None):
        """ Add pairs """
        ppl = self.sim.people
        dt = self.sim.dt

        try:
            p1, p2 = self.match_pairs(ppl)
        except NoPartnersFound:
            return

        # Initialize beta, acts, duration
        beta = pd.Series(1., index=p2)
        dur = pd.Series(dt, index=p2)  # Default duration is dt, replaced for stable matches
        acts = (self.pars.acts.rvs(p2) * dt).astype(int)  # Number of acts does not depend on commitment/risk group
        sw = np.full_like(p1, False, dtype=bool)
        age_p1 = ppl.age[p1]
        age_p2 = ppl.age[p2]

        # If both partners are in the same risk group, determine the probability they'll commit
        for rg in range(self.pars.n_risk_groups):
            matched_risk = (self.risk_group[p1] == rg) & (self.risk_group[p2] == rg)

            # If there are any matched pairs, check if they commit
            if matched_risk.any():

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


        self.append(p1=p1, p2=p2, beta=beta, dur=dur, acts=acts, sw=sw, age_p1=age_p1, age_p2=age_p2)

        # Get sex work values
        p1_sw, p2_sw, beta_sw, dur_sw, acts_sw, sw_sw, age_p1_sw, age_p2_sw = self.add_sex_work(ppl)
        self.append(p1=p1_sw, p2=p1_sw, beta=beta_sw, dur=dur_sw, acts=acts_sw, sw=sw_sw, age_p1=age_p1_sw, age_p2=age_p2_sw)


    def add_sex_work(self, ppl):
        """ Match sex workers to clients """

        dt = self.sim.dt
        # Find people eligible for a relationship
        active_fsw = self.active(ppl) & ppl.female & self.fsw
        active_clients = self.active(ppl) & ppl.male & self.client

        # Find clients who will seek FSW
        self.pars.sw_seeking_dist.pars.p = self.pars.sw_seeking_rate * dt
        m_looking = self.pars.sw_seeking_dist.filter(active_clients.uids)

        # Replace this with choice
        if len(m_looking) > len(active_fsw.uids):  # Replace this - should assign an FSW to all potential clients
            n_pairs = len(active_fsw.uids)
            p2 = active_fsw.uids
            p1 = m_looking[:n_pairs]
        else:
            n_pairs = len(m_looking)
            p2 = active_fsw.uids[:n_pairs]
            p1 = m_looking

        # Beta, acts, duration
        beta = pd.Series(self.pars.sw_beta, index=p2)
        dur = pd.Series(dt, index=p2)  # Assumed instantaneous
        acts = (self.pars.acts.rvs(p2) * dt).astype(int)  # Could alternatively set to 1 and adjust beta
        sw = np.full_like(p1, True, dtype=bool)

        self.lifetime_partners[p1] += 1
        self.lifetime_partners[p2] += 1

        return p1, p2, beta, dur, acts, sw, ppl.age[p1], ppl.age[p2]

    def end_pairs(self):
        people = self.sim.people
        dt = self.sim.dt

        self.contacts.dur = self.contacts.dur - dt

        # Non-alive agents are removed
        alive_bools = people.alive[ss.uids(self.contacts.p1)] & people.alive[ss.uids(self.contacts.p2)]
        active = (self.contacts.dur > 0) & alive_bools

        # For gen pop contacts that are due to expire, decrement the partner count
        inactive_gp = ~active & (~self.contacts.sw)
        self.partners[ss.uids(self.contacts.p1[inactive_gp])] -= 1
        self.partners[ss.uids(self.contacts.p2[inactive_gp])] -= 1

        # For all contacts that are due to expire, remove them from the contacts list
        if len(active) > 0:
            for k in self.meta_keys():
                self.contacts[k] = (self.contacts[k][active])

        return

    def update(self):
        self.end_pairs()
        self.set_network_states(upper_age=self.sim.dt)
        self.add_pairs()

        return
