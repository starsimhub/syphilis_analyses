"""
Define sexual network for syphilis.

Overview:
- Risk groups: agents are randomly assigned into one of 3 main risk groups:
    - 0 = marry and remain married to a single partner throughout their lifetime
    - 1 = marry and then divorce or who have concurrent partner(s) during their marriage
    - 2 = never marry
- In addition, a proportion of each of the groups above engages in sex work.
"""

import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd
import stisim as sti
import scipy.optimize as spo
import scipy.spatial as spsp

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

    def __init__(self, pars=None, key_dict=None, condom_data=None, **kwargs):

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
                adult=(25, np.inf),
            ),

            # Debut
            debut_f=ss.lognorm_ex(20, 3),
            debut_m=ss.lognorm_ex(21, 3),

            # Risk groups
            prop_f1=0.15,
            prop_m1=0.2,
            prop_f2=0.01,
            prop_m2=0.02,
            risk_groups_f = ss.choice(a=3),
            risk_groups_m = ss.choice(a=3),

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
            m0_conc=ss.poisson(lam=0.0001),
            m1_conc=ss.poisson(lam=0.2),
            m2_conc=ss.poisson(lam=0.5),

            # Relationship initiation, stability, and duration
            p_pair_form=ss.bernoulli(p=0.5),  # Probability of a (stable) pair forming between two matched people
            p_matched_stable = [ss.bernoulli(p=0.9),ss.bernoulli(p=0.5),ss.bernoulli(p=0)],  # Probability of a stable pair forming between matched people (otherwise casual)
            p_mismatched_casual = [ss.bernoulli(p=0.5),ss.bernoulli(p=0.5),ss.bernoulli(p=0.5)],  # Probability of a casual pair forming between mismatched people (otherwise instantanous)

            stable_dur_pars=dict(
                teens=[(100, 1),  (8, 2), (1e-4, 1e-4)],  # (mu,stdev) for levels 0, 1, 2
                young=[(100, 1), (10, 3), (1e-4, 1e-4)],
                adult=[(100, 1), (12, 3), (1e-4, 1e-4)],
            ),
            casual_dur_pars=dict(
                teens=[(0.1, 0.25)]*3,  # (mu,stdev) for levels 0, 1, 2
                young=[(0.1, 0.25)]*3,
                adult=[(0.1, 0.25)]*3,
            ),

            # Acts
            acts=ss.lognorm_ex(90, 30),  # Annual acts

            # Sex work parameters
            fsw_shares=ss.bernoulli(p=0.05),
            client_shares=ss.bernoulli(p=0.12),
            sw_seeking_rate=12,  # Annual rate at which clients seek FSWs (12 = 1 new SW partner every month)
            sw_seeking_dist=ss.bernoulli(p=0.5),  # Placeholder value replaced by dt-adjusted sw_seeking_rate
            sw_beta=1,  # Replace with condom use
            sw_intensity=ss.random(),  # At each time step, FSW may work with varying intensity

            # Distributions derived from parameters above - don't adjust
            age_diffs=ss.normal(),
            dur_stable=ss.lognorm_ex(),
            dur_casual=ss.lognorm_ex(),
        )

        self.update_pars(pars=pars, **kwargs)

        # Set condom use
        self.condom_data = None
        if condom_data is not None:
            self.condom_data = self.process_condom_data(condom_data)

        # Add states
        self.participant = ss.BoolArr('participant', default=True)
        self.risk_group = ss.FloatArr('risk_group')     # Which risk group an agent belongs to
        self.fsw = ss.BoolArr('fsw')                    # Whether an agent is a female sex worker
        self.client = ss.BoolArr('client')              # Whether an agent is a client of sex workers
        self.concurrency = ss.FloatArr('concurrency')   # Preferred number of concurrent partners
        self.partners = ss.FloatArr('partners', default=0)  # Actual number of concurrent partners
        self.lifetime_partners = ss.FloatArr('lifetime_partners', default=0)   # Lifetime total number of partners
        self.sw_intensity = ss.FloatArr('sw_intensity')  # Intensity of sex work

        return

    @staticmethod
    def process_condom_data(condom_data):
        if sc.isnumber(condom_data):
            return condom_data
        elif isinstance(condom_data, pd.DataFrame):
            df = condom_data.melt(id_vars=['partnership'])
            dd = dict()
            for pcombo in df.partnership.unique():
                key = tuple(map(int, pcombo[1:-1].split(',')))
                thisdf = df.loc[df.partnership == pcombo]
                dd[key] = dict()
                dd[key]['year'] = thisdf.variable.values.astype(int)
                dd[key]['val'] = thisdf.value.values
        return dd

    def get_age_risk_pars(self, uids, par):
        loc = np.full(uids.shape, fill_value=np.nan, dtype=ss_float_)
        scale = np.full(uids.shape, fill_value=np.nan, dtype=ss_float_)
        for a_label, (age_lower, age_upper) in self.pars.f_age_group_bins.items():
            for rg in range(self.pars.n_risk_groups):
                in_risk_group = (self.sim.people.age[uids] >= age_lower) & (self.sim.people.age[uids] < age_upper) & (self.risk_group[uids] == rg)
                loc[in_risk_group] = par[a_label][rg][0]
                scale[in_risk_group] = par[a_label][rg][1]
        return loc, scale

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.condom_data is not None:
            if isinstance(self.condom_data, dict):
                for rgtuple, valdict in self.condom_data.items():
                    self.condom_data[rgtuple]['simvals'] = sc.smoothinterp(sim.yearvec, valdict['year'], valdict['val'])
        self.init_results()
        return

    def init_results(self):
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'share_active', npts, dtype=float, scale=False),
            ss.Result(self.name, 'partners_f_mean', npts, dtype=float, scale=False),
            ss.Result(self.name, 'partners_m_mean', npts, dtype=float, scale=False),
        ]
        return

    def init_post(self):
        super().init_post(add_pairs=False)
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
        risk_groups_f = np.array([1-self.pars.prop_f1-self.pars.prop_f2, self.pars.prop_f1, self.pars.prop_f2])
        self.pars.risk_groups_f.set(p=risk_groups_f)
        risk_groups_m = np.array([1-self.pars.prop_m1-self.pars.prop_m2, self.pars.prop_m1, self.pars.prop_m2])
        self.pars.risk_groups_m.set(p=risk_groups_m)
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

        if len(f_looking) == 0 or m_eligible.count() == 0:
            raise NoPartnersFound()

        # Get mean age differences and desired ages
        loc, scale = self.get_age_risk_pars(f_looking, self.pars.age_diff_pars)
        self.pars.age_diffs.set(loc=loc, scale=scale)
        age_gaps = self.pars.age_diffs.rvs(f_looking)   # Sample the age differences
        desired_ages = ppl.age[f_looking] + age_gaps    # Desired ages of the male partners
        m_ages = ppl.age[m_eligible]            # Ages of eligible males
        dist_mat = spsp.distance_matrix(m_ages[:, np.newaxis], desired_ages[:, np.newaxis])
        ind_m, ind_f = spo.linear_sum_assignment(dist_mat)
        p1 = m_eligible.uids[ind_m]
        p2 = f_looking[ind_f]

        return p1, p2

    def add_pairs(self, ti=None):
        """ Add pairs """
        ppl = self.sim.people
        dt = self.sim.dt

        # Obtain new pairs
        try:
            p1, p2 = self.match_pairs(ppl)
        except NoPartnersFound:
            return

        # Initialize beta, acts, duration
        condoms = np.zeros(len(p2), dtype=ss_float_)
        dur = np.full(len(p2), dtype=ss_float_, fill_value=dt) # Default duration is dt, replaced for stable matches
        acts = (self.pars.acts.rvs(p2) * dt).astype(int)  # Number of acts does not depend on commitment/risk group
        sw = np.full_like(p1, False, dtype=bool)
        age_p1 = ppl.age[p1]
        age_p2 = ppl.age[p2]

        # First figure out reduction in transmission through condom use
        if self.condom_data is not None:
            if isinstance(self.condom_data, dict):
                for rgm in range(self.pars.n_risk_groups):
                    for rgf in range(self.pars.n_risk_groups):
                        risk_pairing = (self.risk_group[p1] == rgm) & (self.risk_group[p2] == rgf)
                        condoms[risk_pairing] = self.condom_data[(rgm, rgf)]['simvals'][self.sim.ti]
            elif sc.isnumber(self.condom_data):
                condoms[:] = self.condom_data
            else:
                raise Exception("Unknown condom data input type")

        # If both partners are in the same risk group, determine the probability they'll commit
        for rg in range(self.pars.n_risk_groups):
            matched_risk = (self.risk_group[p1] == rg) & (self.risk_group[p2] == rg)
            mismatched_risk = (self.risk_group[p1] == rg) & (self.risk_group[p2] != rg)

            # For matched pairs, there is a probability of forming a stable pair, and failing that, forming a casual pair
            if matched_risk.any():
                stable_dist = self.pars.p_matched_stable[rg]  # To do: let p vary by age
                stable = stable_dist.rvs(p2)
                stable_bool = stable & matched_risk
                casual_bool = ~stable & matched_risk

                if stable_bool.any():
                    uids = p2[stable_bool]
                    loc, scale = self.get_age_risk_pars(uids, self.pars.stable_dur_pars)
                    self.pars.dur_stable.set(loc=loc, scale=scale)
                    dur[stable_bool] = self.pars.dur_stable.rvs(uids) # nb. must use stable_bool on the LHS to support repeated edges. Todo: use different durations for each partnership for the same UID

                if casual_bool.any():
                    uids = p2[casual_bool]
                    loc, scale = self.get_age_risk_pars(uids, self.pars.casual_dur_pars)
                    self.pars.dur_casual.set(loc=loc, scale=scale)
                    dur[casual_bool] = self.pars.dur_casual.rvs(uids)

            # If there are any mismatched pairs, determine the probability they'll have a non-instantaneous partnership
            if mismatched_risk.any():
                casual_dist = self.pars.p_mismatched_casual[rg]  # To do: let p vary by age
                casual = casual_dist.rvs(p2)
                casual_bool = casual & mismatched_risk
                uids = p2[casual_bool]
                loc, scale = self.get_age_risk_pars(uids, self.pars.casual_dur_pars)
                self.pars.dur_casual.set(loc=loc, scale=scale)
                dur[casual_bool] = self.pars.dur_casual.rvs(uids)

        self.append(p1=p1, p2=p2, beta=1-condoms, dur=dur, acts=acts, sw=sw, age_p1=age_p1, age_p2=age_p2)

        # Checks
        if self.sim.people.female[p1].any() or self.sim.people.male[p2].any():
            errormsg = 'Same-sex pairings should not be possible in this network'
            raise ValueError(errormsg)
        if len(p1) != len(p2):
            errormsg = 'Unequal lengths in edge list'
            raise ValueError(errormsg)

        # Get sex work values
        p1_sw, p2_sw, beta_sw, dur_sw, acts_sw, sw_sw, age_p1_sw, age_p2_sw = self.add_sex_work(ppl)

        # Finalize adding the edges to the network
        self.append(p1=p1_sw, p2=p2_sw, beta=beta_sw, dur=dur_sw, acts=acts_sw, sw=sw_sw, age_p1=age_p1_sw, age_p2=age_p2_sw)

        unique_p1, counts_p1 = np.unique(p1, return_counts=True)
        unique_p2, counts_p2 = np.unique(p2, return_counts=True)
        self.partners[unique_p1] += counts_p1
        self.partners[unique_p2] += counts_p2
        self.lifetime_partners[unique_p1] += counts_p1
        self.lifetime_partners[unique_p2] += counts_p2

        return

    def add_sex_work(self, ppl):
        """ Match sex workers to clients """

        dt = self.sim.dt
        # Find people eligible for a relationship
        active_fsw = self.active(ppl) & ppl.female & self.fsw
        active_clients = self.active(ppl) & ppl.male & self.client
        self.sw_intensity[active_fsw.uids] = self.pars.sw_intensity.rvs(active_fsw.uids)

        # Find clients who will seek FSW
        self.pars.sw_seeking_dist.pars.p = np.clip(self.pars.sw_seeking_rate * dt, 0, 1)
        m_looking = self.pars.sw_seeking_dist.filter(active_clients.uids)

        # Attempt to assign a sex worker to every client by repeat sampling the sex workers.
        # FSW with higher work intensity will be sampled more frequently
        if len(m_looking) > len(active_fsw.uids):
            n_repeats = (self.sw_intensity[active_fsw]*10).astype(int)+1
            fsw_repeats = np.repeat(active_fsw.uids, n_repeats)
            if len(fsw_repeats) < len(m_looking):
                fsw_repeats = np.repeat(fsw_repeats, 10)  # 10x the number of clients each sex worker can have

            # Might still not have enough FSW, so form as many pairs as possible
            n_pairs = min(len(fsw_repeats), len(m_looking))
            if len(fsw_repeats) < len(m_looking):
                p1 = m_looking[:n_pairs]
                p2 = fsw_repeats
            else:
                unique_sw, counts_sw = np.unique(fsw_repeats, return_counts=True)
                count_repeats = np.repeat(counts_sw, counts_sw)
                weights = self.sw_intensity[fsw_repeats] / count_repeats
                choices = np.argsort(-weights)[:n_pairs]
                p2 = fsw_repeats[choices]
                p1 = m_looking

        else:
            n_pairs = len(m_looking)
            weights = self.sw_intensity[active_fsw]
            choices = np.argsort(-weights)[:n_pairs]
            p2 = active_fsw.uids[choices]
            p1 = m_looking

        # Beta, acts, duration
        beta = pd.Series(self.pars.sw_beta, index=p2)
        dur = pd.Series(dt, index=p2)  # Assumed instantaneous
        acts = (self.pars.acts.rvs(p2) * dt).astype(int)  # Could alternatively set to 1 and adjust beta
        sw = np.full_like(p1, True, dtype=bool)

        unique_p1, counts_p1 = np.unique(p1, return_counts=True)
        unique_p2, counts_p2 = np.unique(p2, return_counts=True)
        self.lifetime_partners[unique_p1] += counts_p1
        self.lifetime_partners[unique_p2] += counts_p2

        # Check
        if self.sim.people.female[p1].any() or self.sim.people.male[p2].any():
            errormsg = 'Same-sex sex work pairings should not be possible within in this network'
            raise ValueError(errormsg)
        if len(p1) != len(p2):
            errormsg = 'Unequal lengths in edge list'
            raise ValueError(errormsg)

        return p1, p2, beta, dur, acts, sw, ppl.age[p1], ppl.age[p2]

    def end_pairs(self):
        people = self.sim.people
        dt = self.sim.dt

        self.edges.dur = self.edges.dur - dt

        # Non-alive agents are removed
        alive_bools = people.alive[ss.uids(self.edges.p1)] & people.alive[ss.uids(self.edges.p2)]
        active = (self.edges.dur > 0) & alive_bools

        # For gen pop contacts that are due to expire, decrement the partner count
        inactive_gp = ~active & (~self.edges.sw)
        self.partners[ss.uids(self.edges.p1[inactive_gp])] -= 1
        self.partners[ss.uids(self.edges.p2[inactive_gp])] -= 1

        # For all contacts that are due to expire, remove them from the contacts list
        if len(active) > 0:
            for k in self.meta_keys():
                self.edges[k] = (self.edges[k][active])

        return

    def update_results(self):
        ti = self.sim.ti
        partners_active_m = self.partners[(self.sim.people.male & self.active(self.sim.people))]
        partners_active_f = self.partners[(self.sim.people.female & self.active(self.sim.people))]
        self.results.share_active[ti] = len(self.active(self.sim.people).uids)/len(self.sim.people)
        # self.results.partners_f_mean[ti] = np.mean(partners_active_f)
        # self.results.partners_m_mean[ti] = np.mean(partners_active_m)

    def update(self):
        self.end_pairs()
        self.set_network_states(upper_age=self.sim.dt)
        self.add_pairs()
        self.update_results()

        return
