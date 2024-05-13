"""
Define interventions (and analyzers)
"""

import starsim as ss
import sciris as sc
import numpy as np
import pandas as pd
from collections import defaultdict
import pylab as pl
import inspect


def find_timepoint(arr, t=None, interv=None, sim=None, which='first'):
    '''
    Helper function to find if the current simulation time matches any timepoint in the
    intervention. Although usually never more than one index is returned, it is
    returned as a list for the sake of easy iteration.

    Args:
        arr (list/function): list of timepoints in the intervention, or a boolean array; or a function that returns these
        t (int): current simulation time (can be None if a boolean array is used)
        interv (intervention): the intervention object (usually self); only used if arr is callable
        sim (sim): the simulation object; only used if arr is callable
        which (str): what to return: 'first', 'last', or 'all' indices

    Returns:
        inds (list): list of matching timepoints; length zero or one unless which is 'all'
    '''
    if callable(arr):
        arr = arr(interv, sim)
        arr = sc.promotetoarray(arr)
    all_inds = sc.findinds(arr=arr, val=t)
    if len(all_inds) == 0 or which == 'all':
        inds = all_inds
    elif which == 'first':
        inds = [all_inds[0]]
    elif which == 'last':
        inds = [all_inds[-1]]
    else:  # pragma: no cover
        errormsg = f'Argument "which" must be "first", "last", or "all", not "{which}"'
        raise ValueError(errormsg)
    return inds


def select_people(inds, prob=None):
    '''
    Return an array of indices of people to who accept a service being offered

    Args:
        inds: array of indices of people offered a service (e.g. screening, triage, treatment)
        prob: acceptance probability

    Returns: Array of indices of people who accept
    '''
    accept_probs = np.full_like(inds, fill_value=prob, dtype=np.float64)
    accept_inds = np.random.random(accept_probs.shape) < prob
    return inds[accept_inds]


def get_subtargets(subtarget, sim):
    '''
    A small helper function to see if subtargeting is a list of indices to use,
    or a function that needs to be called. If a function, it must take a single
    argument, a sim object, and return a list of indices. Also validates the values.
    Currently designed for use with testing interventions, but could be generalized
    to other interventions. Not typically called directly by the user.

    Args:
        subtarget (dict): dict with keys 'inds' and 'vals'; see test_num() for examples of a valid subtarget dictionary
        sim (Sim): the simulation object
    '''

    # Validation
    if callable(subtarget):
        subtarget = subtarget(sim)

    if 'inds' not in subtarget:  # pragma: no cover
        errormsg = f'The subtarget dict must have keys "inds" and "vals", but you supplied {subtarget}'
        raise ValueError(errormsg)

    # Handle the two options of type
    if callable(subtarget['inds']):  # A function has been provided
        subtarget_inds = subtarget['inds'](sim)  # Call the function to get the indices
    else:
        subtarget_inds = subtarget['inds']  # The indices are supplied directly

    # Validate the values
    if callable(subtarget['vals']):  # A function has been provided
        subtarget_vals = subtarget['vals'](sim)  # Call the function to get the indices
    else:
        subtarget_vals = subtarget['vals']  # The indices are supplied directly
    if sc.isiterable(subtarget_vals):
        if len(subtarget_vals) != len(subtarget_inds):  # pragma: no cover
            errormsg = f'Length of subtargeting indices ({len(subtarget_inds)}) does not match length of values ({len(subtarget_vals)})'
            raise ValueError(errormsg)

    return subtarget_inds, subtarget_vals


__all__ = ['BaseTest']


class BaseTest(ss.Intervention):
    '''
    Base class for screening and triage.

    Args:
         product        (str/Product)   : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of screening strategy
         kwargs         (dict)          : passed to Intervention()
    '''

    def __init__(self, product=None, prob=None, eligibility=None, disease=None, **kwargs):
        ss.Intervention.__init__(self, **kwargs)

        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self.disease = disease
        self._parse_product(product)
        self.screened = ss.BoolArr('screened', default=False)
        self.ti_screened = ss.FloatArr('ti_screened')
        self.outcomes = None
        self.timepoints = []  # The start and end timepoints of the intervention

    def _parse_product(self, product):
        '''
        Parse the product input
        '''
        if isinstance(product, ss.Product):  # No need to do anything
            self.product = product
        elif isinstance(product, str):  # Try to find it in the list of defaults
            try:
                self.product = ss.default_dx(prod_name=product)
            except:
                errormsg = f'Could not find product {product} in the standard list.'
                raise ValueError(errormsg)
        else:
            errormsg = f'Cannot understand format of product {product} - please provide it as either a Product or string matching a default product.'
            raise ValueError(errormsg)
        return

    def initialize(self, sim):
        ss.Intervention.initialize(self, sim)
        self.init_results(sim)
        self.npts = sim.npts
        # self.n_products_used = ss.Result(name=f'Products administered by {self.label}', npts=sim.npts, scale=True)
        self.outcomes = {k: np.array([], dtype=np.int64) for k in self.product.hierarchy}
        return

    def init_results(self, sim):
        self.results += [
            ss.Result(self.name, 'new_screened', sim.npts, dtype=float, scale=True),
            ss.Result(self.name, 'new_screens', sim.npts, dtype=int, scale=True)]

        return

    def apply(self, sim):
        self.outcomes = {k: np.array([], dtype=np.int64) for k in self.product.hierarchy}
        accept_inds = self.deliver(sim)

        # Store results
        idx = sim.ti
        new_screen_inds = accept_inds[self.screened[accept_inds]]  # Figure out people who are getting screened for the first time
        n_new_people = sim.people.scale_flows(new_screen_inds)  # Scale
        n_new_screens = sim.people.scale_flows(accept_inds)  # Scale
        self.results['new_screened'][idx] += n_new_people
        self.results['new_screens'][idx] += n_new_screens

        # Update states
        self.screened[accept_inds] = True
        sim.diseases[self.disease].diagnosed[self.outcomes['positive']] = True
        self.ti_screened[accept_inds] = sim.ti
        sim.diseases[self.disease].ti_diagnosed[self.outcomes['positive']] = sim.ti

        return accept_inds

    def deliver(self, sim):
        '''
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        '''
        ti = np.minimum(len(self.prob) - 1, sc.findinds(np.unique(np.floor(sim.yearvec)), np.floor(sim.year))[0])
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep

        eligible_inds = self.check_eligibility(sim)  # Check eligibility
        accept_inds = select_people(eligible_inds, prob=prob)  # Find people who accept
        if len(accept_inds):
            idx = sim.ti  # int(sim.ti / sim.resfreq)
            # self.n_products_used[idx] += sim.people.scale_flows(accept_inds)
            self.outcomes = self.product.administer(sim, accept_inds)  # Actually administer the diagnostic, filtering people into outcome categories
        return accept_inds

    def check_eligibility(self, sim):
        return self.eligibility(sim).uids

class ART(ss.Intervention):
    """
    ART-treatment intervention by Robyn Stuart, Daniel Klein and Cliff Kerr, edited by Alina Muellenmeister
    """

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):

        pars = ss.dictmergeleft(pars,
                                ART_coverages_df=None,
                                duration_on_ART=ss.normal(loc=18, scale=5),
                                art_efficacy=0.96)

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self._pending_ART = defaultdict(list)
        self.disease = 'hiv'
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.initialized = True
        return

    @staticmethod
    def check_uids(current, date, t, filter_uids=None):
        """
        Return indices for which the current state is false and which meet the date criterion
        """
        if filter_uids is None:
            not_current = current.auids.remove(current.uids)
        else:
            not_current = filter_uids[np.logical_not(current[filter_uids])]
        has_date = not_current[~np.isnan(date[not_current])]
        uids = has_date[t >= date[has_date]]
        return uids

    def apply(self, sim):
        """
        Apply the ART intervention at each time step. Put agents on and off ART and adjust based on data.
        """

        diagnosed = (sim.diseases[self.disease].ti_diagnosed == sim.ti).uids  # Uids of agents diagnosed in this time step

        # Get the current ART coverage. If year is not available, assume 90%
        if len(self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()) > 0:
            ART_coverage_this_year = self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()[0]
        else:
            ART_coverage_this_year = self.pars.ART_coverages_df.Value.iloc[-1]  # Assume last coverage
        ART_coverage = ART_coverage_this_year
        # Schedule ART for a proportion of the newly diagnosed agents:
        diagnosed_to_start_ART = diagnosed[np.random.random(len(diagnosed)) < ART_coverage]

        # Check who is starting ART
        self.start_ART_treatment(sim, diagnosed_to_start_ART)
        # Check who is stopping ART
        self.check_stop_ART_treatment(sim)
        # Apply correction to match ART coverage data:
        self.ART_coverage_correction(sim, ART_coverage * len(sim.diseases[self.disease].diagnosed.uids))

        return

    def start_ART_treatment(self, sim, uids):
        """
        Check who is ready to start ART treatment and put them on ART
        """
        for uid in uids:
            if uid in sim.people.alive.uids:
                sim.diseases[self.disease].on_art[ss.uids(uid)] = True
                sim.diseases[self.disease].ti_art[ss.uids(uid)] = sim.ti
                # Determine when agents goes off ART:
                sim.diseases[self.disease].ti_stop_art[ss.uids(uid)] = sim.ti + int(self.pars.duration_on_ART.rvs(1))
        return

    def check_stop_ART_treatment(self, sim):
        """
        Check who is stopping ART treatment and put them off ART
        """
        stop_uids = self.check_uids(~sim.diseases[self.disease].on_art, sim.diseases[self.disease].ti_stop_art, sim.ti, filter_uids=None)
        sim.diseases[self.disease].on_art[stop_uids] = False
        sim.diseases[self.disease].ti_art[stop_uids] = np.nan
        sim.diseases[self.disease].ti_stop_art[stop_uids] = np.nan
        sim.diseases[self.disease].ti_since_untreated[stop_uids] = sim.ti
        return

    def ART_coverage_correction(self, sim, ART_coverage_this_year):
        """
        Adjust number of people on treatment to match data
        """
        infected_uids_onART = sim.diseases[self.disease].diagnosed & sim.diseases[self.disease].on_art
        infected_uids_not_onART = sim.diseases[self.disease].diagnosed & ~sim.diseases[self.disease].on_art

        # Too many agents on treatment -> remove
        if len(infected_uids_onART.uids) > ART_coverage_this_year:
            # Agents with the highest CD4 counts will go off ART:
            n_agents_to_stop_ART = int(len(infected_uids_onART.uids) - ART_coverage_this_year)
            cd4_counts_onART = sim.diseases[self.disease].cd4[infected_uids_onART]
            # Sort
            uids_onART = infected_uids_onART.uids
            cd4_counts_sort_idx = np.argsort(cd4_counts_onART)
            uids_onART_sorted = uids_onART[cd4_counts_sort_idx]
            # Grab the last n agents with the highest counts
            probabilities = (cd4_counts_onART / np.sum(cd4_counts_onART))
            # Probabilities are increasing with CD4 count
            uids = uids_onART_sorted
            stop_uids = np.random.choice(uids, n_agents_to_stop_ART, p=probabilities, replace=False)
            sim.diseases[self.disease].on_art[ss.uids(stop_uids)] = False
            uids_update_ti_untreated = ss.uids(stop_uids[sim.diseases[self.disease].ti_art[ss.uids(stop_uids)] != sim.ti])
            sim.diseases[self.disease].ti_art[ss.uids(stop_uids)] = np.nan
            sim.diseases[self.disease].ti_stop_art[ss.uids(stop_uids)] = np.nan
            # Only update when agents actually have been on ART:
            sim.diseases[self.disease].ti_since_untreated[uids_update_ti_untreated] = sim.ti

        # Not enough agents on treatment -> add
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


# %% Custom Interventions
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
        for index, uid in enumerate(self.uids):
            self.results += ss.Result(self.name, 'status_' + str(uid), sim.npts, dtype=np.dtype(('U', 10)))
            self.results += ss.Result(self.name, 'ART_status_' + str(uid), sim.npts, dtype=np.dtype(('U', 10)))
            self.results += ss.Result(self.name, 'cd4_count_' + str(uid), sim.npts, dtype=float, scale=False)
            self.results += ss.Result(self.name, 'transmission_' + str(uid), sim.npts, dtype=float, scale=False)

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
