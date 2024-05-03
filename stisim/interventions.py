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
        self.npts = sim.npts
        # self.n_products_used = ss.Result(name=f'Products administered by {self.label}', npts=sim.npts, scale=True)
        self.outcomes = {k: np.array([], dtype=np.int64) for k in self.product.hierarchy}
        return

    def apply(self, sim):
        self.outcomes = {k: np.array([], dtype=np.int64) for k in self.product.hierarchy}

        accept_inds = self.deliver(sim)
        sim.diseases[self.disease].diagnosed[accept_inds] = True
        # sim.people.screens[accept_inds] += 1
        sim.diseases[self.disease].ti_diagnosed[accept_inds] = sim.ti

        # Store results
        idx = sim.ti # int(sim.t / sim.resfreq)
        new_test_inds = accept_inds[np.logical_not(sim.diseases[self.disease].diagnosed[accept_inds])]  # Figure out people who are getting screened for the first time
        n_new_people = sim.people.scale_flows(new_test_inds)  # Scale
        n_new_tests = sim.people.scale_flows(accept_inds)  # Scale
        # sim.results['new_tested'][idx] += n_new_people
        # sim.results['new_tests'][idx] += n_new_tests

        return accept_inds

    def deliver(self, sim):
        '''
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        '''
        #ti = sc.findinds(self.timepoints, sim.ti)[0]
        prob = self.prob[0] #[ti]  # Get the proportion of people who will be tested this timestep
        eligible_inds = self.check_eligibility(sim)  # Check eligibility
        accept_inds = select_people(eligible_inds, prob=prob)  # Find people who accept
        if len(accept_inds):
            idx = sim.ti # int(sim.ti / sim.resfreq)
            # self.n_products_used[idx] += sim.people.scale_flows(accept_inds)
            self.outcomes = self.product.administer(sim, accept_inds)  # Actually administer the diagnostic, filtering people into outcome categories
        return accept_inds

    def check_eligibility(self, sim):
        return ss.true(self.eligibility(sim))


class HIV_testing(ss.Intervention):
    """
    Probability-based testing
    """

    def __init__(self, test_prob, sensitivity, disease, *args, test_delay_mean=None, vac_symp_prob=np.nan,
                 asymp_prob=np.nan, FSW_prob=None, exclude=None, test_delay=None, **kwargs):
        """
        Args:
            **kwargs:
        """

        super().__init__(*args, **kwargs)

        assert (test_delay_mean is None) != (
                test_delay is None), "Either the mean test delay or the absolute test delay must be specified"
        self.results = ss.Results(self.name)

        if asymp_prob == np.nan:
            self.asymp_prob = 0
        else:
            self.asymp_prob = asymp_prob
        if FSW_prob.all() is None:
            self.FSW_prob = test_prob
        else:
            self.FSW_prob = FSW_prob
        self.test_prob = test_prob

        if not isinstance(sensitivity, dict):
            self.sensitivity = {"symptomatic": sensitivity}
        else:
            self.sensitivity = sensitivity
        self.disease = disease
        self.test_delay_mean = test_delay_mean
        self.test_delay = test_delay
        self.vac_symp_prob = vac_symp_prob
        self.test_probs = ss.State('test_prob', float, 0.0)
        self.delays = ss.State('delay', float, np.nan)

        self.n_tests = None
        self.n_positive = None  # Record how many tests were performed that will come back positive
        self.exclude = exclude  # Exclude certain people - mainly to cater for simulations where the index case/incursion should not be diagnosed

        self._scheduled_tests = defaultdict(list)

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'new_tests', sim.npts, dtype=float)
        self.n_tests = np.zeros(sim.npts)
        self.n_positive = np.zeros(sim.npts)
        self.test_probs.initialize(sim.people)
        self.delays.initialize(sim.people)

    def apply(self, sim):
        test_uids = self.select_people(sim)
        self._test(sim, test_uids)

    def schedule_test(self, sim, uids, t: int):
        """
        Schedule a test in the future

        If the test is requested today, then test immediately.

        :param uids: Iterable with person indices to test
        :param t: Simulation day on which to test them
        :return:
        """

        if t == sim.ti:
            # If a person is scheduled to test on the same day (e.g., if they are a household contact and get tested on
            # the same day they are notified)

            not_dead_diag = sim.diseases[self.disease].diagnosed | sim.people.dead
            uids = uids[
                np.logical_not(not_dead_diag[uids])]  # Only test people that haven't been diagnosed and are alive
            self._test(sim, uids)
        else:
            self._scheduled_tests[t] += uids.tolist()

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

    def _test(self, sim, test_uids):
        # After testing (via self.apply or self.schedule_test) perform some post-testing tasks
        # test_uids are the indices of the people that were requested to be tested (i.e. that were
        # passed into sim.people.test, so a test was performed on them
        #
        # CAUTION - this method gets called via both apply() and schedule_test(), therefore it can be
        # called multiple times per timestep, quantities must be incremented rather than overwritten
        if len(test_uids) == 0:
            return

        symp_test_uids = test_uids[sim.diseases[self.disease].symptomatic[test_uids]]
        other_test_uids = test_uids[~sim.diseases[self.disease].symptomatic[test_uids]]

        if len(symp_test_uids):
            sim.diseases[self.disease].test(symp_test_uids, sim.ti, test_sensitivity=self.sensitivity['symptomatic'],
                                            loss_prob=0, test_delay=np.inf)  # Actually test people with mild symptoms
        if len(other_test_uids):
            sim.diseases[self.disease].test(other_test_uids, sim.ti, test_sensitivity=self.sensitivity['symptomatic'],
                                            loss_prob=0, test_delay=np.inf)  # Actually test people without symptoms

        if self.test_delay is not None:
            self.delays[test_uids] = self.test_delay
        else:
            self.delays[test_uids] = np.maximum(1, np.random.poisson(self.test_delay_mean, len(test_uids)))

        # Update the date diagnosed
        positive_today = ss.true(sim.diseases[self.disease].ti_pos_test[test_uids] == sim.ti)

        sim.diseases[self.disease].ti_diagnosed[positive_today] = sim.ti + self.delays[positive_today]

        # Logging
        self.n_positive[sim.ti] = len(positive_today)

        # Store tests performed by this intervention
        # ToDO: would need adjusting for population changes?
        n_tests = len(test_uids) * sim.pars["pop_scale"]
        self.n_tests[sim.ti] += n_tests
        self.results["new_tests"][sim.ti] = n_tests  # Update total test count

    def select_people(self, sim):
        # First, insert any fixed test probabilities
        self.test_probs.values = np.ones(len(sim.people)) * self.asymp_prob
        self.test_probs[
            sim.diseases[self.disease].symptomatic] = self.symp_prob  # Symptomatic people test at a higher rate
        if self.exclude is not None:
            self.test_probs[
                self.exclude] = 0  # If someone is excluded, then they shouldn't test via `apply()` (but can still test via a scheduled test)
        if sim.pars.remove_dead and len(self._scheduled_tests[sim.ti]) > 0:
            self.clean_uid(sim)
        self.test_probs[self._scheduled_tests[
            sim.ti]] = 1  # People scheduled to test (e.g. via contact tracing) are guaranteed to test
        self.test_probs[sim.diseases[self.disease].diagnosed] = 0  # People already diagnosed don't test again
        self.test_probs[sim.diseases[self.disease].dead] = 0  # Dead people don't get tested

        test_uids = ss.true(
            np.random.random(self.test_probs.shape) < self.test_probs)  # Finally, calculate who actually tests
        return test_uids

    def clean_uid(self, sim):
        """
        Removes uids of dead agents if simulation is removing them
        """

        self._scheduled_tests[sim.ti] = [uid for uid in self._scheduled_tests[sim.ti] if uid in sim.people.uid]


class ART(ss.Intervention):
    """
    ART-treatment intervention by Robyn Stuart, Daniel Klein and Cliff Kerr, edited by Alina Muellenmeister
    """

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):

        pars = ss.omergeleft(pars,
                             ART_coverages_df=None,
                             ART_prob=0.9,
                             duration_on_ART=ss.normal(loc=18, scale=5),
                             art_efficacy=0.96)

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self._pending_ART = defaultdict(list)
        self.disease = 'hiv'
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_art', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):

        diagnosed = ss.true(sim.diseases[self.disease].ti_diagnosed == sim.ti)
        self.schedule_ART_treatment(diagnosed, sim.ti)

        # Check who is starting ART
        self.check_start_ART_treatment(sim)
        # Check who is stopping ART
        self.check_stop_ART_treatment(sim)
        # Apply correction to match ART coverage data:
        self.ART_coverage_correction(sim)

        return

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
            self._pending_ART[start_date].append((uid, start_date + period))
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
        for uid, end_day in self._pending_ART[sim.ti]:
            if uid in sim.people.alive.uid:
                sim.diseases[self.disease].on_art[uid] = True
                sim.diseases[self.disease].ti_art[uid] = sim.ti
                # Determine when agents goes off ART:
                sim.diseases[self.disease].ti_stop_art[uid] = sim.ti + int(self.pars.duration_on_ART.rvs(1))
        return

    def check_stop_ART_treatment(self, sim):
        """
        Check who is stopping ART treatment
        """
        stop_uids = self.check_uids(~sim.diseases[self.disease].on_art, sim.diseases[self.disease].ti_stop_art, sim.ti,
                                    filter_uids=None)
        sim.diseases[self.disease].on_art[stop_uids] = False
        sim.diseases[self.disease].ti_art[stop_uids] = np.nan
        sim.diseases[self.disease].ti_stop_art[stop_uids] = np.nan
        sim.diseases[self.disease].ti_since_untreated[stop_uids] = sim.ti
        return

    def ART_coverage_correction(self, sim):
        """
        Adjust number of people on treatment to match data
        """
        infected_uids_onART = sim.diseases[self.disease].diagnosed & sim.diseases[self.disease].infected & sim.diseases[self.disease].on_art
        infected_uids_not_onART = sim.diseases[self.disease].diagnosed & sim.diseases[self.disease].infected & ~sim.diseases[self.disease].on_art

        # Get the current ART coverage. If year is not available, assume 90%
        if len(self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()) > 0:
            ART_coverage_this_year = \
                self.pars.ART_coverages_df[self.pars.ART_coverages_df['Years'] == sim.year]['Value'].tolist()[0]
        else:
            ART_coverage_this_year = int(0.9 * len(ss.true(sim.diseases[self.disease].infected)))

        # Too many agents on treatment -> remove
        if len(ss.true(infected_uids_onART)) > ART_coverage_this_year:
            # Agents with the highest CD4 counts will go off ART:
            n_agents_to_stop_ART = int(len(ss.true(infected_uids_onART)) - ART_coverage_this_year)
            cd4_counts_onART = sim.diseases[self.disease].cd4[infected_uids_onART]
            cd4_counts_onART.sort(axis=0)
            # Grab the last n agents with the highest counts
            probabilities = (cd4_counts_onART / np.sum(cd4_counts_onART)).values
            # Probabilities are increasing with CD4 count
            uids = cd4_counts_onART.uid
            stop_uids = np.random.choice(uids, n_agents_to_stop_ART, p=probabilities, replace=False)
            sim.diseases[self.disease].on_art[stop_uids] = False
            uids_update_ti_untreated = ss.true(sim.diseases[self.disease].ti_art[stop_uids] != sim.ti)
            sim.diseases[self.disease].ti_art[stop_uids] = np.nan
            sim.diseases[self.disease].ti_stop_art[stop_uids] = np.nan
            # Only update when agents actually have been on ART:
            sim.diseases[self.disease].ti_since_untreated[uids_update_ti_untreated] = sim.ti

        # Not enough agents on treatment -> add
        elif len(ss.true(infected_uids_onART)) < ART_coverage_this_year:
            # Agents with the lowest CD4 count will get on ART:
            n_agents_to_start_ART = int(ART_coverage_this_year - len(ss.true(infected_uids_onART)))
            cd4_counts_not_onART = sim.diseases[self.disease].cd4[infected_uids_not_onART]
            cd4_counts_not_onART.sort(axis=0)
            probabilities = (cd4_counts_not_onART / np.sum(cd4_counts_not_onART)).values
            # Probabilities are increasing with CD4 count, therefore flip uid array:
            uids = np.flipud(cd4_counts_not_onART.uid)
            start_uids = np.random.choice(uids, n_agents_to_start_ART, p=probabilities, replace=False)

            # Put them on ART
            sim.diseases[self.disease].on_art[start_uids] = True
            sim.diseases[self.disease].ti_art[start_uids] = sim.ti
            # Determine when agents go off ART:
            # sim.diseases[self.disease].ti_stop_art[start_uids] = sim.ti + self.pars.duration_on_ART.rvs(len(start_uids)).astype(int)

        return


# %% Custom Interventions
__all__ += ['test_ART']


class test_ART(ss.Intervention):
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
            if sim.ti == self.infect_uids_t[index] and uid in sim.people.alive.uid and \
                    sim.diseases[self.disease].infected[uid] == False:
                sim.diseases[self.disease].infected[uid] = True
                sim.diseases[self.disease].ti_infected[uid] = sim.ti
                sim.diseases[self.disease].ti_since_untreated[uid] = sim.ti
                sim.diseases[self.disease].susceptible[uid] = False
                sim.diseases[self.disease].ti_infectious[uid] = sim.ti + 14

                # if self.stop_ART:
                #    sim.diseases[self.disease].ti_stop_art[uid] = sim.ti + sim.diseases[self.disease].pars.avg_duration_stop_ART

            if uid in sim.people.alive.uid:
                # Check if it's time to restart ART treatment:
                # if sim.diseases[self.disease].on_art[uid] and self.stop_ART and self.restart_ART and sim.ti == sim.diseases[self.disease].ti_stop_art[uid]:
                #    sim.diseases[self.disease].schedule_ART_treatment(np.array([uid]), sim.ti + sim.diseases[self.disease].pars.avg_duration_restart_ART)

                if sim.diseases[self.disease].on_art[uid]:
                    ART_status = 'on_ART'
                else:
                    ART_status = 'not_on_ART'
                self.results['cd4_count_' + str(uid)][0] = sim.diseases[self.disease].cd4_start[uid]
                self.results['cd4_count_' + str(uid)][sim.ti] = sim.diseases[self.disease].cd4[uid]
                self.results['ART_status_' + str(uid)][sim.ti] = ART_status
                self.results['status_' + str(uid)][sim.ti] = 'alive'
                self.results['transmission_' + str(uid)][sim.ti] = sim.diseases[self.disease].rel_trans[uid]

            else:
                self.results['cd4_count_' + str(uid)][sim.ti] = np.nan
                self.results['status_' + str(uid)][sim.ti] = 'dead'

        return
