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
    accept_inds = ss.true(ss.binomial_arr(accept_probs))
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


__all__ = ['Analyzer', 'Intervention']


class Analyzer(ss.Module):
    """
    Base class for analyzers. Analyzers are used to provide more detailed information 
    about a simulation than is available by default -- for example, pulling states 
    out of sim.people on a particular timestep before they get updated on the next step.
    
    The key method of the analyzer is ``apply()``, which is called with the sim
    on each timestep.
    
    To retrieve a particular analyzer from a sim, use sim.get_analyzer().
    """

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def initialize(self, sim):
        return super().initialize(sim)

    def apply(self, sim):
        pass

    def finalize(self, sim):
        return super().finalize(sim)


class Intervention:
    '''
    Base class for interventions.

    Args:
        label       (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label in the legend
        do_plot    (bool): whether or not to plot the intervention
        line_args  (dict): arguments passed to pl.axvline() when plotting
    '''

    def __init__(self, label=None, show_label=False, do_plot=None, line_args=None, **kwargs):
        # super().__init__(**kwargs)
        self._store_args()  # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__  # Use the class name if no label is supplied
        self.label = label  # e.g. "Screen"
        self.show_label = show_label  # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else False  # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0),
                                       line_args)  # Do not set alpha by default due to the issue of overlapping interventions
        self.timepoints = []  # The start and end timepoints of the intervention
        self.initialized = False  # Whether or not it has been initialized
        self.finalized = False  # Whether or not it has been initialized
        return

    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k, v in pars.items()])
                output = f"stisim.{which}({parstr})"
            except Exception as E:
                output = f'{type(self)} (error: {str(E)})'  # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'

    def __call__(self, *args, **kwargs):
        # Makes Intervention(sim) equivalent to Intervention.apply(sim)
        if not self.initialized:  # pragma: no cover
            errormsg = f'Intervention (label={self.label}, {type(self)}) has not been initialized'
            raise RuntimeError(errormsg)
        return self.apply(*args, **kwargs)

    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)

    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        f0 = inspect.currentframe()  # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0)  # The list of outer frames
        if self.__class__.__init__ is Intervention.__init__:
            parent = f1[1].frame  # parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        else:
            parent = f1[2].frame  # parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        _, _, _, values = inspect.getargvalues(parent)  # Get the values of the arguments
        if values:
            self.input_args = {}
            for key, value in values.items():
                if key == 'kwargs':  # Store additional kwargs directly
                    for k2, v2 in value.items():  # pragma: no cover
                        self.input_args[k2] = v2  # These are already a dict
                elif key not in ['self', '__class__']:  # Everything else, but skip these
                    self.input_args[key] = value
        return

    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        self.finalized = False
        return

    def finalize(self, sim=None):
        '''
        Finalize intervention

        This method is run once as part of ``sim.finalize()`` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:  # pragma: no cover
            raise RuntimeError(
                'Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return

    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError

    def shrink(self, in_place=False):
        '''
        Remove any excess stored data from the intervention; for use with sim.shrink().

        Args:
            in_place (bool): whether to shrink the intervention (else shrink a copy)
        '''
        if in_place:  # pragma: no cover
            return self
        else:
            return sc.dcp(self)

    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines at timepoints when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.

        Note 2: By default, the intervention is plotted at the timepoints stored in self.timepoints.
        However, if there is a self.plot_timepoints attribute, this will be used instead.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_timepoints'):
                timepoints = self.plot_timepoints
            else:
                timepoints = self.timepoints
            if sc.isiterable(timepoints):
                label_shown = False  # Don't show the label more than once
                for timepoint in timepoints:
                    if sc.isnumber(timepoint):
                        if self.show_label and not label_shown:  # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        date = sim.yearvec[timepoint]
                        ax.axvline(date, label=label, **line_args)
        return

    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its ``to_json`` method will need to handle those.

        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output


# %% Template classes for routine and campaign delivery
__all__ += ['RoutineDelivery']


class RoutineDelivery(Intervention):
    '''
    Base class for any intervention that uses routine delivery; handles interpolation of input years.
    '''

    def __init__(self, years=None, start_year=None, end_year=None, prob=None, annual_prob=True):
        self.years = years
        self.start_year = start_year
        self.end_year = end_year
        self.prob = sc.promotetoarray(prob)
        self.annual_prob = annual_prob  # Determines whether the probability is annual or per timestep
        return

    def initialize(self, sim):

        # Validate inputs
        if (self.years is not None) and (self.start_year is not None or self.end_year is not None):
            errormsg = 'Provide either a list of years or a start year, not both.'
            raise ValueError(errormsg)

        # If start_year and end_year are not provided, figure them out from the provided years or the sim
        if self.years is None:
            if self.start_year is None: self.start_year = sim['start']
            if self.end_year is None:   self.end_year = sim['end']
        else:
            self.start_year = self.years[0]
            self.end_year = self.years[-1]

        # More validation
        if (self.start_year not in sim.yearvec) or (self.end_year not in sim.yearvec):
            errormsg = 'Years must be within simulation start and end dates.'
            raise ValueError(errormsg)

        # Adjustment to get the right end point
        adj_factor = int(1 / sim['dt']) - 1 if sim['dt'] < 1 else 1

        # Determine the timepoints at which the intervention will be applied
        self.start_point = sc.findinds(sim.yearvec, self.start_year)[0]
        self.end_point = sc.findinds(sim.yearvec, self.end_year)[0] + adj_factor
        self.years = sc.inclusiverange(self.start_year, self.end_year)
        self.timepoints = sc.inclusiverange(self.start_point, self.end_point)
        self.yearvec = np.arange(self.start_year, self.end_year + adj_factor, sim['dt'])

        # Get the probability input into a format compatible with timepoints
        if len(self.years) != len(self.prob):
            if len(self.prob) == 1:
                self.prob = np.array([self.prob[0]] * len(self.timepoints))
            else:
                errormsg = f'Length of years incompatible with length of probabilities: {len(self.years)} vs {len(self.prob)}'
                raise ValueError(errormsg)
        else:
            self.prob = sc.smoothinterp(self.yearvec, self.years, self.prob, smoothness=0)

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.annual_prob: self.prob = 1 - (1 - self.prob) ** sim['dt']

        return


# %% Screening and triage
__all__ += ['BaseTest', 'BaseScreening', 'routine_screening']

class BaseTest(Intervention):
    '''
    Base class for screening and triage.

    Args:
         product        (str/Product)   : the diagnostic to use
         prob           (float/arr)     : annual probability of eligible people receiving the diagnostic
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of screening strategy
         kwargs         (dict)          : passed to Intervention()
    '''

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)

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
        Intervention.initialize(self)
        self.npts = sim.res_npts
        self.n_products_used = ss.Result(name=f'Products administered by {self.label}', npts=sim.res_npts, scale=True)
        self.outcomes = {k: np.array([], dtype=np.int64) for k in self.product.hierarchy}
        return

    def deliver(self, sim):
        '''
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        '''
        ti = sc.findinds(self.timepoints, sim.t)[0]
        prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
        eligible_inds = self.check_eligibility(sim)  # Check eligibility
        accept_inds = select_people(eligible_inds, prob=prob)  # Find people who accept
        if len(accept_inds):
            idx = int(sim.t / sim.resfreq)
            self.n_products_used[idx] += sim.people.scale_flows(accept_inds)
            self.outcomes = self.product.administer(sim,
                                                    accept_inds)  # Actually administer the diagnostic, filtering people into outcome categories
        return accept_inds

    def check_eligibility(self, sim):
        raise NotImplementedError


class BaseScreening(BaseTest):
    '''
    Base class for screening.

    Args:
        age_range (list/tuple/arr)  : age range for screening, e.g. [30,50]
        kwargs    (dict)            : passed to BaseTest
    '''

    def __init__(self, age_range=None, **kwargs):
        BaseTest.__init__(self, **kwargs)  # Initialize the BaseTest object
        self.age_range = age_range or [30, 50]  # This is later filtered to exclude people not yet sexually active

    def check_eligibility(self, sim):
        '''
        Return an array of indices of agents eligible for screening at time t, i.e. sexually active
        females in age range, plus any additional user-defined eligibility, which often includes
        the screening interval.
        '''
        adult_females = sim.people.is_female_adult
        in_age_range = (sim.people.age >= self.age_range[0]) * (sim.people.age <= self.age_range[1])
        conditions = (adult_females * in_age_range).astype(bool)
        if self.eligibility is not None:
            other_eligible = sc.promotetoarray(self.eligibility(sim))
            conditions = conditions * other_eligible
        return ss.true(conditions)

    def apply(self, sim):
        '''
        Perform screening by finding who's eligible, finding who accepts, and applying the product.
        '''
        self.outcomes = {k: np.array([], dtype=np.int64) for k in self.product.hierarchy}
        accept_inds = np.array([])
        if sim.ti in self.timepoints:
            accept_inds = self.deliver(sim)
            sim.people.screened[accept_inds] = True
            sim.people.screens[accept_inds] += 1
            sim.people.date_screened[accept_inds] = sim.t

            # Store results
            idx = int(sim.t / sim.resfreq)
            new_screen_inds = ss.ifalsei(sim.people.screened,
                                         accept_inds)  # Figure out people who are getting screened for the first time
            n_new_people = sim.people.scale_flows(new_screen_inds)  # Scale
            n_new_screens = sim.people.scale_flows(accept_inds)  # Scale
            sim.results['new_screened'][idx] += n_new_people
            sim.results['new_screens'][idx] += n_new_screens

        return accept_inds


class routine_screening(BaseScreening, RoutineDelivery):
    '''
    Routine screening - an instance of base screening combined with routine delivery.
    See base classes for a description of input arguments.

    **Examples**::

        screen1 = hpv.routine_screening(product='hpv', prob=0.02) # Screen 2% of the eligible population every year
        screen2 = hpv.routine_screening(product='hpv', prob=0.02, start_year=2020) # Screen 2% every year starting in 2020
        screen3 = hpv.routine_screening(product='hpv', prob=np.linspace(0.005,0.025,5), years=np.arange(2020,2025)) # Scale up screening over 5 years starting in 2020
    '''

    def __init__(self, product=None, prob=None, eligibility=None, age_range=None,
                 years=None, start_year=None, end_year=None, **kwargs):
        BaseScreening.__init__(self, product=product, age_range=age_range, eligibility=eligibility, **kwargs)
        RoutineDelivery.__init__(self, prob=prob, start_year=start_year, end_year=end_year, years=years)

    def initialize(self, sim):
        RoutineDelivery.initialize(self, sim)  # Initialize this first, as it ensures that prob is interpolated properly
        BaseScreening.initialize(self, sim)  # Initialize this next


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


# %% Treatment interventions
__all__ += ['BaseTreatment', 'treat_num', 'ART']


class BaseTreatment(Intervention):
    """
    Base treatment class.

    Args:
         product        (str/Product)   : the treatment product to use
         prob           (float/arr)     : probability of treatment aong those eligible
         eligibility    (inds/callable) : indices OR callable that returns inds
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, product=None, prob=None, eligibility=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = sc.promotetoarray(prob)
        self.eligibility = eligibility
        self._parse_product(product)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def initialize(self, sim):
        Intervention.initialize(self, sim)
        self.outcomes = {k: np.array([], dtype=int) for k in
                         ['unsuccessful', 'successful']}  # Store outcomes on each timestep
        return

    def get_accept_inds(self, sim):
        """
        Get indices of people who will acccept treatment; these people are then added to a queue or scheduled for receiving treatment
        """
        accept_uids = np.array([], dtype=int)
        eligible_uids = self.check_eligibility(sim)  # Apply eligiblity
        if len(eligible_uids):
            self.coverage_dist.set(p=self.prob[0])
            accept_uids = self.coverage_dist.filter(eligible_uids)
        return accept_uids

    def get_candidates(self, sim):
        """
        Get candidates for treatment on this timestep. Implemented by derived classes.
        """
        raise NotImplementedError

    def apply(self, sim):
        """
        Perform treatment by getting candidates, checking their eligibility, and then treating them.
        """
        # Get indices of who will get treated
        treat_candidates = self.get_candidates(sim)  # NB, this needs to be implemented by derived classes
        still_eligible = self.check_eligibility(sim)
        treat_uids = np.intersect1d(treat_candidates, still_eligible)
        if len(treat_uids):
            self.outcomes = self.product.administer(sim, treat_uids)
        return treat_uids


class treat_num(BaseTreatment):
    """
    Treat a fixed number of people each timestep.

    Args:
         max_capacity (int): maximum number who can be treated each timestep
    """

    def __init__(self, max_capacity=None, **kwargs):
        super().__init__(**kwargs)
        self.queue = []
        self.max_capacity = max_capacity
        return

    def add_to_queue(self, sim):
        """
        Add people who are willing to accept treatment to the queue
        """
        accept_inds = self.get_accept_inds(sim)
        if len(accept_inds): self.queue += accept_inds.tolist()
        return

    def get_candidates(self, sim):
        """
        Get the indices of people who are candidates for treatment
        """
        treat_candidates = np.array([], dtype=int)
        if len(self.queue):
            if self.max_capacity is None or (self.max_capacity > len(self.queue)):
                treat_candidates = self.queue[:]
            else:
                treat_candidates = self.queue[:self.max_capacity]
        return sc.promotetoarray(treat_candidates)

    def apply(self, sim):
        """
        Apply treatment. On each timestep, this method will add eligible people who are willing to accept treatment to a
        queue, and then will treat as many people in the queue as there is capacity for.
        """
        self.add_to_queue(sim)
        treat_inds = BaseTreatment.apply(self, sim)  # Apply method from BaseTreatment class
        self.queue = [e for e in self.queue if
                      e not in treat_inds]  # Recreate the queue, removing people who were treated
        return treat_inds


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
        infected_uids_onART = sim.diseases[self.disease].infected & sim.diseases[self.disease].on_art
        infected_uids_not_onART = sim.diseases[self.disease].infected & ~sim.diseases[self.disease].on_art

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


# %% Vaccination
__all__ += ['BaseVaccination', 'routine_vx', 'campaign_vx']


class BaseVaccination(Intervention):
    """
    Base vaccination class for determining who will receive a vaccine.

    Args:
         product        (str/Product)   : the vaccine to use
         prob           (float/arr)     : annual probability of eligible population getting vaccinated
         eligibility    (inds/callable) : indices OR callable that returns inds
         label          (str)           : the name of vaccination strategy
         kwargs         (dict)          : passed to Intervention()
    """

    def __init__(self, product=None, prob=None, label=None, **kwargs):
        Intervention.__init__(self, **kwargs)
        self.prob = sc.promotetoarray(prob)
        self.label = label
        self._parse_product(product)
        self.vaccinated = ss.State('vaccinated', bool, False)
        self.n_doses = ss.State('doses', int, 0)
        self.ti_vaccinated = ss.State('ti_vaccinated', int, ss.INT_NAN)
        self.coverage_dist = ss.bernoulli(p=0)  # Placeholder
        return

    def apply(self, sim):
        """
        Deliver the diagnostics by finding who's eligible, finding who accepts, and applying the product.
        """
        accept_uids = np.array([])
        if sim.ti in self.timepoints:

            ti = sc.findinds(self.timepoints, sim.ti)[0]
            prob = self.prob[ti]  # Get the proportion of people who will be tested this timestep
            is_eligible = self.check_eligibility(sim)  # Check eligibility
            self.coverage_dist.set(p=prob)
            accept_uids = self.coverage_dist.filter(ss.true(is_eligible))

            if len(accept_uids):
                self.product.administer(sim.people, accept_uids)

                # Update people's state and dates
                self.vaccinated[accept_uids] = True
                self.ti_vaccinated[accept_uids] = sim.ti
                self.n_doses[accept_uids] += 1

        return accept_uids


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
