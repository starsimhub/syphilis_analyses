import numpy as np
import pandas as pd
import sciris as sc
import pickle as pkl
import starsim as ss
import starsim.settings as ssd
from starsim import utils as ssg
from starsim import version as ssv
from starsim.sim import Sim
from stisim import plotting as sspl
from starsim.settings import options as sso
from starsim.results import Result


class FlexPretty(sc.prettyobj):
    '''
    A class that supports multiple different display options: namely obj.brief()
    for a one-line description and obj.disp() for a full description.
    '''

    def __repr__(self):
        ''' Use brief repr by default '''
        try:
            string = self._brief()
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, something went wrong printing object:\n{str(E)}'
        return string

    def _disp(self):
        ''' Verbose output -- use Sciris' pretty repr by default '''
        return sc.prepr(self)

    def disp(self, output=False):
        ''' Print or output verbose representation of the object '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string

    def _brief(self):
        ''' Brief output -- use a one-line output, a la Python's default '''
        return sc.objectid(self)

    def brief(self, output=False):
        ''' Print or output a brief representation of the object '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string

class MultiSim(FlexPretty):
    '''
    Class for running multiple copies of a simulation. The parameter n_runs
    controls how many copies of the simulation there will be, if a list of sims
    is not provided. This is the main class that's used to run multiple versions
    of a simulation (e.g., with different random seeds).

    Args:
        sims      (Sim/list) : a single sim or a list of sims
        base_sim  (Sim)      : the sim used for shared properties; if not supplied, the first of the sims provided
        label      (str)     : the name of the multisim
        initialize (bool)    : whether or not to initialize the sims (otherwise, initialize them during run)
        kwargs    (dict)     : stored in run_args and passed to run()

    Returns:
        msim: a MultiSim object

    **Examples**::

        sim = ss.Sim() # Create the sim
        msim = ssm.MultiSim(sim, n_runs=5) # Create the multisim
        msim.run() # Run them in parallel
        msim.combine() # Combine into one sim
        msim.plot() # Plot results

        sim = ss.Sim() # Create the sim
        msim = ssm.MultiSim(sim, n_runs=11, noise=0.1, keep_people=True) # Set up a multisim with noise
        msim.run() # Run
        msim.reduce() # Compute statistics
        msim.plot() # Plot

        sims = [ss.Sim(beta=0.015*(1+0.02*i)) for i in range(5)] # Create sims
        for sim in sims: sim.run() # Run sims in serial
        msim = ssm.MultiSim(sims) # Convert to multisim
        msim.plot() # Plot as single sim
    '''

    def __init__(self, sims=None, base_sim=None, label=None, initialize=False, **kwargs):

        # Handle inputs
        if base_sim is None:
            if isinstance(sims, Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, list):
                base_sim = sims[0]
            else:
                errormsg = f'If base_sim is not supplied, sims must be either a single sim (treated as base_sim) or a list of sims, not {type(sims)}'
                raise TypeError(errormsg)

        # Set properties
        self.sims      = sims
        self.base_sim  = base_sim
        self.label     = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args  = sc.mergedicts(kwargs)
        self.results   = None
        self.which     = None # Whether the multisim is to be reduced, combined, etc.
        set_metadata(self) # Set version, date, and git info

        # Optionally initialize
        if initialize:
            self.init_sims()

        return


    def __len__(self):
        try:
            return len(self.sims)
        except:
            return 0


    def result_keys(self):
        ''' Attempt to retrieve the results keys from the sims '''
        try:
            keys = self.base_sim.results.keys()
        except Exception as E:
            errormsg = f'Could not retrieve result keys since base sim not accessible: {str(E)}'
            raise ValueError(errormsg)
        return keys


    def init_sims(self, **kwargs):
        '''
        Initialize the sims, but don't actually run them. Syntax is the same
        as MultiSim.run(). Note: in most cases you can just call run() directly,
        there is no need to call this separately.

        Args:
            kwargs  (dict): passed to multi_run()
        '''

        # Handle which sims to use
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

        # Initialize the sims but don't run them
        kwargs = sc.mergedicts(self.run_args, kwargs, {'do_run':False}) # Never run, that's the point!
        self.sims = multi_run(sims, **kwargs)

        return


    def run(self, reduce=False, combine=False, **kwargs):
        '''
        Run the actual sims

        Args:
            reduce  (bool): whether or not to reduce after running (see reduce())
            combine (bool): whether or not to combine after running (see combine(), not compatible with reduce)
            kwargs  (dict): passed to multi_run(); use run_args to pass arguments to sim.run()

        Returns:
            None (modifies MultiSim object in place)

        **Examples**::

            msim.run()
            msim.run(run_args=dict(until='2020-0601', restore_pars=False))
        '''
        # Handle which sims to use -- same as init_sims()
        if self.sims is None:
            sims = self.base_sim
        else:
            sims = self.sims

            # Handle missing labels
            for s,sim in enumerate(sims):
                if sim.label is None:
                    sim.label = f'Sim {s}'

        # Run
        kwargs = sc.mergedicts(self.run_args, kwargs)
        self.sims = multi_run(sims, **kwargs)

        # Reduce or combine
        if reduce:
            self.reduce()
        elif combine:
            self.combine()

        return self


    def _has_orig_sim(self):
        ''' Helper method for determining if an original base sim is present '''
        return hasattr(self, 'orig_base_sim')


    def _rm_orig_sim(self, reset=False):
        ''' Helper method for removing the original base sim, if present '''
        if self._has_orig_sim():
            if reset:
                self.base_sim = self.orig_base_sim
            delattr(self, 'orig_base_sim')
        return


    def shrink(self, **kwargs):
        '''
        Not to be confused with reduce(), this shrinks each sim in the msim;
        see sim.shrink() for more information.

        Args:
            kwargs (dict): passed to sim.shrink() for each sim
        '''
        self.base_sim.shrink(**kwargs)
        self._rm_orig_sim()
        for sim in self.sims:
            sim.shrink(**kwargs)
        return


    def reset(self):
        ''' Undo a combine() or reduce() by resetting the base sim, which, and results '''
        self._rm_orig_sim(reset=True)
        self.which = None
        self.results = None
        return


    def reduce(self, quantiles=None, use_mean=False, bounds=None, output=False):
        '''
        Combine multiple sims into a single sim statistically: by default, use
        the median value and the 10th and 90th percentiles for the lower and upper
        bounds. If use_mean=True, then use the mean and ±2 standard deviations
        for lower and upper bounds.

        Args:
            quantiles (dict): the quantiles to use, e.g. [0.1, 0.9] or {'low : '0.1, 'high' : 0.9}
            use_mean (bool): whether to use the mean instead of the median
            bounds (float): if use_mean=True, the multiplier on the standard deviation for upper and lower bounds (default 2)
            output (bool): whether to return the "reduced" sim (in any case, modify the multisim in-place)

        **Example**::

            msim = cv.MultiSim(cv.Sim())
            msim.run()
            msim.reduce()
            msim.summarize()
        '''

        if use_mean:
            if bounds is None:
                bounds = 2
        else:
            if quantiles is None:
                quantiles = make_metapars()['quantiles']
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low':float(quantiles[0]), 'high':float(quantiles[1])}
                except Exception as E:
                    errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                    raise ValueError(errormsg)

        # Store information on the sims
        n_runs = len(self)
        reduced_sim = sc.dcp(self.sims[0])
        reduced_sim.metadata = dict(parallelized=True, combined=False, n_runs=n_runs, quantiles=quantiles, use_mean=use_mean, bounds=bounds) # Store how this was parallelized

        # Perform the statistics
        raw = {}
        mainkeys = reduced_sim.results.keys()
        for reskey in mainkeys:
            if isinstance(self.base_sim.results[reskey], MultiSimResult):
                raw[reskey] = np.zeros((reduced_sim.npts, len(self.sims)))
                for s,sim in enumerate(self.sims):
                    vals = sim.results[reskey].values
                    raw[reskey][:, s] = vals
            elif isinstance(self.base_sim.results[reskey], ss.ndict):
                for nreskey in self.base_sim.results[reskey].keys():
                    raw[nreskey] = np.zeros((reduced_sim.npts, len(self.sims)))
                    for s, sim in enumerate(self.sims):
                        vals = sim.results[reskey][nreskey].values
                        raw[nreskey][:, s] = vals

        for reskey in mainkeys:
            if isinstance(self.sims[0].results[reskey], MultiSimResult):
                axis = 1
                results = reduced_sim.results
                if use_mean:
                    r_mean = np.mean(raw[reskey].values, axis=axis)
                    r_std = np.std(raw[reskey].values, axis=axis)
                    results[reskey].values[:] = r_mean
                    results[reskey].low = r_mean - bounds * r_std
                    results[reskey].high = r_mean + bounds * r_std
                else:
                    results[reskey].values[:] = np.quantile(raw[reskey], q=0.5, axis=axis)
                    results[reskey].low = np.quantile(raw[reskey], q=quantiles['low'], axis=axis)
                    results[reskey].high = np.quantile(raw[reskey], q=quantiles['high'], axis=axis)
            elif isinstance(self.base_sim.results[reskey], ss.ndict):
                for nreskey in self.base_sim.results[reskey].keys():
                    axis = 1
                    results = reduced_sim.results
                    if use_mean:
                        r_mean = np.mean(raw[nreskey].values, axis=axis)
                        r_std = np.std(raw[nreskey].values, axis=axis)
                        results[reskey][nreskey].values[:] = r_mean
                        results[reskey][nreskey].low = r_mean - bounds * r_std
                        results[reskey][nreskey].high = r_mean + bounds * r_std
                    else:
                        results[reskey][nreskey].values[:] = np.quantile(raw[nreskey], q=0.5, axis=axis)
                        results[reskey][nreskey].low = np.quantile(raw[nreskey], q=quantiles['low'], axis=axis)
                        results[reskey][nreskey].high = np.quantile(raw[nreskey], q=quantiles['high'], axis=axis)

        # Compute and store final results
        summary = compute_summary(reduced_sim, output=True)
        self.orig_base_sim = self.base_sim
        self.base_sim = reduced_sim
        self.results = reduced_sim.results
        self.summary = summary
        self.which = 'reduced'

        if output:
            return self.base_sim
        else:
            return


    def mean(self, bounds=None, **kwargs):
        '''
        Alias for reduce(use_mean=True). See reduce() for full description.

        Args:
            bounds (float): multiplier on the standard deviation for the upper and lower bounds (default, 2)
            kwargs (dict): passed to reduce()
        '''
        return self.reduce(use_mean=True, bounds=bounds, **kwargs)


    def median(self, quantiles=None, **kwargs):
        '''
        Alias for reduce(use_mean=False). See reduce() for full description.

        Args:
            quantiles (list or dict): upper and lower quantiles (default, 0.1 and 0.9)
            kwargs (dict): passed to reduce()
        '''
        return self.reduce(use_mean=False, quantiles=quantiles, **kwargs)


    def combine(self, output=False):
        '''
        Combine multiple sims into a single sim with scaled results.

        **Example**::

            msim = cv.MultiSim(cv.Sim())
            msim.run()
            msim.combine()
            msim.summarize()
        '''

        n_runs = len(self)
        combined_sim = sc.dcp(self.sims[0])
        combined_sim.parallelized = dict(parallelized=True, combined=True, n_runs=n_runs)  # Store how this was parallelized

        for s,sim in enumerate(self.sims[1:]): # Skip the first one
            if combined_sim.people: # If the people are there, add them and increment the population size accordingly
                combined_sim.people += sim.people
                combined_sim['pop_size'] = combined_sim.people.pars['pop_size']
            else: # If not, manually update population size
                combined_sim['pop_size'] += sim['pop_size']  # Record the number of people
            for key in sim.result_keys():
                vals = sim.results[key].values
                if len(vals) != combined_sim.npts:
                    errormsg = f'Cannot combine sims with inconsistent numbers of days: {combined_sim.npts} vs. {len(vals)}'
                    raise ValueError(errormsg)
                combined_sim.results[key].values += vals

        # For non-count results (scale=False), rescale them
        for key in combined_sim.result_keys():
            if not combined_sim.results[key].scale:
                combined_sim.results[key].values /= n_runs

        # Compute and store final results
        combined_sim.compute_summary()
        self.orig_base_sim = self.base_sim
        self.base_sim = combined_sim
        self.results = combined_sim.results
        self.summary = combined_sim.summary

        self.which = 'combined'

        if output:
            return self.base_sim
        else:
            return


    def compare(self, t=None, sim_inds=None, output=False, do_plot=False, **kwargs):
        '''
        Create a dataframe compare sims at a single point in time.

        Args:
            t        (int/str) : the day (or date) to do the comparison; default, the end
            sim_inds (list)    : list of integers of which sims to include (default: all)
            output   (bool)    : whether or not to return the comparison as a dataframe
            do_plot  (bool)    : whether or not to plot the comparison (see also plot_compare())
            kwargs   (dict)    : passed to plot_compare()

        Returns:
            df (dataframe): a dataframe comparison
        '''

        # Handle time
        if t is None:
            t = -1
            daystr = 'the last day'
        else:
            daystr = f'day {t}'

        # Handle the indices
        if sim_inds is None:
            sim_inds = list(range(len(self.sims)))

        # Move the results to a dictionary
        resdict = sc.ddict(dict)
        for i,s in enumerate(sim_inds):
            sim = self.sims[s]
            day = sim.day(t) # Unlikely, but different sims might have different start days
            label = sim.label
            if not label: # Give it a label if it doesn't have one
                label = f'Sim {i}'
            if label in resdict: # Avoid duplicates
                label += f' ({i})'
            for reskey in sim.result_keys():
                res = sim.results[reskey]
                val = res.values[day]
                if res.scale: # Results that are scaled by population are ints
                    val = int(val)
                resdict[label][reskey] = val

        if do_plot:
            self.plot_compare(**kwargs)

        df = pd.DataFrame.from_dict(resdict).astype(object) # astype is necessary to prevent type coercion
        if not output:
            print(f'Results for {daystr} in each sim:')
            print(df)
        else:
            return df


    def plot(self, to_plot=None, inds=None, plot_sims=False, color_by_sim=None, max_sims=5, colors=None, labels=None, alpha_range=None, plot_args=None, show_args=None, **kwargs):
        '''
        Plot all the sims  -- arguments passed to Sim.plot(). The
        behavior depends on whether or not combine() or reduce() has been called.
        If so, this function by default plots only the combined/reduced sim (which
        you can override with plot_sims=True). Otherwise, it plots a separate line
        for each sim.

        Note that this function is complex because it aims to capture the flexibility
        of both sim.plot() and scens.plot(). By default, if combine() or reduce()
        has been used, it will resemble sim.plot(); otherwise, it will resemble
        scens.plot(). This can be changed via color_by_sim, together with the
        other options.

        Args:
            to_plot      (list) : list or dict of which results to plot; see cv.get_default_plots() for structure
            inds         (list) : if not combined or reduced, the indices of the simulations to plot (if None, plot all)
            plot_sims    (bool) : whether to plot individual sims, even if combine() or reduce() has been used
            color_by_sim (bool) : if True, set colors based on the simulation type; otherwise, color by result type; True implies a scenario-style plotting, False implies sim-style plotting
            max_sims     (int)  : maximum number of sims to use with color-by-sim; can be overridden by other options
            colors       (list) : if supplied, override default colors for color_by_sim
            labels       (list) : if supplied, override default labels for color_by_sim
            alpha_range  (list) : a 2-element list/tuple/array providing the range of alpha values to use to distinguish the lines
            plot_args    (dict) : passed to sim.plot()
            show_args    (dict) : passed to sim.plot()
            kwargs       (dict) : passed to sim.plot()

        Returns:
            fig: Figure handle

        **Examples**::

            sim = cv.Sim()
            msim = cv.MultiSim(sim)
            msim.run()
            msim.plot() # Plots individual sims
            msim.reduce()
            msim.plot() # Plots the combined sim
        '''

        # Plot a single curve, possibly with a range
        if not plot_sims and self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot(to_plot=to_plot, colors=colors, **kwargs)

        # PLot individual sims on top of each other
        else:

            # Initialize
            fig          = kwargs.pop('fig', None)
            orig_show    = kwargs.get('do_show', None)
            orig_close   = sso.close
            orig_setylim = kwargs.get('setylim', True)
            kwargs['legend_args'] = sc.mergedicts({'show_legend':True}, kwargs.get('legend_args')) # Only plot the legend the first time

            # Handle indices
            if inds is None:
                inds = np.arange(len(self.sims))
            n_sims = len(inds)

            # Handle what style of plotting to use:
            if color_by_sim is None:
                if n_sims <= max_sims:
                    color_by_sim = True
                else:
                    color_by_sim = False

            # Handle what to plot
            if to_plot is None:
                kind = 'scens' if color_by_sim else 'sim'
                to_plot = ssd.get_default_plots(kind=kind)

            # Handle colors
            if colors is None:
                if color_by_sim:
                    colors = sc.gridcolors(ncolors=n_sims)
                else:
                    colors = [None]*n_sims # So we can iterate over it
            else:
                colors = [colors]*n_sims # Again, for iteration

            # Handle alpha if not using colors
            if alpha_range is None:
                if color_by_sim:
                    alpha_range = [0.8, 0.8] # We're using color to distinguish sims, so don't need alpha
                else:
                    alpha_range = [0.8, 0.3] # We're using alpha to distinguish sims
            alphas = np.linspace(alpha_range[0], alpha_range[1], n_sims)

            # Plot
            for s,ind in enumerate(inds):

                sim = self.sims[ind]
                final_plot = (s == n_sims-1) # Check if this is the final plot

                # Handle the legend and labels
                if final_plot:
                    merged_show_args  = sc.mergedicts(dict(returnfig=sso.returnfig), show_args)
                    kwargs['do_show'] = orig_show
                    kwargs['setylim'] = orig_setylim
                    sso.set(close=orig_close) # Reset original closing settings
                else:
                    merged_show_args  = dict(annotations=False, returnfig=True) # Only show things like data the last time it's plotting
                    kwargs['do_show'] = False # On top of that, don't show the plot at all unless it's the last time
                    kwargs['setylim'] = False # Don't set the y limits until we have all the data
                    sso.set(close=False) # Do not close figures if we're in the middle of plotting

                # Optionally set the label for the first max_sims sims
                if color_by_sim is True and s<max_sims:
                    if labels is None:
                        merged_labels = sim.label
                    else:
                        merged_labels = labels[s]
                elif final_plot and not color_by_sim:
                    merged_labels = labels
                else:
                    merged_labels = ''

                # Actually plot
                merged_plot_args = sc.mergedicts({'alpha':alphas[s]}, plot_args) # Need a new variable to avoid overwriting
                fig = sim.plot(fig=fig, to_plot=('scens', to_plot), colors=colors[s], labels=merged_labels, plot_args=merged_plot_args, show_args=merged_show_args, **kwargs)

        return sspl.handle_show_return(fig=fig)


    def plot_result(self, key, colors=None, labels=None, *args, **kwargs):
        ''' Convenience method for plotting -- arguments passed to sim.plot_result() '''
        if self.which in ['combined', 'reduced']:
            fig = self.base_sim.plot_result(key, *args, **kwargs)
        else:
            fig = None
            if colors is None:
                colors = sc.gridcolors(len(self))
            if labels is None:
                labels = [sim.label for sim in self.sims]
            orig_setylim = kwargs.get('setylim', True)
            for s,sim in enumerate(self.sims):
                if s == len(self.sims)-1:
                    kwargs['setylim'] = orig_setylim
                else:
                    kwargs['setylim'] = False
                fig = sim.plot_result(key=key, fig=fig, color=colors[s], label=labels[s], *args, **kwargs)
        return sspl.handle_show_return(fig=fig)


    def plot_compare(self, t=-1, sim_inds=None, log_scale=True, **kwargs):
        '''
        Plot a comparison between sims, using bars to show different values for
        each result. For an explanation of other available arguments, see Sim.plot().

        Args:
            t         (int)  : index of results, passed to compare()
            sim_inds  (list) : which sims to include, passed to compare()
            log_scale (bool) : whether to plot with a logarithmic x-axis
            kwargs    (dict) : standard plotting arguments, see Sim.plot() for explanation

        Returns:
            fig: Figure handle
        '''
        df = self.compare(t=t, sim_inds=sim_inds, output=True)
        return sspl.plot_compare(df, log_scale=log_scale, **kwargs)


    def save(self, filename=None, keep_people=False, **kwargs):
        '''
        Save to disk as a gzipped pickle. Load with cv.load(filename) or
        cv.MultiSim.load(filename).

        Args:
            filename    (str)  : the name or path of the file to save to; if None, uses default
            keep_people (bool) : whether or not to store the population in the Sim objects (NB, very large)
            kwargs      (dict) : passed to ``sc.makefilepath()``

        Returns:
            scenfile (str): the validated absolute path to the saved file

        **Example**::

            msim.save() # Saves to an .msim file
        '''
        if filename is None:
            filename = 'covasim.msim'
        msimfile = sc.makefilepath(filename=filename, **kwargs)
        self.filename = filename # Store the actual saved filename

        # Store sims separately
        sims = self.sims
        self.sims = None # Remove for now

        obj = sc.dcp(self) # This should be quick once we've removed the sims
        if keep_people:
            obj.sims = sims # Just restore the object in full
            print('Note: saving people, which may produce a large file!')
        else:
            obj.base_sim.shrink(in_place=True)
            obj.sims = []
            for sim in sims:
                obj.sims.append(sim.shrink(in_place=False))

        ssg.save(filename=msimfile, obj=obj) # Actually save

        self.sims = sims # Restore
        return msimfile


    @staticmethod
    def load(msimfile, *args, **kwargs):
        '''
        Load from disk from a gzipped pickle.

        Args:
            msimfile (str): the name or path of the file to load from
            kwargs: passed to cv.load()

        Returns:
            msim (MultiSim): the loaded MultiSim object

        **Example**::

            msim = cv.MultiSim.load('my-multisim.msim')
        '''
        msim = ssg.load(msimfile, *args, **kwargs)
        if not isinstance(msim, MultiSim):
            errormsg = f'Cannot load object of {type(msim)} as a MultiSim object'
            raise TypeError(errormsg)
        return msim


    @staticmethod
    def merge(*args, base=False):
        '''
        Convenience method for merging two MultiSim objects.

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate)
            base (bool): if True, make a new list of sims from the multisim's two base sims; otherwise, merge the multisim's lists of sims

        Returns:
            msim (MultiSim): a new MultiSim object

        **Examples**:

            mm1 = cv.MultiSim.merge(msim1, msim2, base=True)
            mm2 = cv.MultiSim.merge([m1, m2, m3, m4], base=False)
        '''

        # Handle arguments
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0] # A single list of MultiSims has been provided

        # Create the multisim from the base sim of the first argument
        msim = MultiSim(base_sim=sc.dcp(args[0].base_sim), label=args[0].label)
        msim.sims = []
        msim.chunks = [] # This is used to enable automatic splitting later

        # Handle different options for combining
        if base: # Only keep the base sims
            for i,ms in enumerate(args):
                sim = sc.dcp(ms.base_sim)
                sim.label = ms.label
                msim.sims.append(sim)
                msim.chunks.append([[i]])
        else: # Keep all the sims
            for ms in args:
                len_before = len(msim.sims)
                msim.sims += sc.dcp(ms.sims)
                len_after= len(msim.sims)
                msim.chunks.append(list(range(len_before, len_after)))

        return msim


    def split(self, inds=None, chunks=None):
        '''
        Convenience method for splitting one MultiSim into several. You can specify
        either individual indices of simulations to extract, via inds, or consecutive
        chunks of indices, via chunks. If this function is called on a merged MultiSim,
        the chunks can be retrieved automatically and no arguments are necessary.

        Args:
            inds (list): a list of lists of indices, with each list turned into a MultiSim
            chunks (int or list): if an int, split the MultiSim into that many chunks; if a list return chunks of that many sims

        Returns:
            A list of MultiSim objects

        **Examples**::

            m1 = cv.MultiSim(cv.Sim(label='sim1'), initialize=True)
            m2 = cv.MultiSim(cv.Sim(label='sim2'), initialize=True)
            m3 = cv.MultiSim.merge(m1, m2)
            m3.run()
            m1b, m2b = m3.split()

            msim = cv.MultiSim(cv.Sim(), n_runs=6)
            msim.run()
            m1, m2 = msim.split(inds=[[0,2,4], [1,3,5]])
            mlist1 = msim.split(chunks=[2,4]) # Equivalent to inds=[[0,1], [2,3,4,5]]
            mlist2 = msim.split(chunks=2) # Equivalent to inds=[[0,1,2], [3,4,5]]
        '''

        # Process indices and chunks
        if inds is None: # Indices not supplied
            if chunks is None: # Chunks not supplied
                if hasattr(self, 'chunks'): # Created from a merged MultiSim
                    inds = self.chunks
                else: # No indices or chunks and not created from a merge
                    errormsg = 'If a MultiSim has not been created via merge(), you must supply either inds or chunks to split it'
                    raise ValueError(errormsg)
            else: # Chunks supplied, but not inds
                inds = [] # Initialize
                sim_inds = np.arange(len(self)) # Indices for the simulations
                if sc.isiterable(chunks): # e.g. chunks = [2,4]
                    chunk_inds = np.cumsum(chunks)[:-1]
                    inds = np.split(sim_inds, chunk_inds)
                else: # e.g. chunks = 3
                    inds = np.split(sim_inds, chunks) # This will fail if the length is wrong

        # Do the conversion
        mlist = []
        for indlist in inds:
            sims = sc.dcp([self.sims[i] for i in indlist])
            msim = MultiSim(sims=sims)
            mlist.append(msim)

        return mlist


    def disp(self, output=False):
        '''
        Display a verbose description of a multisim. See also multisim.summarize()
        (medium length output) and multisim.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = cv.MultiSim(cv.Sim(verbose=0), label='Example multisim')
            msim.run()
            msim.disp() # Displays detailed output
        '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string


    def summarize(self, output=False):
        '''
        Print a moderate length summary of the MultiSim. See also multisim.disp()
        (detailed output) and multisim.brief() (short output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = cv.MultiSim(cv.Sim(verbose=0), label='Example multisim')
            msim.run()
            msim.summarize() # Prints moderate length output
        '''
        labelstr = f' "{self.label}"' if self.label else ''
        simlenstr = f'{len(self.sims)}' if self.sims else '0'
        string  = f'MultiSim{labelstr} summary:\n'
        string += f'  Number of sims: {simlenstr}\n'
        string += f'  Reduced/combined: {self.which}\n'
        string += f'  Base: {self.base_sim.brief(output=True)}\n'
        if self.sims:
            string += '  Sims:\n'
            for s,sim in enumerate(self.sims):
                string += f'    {s}: {sim.brief(output=True)}\n'
        if not output:
            print(string)
        else:
            return string


    def _brief(self):
        '''
        Return a brief description of a multisim -- used internally and by repr();
        see multisim.brief() for the user version.
        '''
        try:
            labelstr = f'"{self.label}"; ' if self.label else ''
            n_sims = 0 if not self.sims else len(self.sims)
            string   = f'MultiSim({labelstr}n_sims: {n_sims}; base: {self.base_sim.brief(output=True)})'
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, multisim appears to be malformed:\n{str(E)}'
        return string


    def brief(self, output=False):
        '''
        Print a compact representation of the multisim. See also multisim.disp()
        (detailed output) and multisim.summarize() (medium length output).

        Args:
            output (bool): if true, return a string instead of printing output

        **Example**::

            msim = cv.MultiSim(cv.Sim(verbose=0), label='Example multisim')
            msim.run()
            msim.brief() # Prints one-line output
         '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string


    def to_json(self, *args, **kwargs):
        ''' Shortcut for base_sim.to_json() '''
        if not self.base_sim.results_ready: # pragma: no cover
            errormsg = 'JSON export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_json(*args, **kwargs)


    def to_excel(self, *args, **kwargs):
        ''' Shortcut for base_sim.to_excel() '''
        if not self.base_sim.results_ready: # pragma: no cover
            errormsg = 'Excel export only available for reduced sim; please run msim.mean() or msim.median() first'
            raise RuntimeError(errormsg)
        return self.base_sim.to_excel(*args, **kwargs)

def compute_summary(sim, t=None, update=True, output=False, require_run=False):
    '''
    Compute the summary dict and string for the sim. Used internally; see
    sim.summarize() for the user version.

    Args:
        full (bool): whether or not to print all results (by default, only cumulative)
        t (int/str): day or date to compute summary for (by default, the last point)
        update (bool): whether to update the stored sim.summary
        output (bool): whether to return the summary
        require_run (bool): whether to raise an exception if simulations have not been run yet
    '''
    if t is None:
        t = sim.ti

    # Compute the summary
    if require_run and not sim.results_ready:
        errormsg = 'Simulation not yet run'
        raise RuntimeError(errormsg)

    summary = sc.objdict()
    for key in sim.results.keys():
        if isinstance(sim.results[key], MultiSimResult):
            summary[key] = sim.results[key].values[t]
        elif isinstance(sim.results[key], Result):
            summary[key] = sim.results[key][t]
        elif isinstance(sim.results[key], ss.ndict):
            for nkey in sim.results[key].keys():
                summary[key+'-'+nkey] = sim.results[key][nkey].values[t]
        else:
            summary[key] = sim.results[key][-1]

    for interv in sim.interventions.keys():
        if len(sim.interventions[interv].results) > 0:
            for key in sim.interventions[interv].results.keys():
                summary[key] = sim.interventions[interv].results[key].values[t]


    # Update the stored state
    if update:
        sim.summary = summary

    # Optionally return
    if output:
        return summary
    else:
        return

def export_results(sim):
    '''
    Convert results to dict.

    The results written to Excel must have a regular table shape.

    Args:

    Returns:
        resdict (dict): dictionary representation of the results

    '''

    if not sim.results_ready: # pragma: no cover
        errormsg = 'Please run the sim before exporting the results'
        raise RuntimeError(errormsg)

    resdict = {}
    resdict['t'] = sim.tivec

    for key,res in sim.results.items():
        if isinstance(res, MultiSimResult):
            resdict[key] = res.values
        elif isinstance(sim.results[key], ss.ndict):
            for nkey in sim.results[key].keys():
                resdict[key+'-'+nkey] = sim.results[key][nkey].values
    for interv in sim.interventions.keys():
        if len(sim.interventions[interv].results) > 0:
            for key in sim.interventions[interv].results.keys():
                resdict[key] = sim.interventions[interv].results[key].values
    return resdict

def single_run(sim, ind=0, reseed=True, noise=0.0, noisepar=None, keep_people=False, run_args=None, sim_args=None, verbose=None, do_run=True, **kwargs):
    '''
    Convenience function to perform a single simulation run. Mostly used for
    parallelization, but can also be used directly.

    Args:
        sim         (Sim)   : the sim instance to be run
        ind         (int)   : the index of this sim
        reseed      (bool)  : whether or not to generate a fresh seed for each run
        noise       (float) : the amount of noise to add to each run
        noisepar    (str)   : the name of the parameter to add noise to
        keep_people (bool)  : whether to keep the people after the sim run
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim, e.g. 'n_infected'
        verbose     (int)   : detail to print
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        kwargs      (dict)  : also passed to the sim

    Returns:
        sim (Sim): a single sim object with results

    **Example**::

        import covasim as cv
        sim = ss.Sim() # Create a default simulation
        sim = ssm.single_run(sim) # Run it, equivalent(ish) to sim.run()
    '''

    # Set sim and run arguments
    sim_args = sc.mergedicts(sim_args, kwargs)
    run_args = sc.mergedicts({'verbose':verbose}, run_args)
    if verbose is None:
        verbose = sim.pars['verbose']

    if not sim.label:
        sim.label = f'Sim {ind}'

    if reseed:
        sim.pars['rand_seed'] += ind # Reset the seed, otherwise no point of parallel runs
        ssg.set_seed()

    # If the noise parameter is not found, guess what it should be
    if noisepar is None:
        noisepar = 'beta'
        if noisepar not in sim.pars.keys():
            raise sc.KeyNotFoundError(f'Noise parameter {noisepar} was not found in sim parameters')

    # Handle noise -- normally distributed fractional error
    noiseval = noise*np.random.normal()
    if noiseval > 0:
        noisefactor = 1 + noiseval
    else:
        noisefactor = 1/(1-noiseval)
    sim.pars[noisepar] *= noisefactor

    if verbose>=1:
        verb = 'Running' if do_run else 'Creating'
        print(f'{verb} a simulation using seed={sim.pars["rand_seed"]} and noise={noiseval}')

    # Handle additional arguments
    for key,val in sim_args.items():
        print(f'Processing {key}:{val}')
        if key in sim.pars.keys():
            if verbose>=1:
                print(f'Setting key {key} from {sim.pars[key]} to {val}')
                sim.pars[key] = val
        else:
            raise sc.KeyNotFoundError(f'Could not set key {key}: not a valid parameter name')

    # Run
    if do_run:
        sim.run(**run_args)

    # Shrink the sim to save memory
    if not keep_people:
        sim.shrink()

    return sim

class MultiSimResult(object):
    def __init__(self, module=None, name=None, npts=None, dtype=None, color=None):
        self.module = module
        self.name = name  # Name of this result
        if color is None:
            color = '#b62413'
        self.color = color  # Default color
        if npts is None:
            npts = 0
        npts = int(npts)
        if dtype is None:
            dtype = ss.float_

        self.values = np.zeros(npts, dtype=dtype)
        self.low = None
        self.high = None
        return

    def __repr__(self):
        ''' Use pretty repr, like sc.prettyobj, but displaying full values '''
        output = sc.prepr(self, skip=['values', 'low', 'high'], use_repr=False)
        output += 'values:\n' + repr(self.values)
        if self.low is not None:
            output += '\nlow:\n' + repr(self.low)
        if self.high is not None:
            output += '\nhigh:\n' + repr(self.high)
        return output

    def __getitem__(self, key):
        ''' To allow e.g. result['high'] instead of result.high, and result[5] instead of result.values[5] '''
        if isinstance(key, str):
            output = getattr(self, key)
        else:
            output = self.values.__getitem__(key)
        return output

    def __setitem__(self, key, value):
        ''' To allow e.g. result[:] = 1 instead of result.values[:] = 1 '''
        if isinstance(key, str):
            setattr(self, key, value)
        else:
            self.values.__setitem__(key, value)
        return

    def __len__(self):
        ''' To allow len(result) instead of len(result.values) '''
        return len(self.values)

    @property
    def npts(self):
        return len(self.values)


def multi_run(sim, n_runs=4, reseed=None, noise=0.0, noisepar=None, iterpars=None, keep_people=None, run_args=None,
              sim_args=None, par_args=None, do_run=True, parallel=True, n_cpus=None, verbose=None, retry='warn', **kwargs):
    '''
    For running multiple runs in parallel. If the first argument is a list of sims,
    exactly these will be run and most other arguments will be ignored.

    Args:
        sim         (Sim)   : the sim instance to be run, or a list of sims.
        n_runs      (int)   : the number of parallel runs
        reseed      (bool)  : whether or not to generate a fresh seed for each run (default: true for single, false for list of sims)
        noise       (float) : the amount of noise to add to each run
        noisepar    (str)   : the name of the parameter to add noise to
        iterpars    (dict)  : any other parameters to iterate over the runs; see sc.parallelize() for syntax
        combine     (bool)  : whether or not to combine all results into one sim, rather than return multiple sim objects
        keep_people (bool)  : whether to keep the people after the sim run (default false)
        run_args    (dict)  : arguments passed to sim.run()
        sim_args    (dict)  : extra parameters to pass to the sim
        par_args    (dict)  : arguments passed to sc.parallelize()
        do_run      (bool)  : whether to actually run the sim (if not, just initialize it)
        parallel    (bool)  : whether to run in parallel using multiprocessing (else, just run in a loop)
        n_cpus      (int)   : the number of CPUs to run on (if blank, set automatically; otherwise, passed to par_args)
        verbose     (int)   : detail to print
        retry       (str)   : what to do if default parallelizer fails: choices are 'warn' (default), 'die' (raise exception), or 'silent' (keep going)
        kwargs      (dict)  : also passed to the sim

    Returns:
        If combine is True, a single sim object with the combined results from each sim.
        Otherwise, a list of sim objects (default).

    **Example**::

        import covasim as cv
        sim = ss.Sim()
        sims = ssm.multi_run(sim, n_runs=6, noise=0.2)
    '''

    # Handle inputs
    sim_args = sc.mergedicts(sim_args, kwargs)  # Handle blank
    par_args = sc.mergedicts({'ncpus': n_cpus, 'parallelizer': 'concurrent.futures'}, par_args)  # Handle blank

    # Handle iterpars
    if iterpars is None:
        iterpars = {}
    else:
        n_runs = None  # Reset and get from length of dict instead
        for key, val in iterpars.items():
            new_n = len(val)
            if n_runs is not None and new_n != n_runs:
                raise ValueError(f'Each entry in iterpars must have the same length, not {n_runs} and {len(val)}')
            else:
                n_runs = new_n

    # Run the sims
    if isinstance(sim, Sim):  # One sim
        if reseed is None: reseed = True
        iterkwargs = dict(ind=np.arange(n_runs))
        iterkwargs.update(iterpars)
        kwargs = dict(sim=sim, reseed=reseed, noise=noise, noisepar=noisepar, verbose=verbose, keep_people=keep_people,
                      sim_args=sim_args, run_args=run_args, do_run=do_run)
    elif isinstance(sim, list):  # List of sims
        if reseed is None: reseed = False
        iterkwargs = dict(sim=sim, ind=np.arange(len(sim)))
        kwargs = dict(reseed=reseed, verbose=verbose, keep_people=keep_people, sim_args=sim_args, run_args=run_args,
                      do_run=do_run)
    else:
        errormsg = f'Must be Sim object or list, not {type(sim)}'
        raise TypeError(errormsg)

    # Actually run!
    if parallel:
        kw = dict(iterkwargs=iterkwargs, kwargs=kwargs, **par_args)
        try:
            sims = sc.parallelize(single_run, **kw)  # Run in parallel
        except RuntimeError as E:  # Handle if run outside of __main__ on Windows
            if 'freeze_support' in E.args[0]:  # For this error, add additional information
                errormsg = '''
 Uh oh! It appears you are trying to run with multiprocessing on Windows outside
 of the __main__ block; please see https://docs.python.org/3/library/multiprocessing.html
 for more information. The correct syntax to use is e.g.

     import covasim as cv
     sim = ss.Sim()
     msim = ssm.MultiSim(sim)

     if __name__ == '__main__':
         msim.run()

Alternatively, to run without multiprocessing, set parallel=False.
 '''
                raise RuntimeError(errormsg) from E
            else:  # For all other runtime errors, raise the original exception
                raise E
        except pkl.PicklingError as E:
            parallelizer = par_args.get('parallelizer')
            if retry in ['warn', 'silent'] and parallelizer != 'multiprocess':
                if retry == 'warn':
                    warnmsg = f'multi_run() failed with parallelizer={parallelizer}, trying more robust "multiprocess"...'
                    ssg.warn(warnmsg)
                kw['parallelizer'] = 'multiprocess'
                sims = sc.parallelize(single_run, **kw)  # Try again to run in parallel
            else:
                errormsg = 'Parallel run failed due to a pickling error; this is usually due to including a lambda function or other complex object'
                raise pkl.PicklingError(errormsg) from E

    else:  # Run in serial, not in parallel
        sims = []
        n_sims = len(list(iterkwargs.values())[0])  # Must have length >=1 and all entries must be the same length
        for s in range(n_sims):
            this_iter = {k: v[s] for k, v in iterkwargs.items()}  # Pull out items specific to this iteration
            this_iter.update(kwargs)  # Merge with the kwargs
            this_iter['sim'] = this_iter['sim'].copy()  # Ensure we have a fresh sim; this happens implicitly on pickling with multiprocessing
            sim = single_run(**this_iter)  # Run in series
            sims.append(sim)

    return sims

def set_metadata(obj, **kwargs):
    ''' Set standard metadata for an object '''
    obj.created = kwargs.get('created', sc.now())
    obj.version = kwargs.get('version', ssv.__version__)
    obj.git_info = kwargs.get('git_info', ssg.git_info()) # 4 = 2 (default) + base + caller
    return

def make_metapars():
    ''' Create default metaparameters for a Scenarios run '''
    metapars = sc.objdict(
        n_runs    = 3, # Number of parallel runs; change to 3 for quick, 11 for real
        noise     = 0.0, # Use noise, optionally
        noisepar  = 'beta',
        rand_seed = 1,
        quantiles = {'low':0.1, 'high':0.9},
        verbose   = sso.verbose,
    )
    return metapars