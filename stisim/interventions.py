"""
Define interventions (and analyzers)
"""

import starsim as ss
# import sciris as sc
import numpy as np
import pandas as pd
from collections import defaultdict

# %% Custom Interventions
__all__ = ['DualTest', 'test_ART']


class DualTest(ss.Intervention):
    """ Dual test for diagnosing HIV and syphilis """
    def __init__(self, pars=None):
        return

    def initialize(self, sim):
        return

    def apply(self, sim):
        return



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
            self.results += ss.Result(self.name, 'viral_load_' + str(uid), sim.npts, dtype=float, scale=False)

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
            if sim.ti == self.infect_uids_t[index] and uid in sim.people.alive.uid:
                sim.diseases[self.disease].infected[uid] = True
                sim.diseases[self.disease].ti_infected[uid] = sim.ti
                sim.diseases[self.disease].ti_since_untreated[uid] = sim.ti
                sim.diseases[self.disease].susceptible[uid] = False
                sim.diseases[self.disease].ti_infectious[uid] = sim.ti + 14
                sim.diseases[self.disease].max_n_start_ART[uid] = sim.diseases[self.disease].pars.n_ART_start.rvs(1).astype(int)

                #if self.stop_ART:
                #    sim.diseases[self.disease].ti_stop_art[uid] = sim.ti + sim.diseases[self.disease].pars.avg_duration_stop_ART

            if uid in sim.people.alive.uid:
                # Check if it's time to restart ART treatment:
                #if sim.diseases[self.disease].on_art[uid] and self.stop_ART and self.restart_ART and sim.ti == sim.diseases[self.disease].ti_stop_art[uid]:
                #    sim.diseases[self.disease].schedule_ART_treatment(np.array([uid]), sim.ti + sim.diseases[self.disease].pars.avg_duration_restart_ART)

                if sim.diseases[self.disease].on_art[uid]:
                    ART_status = 'on_ART'
                else:
                    ART_status = 'not_on_ART'

                self.results['cd4_count_' + str(uid)][sim.ti] = sim.diseases[self.disease].cd4[uid]
                self.results['viral_load_' + str(uid)][sim.ti] = sim.diseases[self.disease].virus[uid]
                self.results['ART_status_' + str(uid)][sim.ti] = ART_status
                self.results['status_' + str(uid)][sim.ti] = 'alive'

            else:
                self.results['cd4_count_' + str(uid)][sim.ti] = np.nan
                self.results['viral_load_' + str(uid)][sim.ti] = np.nan
                self.results['status_' + str(uid)][sim.ti] = 'dead'

        return


