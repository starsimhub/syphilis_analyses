"""
Define default HIV disease module and related interventions
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['HIV']


class HIV(ss.Infection):

    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        # States
        self.add_states(
            ss.State('on_art', bool, False),
            ss.State('ti_art', int, ss.INT_NAN),
            # ss.State('cd4', float, 500),  # cells/uL
            ss.State('cd4', float, 1000),  #
            ss.State('infected_cells', float, 1e-3),  # cells per uL
            ss.State('virus', float, 0),  #
            ss.State('viral_load', float, 1e7),  # RNA copies/mL
            ss.State('ti_dead', int, ss.INT_NAN),  # Time of HIV-cause death
        )

        pars = ss.omergeleft(pars,
                             cd4_min=100,
                             cd4_max=500,
                             # From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6155466/#imr12698-bib-0019
                             d_T=0.1 / 24,  # 0.1 cells/uL/day
                             lambda_T=100 / 24,  # 100 cells/uL/day
                             d_I=1 / 24,  # 1 cell/uL/day
                             k=250 / 24,  # 250 virus/cell/day
                             c=25 / 24,  # 25 virus/day
                             beta_noART=(3 * 1e-4) / 24,  # 3 * 1e-7 virus/ml/day -> 3 * 1e-4 virus/uL/day
                             init_prev=0.05,
                             eff_condoms=0.7,
                             art_efficacy=0.96,
                             death_prob=0.05,
                             )

        par_dists = ss.omergeleft(par_dists,
                                  init_prev=ss.bernoulli,
                                  death_prob=ss.bernoulli,
                                  )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)
        self.death_prob_data = sc.dcp(self.pars.death_prob)
        self.pars.death_prob = self.make_death_prob

        return

    @staticmethod
    def make_death_prob(module, sim, uids):
        p = module.pars
        out = sim.dt * module.death_prob_data / (p.cd4_min - p.cd4_max) ** 2 * (module.cd4[uids] - p.cd4_max) ** 2
        out = np.array(out)
        return out

    def update_pre(self, sim):
        """
        Update viral dynamics
        """

        infected_uids = sim.people.alive & self.infected

        if sim.ti > 0:
            current_uninfected_cells = self.cd4[infected_uids]
            current_infected_cells = self.infected_cells[infected_uids]
            current_virus = self.virus[infected_uids]
            # E.g. If time step is 1-day, this will update the viral dynamics over 24hrs.
            for hour in np.arange(0, 8760 * sim.dt, 1):
                # Update beta for agents on ART
                use_beta = [self.pars.beta_noART * (1-self.pars.art_efficacy) if self.on_art[uid] else self.pars.beta_noART
                            for uid in current_uninfected_cells.uid]

                # Calculate the CD4 and infected cell counts, and the viral load changes
                new_uninfected_cells = self.pars.lambda_T - use_beta * current_uninfected_cells * current_virus - current_uninfected_cells * self.pars.d_T
                new_infected_cells = use_beta * current_uninfected_cells * current_virus - current_infected_cells * self.pars.d_I
                new_virus = self.pars.k * current_infected_cells - current_virus * self.pars.c

                # Update free virus
                current_virus += new_virus
                # Update infected cells
                current_infected_cells += new_infected_cells
                # Update uninfected target cells
                current_uninfected_cells += new_uninfected_cells

            # Update values at the end of the time step
            self.cd4[infected_uids] = current_uninfected_cells
            self.infected_cells[infected_uids] = current_infected_cells
            self.virus[infected_uids] = current_virus

        self.rel_trans[sim.people.alive & self.infected & self.on_art] = 1 - self.pars['art_efficacy']

        can_die = ss.true(sim.people.alive & sim.people.hiv.infected)
        hiv_deaths = self.pars.death_prob.filter(can_die)

        sim.people.request_death(hiv_deaths)
        self.ti_dead[hiv_deaths] = sim.ti
        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'new_deaths', sim.npts, dtype=int)
        # Save the V, T and I level for a few example agents not on Art:
        # infected_and_on_art_uids = [uid for uid, flag in enumerate(sim.people.alive & self.infected & self.on_art) if flag]
        # infected_and_not_on_art_uids = [uid for uid, flag in enumerate(sim.people.alive & self.infected & ~self.on_art) if flag]
        for agent in [11, 36]:
            self.results += ss.Result(self.name, 'Agent' + str(agent) + '_V_level', sim.npts, dtype=int)
            self.results += ss.Result(self.name, 'Agent' + str(agent) + '_T_level', sim.npts, dtype=int)
            self.results += ss.Result(self.name, 'Agent' + str(agent) + '_I_level', sim.npts, dtype=int)
        return

    def update_results(self, sim):
        super().update_results(sim)
        self.results['new_deaths'][sim.ti] = np.count_nonzero(self.ti_dead == sim.ti)
        # Save the V, T and I level for two example agents:
        # infected_and_on_art_uids = [uid for uid, flag in enumerate(sim.people.alive & self.infected & self.on_art) if
        #                             flag]
        # infected_and_not_on_art_uids = [uid for uid, flag in enumerate(sim.people.alive & self.infected & ~self.on_art)
        #                                 if flag]
        for agent in [11, 36]:
            # Save the V, T and I level for two example agents:
            self.results['Agent' + str(agent) + '_V_level'][sim.ti] = round(self.virus[agent].astype(dtype=np.int64), 4)
            self.results['Agent' + str(agent) + '_T_level'][sim.ti] = round(self.cd4[agent].astype(dtype=np.int64), 4)
            self.results['Agent' + str(agent) + '_I_level'][sim.ti] = round(self.infected_cells[agent].astype(dtype=np.int64), 4)
        return

    def make_new_cases(self, sim):
        # eff_condoms = sim.pars[self.name]['eff_condoms'] # TODO figure out how to add this
        super().make_new_cases(sim)
        return

    def set_prognoses(self, sim, uids, source_uids=None):
        super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti
        return

    def set_congenital(self, sim, target_uids, source_uids):
        return self.set_prognoses(sim, target_uids, source_uids)
