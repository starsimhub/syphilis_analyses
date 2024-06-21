"""
Test that the population size doesn't depend on the number of agents in the simulation
"""

# Imports
import sciris as sc 
import starsim as ss
import stisim as sti
import pandas as pd
import numpy as np
import pylab as pl


class network_stats(ss.Analyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'network_stats'
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.init_results()
        return

    def init_post(self):
        super().init_post()
        return

    def init_results(self):
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'share_active', npts, dtype=float, scale=False),
            ss.Result(self.name, 'partners_f_mean', npts, dtype=float, scale=False),
            ss.Result(self.name, 'partners_m_mean', npts, dtype=float, scale=False),
        ]
        return

    def apply(self, sim):
        ti = sim.ti
        nw = sim.networks.structuredsexual

        partners_active_m = nw.partners[(self.sim.people.male & nw.active(sim.people))]
        partners_active_f = nw.partners[(self.sim.people.female & nw.active(sim.people))]

        self.results.share_active[ti] = len(nw.active(sim.people).uids)/len(sim.people)
        self.results.partners_f_mean[ti] = np.mean(partners_active_f)
        self.results.partners_m_mean[ti] = np.mean(partners_active_m)

        return


def make_sim(n_agents=500, dt=1):

    fertility_rates = {'fertility_rate': pd.read_csv('test_data/zimbabwe_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv('test_data/zimbabwe_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)
    ppl = ss.People(n_agents, age_data=pd.read_csv('test_data/zimbabwe_age.csv'))
    sexual = sti.StructuredSexual()

    sim = ss.Sim(
        dt=dt,
        start=1990,
        total_pop=9980999,
        n_years=35,
        people=ppl,
        networks=sexual,
        demographics=[pregnancy, death],
        analyzers=network_stats()
    )

    return sim


def test_n_agents():
    sc.heading('Test pop sizes with varying n_agents')
    share_active = dict()
    partners_m_mean = dict()
    partners_f_mean = dict()
    sims = sc.autolist()
    n_agent_list = [10e3]
    for n_agents in n_agent_list:
        sim = make_sim(n_agents=n_agents, dt=1/12)
        sim.run()
        ares = sim.analyzers[0].results
        share_active[n_agents] = ares.share_active
        partners_f_mean[n_agents] = ares.partners_f_mean
        partners_m_mean[n_agents] = ares.partners_m_mean
        sims += sim

    fig, axes = pl.subplots(3, 1)
    axes = axes.ravel()
    for n_agents in n_agent_list:
        axes[0].plot(sim.yearvec, share_active[n_agents], label=int(n_agents))
        axes[1].plot(sim.yearvec, partners_f_mean[n_agents], label=int(n_agents))
        axes[2].plot(sim.yearvec, partners_m_mean[n_agents], label=int(n_agents))
    axes[0].legend()

    axes[0].set_title('Share active')
    axes[1].set_title('Mean partners - females')
    axes[2].set_title('Mean partners - males')

    sc.figlayout()
    pl.show()

    return sims


if __name__ == '__main__':
    sims = test_n_agents()
