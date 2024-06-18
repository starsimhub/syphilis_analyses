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
 
def make_sim(n_agents=500, dt=1):

    fertility_rates = {'fertility_rate': pd.read_csv('test_data/zimbabwe_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv('test_data/zimbabwe_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)
    ppl = ss.People(n_agents, age_data=pd.read_csv('test_data/zimbabwe_age.csv'))

    sim = ss.Sim(
        dt=dt,
        start=1990,
        total_pop=9980999,
        n_years=35,
        people=ppl,
        demographics=[pregnancy, death],
    )

    return sim

def test_n_agents():
    sc.heading('Test pop sizes with varying n_agents')
    results = dict()
    sims = sc.autolist()
    n_agent_list = [5e3, 10e3, 20e3]
    for n_agents in n_agent_list:
        sim = make_sim(n_agents=n_agents, dt=1)
        sim.run()
        results[n_agents] = sim.results.n_alive
        sims += sim

    fig, ax = pl.subplots(1, 1)
    for n_agents in n_agent_list:
        ax.plot(sim.yearvec, results[n_agents], label=int(n_agents))
    ax.legend()

    return sims


if __name__ == '__main__':
    sims = test_n_agents()
