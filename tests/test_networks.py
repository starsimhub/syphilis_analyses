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


class partner_count(ss.Analyzer):
    def __init__(self):
        return

    def apply(self, sim):
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
    )

    return sim

def test_n_agents():
    sc.heading('Test pop sizes with varying n_agents')
    results = dict()
    sims = sc.autolist()
    for n_agents in [1e3, 5e3, 10e3]:
        sim = make_sim(n_agents=500, dt=1/)
        sim.run()
        sims += sim

    return sims


if __name__ == '__main__':
    sims = test_n_agents()
