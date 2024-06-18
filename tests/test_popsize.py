"""
Test that the population size doesn't depend on the number of agents in the simulation
"""

# Imports
import sciris as sc
import starsim as ss
import stisim as sti
import pandas as pd
import numpy as np

 
def make_sim(n_agents=500, dt=1):

    pregnancy = ss.Pregnancy(fertility_rate=10)
    death = ss.Deaths(death_rate=10)
    sim = ss.Sim(
        dt=dt,
        start=1990,
        total_pop=9980999,
        n_years=35,
        n_agents=n_agents,
        demographics=[pregnancy, death],
    )

    return sim

def test_n_agents():
    sc.heading('Test pop sizes with varying n_agents')
    results = dict()
    sims = sc.autolist()
    for n_agents in [500, 1e3, 5e3]:
        sim = make_sim(n_agents=500, dt=1)
        sim.run()
        results[n_agents] = sim.results.n_alive
        sims += sim

    fig, ax = pl.subplots(1, 1)
    for n_agents in [500, 1e3, 5e3]:
        ax.plot(sim.yearvec, results[n_agents], label=int(n_agents))
    ax.legend()

    return sims


if __name__ == '__main__':
    sims = test_n_agents()
