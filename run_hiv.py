"""
Run hiv
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc
from stisim.networks import StructuredSexual
from stisim.diseases.hiv import HIV

quick_run = False
ss.options['multirng']=False


def make_hiv_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075):
    """ Make a sim with HIV """
    hiv = HIV()
    hiv.pars['beta'] = {'structuredsexual': [0.95, 0.5], 'maternal': [0.99, 0]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.1)
    #hiv.pars['rel_trans']['latent_temp'] = latent_trans
    #hiv.pars['rel_trans']['latent_long'] = latent_trans


    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(f'data/{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(f'data/{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(n_agents, age_data=pd.read_csv(f'data/{location}_age.csv'))
    sexual = StructuredSexual()
    maternal = ss.MaternalNet()

    sim_kwargs = dict(
        dt=dt,
        total_pop=total_pop,
        start=1990,
        n_years=1,
        people=ppl,
        diseases=hiv,
        networks=ss.ndict(sexual, maternal),
        demographics=[pregnancy, death],
    )

    return sim_kwargs


def run_hiv(location='zimbabwe', total_pop=100e6, dt=1.0, n_agents=500):

    sim_kwargs = make_hiv_sim(location=location, total_pop=total_pop, dt=dt, n_agents=n_agents)
    sim = ss.Sim(**sim_kwargs)
    sim.run()
    df_res = sim.export_df()
    df = pd.DataFrame.from_dict(df_res)
    return sim, df

def plot_cd4_count(sim_output):
    '''
    Add a plot of time vs CD4 count for a subset of agents to make sure CD4 counts work sensibly.
    '''

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for agent in np.arange(0, 5):
        plt.plot(sim_output.index, sim_output['hiv.Agent' + str(agent) + '_cd4'])

    ax.set_xlabel('Time')
    ax.set_ylabel('CD4 count (cells/uL)')
    ax.set_title('CD4 counts')
    fig.tight_layout()
    sc.savefig("figures/cd4_count.png", dpi=100)

if __name__ == '__main__':

    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]
    sim, output = run_hiv(location=location, total_pop=total_pop, dt=1/365, n_agents=int(10e3))
    output.to_csv("HIV_output.csv")

    # plot_cd4_count(output)

    sc.saveobj(f'sim_{location}.obj', sim)


