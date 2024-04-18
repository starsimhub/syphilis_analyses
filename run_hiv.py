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
from stisim.interventions import ART

quick_run = False
ss.options['multirng'] = False


def make_hiv_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075):
    """ Make a sim with HIV """
    hiv = HIV()
    hiv.pars['beta'] = {'structuredsexual': [0.95, 0.5], 'maternal': [0.99, 0]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.1)
    # hiv.pars['rel_trans']['latent_temp'] = latent_trans
    # hiv.pars['rel_trans']['latent_long'] = latent_trans

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
        interventions=ART(year=1990, coverage=0.5),
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

def plot_T_level(sim_output):
    """
    Single plot of CD4 counts
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # Plot the first 40 days only:
    sim_output_sub = sim_output.iloc[0:40]
    plt.plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(11) + '_T_level'])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('CD4-count (cells/uL)')
    ax.set_title('CD4-count (cells/uL)')
    fig.tight_layout()
    sc.savefig("figures/cd4_counts.png", dpi=100)


def plot_V_level(sim_output):
    """
    Single Plot of viral load
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # Plot the first 40 days only:
    sim_output_sub = sim_output.iloc[0:40]
    plt.plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(11) + '_V_level'])
    ax.set_xlabel('Time (days)')
    ax.set_yscale('log')
    ax.set_ylabel('Viral load (copies/mL)')
    ax.set_title('Viral load (copies/mL)')
    fig.tight_layout()
    sc.savefig("figures/viral_loads.png", dpi=100)


def plot_viral_dynamics(sim_output):
    """
    3 Subplots to show viral load, CD4 count and infected cells count
    """
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    # Plot the first 40 days only:
    sim_output_sub = sim_output.iloc[0:40]

    # Viral Load:
    ax[0].plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(11) + '_V_level'] * 1e3, label='on ART after 30 days')
    ax[0].plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(36) + '_V_level'] * 1e3, label='not on ART')
    ax[0].set_xlabel('Time (days)')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Viral load (copies/mL)')
    ax[0].set_title('Viral load')

    # CD4-count
    ax[1].plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(11) + '_T_level'], label='on ART after 30 days')
    ax[1].plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(36) + '_T_level'], label='not on ART')
    ax[1].set_xlabel('Time (days)')
    ax[1].set_ylabel('CD4-count (cells/uL)')
    ax[1].set_title('CD4-count')

    # Infected cells count:
    ax[2].plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(11) + '_I_level'], label='on ART after 30 days')
    ax[2].plot(np.arange(0, 40, 1), sim_output_sub['hiv.Agent' + str(36) + '_I_level'], label='not on ART')
    ax[2].set_xlabel('Time (days)')
    ax[2].set_ylabel('Infected cell count (cells/uL)')
    ax[2].set_title('Infected cells')

    ax[0].legend()
    fig.tight_layout()
    sc.savefig("figures/viral_dynamics.png", dpi=100)


if __name__ == '__main__':

    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]
    sim, output = run_hiv(location=location, total_pop=total_pop, dt=1 / 365, n_agents=int(10e3))
    output.to_csv("HIV_output.csv")

    # Plot viral dynamics of one infected agent not on ART (hard-coded) to check if dynamics work sensibly
    if sim.dt == 1 / 365:
        # plot_T_level(output)
        # plot_V_level(output)
        plot_viral_dynamics(output)

    sc.saveobj(f'sim_{location}.obj', sim)
