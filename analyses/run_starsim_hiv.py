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
from stisim.interventions import ART, HIV_testing

quick_run = False
ss.options['multirng'] = False


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
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Plot the first 40 days only:
    sim_output_sub = sim_output.iloc[0:]

    # Viral Load:
    ax[0].plot(sim_output_sub.index, sim_output_sub['hiv.Agent' + str(2) + '_V_level'], label='not on ART')
    #ax[0].plot(np.arange(0, 40), sim_output_sub['hiv.Agent' + str(8) + '_V_level'] * 1e3, label='not on ART')
    ax[0].set_xlabel('Time (days)')
    # ax[0].set_yscale('log')
    ax[0].set_ylabel('Viral load')
    ax[0].set_title('Viral load')

    # CD4-count
    ax[1].plot(sim_output_sub.index, sim_output_sub['hiv.Agent' + str(2) + '_T_level'], label='not on ART')
    #ax[1].plot(np.arange(0, 40), sim_output_sub['hiv.Agent' + str(8) + '_T_level'], label='on ART')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('CD4-count')
    ax[1].set_title('CD4-count')

    ax[0].legend()
    fig.tight_layout()
    sc.savefig("figures/viral_dynamics.png", dpi=100)


def plot_hiv(sim_output):
    """
    Plot:
    - new infections
    - cum infections
    - new deaths
    - cum deaths
    - HIV prevalence, HIV prevalence in FSW
    - n on ART
    """
    fig, ax = plt.subplots(4, 2, figsize=(15, 15))

    # sim_output = sim_output.iloc[1:]
    ax[0, 0].plot(sim_output.index, sim_output['hiv.new_infections'])
    ax[0, 0].set_title('New infections')
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_infections'])
    ax[0, 1].set_title('Cumulative infections')

    ax[1, 0].plot(sim_output.index, sim_output['deaths.new'])
    ax[1, 0].set_title('New Deaths')
    ax[1, 1].plot(sim_output.index, sim_output['deaths.cumulative'])
    ax[1, 1].set_title('Cumulative Deaths')

    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence'], label='General Population')
    ax[2, 0].set_title('HIV prevalence')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_sw'], label='FSW')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_client'], label='Client')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_0'], label='Risk Group 0')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_1'], label='Risk Group 1')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_2'], label='Risk Group 2')
    ax[2, 0].legend()

    ax[3, 0].plot(sim_output.index, sim_output['hiv.new_on_art'])
    ax[3, 0].set_title('New on Art')
    ax[3, 1].plot(sim_output.index, sim_output['hiv.cum_on_art'])
    ax[3, 1].set_title('Cumulative ART')


    fig.tight_layout()
    # plt.show()
    sc.savefig("figures/hiv_plots.png", dpi=100)


def make_hiv_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075):
    """
    Make a sim with HIV
    """

    hiv = HIV()
    hiv.pars['beta'] = {'structuredsexual': [0.95, 0.95], 'maternal': [0.08, 0.5]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.3)
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
        n_years=20,
        people=ppl,
        diseases=hiv,
        networks=ss.ndict(sexual, maternal),
        interventions=[HIV_testing(disease='hiv',
                                   symp_prob=0.01,
                                   sensitivity=0.9,
                                   test_delay_mean=1)],
        demographics=[pregnancy, death],
    )

    return sim_kwargs


def run_hiv(location='zimbabwe', total_pop=100e6, dt=1.0, n_agents=500):
    """
    Make and run the sim
    """
    sim_kwargs = make_hiv_sim(location=location, total_pop=total_pop, dt=dt, n_agents=n_agents)
    sim = ss.Sim(**sim_kwargs)
    sim.run()
    df_res = sim.export_df()
    df = pd.DataFrame.from_dict(df_res)
    return sim, df


if __name__ == '__main__':

    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]
    sim, output = run_hiv(location=location, total_pop=total_pop, dt=1 / 12, n_agents=int(10e3))
    output.to_csv("HIV_output.csv")

    # Plot viral dynamics of one infected agent not on ART (hard-coded) to check if dynamics work sensibly
    # plot_T_level(output)
    # plot_V_level(output)
    plot_viral_dynamics(output)
    plot_hiv(output)

    sc.saveobj(f'sim_{location}.obj', sim)
