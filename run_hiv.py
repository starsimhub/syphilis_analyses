"""
Run hiv
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc
import seaborn as sns
from stisim.networks import StructuredSexual
from stisim.diseases.hiv import HIV
from stisim.interventions import ART, HIV_testing, test_ART, BaseTest
from matplotlib.ticker import FuncFormatter

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


def plot_viral_dynamics(output, save_agents):
    """
    2 Subplots to show viral load, CD4 count
    """
    for index, agent in enumerate(save_agents):
        fig, ax = plt.subplots(1, 2, figsize=(25, 8))
        df_this_agent = output[output.columns[output.columns.str.contains('_' + str(agent))]].dropna()
        sns.scatterplot(ax=ax[0], x=df_this_agent.index, y="transmission_" + str(agent), data=df_this_agent,
                        hue='ART_status_' + str(agent))
        sns.scatterplot(ax=ax[1], x=df_this_agent.index, y="cd4_count_" + str(agent), data=df_this_agent,
                        hue='ART_status_' + str(agent))

        ax[0].set_xlabel('Time (months)')
        # ax[0].set_yscale('log')
        ax[0].set_ylim([0, 7])
        ax[0].set_xlim([0, len(output)])
        ax[0].set_ylabel('Transmission probability')
        ax[0].set_title('Transmission probability')

        # CD4-count
        ax[1].set_xlabel('Time (months)')
        ax[1].set_ylabel('CD4-count')
        ax[1].set_title('CD4-count')
        ax[1].set_ylim([0, 1000])
        ax[1].set_xlim([0, len(output)])

        ax[0].legend()
        fig.tight_layout()
        sc.savefig("figures/individual_viral_dynamics/viral_dynamics_agent_" + str(agent) + ".png", dpi=100)
        plt.close()


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

    ART_coverages_raw = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Testing & treatment',
                                      skiprows=28).iloc[0:1, 2:43]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
    pop_scale = total_pop / int(10e3)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.columns.values[
                                                             ~pd.isna(ART_coverages_raw.values)[0]].astype(int),
                                                         ART_coverages_raw.values[
                                                             ~pd.isna(ART_coverages_raw.values)]) / pop_scale).astype(
                                         int)})

    fig, ax = plt.subplots(6, 2, figsize=(20, 15))

    # sim_output = sim_output.iloc[1:]
    ax[0, 0].plot(sim_output.iloc[1:].index, sim_output.iloc[1:]['hiv.new_infections'])
    ax[0, 0].set_title('New infections')
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_infections'])
    ax[0, 1].set_title('Cumulative infections')

    ax[1, 0].plot(sim_output.iloc[1:].index, sim_output.iloc[1:]['deaths.new'])
    ax[1, 0].set_title('New Deaths')
    # ax[1, 1].plot(sim_output.index, sim_output['deaths.cumulative'])
    # ax[1, 1].set_title('Cumulative Deaths')
    ax[1, 1].plot(sim_output.index, sim_output['hiv.new_deaths'])
    ax[1, 1].set_title('HIV Deaths')

    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence'], label='General Population')
    ax[2, 0].set_title('HIV prevalence')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_sw'], label='FSW')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_client'], label='Client')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_0'], label='Risk Group 0')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_1'], label='Risk Group 1')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_2'], label='Risk Group 2')
    ax[2, 0].legend()

    ax[3, 0].plot(sim_output.iloc[1:].index, sim_output.iloc[1:]['hiv.n_on_art'], label='Modelled')
    ax[3, 0].set_title('Number of people on ART (Mio)')
    ax[3, 0].plot(ART_coverages_df["Years"][1:], ART_coverages_df["Value"][1:] * pop_scale, label="Data")
    ax[3, 0].legend()
    ax[3, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[3, 0].set_xlim([2000.0, max(sim_output.iloc[1:].index)])

    ax[4, 0].plot(sim_output.iloc[1:].index, sim_output.iloc[1:]['hiv.new_diagnoses'])
    ax[4, 0].set_title('New Diagnoses')
    ax[4, 1].plot(sim_output.index, sim_output['hiv.cum_diagnoses'])
    ax[4, 1].set_title('Cumulative Diagnoses')

    ax[5, 0].plot(sim_output.iloc[1:].index, sim_output.iloc[1:]['n_alive'])
    ax[5, 0].set_title('Population')
    ax[5, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[5, 0].set_ylabel('Mio.')

    fig.tight_layout()
    # plt.show()
    sc.savefig("figures/hiv_plots.png", dpi=100)


def get_testing_products():

    # Load HIV test data:
    HIV_tests_data_raw = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Testing & treatment',
                                       skiprows=1).iloc[0:15, 1:43]
    HIV_tests_data_raw.index = HIV_tests_data_raw.iloc[:, 0]
    HIV_tests_data_raw = HIV_tests_data_raw.iloc[:, 1:]
    HIV_tests_data_raw.loc["Other_avg"] = HIV_tests_data_raw[HIV_tests_data_raw.index != "FSW"].mean()
    tivec = np.arange(start=1990, stop=2020 + 1, step=1)
    FSW_prop = np.interp(tivec,
                         HIV_tests_data_raw.loc["FSW"].index[~pd.isna(HIV_tests_data_raw.loc["FSW"].values)].astype(int),
                         HIV_tests_data_raw.loc["FSW"].values[~pd.isna(HIV_tests_data_raw.loc["FSW"].values)])
    other_prop = np.interp(tivec,
                           HIV_tests_data_raw.loc["Other_avg"].index[~pd.isna(HIV_tests_data_raw.loc["Other_avg"].values)].astype(int),
                           HIV_tests_data_raw.loc["Other_avg"].values[~pd.isna(HIV_tests_data_raw.loc["Other_avg"].values)])

    ####################################################################################################################
    # Product
    ####################################################################################################################

    testing_data = pd.DataFrame(
        {'name': 'simple_testing',
         'state': ['susceptible', 'susceptible', 'infected', 'infected'],
         'disease': 'hiv',
         'probability': [0, 1, 1, 0],
         'result': ['positive', 'negative', 'positive', 'negative']})

    simple_test = Dx(df=testing_data)

    ####################################################################################################################
    # FSW Testing
    ####################################################################################################################

    FSW_eligible = lambda sim: sim.networks.structuredsexual.fsw & (np.isnan(sim.get_intervention('fsw_testing').ti_screened) | (sim.ti > (sim.get_intervention('fsw_testing').ti_screened + 12)))
    FSW_testing = BaseTest(prob=FSW_prop,
                           product=simple_test,
                           eligibility=FSW_eligible,
                           label='fsw_testing',
                           disease='hiv')

    ####################################################################################################################
    # Remaining population testing
    ####################################################################################################################

    other_eligible = lambda sim: ~sim.networks.structuredsexual.fsw & (np.isnan(sim.get_intervention('other_testing').ti_screened) | (sim.ti > (sim.get_intervention('other_testing').ti_screened + 12)))
    other_testing = BaseTest(prob=other_prop,
                             product=simple_test,
                             eligibility=other_eligible,
                             label='other_testing',
                             disease='hiv')

    # TODO Add testing for agents with CD4 count <200

    return FSW_testing, other_testing


def make_hiv_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075,
                 save_agents=np.array([0])):
    """
    Make a sim with HIV
    """

    hiv = HIV()
    hiv.pars['beta'] = {'structuredsexual': [0.95, 0.95], 'maternal': [0.08, 0.5]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.3)
    hiv.pars['cd4_start_mean'] = 800
    hiv.pars['primary_acute_inf_dur'] = 2.9  # in months
    hiv.pars['transmission_sd'] = 0.00  # Standard Deviation of normal distribution for transmission.
    hiv.pars['death_data'] = [(500, 0.0036 / 12),
                              (range(350, 500), 0.0036 / 12),
                              (range(200, 350), 0.0088 / 12),
                              (range(50, 200), 0.059 / 12),
                              (range(0, 50), 0.323 / 12)]

    # Read in treatment data:
    ART_coverages_raw = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Testing & treatment',
                                      skiprows=28).iloc[0:1, 2:43]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
    pop_scale = total_pop / n_agents
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.columns.values[
                                                             ~pd.isna(ART_coverages_raw.values)[0]].astype(int),
                                                         ART_coverages_raw.values[
                                                             ~pd.isna(ART_coverages_raw.values)]) / pop_scale).astype(
                                         int)})
    hiv.pars['ART_coverages_df'] = ART_coverages_df

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

    simple_testing = get_testing_products()

    sim_kwargs = dict(
        dt=dt,
        total_pop=total_pop,
        start=1990,
        n_years=40,
        people=ppl,
        remove_dead=1,  # How many timesteps to go between removing dead agents (0 to not remove)
        diseases=hiv,
        networks=ss.ndict(sexual, maternal),
        interventions=[fsw_testing,
                       other_testing,
                       ART(ART_coverages_df=ART_coverages_df,
                           duration_on_ART=ss.normal(loc=18, scale=5),# https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-021-10464-x
                           art_efficacy=0.96),
                       validate_ART(disease='hiv',
                                uids=save_agents,
                                infect_uids_t=np.repeat(100, len(save_agents)),
                                stop_ART=True,
                                restart_ART=True)],
        demographics=[pregnancy, death])

    return sim_kwargs


def run_hiv(location='zimbabwe', total_pop=100e6, dt=1.0, n_agents=500, save_agents=np.array([0])):
    """
    Make and run the sim
    """
    sim_kwargs = make_hiv_sim(location=location, total_pop=total_pop, dt=dt, n_agents=n_agents, save_agents=save_agents)
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
    save_agents = np.arange(0, 40)

    sim, output = run_hiv(location=location, total_pop=total_pop, dt=1 / 12, n_agents=int(10e3),
                          save_agents=save_agents)
    output.to_csv("HIV_output.csv")

    # Call method in validate_ART intervention:
    # sim.get_interventions(validate_ART)[0].save_viral_histories(sim)
    #viral_histories = pd.read_csv("viral_histories.csv", index_col=0)
    #plot_viral_dynamics(viral_histories, save_agents)

    plot_hiv(output)

    sc.saveobj(f'sim_{location}.obj', sim)
