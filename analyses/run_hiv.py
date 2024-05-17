"""
Run hiv
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc
from io import StringIO
import seaborn as sns
import stisim as sti
from stisim.networks import StructuredSexual
from stisim.products import Dx
from stisim.diseases.hiv import HIV
from stisim.interventions import ART, validate_ART, BaseTest
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

quick_run = False
ss.options['multirng'] = False
datadir = sti.root/'analyses'/'data'

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
    2 Subplots to show transmission probability and CD4 counts
    """
    tivec = np.arange(start=1990, stop=2030 + 1 / 12, step=1 / 12)
    for index, agent in enumerate(save_agents):
        fig, ax = plt.subplots(1, 2, figsize=(25, 8))
        df_this_agent = output[output.columns[output.columns.str.contains('_' + str(agent))]].dropna()
        sns.scatterplot(ax=ax[0], x=df_this_agent.index, y="transmission_" + str(agent), data=df_this_agent,
                        hue='ART_status_' + str(agent))
        sns.scatterplot(ax=ax[1], x=df_this_agent.index, y="cd4_count_" + str(agent), data=df_this_agent,
                        hue='ART_status_' + str(agent))

        ax[0].xaxis.set_ticks(df_this_agent.index[::12 * 5])
        ax[0].set_xticklabels(np.ceil(tivec[df_this_agent.index][::12 * 5]).astype(int))
        ax[1].xaxis.set_ticks(df_this_agent.index[::12 * 5])
        ax[1].set_xticklabels(np.ceil(tivec[df_this_agent.index][::12 * 5]).astype(int))

        ax[0].set_xlabel('Time')
        # ax[0].set_yscale('log')
        ax[0].set_ylim([0, 7])
        ax[0].set_xlim([0, len(output)])
        ax[0].set_ylabel('Transmission probability')
        ax[0].set_title('Transmission probability')

        # CD4-count
        ax[1].set_xlabel('Time')
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

    fig, ax = plt.subplots(7, 2, figsize=(20, 15))

    #####################################################################################################################
    # n infections, diagnosed and on ART and cumulative states
    #####################################################################################################################
    n_infections_data = pd.read_excel(datadir/f'{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=33).iloc[[1], 2:36]
    ax[0, 0].scatter(n_infections_data.columns, n_infections_data.values, label='Data - number of PLHIV', color='tab:red')
    ax[0, 0].plot(sim_output.index, sim_output['hiv.n_infected'], label='n infected')
    ax[0, 0].plot(sim_output.index, sim_output['hiv.n_diagnosed'], label='n diagnosed')
    ax[0, 0].plot(sim_output.index, sim_output['hiv.n_on_art'], label='n on ART')
    ax[0, 0].set_ylabel('Mio.')
    ax[0, 0].legend()

    #####################################################################################################################
    # Cumulative infections, diagnosed and on ART and cumulative states
    #####################################################################################################################

    ax[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_infections'], label='Cumulative Infections')
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_diagnoses'], label='Cumulative Diagnoses')
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_agents_on_art'], label='Cumulative on ART')
    ax[0, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[0, 1].set_ylabel('Mio.')
    ax[0, 1].legend()

    #####################################################################################################################
    # New Deaths and HIV Deaths
    #####################################################################################################################
    hiv_deaths_data = pd.read_excel(datadir/f'{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=41).iloc[[1], 2:36]
    ax[1, 0].scatter(hiv_deaths_data.columns, hiv_deaths_data.values[0], color='tab:red', label='Data - yearly HIV deaths')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    ax[1, 0].plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['hiv.new_deaths'].sum(), color='tab:blue', label='Modelled - yearly HIV deaths')
    ax[1, 0].set_title('Deaths')
    ax[1, 0].plot(sim_output.index, sim_output['deaths.new'], color='tab:orange', label='New Deaths')
    ax[1, 0].plot(sim_output.index, sim_output['hiv.new_deaths'], color='tab:blue', label='New HIV Deaths')
    ax[1, 0].legend()

    #####################################################################################################################
    # New Diagnoses, Infections and on ART
    #####################################################################################################################
    ax[1, 1].set_title('New Diagnoses and on ART')
    # ax[1, 1].plot(sim_output.index, sim_output['hiv.new_infections'], label='New Infections')
    ax[1, 1].plot(sim_output.index, sim_output['hiv.new_diagnoses'], label='New Diagnoses')
    ax[1, 1].plot(sim_output.index, sim_output['hiv.new_agents_on_art'], label='New on ART')
    ax[1, 1].legend()

    #####################################################################################################################
    # HIV Prevalence
    #####################################################################################################################
    prevalence_data = pd.read_csv(datadir/f'world_bank_hiv_prevalence.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    ax[2, 0].scatter(prevalence_data.index.astype(int), prevalence_data.values, color='tab:red', label='Data (15-49)')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence'] * 100, label='General Population')
    ax[2, 0].set_title('HIV prevalence (%)')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_sw'] * 100, label='FSW')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_client'] * 100, label='Client')
    ax[2, 0].legend()

    ax[2, 1].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_0_female'] * 100, color='tab:blue', label='Risk Group 0 - Female')
    ax[2, 1].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_1_female'] * 100, color='tab:green', label='Risk Group 1- Female')
    ax[2, 1].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_2_female'] * 100, color='tab:orange', label='Risk Group 2- Female')
    ax[2, 1].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_0_male'] * 100, color='tab:blue', linestyle='--', label='Risk Group 0 - Male')
    ax[2, 1].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_1_male'] * 100, color='tab:green', linestyle='--', label='Risk Group 1- Male')
    ax[2, 1].plot(sim_output.index, sim_output['hiv.prevalence_risk_group_2_male'] * 100, color='tab:orange', linestyle='--', label='Risk Group 2- Male')
    blue_patch = mpatches.Patch(color='tab:blue', label='Risk Group 0')
    green_patch = mpatches.Patch(color='tab:green', label='Risk Group 1')
    orange_patch = mpatches.Patch(color='tab:orange', label='Risk Group 2')
    females = Line2D([0], [0], label='females', color='k', linestyle='-')
    males = Line2D([0], [0], label='males', color='k', linestyle='--')
    ax[2, 1].legend(handles=[blue_patch, green_patch, orange_patch, females, males])
    ax[2, 1].set_title('HIV prevalence (%) by Risk Group and gender')

    #####################################################################################################################
    # ART Coverage
    #####################################################################################################################
    ART_coverages_raw = pd.read_csv(datadir/f'world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.index.astype(int).tolist(),
                                                         (ART_coverages_raw.values / 100).tolist()))})
    ART_coverages_raw = pd.read_excel(datadir/f'{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=28).iloc[0:1, 2:43]
    pop_scale = total_pop / int(10e3)
    ART_coverages_df_numbers = pd.DataFrame({"Years": tivec,
                                             "Value": (np.interp(tivec,
                                                                 ART_coverages_raw.columns.values[~pd.isna(ART_coverages_raw.values)[0]].astype(int),
                                                                 ART_coverages_raw.values[~pd.isna(ART_coverages_raw.values)]) / pop_scale).astype(int)})

    ax[3, 0].plot(sim_output.index, sim_output['hiv.n_on_art'] / sim_output['hiv.n_diagnosed'], label='Modelled')
    ax[3, 0].set_title('ART coverage (%)')
    ax[3, 0].scatter(ART_coverages_df["Years"], ART_coverages_df["Value"], color='tab:red', label="Data")
    ax[3, 0].legend()
    ax[3, 0].set_xlim([2000.0, max(sim_output.iloc[1:].index)])

    ax[3, 1].plot(sim_output.index, sim_output['hiv.n_on_art'], label='Modelled')
    ax[3, 1].set_title('Number of people on ART (in Mio)')
    ax[3, 1].scatter(ART_coverages_df_numbers["Years"], ART_coverages_df_numbers["Value"] * pop_scale, color='tab:red', label="Data")
    ax[3, 1].legend()
    ax[3, 1].set_ylabel('Mio.')
    ax[3, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[3, 1].set_xlim([2000.0, max(sim_output.iloc[1:].index)])

    #####################################################################################################################
    # New Screened
    #####################################################################################################################
    ax[4, 0].plot(sim_output.index, sim_output['fsw_testing.new_screened'], color='tab:orange', label='FSW')
    ax[4, 0].plot(sim_output.index, sim_output['other_testing.new_screened'], color='tab:blue', label='General Population')
    ax[4, 0].plot(sim_output.index, sim_output['low_cd4_testing.new_screened'], color='tab:green', label='CD4 count <200')
    ax[4, 0].plot(sim_output.index, sim_output['fsw_testing.new_screens'], color='tab:orange', linestyle='--', label='FSW')
    ax[4, 0].plot(sim_output.index, sim_output['other_testing.new_screens'], color='tab:blue', linestyle='--', label='General Population')
    ax[4, 0].plot(sim_output.index, sim_output['low_cd4_testing.new_screens'], color='tab:green', linestyle='--', label='CD4 count <200')
    ax[4, 0].set_title('New Screened and New Screens')
    orange_patch = mpatches.Patch(color='tab:orange', label='FSW')
    blue_patch = mpatches.Patch(color='tab:blue', label='General Population')
    green_patch = mpatches.Patch(color='tab:green', label='CD4 count <200')
    screened = Line2D([0], [0], label='new screened', color='k', linestyle='-')
    screens = Line2D([0], [0], label='new screens', color='k', linestyle='--')
    ax[4, 0].legend(handles=[blue_patch, green_patch, orange_patch, screened, screens])

    #####################################################################################################################
    # Yearly Tests
    #####################################################################################################################
    n_tests_data = pd.read_excel(datadir/f'{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=1).iloc[[1], 2:36]
    # Calculate yearly tests in the model:
    ax[4, 1].scatter(n_tests_data.columns, n_tests_data.values[0], color='tab:red', label='Data')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['fsw_testing.new_screens'].sum(), marker='o', label='FSW')
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['other_testing.new_screens'].sum(), marker='o', label='General Population')
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['low_cd4_testing.new_screens'].sum(), marker='o', label='CD4 count <200')
    ax[4, 1].plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['low_cd4_testing.new_screens'].sum()
                  + sim_output.groupby(by='year')['other_testing.new_screens'].sum()
                  + sim_output.groupby(by='year')['fsw_testing.new_screens'].sum(), label='FSW + Other + Low CD4 count testing')
    ax[4, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[4, 1].set_title('Yearly Tests (in Mio)')
    ax[4, 1].legend()

    #####################################################################################################################
    # Pregnancies & Births
    #####################################################################################################################
    ax[5, 0].plot(sim_output.index, sim_output['pregnancy.pregnancies'], label='Pregnancies')
    ax[5, 0].plot(sim_output.index, sim_output['pregnancy.births'], label='Births')
    ax[5, 0].set_title('Pregnancies and Births')
    ax[5, 0].legend()

    #####################################################################################################################
    # Population
    #####################################################################################################################
    population_data = pd.read_excel(datadir/f'{location}_20230725.xlsx', sheet_name='Population size', skiprows=72).iloc[0:1, 3:36]
    ax[5, 1].scatter(np.arange(1990, 2022 + 1, 1), population_data.loc[0].values, color='tab:red', label='Data')
    ax[5, 1].plot(sim_output.index, sim_output['n_alive'], label='Modelled')
    ax[5, 1].set_title('Population')
    ax[5, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[5, 1].set_ylabel('Mio.')
    ax[5, 1].legend()

    #####################################################################################################################
    # Syphilis Prevalence
    #####################################################################################################################
    ax[6, 0].plot(sim_output.index, 100 * sim_output['hiv.n_syphilis_inf']/sim_output['n_alive'], label='Syphilis prevalence')
    ax[6, 0].set_title('Syphilis prevalence')

    fig.tight_layout()
    # plt.show()
    sc.savefig("figures/hiv_plots.png", dpi=100)


def get_testing_products():
    """
    Define HIV products and testing interventions
    """
    # Load HIV test data:
    hiv_testing_data = pd.read_excel(datadir/f'{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=1)
    HIV_tests_data_raw = hiv_testing_data.iloc[0:15, 1:43]
    HIV_low_cd4count_data_raw = hiv_testing_data.iloc[21:22, 1:43]
    HIV_low_cd4count_data_raw = HIV_low_cd4count_data_raw.iloc[:, 1:]
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
    low_cd4count_prop = np.interp(tivec,
                                  HIV_low_cd4count_data_raw.iloc[0].index[~pd.isna(HIV_low_cd4count_data_raw.iloc[0].values)].astype(int),
                                  HIV_low_cd4count_data_raw.iloc[0].values[~pd.isna(HIV_low_cd4count_data_raw.iloc[0].values)])

    ####################################################################################################################
    # Product
    ####################################################################################################################

    testing_data = pd.read_csv(StringIO("""name,state,disease,result,probability
                    simple testing,susceptible,hiv,positive,0
                    simple testing,susceptible,hiv,negative,1
                    simple testing,infected,hiv,positive,1
                    simple testing,infected,hiv,negative,0
                    """), sep=",")
    simple_test = Dx(df=testing_data, name='simple_test')

    ####################################################################################################################
    # FSW Testing
    ####################################################################################################################
    # Eligible for testing are FSW agents, who haven't been diagnosed yet and haven't been screened yet or last screening was 12months ago.

    FSW_eligible = lambda sim: sim.networks.structuredsexual.fsw & ~sim.diseases['hiv'].diagnosed & \
                               (np.isnan(sim.get_intervention('fsw_testing').ti_screened) | (sim.ti > (sim.get_intervention('fsw_testing').ti_screened + 12)))
    FSW_testing = BaseTest(prob=FSW_prop / 12,
                           name='fsw_testing',
                           product=simple_test,
                           eligibility=FSW_eligible,
                           label='fsw_testing',
                           disease='hiv')

    ####################################################################################################################
    # Remaining population testing
    ####################################################################################################################

    # Eligible for testing are non-FSW agents, who haven't been diagnosed yet and haven't been screened yet or last screening was 12months ago.
    other_eligible = lambda sim: ~sim.networks.structuredsexual.fsw & ~sim.diseases['hiv'].diagnosed & \
                                 (np.isnan(sim.get_intervention('other_testing').ti_screened) | (sim.ti > (sim.get_intervention('other_testing').ti_screened + 12)))
    other_testing = BaseTest(prob=other_prop / 12,
                             name='other_testing',
                             product=simple_test,
                             eligibility=other_eligible,
                             label='other_testing',
                             disease='hiv')

    ####################################################################################################################
    # CD4 count < 50 testing
    ####################################################################################################################
    # Eligible for testing are agents, who haven't been diagnosed yet and whose CD4 count is below 50.

    low_cd4_eligibe = lambda sim: (sim.diseases['hiv'].cd4 < 200) & ~sim.diseases['hiv'].diagnosed
    low_cd4_testing = BaseTest(prob=low_cd4count_prop / 12,
                               name='low_cd4_testing',
                               product=simple_test,
                               eligibility=low_cd4_eligibe,
                               label='low_cd4_testing',
                               disease='hiv')

    return FSW_testing, other_testing, low_cd4_testing


def make_hiv_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075,
                 save_agents=np.array([0])):
    """
    Make a sim with HIV
    """
    ####################################################################################################################
    # HIV Params
    ####################################################################################################################
    hiv = HIV()
    hiv.pars['beta'] = {'structuredsexual': [0.95, 0.95], 'maternal': [0.95, 0.]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.15) #ss.bernoulli(p=0.15)
    hiv.pars['cd4_start_dist'] = ss.normal(loc=800, scale=10)
    hiv.pars['init_diagnosed'] = ss.bernoulli(p=0.15)  # Proportion of initially infected agents who start out as diagnosed
    hiv.pars['primary_acute_inf_dur'] = 2.9  # in months
    hiv.pars['transmission_sd'] = 0.0  # Standard Deviation of normal distribution for randomness in transmission.
    hiv.pars['dist_sus_with_syphilis'] = ss.normal(loc=1.5, scale=0.25) # TODO Data?
    hiv.pars['dist_trans_with_syphilis'] = ss.normal(loc=1.5, scale=0.025) # TODO Data? 
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
    hiv.pars['syphilis_prev'] = pd.DataFrame({"Years": tivec,
                                             "Value": (np.interp(tivec,
                                                         np.arange(1990, 2021+1, 1),
                                                         np.repeat(0.05, len(np.arange(1990, 2021+1, 1)))))})
    ####################################################################################################################
    # Treatment Data
    ####################################################################################################################
    ART_coverages_raw = pd.read_csv(datadir/f'world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.index.astype(int).tolist(),
                                                         (ART_coverages_raw.values / 100).tolist()))})

    hiv.pars['ART_coverages_df'] = ART_coverages_df

    ####################################################################################################################
    # Make demographic modules
    ####################################################################################################################

    fertility_rates = {'fertility_rate': pd.read_csv(datadir/f'{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(datadir/f'{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    ####################################################################################################################
    # Make people and networks
    ####################################################################################################################
    ss.set_seed(1)
    ppl = ss.People(n_agents, age_data=pd.read_csv(datadir/f'{location}_age.csv'))
    sexual = StructuredSexual()
    maternal = ss.MaternalNet()

    ####################################################################################################################
    # Products and testing
    ####################################################################################################################
    fsw_testing, other_testing, low_cd4_testing = get_testing_products()

    ####################################################################################################################
    # Sim
    ####################################################################################################################
    sim_kwargs = dict(
        dt=dt,
        total_pop=total_pop,
        start=1990,
        n_years=40,
        people=ppl,
        diseases=hiv,
        networks=ss.ndict(sexual, maternal),
        interventions=[fsw_testing,
                       other_testing,
                       low_cd4_testing,
                       ART(ART_coverages_df=ART_coverages_df,
                           duration_on_ART=ss.normal(loc=18, scale=5),  # https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-021-10464-x
                           art_efficacy=0.96),
                       validate_ART(disease='hiv',
                                    uids=save_agents,
                                    infect_uids_t=np.repeat(200, len(save_agents)),
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

    sim, output = run_hiv(location=location, total_pop=total_pop, dt=1 / 12, n_agents=int(1e4),
                          save_agents=save_agents)
    output.to_csv("HIV_output.csv")

    # Call method in validate_ART intervention:
    sim.get_interventions(validate_ART)[0].save_viral_histories(sim)
    viral_histories = pd.read_csv("viral_histories.csv", index_col=0)
    plot_viral_dynamics(viral_histories, save_agents)

    plot_hiv(output)

    sc.saveobj(f'sim_{location}.obj', sim)
