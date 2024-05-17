####### PLOT RESULTS
import matplotlib.pyplot as plt
import numpy as np
import stisim as ss
from stisim import interventions as ssi
import sciris as sc
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

def plot_cum_infections(sim, ax, disease):
    if sim.results[disease]["cum_infections"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["cum_infections"].low, sim.results[disease]["cum_infections"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["cum_infections"][:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " infections")

def plot_cum_diagnoses(sim, ax, disease):
    if sim.results[disease]["cum_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["cum_diagnoses"].low, sim.results[disease]["cum_diagnoses"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["cum_diagnoses"][:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " diagnoses")

def plot_cum_deaths(sim, ax, disease):
    if sim.results[disease]["cum_deaths"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["cum_deaths"].low, sim.results[disease]["cum_deaths"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["cum_deaths"][:], color="b", alpha=1)

    ax.set_title("Cumulative " + disease + " deaths")

def plot_new_infections(sim, ax, disease):
    if sim.results[disease]["new_infections"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["new_infections"].low, sim.results[disease]["new_infections"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["new_infections"][:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " infections")


def plot_new_diagnoses(sim, ax, disease):
    if sim.results[disease]["new_diagnoses"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["new_diagnoses"].low, sim.results[disease]["new_diagnoses"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["new_diagnoses"][:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " diagnoses")


def plot_new_deaths(sim, ax, disease):
    if sim.results[disease]["new_deaths"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["new_deaths"].low, sim.results[disease]["new_deaths"].high, **fill_args)
    ax.plot(sim.tivec, sim.results[disease]["new_deaths"][:], color="b", alpha=1)

    ax.set_title("Daily " + disease + " deaths")
def plot_prevalence(sim, ax, disease='hiv'):
    if sim.results[disease]["prevalence"].low is not None:
        fill_args = {"alpha": 0.3}
        ax.fill_between(sim.tivec, sim.results[disease]["prevalence"].low, sim.results[disease]["prevalence"].high, **fill_args)

    ax.plot(sim.tivec, sim.results[disease]["prevalence"][:], color="b", alpha=1)

    ax.set_title("Prevalence of HIV in the population")

def diagnostic_plots(sim, disease):

    # MAIN FIGURE
    fig, ax = plt.subplots(2, 2)

    fig.set_size_inches(10, 10)
    fig.tight_layout(pad=5.0)
    plot_cum_infections(sim, ax[0, 0], disease)
    plot_cum_diagnoses(sim, ax[0, 1], disease)
    #plot_cum_tests(sim, ax[1, 0])
    plot_cum_deaths(sim, ax[1, 0], disease)
    #plot_cum_sb(sim, ax[1, 0])
    plot_prevalence(sim, ax[1, 1], disease)

    fig.tight_layout()
    fig.canvas.manager.set_window_title("Key outputs")
    fig.savefig(disease + '_output_check', transparent=True)

def plot_hiv(sim_output, location='zimbabwe', total_pop=9980999, save='test'):
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
    n_infections_data = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=33).iloc[[1], 2:36]
    ax[0, 0].scatter(n_infections_data.columns, n_infections_data.values, label='Data - number of PLHIV', color='tab:red')
    ax[0, 0].plot(sim_output.index, sim_output['hiv.n_infected'], label='n infected')
    ax[0, 0].plot(sim_output.index, sim_output['hiv.n_diagnosed'], label='n diagnosed')
    ax[0, 0].plot(sim_output.index, sim_output['hiv.n_on_art'], label='n on ART')
    ax[0, 0].set_ylim([0, 6000000])
    ax[0, 0].set_ylabel('Mio.')
    ax[0, 0].legend()


    #####################################################################################################################
    # Cumulative infections, diagnosed and on ART and cumulative states
    #####################################################################################################################

    ax[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_infections'], label='Cumulative Infections')
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_diagnoses'], label='Cumulative Diagnoses')
    ax[0, 1].plot(sim_output.index, sim_output['hiv.cum_agents_on_art'], label='Cumulative on ART')
    ax[0, 1].set_ylim([0, 20000000])
    ax[0, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax[0, 1].set_ylabel('Mio.')
    ax[0, 1].legend()


    #####################################################################################################################
    # New Deaths and HIV Deaths
    #####################################################################################################################
    hiv_deaths_data = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=41).iloc[[1], 2:36]
    ax[1, 0].scatter(hiv_deaths_data.columns, hiv_deaths_data.values[0], color='tab:red', label='Data - yearly HIV deaths')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    ax[1, 0].plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['hiv.new_deaths'].sum(), color='tab:blue', label='Modelled - yearly HIV deaths')
    ax[1, 0].set_title('Deaths')
    ax[1, 0].plot(sim_output.index, sim_output['deaths.new'], color='tab:orange', label='New Deaths')
    ax[1, 0].plot(sim_output.index, sim_output['hiv.new_deaths'], color='tab:blue', label='New HIV Deaths')
    ax[1, 0].legend()
    ax[1, 0].set_ylim([0, 1000000])

    #####################################################################################################################
    # New Diagnoses, Infections and on ART
    #####################################################################################################################
    ax[1, 1].set_title('New Infections, Diagnoses and on ART')
    ax[1, 1].plot(sim_output.index, sim_output['hiv.new_infections'], label='New Infections')
    hiv_new_infections_data = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=17).iloc[[1], 2:36]
    ax[1, 1].scatter(hiv_new_infections_data.columns, hiv_new_infections_data.values[0],  color='tab:red', label='Data - new infections per year')
    ax[1, 1].plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['hiv.new_infections'].sum(), color='tab:blue', label='Modelled - new infections per year')

    ax[1, 1].plot(sim_output.index, sim_output['hiv.new_diagnoses'], label='New Diagnoses')
    ax[1, 1].plot(sim_output.index, sim_output['hiv.new_agents_on_art'], label='New on ART')
    ax[1, 1].legend()
    # ax[1, 1].set_ylim([0, 500000])

    #####################################################################################################################
    # HIV Prevalence
    #####################################################################################################################
    prevalence_data = pd.read_csv(f'data/world_bank_hiv_prevalence.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    ax[2, 0].scatter(prevalence_data.index.astype(int), prevalence_data.values, color='tab:red', label='Data (15-49)')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence'] * 100, label='General Population')
    ax[2, 0].set_title('HIV prevalence (%)')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_sw'] * 100, label='FSW')
    ax[2, 0].plot(sim_output.index, sim_output['hiv.prevalence_client'] * 100, label='Client')
    ax[2, 0].legend()
    ax[2, 0].set_ylim([0, 100])

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
    ax[2, 1].set_ylim([0, 100])

    #####################################################################################################################
    # ART Coverage
    #####################################################################################################################
    ART_coverages_raw = pd.read_csv(f'data/world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.index.astype(int).tolist(),
                                                         (ART_coverages_raw.values / 100).tolist()))})
    ART_coverages_raw = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=28).iloc[0:1, 2:43]
    pop_scale = total_pop / int(10e3)
    ART_coverages_df_numbers = pd.DataFrame({"Years": tivec,
                                             "Value": (np.interp(tivec,
                                                                 ART_coverages_raw.columns.values[~pd.isna(ART_coverages_raw.values)[0]].astype(int),
                                                                 ART_coverages_raw.values[~pd.isna(ART_coverages_raw.values)]) / pop_scale).astype(int)})

    ax[3, 0].plot(sim_output.index, 100 * (sim_output['hiv.n_on_art'] / sim_output['hiv.n_diagnosed']), label='Modelled')
    ax[3, 0].set_title('ART coverage (%)')
    ax[3, 0].scatter(ART_coverages_df["Years"], 100 * ART_coverages_df["Value"], color='tab:red', label="Data")
    ax[3, 0].legend()
    ax[3, 0].set_xlim([2000.0, max(sim_output.iloc[1:].index)])
    ax[3, 0].set_ylim([0, 100])

    ax[3, 1].plot(sim_output.index, sim_output['hiv.n_on_art'], label='Modelled')
    ax[3, 1].set_title('Number of people on ART (in Mio)')
    ax[3, 1].scatter(ART_coverages_df_numbers["Years"], ART_coverages_df_numbers["Value"] * pop_scale, color='tab:red', label="Data")
    ax[3, 1].legend()
    ax[3, 1].set_ylabel('Mio.')
    ax[3, 1].set_ylim([0, 5000000])
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
    #ax[4, 0].set_ylim([0, 100000])

    #####################################################################################################################
    # Yearly Tests
    #####################################################################################################################
    n_tests_data = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=1).iloc[[1], 2:36]
    # Calculate yearly tests in the model:
    ax[4, 1].scatter(n_tests_data.columns, n_tests_data.values[0], color='tab:red', label='Data')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['fsw_testing.new_screens'].sum(), marker='o', label='FSW')
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['other_testing.new_screens'].sum(), marker='o', label='General Population')
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['low_cd4_testing.new_screens'].sum(), marker='o', label='CD4 count <200')
    ax[4, 1].plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['low_cd4_testing.new_screens'].sum()
                  + sim_output.groupby(by='year')['other_testing.new_screens'].sum()
                  + sim_output.groupby(by='year')['fsw_testing.new_screens'].sum(), label='FSW + Other + Low CD4 count testing')
    #ax[4, 1].set_ylim([0, 4000000])
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
    population_data = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Population size', skiprows=72).iloc[0:1, 3:36]
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
    sc.savefig("figures/calibration_plots/hiv_" + str(save) + ".png", dpi=100)
    plt.close()