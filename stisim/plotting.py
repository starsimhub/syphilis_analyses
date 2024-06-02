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
import stisim as sti

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

def plot_hiv(sim_output, save, location="zimbabwe"):
    """
    Plot:
    - new infections
    - cum infections
    - new deaths
    - cum deaths
    - HIV prevalence, HIV prevalence in FSW
    - n on ART
    """

    fig, axes = plt.subplots(5, 2, figsize=(20, 15))
    axes = axes.ravel()
    data = pd.read_csv(sti.data / f'{location}_calib.csv')

    #####################################################################################################################
    # Population
    #####################################################################################################################
    ax = axes[0]
    ax.scatter(data.year, data.pop_size, color='tab:red', label='Data')
    ax.plot(sim_output.index, sim_output['n_alive'], label='Modelled')
    ax.set_title('Population')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax.set_ylabel('Mio.')
    ax.legend()

    #####################################################################################################################
    # PLHIV: total, diagnosed, and treated
    #####################################################################################################################
    ax = axes[1]
    ax.scatter(data.year, data.plhiv, label='PLHIV (data)', color='tab:red')
    ax.plot(sim_output.index, sim_output['hiv.n_infected'], label='PLHIV (modelled)')
    ax.plot(sim_output.index, sim_output['hiv.n_diagnosed'], label='Diagnosed PLHIV')
    ax.plot(sim_output.index, sim_output['hiv.n_on_art'], label='Treated PLHIV')
    ax.set_title('PLHIV: total, diagnosed, and treated')
    ax.set_ylabel('Mio.')
    ax.legend()

    #####################################################################################################################
    # HIV Prevalence
    #####################################################################################################################
    ax = axes[2]
    ax.scatter(data.year, data.hiv_prev * 100, color='tab:red', label='Overall')
    ax.plot(sim_output.index, sim_output['hiv.prevalence'] * 100, label='Overall')
    ax.set_title('HIV prevalence (%)')
    ax.plot(sim_output.index, sim_output['hiv.prevalence_sw'] * 100, label='FSW')
    ax.plot(sim_output.index, sim_output['hiv.prevalence_client'] * 100, label='Client')
    ax.legend()

    ax = axes[3]
    ax.plot(sim_output.index, sim_output['hiv.prevalence_risk_group_0_female'] * 100, color='tab:blue', label='Risk Group 0 - Female')
    ax.plot(sim_output.index, sim_output['hiv.prevalence_risk_group_2_female'] * 100, color='tab:orange', label='Risk Group 2- Female')
    ax.plot(sim_output.index, sim_output['hiv.prevalence_risk_group_0_male'] * 100, color='tab:blue', linestyle='--', label='Risk Group 0 - Male')
    ax.plot(sim_output.index, sim_output['hiv.prevalence_risk_group_1_male'] * 100, color='tab:green', linestyle='--', label='Risk Group 1- Male')
    ax.plot(sim_output.index, sim_output['hiv.prevalence_risk_group_2_male'] * 100, color='tab:orange', linestyle='--', label='Risk Group 2- Male')
    blue_patch = mpatches.Patch(color='tab:blue', label='Risk Group 0')
    green_patch = mpatches.Patch(color='tab:green', label='Risk Group 1')
    orange_patch = mpatches.Patch(color='tab:orange', label='Risk Group 2')
    females = Line2D([0], [0], label='females', color='k', linestyle='-')
    males = Line2D([0], [0], label='males', color='k', linestyle='--')
    ax.legend(handles=[blue_patch, green_patch, orange_patch, females, males])
    ax.set_title('HIV prevalence (%) by Risk Group and gender')

    #####################################################################################################################
    # HIV Deaths
    #####################################################################################################################
    ax = axes[4]
    ax.scatter(data.year, data.new_deaths, color='tab:red', label='Data')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    ax.plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['hiv.new_deaths'].sum(), color='tab:blue', label='Modelled')
    ax.set_title('Deaths')
    ax.legend()

    #####################################################################################################################
    # New infections
    #####################################################################################################################
    ax = axes[5]
    ax.scatter(data.year, data.new_infections, color='tab:red', label='Data')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    ax.plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['hiv.new_infections'].sum(), color='tab:blue', label='Modelled')
    ax.set_title('New infections')
    ax.legend()

    #####################################################################################################################
    # ART Coverage
    #####################################################################################################################
    ART_coverages_raw = pd.read_csv(sti.data / f'world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.index.astype(int).tolist(),
                                                         (ART_coverages_raw.values / 100).tolist()))})
    ART_coverages_raw = pd.read_excel(sti.data / f'{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=28).iloc[0:1, 2:43]
    pop_scale = total_pop / int(10e3)
    ART_coverages_df_numbers = pd.DataFrame({"Years": tivec,
                                             "Value": (np.interp(tivec,
                                                                 ART_coverages_raw.columns.values[~pd.isna(ART_coverages_raw.values)[0]].astype(int),
                                                                 ART_coverages_raw.values[~pd.isna(ART_coverages_raw.values)]) / pop_scale).astype(int)})
    ax = axes[6]
    ax.plot(sim_output.index, sim_output['hiv.n_on_art'] / sim_output['hiv.n_infected'], label='Modelled')
    ax.set_title('ART coverage (%)')
    ax.scatter(ART_coverages_df["Years"], ART_coverages_df["Value"], color='tab:red', label="Data")
    ax.legend()
    ax.set_xlim([2000.0, max(sim_output.iloc[1:].index)])

    ax = axes[7]
    ax.plot(sim_output.index, sim_output['hiv.n_on_art'], label='Modelled')
    ax.set_title('Number of people on ART (in Mio)')
    ax.scatter(ART_coverages_df_numbers["Years"], ART_coverages_df_numbers["Value"] * pop_scale, color='tab:red', label="Data")
    ax.legend()
    ax.set_ylabel('Mio.')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax.set_xlim([2000.0, max(sim_output.iloc[1:].index)])

    #####################################################################################################################
    # New tested
    #####################################################################################################################
    ax = axes[8]
    ax.plot(sim_output.index, sim_output['fsw_testing.new_tests'], color='tab:orange', label='FSW')
    ax.plot(sim_output.index, sim_output['other_testing.new_tests'], color='tab:blue', label='General Population')
    ax.plot(sim_output.index, sim_output['low_cd4_testing.new_tests'], color='tab:green', label='CD4 count <200')
    ax.plot(sim_output.index, sim_output['fsw_testing.new_diagnoses'], color='tab:orange', linestyle='--', label='FSW')
    ax.plot(sim_output.index, sim_output['other_testing.new_diagnoses'], color='tab:blue', linestyle='--', label='General Population')
    ax.plot(sim_output.index, sim_output['low_cd4_testing.new_diagnoses'], color='tab:green', linestyle='--', label='CD4 count <200')
    ax.set_title('New tests and diagnoses')
    orange_patch = mpatches.Patch(color='tab:orange', label='FSW')
    blue_patch = mpatches.Patch(color='tab:blue', label='General Population')
    green_patch = mpatches.Patch(color='tab:green', label='CD4 count <200')
    screened = Line2D([0], [0], label='new tests', color='k', linestyle='-')
    screens = Line2D([0], [0], label='new diagnoses', color='k', linestyle='--')
    ax.legend(handles=[blue_patch, green_patch, orange_patch, screened, screens])

    #####################################################################################################################
    # Yearly Tests
    #####################################################################################################################
    ax = axes[9]
    n_tests_data = pd.read_excel(sti.data / f'{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=1).iloc[[1], 2:36]
    # Calculate yearly tests in the model:
    ax.scatter(n_tests_data.columns, n_tests_data.values[0], color='tab:red', label='Data')
    sim_output['year'] = np.floor(np.round(sim_output.index, 1)).astype(int)
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['fsw_testing.new_screens'].sum(), marker='o', label='FSW')
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['other_testing.new_screens'].sum(), marker='o', label='General Population')
    # ax[4, 1].scatter(np.unique(sim_output['year']), sim_output.groupby(by='year')['low_cd4_testing.new_screens'].sum(), marker='o', label='CD4 count <200')
    ax.plot(np.unique(sim_output['year']), sim_output.groupby(by='year')['low_cd4_testing.new_tests'].sum()
            + sim_output.groupby(by='year')['other_testing.new_tests'].sum()
            + sim_output.groupby(by='year')['fsw_testing.new_tests'].sum(), label='FSW + Other + Low CD4 count testing')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x * 1e-6:0.1f}'))
    ax.set_title('Yearly Tests (in Mio)')
    ax.legend()

    fig.tight_layout()
    # plt.show()
    sc.savefig("figures/calibration_plots/hiv_plots" + str(save) + ".png", dpi=100)
    plt.close()