"""
Run hiv
"""

# Additions to handle numpy multithreading
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)
# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc
from io import StringIO
import seaborn as sns
import stisim as sti
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# Run settings
debug = False
n_trials    = [1000, 2][debug]  # How many trials to run for calibration
n_workers   = [40, 2][debug]    # How many cores to use
storage     = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations


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

    fig, axes = plt.subplots(5, 2, figsize=(20, 15))
    axes = axes.ravel()
    data = pd.read_csv(sti.data/f'{location}_calib.csv')

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
    ax.scatter(data.year, data.hiv_prev*100, color='tab:red', label='Overall')
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
    ART_coverages_raw = pd.read_csv(sti.data/f'world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.index.astype(int).tolist(),
                                                         (ART_coverages_raw.values / 100).tolist()))})
    ART_coverages_raw = pd.read_excel(sti.data/f'{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=28).iloc[0:1, 2:43]
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
    n_tests_data = pd.read_excel(sti.data/f'{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=1).iloc[[1], 2:36]
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
    sc.savefig("figures/hiv_plots.png", dpi=100)


def get_testing_products():
    """
    Define HIV products and testing interventions
    """
    # Load HIV test data:
    hiv_testing_data = pd.read_excel(sti.data/f'{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=1)
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
    # FSW Testing
    ####################################################################################################################

    # Eligible for testing are FSW agents who haven't been diagnosed or treated yet
    fsw_eligible = lambda sim: sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    fsw_testing = sti.HIVTest(
        years=tivec,
        test_prob_data=FSW_prop,
        name='fsw_testing',
        eligibility=fsw_eligible,
        label='fsw_testing',
       )

    ####################################################################################################################
    # Remaining population testing
    ####################################################################################################################

    # Eligible for testing are non-FSW agents who haven't been diagnosed or treated yet
    other_eligible = lambda sim: ~sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    other_testing = sti.HIVTest(
        years=tivec,
        test_prob_data=other_prop,
        name='other_testing',
        eligibility=other_eligible,
        label='other_testing',
    )

    ####################################################################################################################
    # Low CD4 count testing
    ####################################################################################################################
    # Eligible for testing are agents, who haven't been diagnosed yet and whose CD4 count is below 200.
    low_cd4_eligible = lambda sim: (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed
    low_cd4_testing = sti.HIVTest(
        years=tivec,
        test_prob_data=low_cd4count_prop,
        name='low_cd4_testing',
        eligibility=low_cd4_eligible,
        label='low_cd4_testing',
    )

    return fsw_testing, other_testing, low_cd4_testing


def make_hiv_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, save_agents=np.array([0])):
    """
    Make a sim with HIV
    """
    ####################################################################################################################
    # HIV Params
    ####################################################################################################################
    hiv = sti.HIV(
        beta={'structuredsexual': [0.05, 0.025], 'maternal': [0.05, 0.]},
        init_prev=0.07,
        beta_m2f=0.05,
        beta_f2m=0.025,
        beta_m2c=0.025,
        init_diagnosed=0.0,  # Proportion of initially infected agents who start out as diagnosed
    )

    ####################################################################################################################
    # Make demographic modules
    ####################################################################################################################
    fertility_rates = {'fertility_rate': pd.read_csv(sti.data/f'{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(sti.data/f'{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    ####################################################################################################################
    # Make people and networks
    ####################################################################################################################
    ss.set_seed(1)
    ppl = ss.People(n_agents, age_data=pd.read_csv(sti.data/f'{location}_age.csv'))
    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    ####################################################################################################################
    # Testing and treatment
    ####################################################################################################################
    n_art = pd.read_csv(sti.data/'zimbabwe_art.csv').set_index('year')
    fsw_testing, other_testing, low_cd4_testing = get_testing_products()
    art = sti.ART(
        coverage_data=n_art,
        dur_on_art=ss.lognorm_ex(15, 3),  # https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-021-10464-x
    )

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
        interventions=[
            fsw_testing,
            other_testing,
            low_cd4_testing,
            art,
        ],
        demographics=[pregnancy, death])

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


def run_calibration(n_trials=None, n_workers=None, do_save=True):

    # Define the calibration parameters
    calib_pars = dict(
        diseases = dict(
            hiv = dict(
                beta_m2f = [0.05, 0.01, 0.10],
                beta_f2m = [0.025, 0.005, 0.05],
            ),
        ),
        networks = dict(
            structuredsexual = dict(
                prop_f1 = [0.15, 0.1, 0.45],
                prop_m1 = [0.21, 0.15, 0.5],
                f1_conc = [0.01, 0.005, 0.1],
                m1_conc = [0.01, 0.005, 0.1],
                p_pair_form = [0.5, 0.4, 0.9],
            ),
        ),
    )

    # Make the sim
    sim_kwargs = make_hiv_sim(location='zimbabwe', total_pop=9980999, dt=1/12, n_agents=10e3)
    sim = ss.Sim(**sim_kwargs)

    # Weight the data
    weights = {
        'n_alive': 1,
        'hiv.prevalence': 1,
        'hiv.n_infected': 1,
        'hiv.new_infections': 1,
        'hiv.new_deaths': 1,
        }

    # Make the calibration
    calib = sti.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        datafile=sti.data/'zimbabwe_calib.csv',
        weights=weights,
        total_trials=2, n_workers=1, die=True
    )

    calib.calibrate()
    filename = f'zim_calib{filestem}'
    sc.saveobj(f'results/{filename}.obj', calib)
    print(f'Best pars are {calib.best_pars}')

    return sim, calib


if __name__ == '__main__':
    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]

    # sim, output = run_hiv(location=location, total_pop=total_pop, dt=1 / 12, n_agents=int(1e4))
    # output.to_csv("HIV_output.csv")

    # Calibration
    sim, calib = run_calibration(n_trials=n_trials, n_workers=n_workers)



    plot_hiv(output)

    sc.saveobj(f'sim_{location}.obj', sim)
