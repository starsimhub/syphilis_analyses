"""
Run hiv
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import sciris as sc
from io import StringIO
from starsim import utils as ssu
from stisim.networks import StructuredSexual
from stisim.products import Dx
from stisim.diseases.hiv import HIV
from stisim.interventions import ART, validate_ART, BaseTest

quick_run = False
ss.options['multirng'] = False

def get_testing_products(location):
    """
    Define HIV products and testing interventions
    """
    # Load HIV test data:
    hiv_testing_data = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=1)
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


def make_hiv_sim(seed, location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, ss_hiv_beta=0.95, maternal_hiv_beta=0.5,
                 init_prev=ss.bernoulli(p=0.15),
                 duration_on_ART=ss.lognorm_ex(mean=18, stdev=5),
                 cd4_start_dist=ss.lognorm_ex(mean=800, stdev=10),
                 risk_groups_f=ss.choice(a=3, p=np.array([0.85, 0.14, 0.01])),
                 risk_groups_m=ss.choice(a=3, p=np.array([0.78, 0.21, 0.01])),
                 p_pair_form=0.5,
                 conc={'f': [0.0001, 0.01, 0.1],
                                  'm': [0.01, 0.2, 0.5]},
                 **kwargs):
    """
    Make a sim with HIV
    """
    # Set the seed
    ss.set_seed(seed)
    ####################################################################################################################
    # HIV Params
    ####################################################################################################################
    hiv = HIV()
    hiv.pars['init_prev'] = init_prev
    hiv.pars['cd4_start_dist'] = cd4_start_dist
    hiv.pars['beta'] = {'structuredsexual': [ss_hiv_beta, ss_hiv_beta],
                        'maternal': [maternal_hiv_beta, 0.0]}

    hiv.pars['init_diagnosed'] = ss.bernoulli(p=0.15)  # Proportion of initially infected agents who start out as diagnosed
    hiv.pars['primary_acute_inf_dur'] = 2.9  # in months
    hiv.pars['transmission_sd'] = 0.0  # Standard Deviation of normal distribution for randomness in transmission.
    ####################################################################################################################
    # Add Syphilis
    ####################################################################################################################
    hiv.pars['dist_sus_with_syphilis'] = ss.lognorm_ex(mean=2, stdev=0.25)  # TODO Data?
    hiv.pars['dist_trans_with_syphilis'] = ss.lognorm_ex(mean=2, stdev=0.025)  # TODO Data?
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
    hiv.pars['syphilis_prev'] = pd.DataFrame({"Years": tivec,
                                              "Value": (np.interp(tivec,
                                                                  np.arange(1990, 2021 + 1, 1),
                                                                  np.repeat(0.05, len(np.arange(1990, 2021 + 1, 1)))))})
    ####################################################################################################################
    # Treatment Data
    ####################################################################################################################
    ART_coverages_raw = pd.read_csv(f'data/world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
    tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
    ART_coverages_df = pd.DataFrame({"Years": tivec,
                                     "Value": (np.interp(tivec,
                                                         ART_coverages_raw.index.astype(int).tolist(),
                                                         (ART_coverages_raw.values / 100).tolist()))})

    hiv.pars['ART_coverages_df'] = ART_coverages_df

    ####################################################################################################################
    # Add coverage of pregnant women who receive ARV for PMTCT
    ####################################################################################################################
    pmtct_coverages = pd.read_excel(f'data/{location}_20230725.xlsx', sheet_name='Optional indicators', skiprows=81).iloc[[1], 3:36]
    pmtct_coverages[2006] = 0  # Hard coded
    years = np.arange(np.min(pmtct_coverages.columns.values[~pd.isna(pmtct_coverages.values)[0]].astype(int)),
                      np.max(pmtct_coverages.columns.values[~pd.isna(pmtct_coverages.values)[0]].astype(int)) + 1 / 12, 1 / 12)
    pmtct_coverages_df = pd.DataFrame({"Years": years,
                                       "Value": (np.interp(years,
                                                           pmtct_coverages.columns.values[~pd.isna(pmtct_coverages.values)[0]].astype(int),
                                                           pmtct_coverages.values[~pd.isna(pmtct_coverages.values)]))})
    hiv.pars['maternal_beta_pmtct_df'] = pd.DataFrame({"Years": years,
                                                       "Value": (1 - pmtct_coverages_df['Value']) * hiv.pars.beta['maternal'][0]})

    ####################################################################################################################
    # Make demographic modules
    ####################################################################################################################

    fertility_rates = {'fertility_rate': pd.read_csv(f'data/{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(f'data/{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    ####################################################################################################################
    # Make people and networks
    ####################################################################################################################
    ppl = ss.People(n_agents, age_data=pd.read_csv(f'data/{location}_age.csv'))
    sexual = StructuredSexual()
    sexual.pars.risk_groups_f = risk_groups_f
    sexual.pars.risk_groups_m = risk_groups_m
    sexual.pars.p_pair_form = p_pair_form
    sexual.pars.f0_conc.pars.lam = conc['f'][0]
    sexual.pars.f1_conc.pars.lam = conc['f'][1]
    sexual.pars.f2_conc.pars.lam = conc['f'][2]
    sexual.pars.m0_conc.pars.lam = conc['m'][0]
    sexual.pars.m1_conc.pars.lam = conc['m'][1]
    sexual.pars.m2_conc.pars.lam = conc['m'][2]

    maternal = ss.MaternalNet()

    ####################################################################################################################
    # Products and testing
    ####################################################################################################################
    fsw_testing, other_testing, low_cd4_testing = get_testing_products(location)

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
                           duration_on_ART=duration_on_ART,  # https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-021-10464-x
                           art_efficacy=0.96)],
        demographics=[pregnancy, death])

    sim = ss.Sim(**sim_kwargs)
    sim.initialize()

    return sim

