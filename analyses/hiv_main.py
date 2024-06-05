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
from stisim.interventions import ART, HIVTest
import stisim as sti
from functools import partial
quick_run = False
ss.options['multirng'] = False

def make_hiv_sim(seed, location='zimbabwe', total_pop=100e6, dt=1, n_agents=500,
                 ss_hiv_beta_m2f=0.5, ss_hiv_beta_f2m=0.5, maternal_hiv_beta=0.5,
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
    hiv.pars['beta_m2f'] = ss_hiv_beta_m2f
    hiv.pars['beta_f2m'] = ss_hiv_beta_f2m
    hiv.pars['beta_m2c'] = maternal_hiv_beta

    ####################################################################################################################
    # Add Syphilis
    ####################################################################################################################
    syphilis = sti.SyphilisPlaceholder(prevalence=None)

    ####################################################################################################################
    # Make demographic modules
    ####################################################################################################################
    fertility_rates = {'fertility_rate': pd.read_csv(sti.data / f'{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(sti.data / f'{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    ####################################################################################################################
    # Make people and networks
    ####################################################################################################################
    ppl = ss.People(n_agents, age_data=pd.read_csv(sti.data / f'{location}_age.csv'))
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
    hiv_testing_data = pd.read_excel(sti.data / f'{location}_20230725.xlsx', sheet_name='Testing & treatment', skiprows=1)
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
    #global fsw_eligibility
    def fsw_eligibility(sim):
       return sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    # fsw_eligible = lambda sim: sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    fsw_eligible = partial(fsw_eligibility)
    fsw_testing = HIVTest(
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
    #global other_eligibility
    def other_eligibility(sim):
        return sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    #other_eligible = lambda sim: ~sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
    other_eligible = partial(other_eligibility)
    other_testing = HIVTest(
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
    global low_cd4_eligibility
    def low_cd4_eligibility(sim):
        return (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed

    low_cd4_eligible = partial(low_cd4_eligibility)
    #low_cd4_eligible = lambda sim: (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed
    low_cd4_testing = HIVTest(
        years=tivec,
        test_prob_data=low_cd4count_prop,
        name='low_cd4_testing',
        eligibility=low_cd4_eligible,
        label='low_cd4_testing',
    )

    ####################################################################################################################
    # Testing and treatment
    ####################################################################################################################
    n_art = pd.read_csv(sti.data/'zimbabwe_art.csv').set_index('year')
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
        diseases=[hiv, syphilis],
        networks=ss.ndict(sexual, maternal),
        interventions=[
            fsw_testing,
            other_testing,
            low_cd4_testing,
            art],
        demographics=[pregnancy, death])

    sim = ss.Sim(**sim_kwargs)
    sim.initialize()

    return sim

