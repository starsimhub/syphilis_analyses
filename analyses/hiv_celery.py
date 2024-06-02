# FUNCTIONS TO RUN MODELS ON CLUSTER
import pandas as pd
from celery import Celery
import sciris as sc
from starsim import utils
from celery.signals import after_setup_task_logger
import logging
from stisim.networks import StructuredSexual
import starsim as ss
import stisim as sti
from stisim.products import Dx
from stisim.diseases.hiv import HIV
import numpy as np
from stisim.interventions import ART, HIVTest
# from hiv_main import make_hiv_sim
utils.git_info = lambda: None  # Disable this function to increase performance slightly


import os

broker = os.getenv("GAVI_OB_REDIS_URL", "redis://127.0.0.1:6379")

# Create celery app
celery = Celery("stisim_hiv")
celery.conf.broker_url = broker
celery.conf.result_backend = broker
celery.conf.task_default_queue = "stisim_hiv"
celery.conf.accept_content = ["pickle", "json"]
celery.conf.task_serializer = "pickle"
celery.conf.result_serializer = "pickle"
celery.conf.worker_prefetch_multiplier = 1
celery.conf.task_acks_late = True  # Allow other servers to pick up tasks in case they are faster
celery.conf.worker_max_tasks_per_child = 5
celery.conf.worker_max_memory_per_child = 3000000


# Quieter tasks
@after_setup_task_logger.connect
def setup_task_logger(logger, *args, **kwargs):
    logger.setLevel(logging.WARNING)

@celery.task()
def run_sim(seed, location, total_pop, dt, n_agents, disease='hiv', return_sim=False, **kwargs):
    """
    Run the calibration sim (use run_sim for non-calibration runs)

    Args:
        beta:
        seed:

    Returns:

    """
    if disease.lower() == 'hiv':
        """
            Make a sim with HIV
            """
        # Set the seed
        ss.set_seed(seed)
        ####################################################################################################################
        # HIV Params
        ####################################################################################################################
        hiv = HIV()
        hiv.pars['init_prev'] = kwargs['init_prev']
        hiv.pars['cd4_start_dist'] = kwargs['cd4_start_dist']
        hiv.pars['beta'] = {'structuredsexual': [kwargs['ss_hiv_beta'], kwargs['ss_hiv_beta']],
                            'maternal': [kwargs['maternal_hiv_beta'], 0.0]}

        hiv.pars['init_diagnosed'] = ss.bernoulli(p=0.15)  # Proportion of initially infected agents who start out as diagnosed
        hiv.pars['primary_acute_inf_dur'] = 2.9  # in months
        hiv.pars['transmission_sd'] = 0.0  # Standard Deviation of normal distribution for randomness in transmission.
        ####################################################################################################################
        # Add Syphilis
        ####################################################################################################################
        # syphilis = sti.SyphilisPlaceholder(prevalence=None)

        ####################################################################################################################
        # Treatment Data
        ####################################################################################################################
        ART_coverages_raw = pd.read_csv(sti.data / f'world_bank_art_coverages.csv', skiprows=4).set_index('Country Name').loc[location.capitalize()].dropna()[3:]
        tivec = np.arange(start=1990, stop=2021 + 1 / 12, step=1 / 12)
        ART_coverages_df = pd.DataFrame({"Years": tivec,
                                         "Value": (np.interp(tivec,
                                                             ART_coverages_raw.index.astype(int).tolist(),
                                                             (ART_coverages_raw.values / 100).tolist()))})

        hiv.pars['ART_coverages_df'] = ART_coverages_df

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
        sexual.pars.risk_groups_f = kwargs['risk_groups_f']
        sexual.pars.risk_groups_m = kwargs['risk_groups_m']
        sexual.pars.p_pair_form =kwargs['p_pair_form']
        sexual.pars.f0_conc.pars.lam = kwargs['conc']['f'][0]
        sexual.pars.f1_conc.pars.lam =  kwargs['conc']['f'][1]
        sexual.pars.f2_conc.pars.lam =  kwargs['conc']['f'][2]
        sexual.pars.m0_conc.pars.lam =  kwargs['conc']['m'][0]
        sexual.pars.m1_conc.pars.lam =  kwargs['conc']['m'][1]
        sexual.pars.m2_conc.pars.lam =  kwargs['conc']['m'][2]

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
        fsw_eligible = lambda sim: sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
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
        other_eligible = lambda sim: ~sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art
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
        low_cd4_eligible = lambda sim: (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed
        low_cd4_testing = HIVTest(
            years=tivec,
            test_prob_data=low_cd4count_prop,
            name='low_cd4_testing',
            eligibility=low_cd4_eligible,
            label='low_cd4_testing',
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
            diseases=[hiv],
            networks=ss.ndict(sexual, maternal),
            interventions=[
                fsw_testing,
                other_testing,
                low_cd4_testing,
                ART(ART_coverages_df=ART_coverages_df,
                    dur_on_art=ss.normal(loc=18, scale=5))],
            demographics=[pregnancy, death])

        sim = ss.Sim(**sim_kwargs)
        sim.initialize()
    else:
        print(disease + ' is not a supported disease yet.')
        return

    sim.run()

    for result in sim.results:
        # Convert to multisim result
        result
                        
    df_res = sim.export_df() #ssm.export_results(sim)    #ss.export_df
    df = pd.DataFrame.from_dict(df_res)
    summary = {}

    for dataframe in [df, summary]:
        dataframe['hiv_beta'] = kwargs['ss_hiv_beta']
        dataframe['maternal_hiv_beta'] = kwargs['maternal_hiv_beta']
        dataframe['init_prev'] = kwargs['init_prev'].pars.p
        dataframe['duration_on_ART'] = kwargs['duration_on_ART'].pars.mean
        dataframe['cd4_start_dist'] = kwargs['cd4_start_dist'].pars.mean
        dataframe['f0_prob'] = kwargs['risk_groups_f'].pars.p[0]
        dataframe['f1_prob'] = kwargs['risk_groups_f'].pars.p[1]
        dataframe['f2_prob'] = kwargs['risk_groups_f'].pars.p[2]
        dataframe['m0_prob'] = kwargs['risk_groups_m'].pars.p[0]
        dataframe['m1_prob'] = kwargs['risk_groups_m'].pars.p[1]
        dataframe['m2_prob'] = kwargs['risk_groups_m'].pars.p[2]
        dataframe['f0_conc'] = kwargs['conc']['f'][0]
        dataframe['f1_conc'] = kwargs['conc']['f'][1]
        dataframe['f2_conc'] = kwargs['conc']['f'][2]
        dataframe['m0_conc'] = kwargs['conc']['m'][0]
        dataframe['m1_conc'] = kwargs['conc']['m'][1]
        dataframe['m2_conc'] = kwargs['conc']['m'][2]
        dataframe['p_pair_form'] = kwargs['p_pair_form'].pars.p

    #summary = {} # sc.dcp(kwargs)
    summary["seed"] = seed

    final_day_quantities = ["cum_diagnoses",
                            "cum_infections",
                            "cum_deaths"]
    for quantity in final_day_quantities:
        summary[quantity] = df.iloc[-1][disease.lower() + "." + quantity]

    if return_sim:
        return df, summary, sim
    else:
        return df, summary