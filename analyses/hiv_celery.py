# FUNCTIONS TO RUN MODELS ON CLUSTER
import pandas as pd
from celery import Celery
import sciris as sc
from starsim import utils
from celery.signals import after_setup_task_logger
import logging
from hiv_main import make_hiv_sim
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
        sim = make_hiv_sim(seed,
                           location=location,
                           total_pop=total_pop,
                           dt=dt,
                           n_agents=n_agents,
                           **kwargs)
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
        dataframe['ss_hiv_beta_m2f'] = kwargs['ss_hiv_beta_m2f']
        dataframe['ss_hiv_beta_fm2'] = kwargs['ss_hiv_beta_f2m']
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

    final_day_quantities = ["n_alive", "hiv.n_infected",
                            "hiv.n_diagnosed",
                            "hiv.prevalence",
                            "hiv.new_diagnoses",
                            "hiv.new_infections",
                            "hiv.new_deaths"]
    for quantity in final_day_quantities:
        summary[quantity] = df.iloc[-1][quantity]

    if return_sim:
        return df, summary, sim
    else:
        return df, summary