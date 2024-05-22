# FUNCTIONS TO RUN MODELS ON CLUSTER
import pandas as pd
from celery import Celery
import starsim as ss
import sciris as sc
from starsim import utils
from stisim import multisim as ssm
from celery.signals import after_setup_task_logger
import logging
from stisim.diseases.hiv_main import make_hiv_sim
import numpy as np
utils.git_info = lambda: None  # Disable this function to increase performance slightly


import os

broker = os.getenv("GAVI_OB_REDIS_URL", "redis://127.0.0.1:6379")

# Create celery app
celery = Celery("stisim-hiv")
celery.conf.broker_url = broker
celery.conf.result_backend = broker
celery.conf.task_default_queue = "stisim-hiv"
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

    summary = sc.dcp(kwargs)
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