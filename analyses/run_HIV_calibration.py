"""
Run hiv
"""

# %% Imports and settings
import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import starsim as ss
from hiv_celery import run_sim
from celery import group
from hiv_celery import celery
from starsim import Samples
import sciris as sc
from stisim.plotting import plot_hiv
import pandas as pd
from socket import error as SocketError
import errno
import urllib.request

quick_run = False
ss.options['multirng'] = False
debug_mode = False
result_dir = Path("results/Calibration")
to_run = []
sd_hiv_beta = 0.0025


def get_betas(mean_beta, std_beta, nruns=None):
    seeds = np.arange(nruns)
    betas = np.random.normal(mean_beta, std_beta, nruns)
    return seeds, betas


def run_scenario(kwargs, filter=False):
    nruns = kwargs['nruns']
    del kwargs['nruns']

    seeds, hiv_betas = get_betas(kwargs['ss_hiv_beta'], sd_hiv_beta, nruns)
    description_baseline = kwargs['save_as']

    if not hasattr(thread_local, "pbar"):
        thread_local.pbar = tqdm(total=len(seeds))
    pbar = thread_local.pbar
    pbar.set_description(description_baseline)
    pbar.n = 0
    pbar.refresh()
    pbar.unpause()

    fname = description_baseline

    if (result_dir / fname).exists():
        return
    tasks = []
    for seed in seeds:
        kwargs['ss_hiv_beta'] = hiv_betas[seed]
        tasks.append(run_sim.s(seed, **kwargs))

    job = group(*tasks)
    result = job.apply_async()
    ready = False

    while not ready:
        time.sleep(1)
        n_ready = sum(int(result.ready()) for result in result.results)
        ready = n_ready == len(seeds)
        pbar.n = n_ready
        if pbar.n == 0:
            pbar.reset(total=len(seeds))
        else:
            pbar.refresh()

    if result.successful():
        outputs = result.join()
        if filter:
            Samples.new(result_dir, outputs, ["seed"], fname="calibration_filtered.zip")
        else:
            Samples.new(result_dir, outputs, ["seed"], fname=fname + '.zip')
    else:
        pbar.set_description("baseline ERROR")
        for x in result.results:
            if x.failed():
                with open(result_dir / f"error_{x.id}.txt", "w") as log:
                    log.write(str(x.__dict__))

    try:
        time.sleep(2)
        result.forget()

    except:
        # if err.errno != errno.ECONNRESET:
        # The error is NOT a ConnectionResetError
        #    raise
        time.sleep(10)
        print('Connection Error. Trying again.')
        result.forget()  # Try again?

    return True


if __name__ == '__main__':
    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]

    # Load calibration csv:
    calibration_scenarios = pd.read_csv("calibration_scenarios.csv")
    ss_hiv_betas = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    maternal_hiv_betas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    dur_on_ART_means = [18]
    mean_initial_cd4s = [400, 600, 800, 1000]
    risk_groups_f_probs = [np.array([0.85, 0.14, 0.01])]
    risk_groups_m_probs = [np.array([0.78, 0.21, 0.01])]
    p_pair_forms = [0.5]
    concs = [{'f': [0.0001, 0.01, 0.1],
              'm': [0.01, 0.2, 0.5]}]
    idx = 0
    for ss_hiv_beta in ss_hiv_betas:
        for maternal_hiv_beta in maternal_hiv_betas:
            for dur_on_ART_mean in dur_on_ART_means:
                for mean_initial_cd4 in mean_initial_cd4s:
                    for risk_groups_f_prob in risk_groups_f_probs:
                        for risk_groups_m_prob in risk_groups_m_probs:
                            for p_pair_form in p_pair_forms:
                                for conc in concs:
                                    init_prev = 0.15
                                    ss_hiv_beta = ss_hiv_beta
                                    maternal_hiv_beta = maternal_hiv_beta
                                    mean_duration_onART = dur_on_ART_mean
                                    mean_initial_cd4 = mean_initial_cd4
                                    calibration_name = 'scenario_' + str(idx)

                                    kwargs = {'location': location,
                                              'total_pop': total_pop,
                                              'dt': 1 / 12,
                                              'n_agents': int(1e4),
                                              'init_prev': ss.bernoulli(p=init_prev),
                                              'cd4_start_dist': ss.lognorm_ex(mean=mean_initial_cd4, stdev=10),
                                              'ss_hiv_beta': ss_hiv_beta,
                                              'risk_groups_f': ss.choice(a=3, p=risk_groups_f_prob),
                                              'risk_groups_m': ss.choice(a=3, p=risk_groups_m_prob),
                                              'p_pair_form': ss.bernoulli(p=p_pair_form),
                                              'maternal_hiv_beta': maternal_hiv_beta,
                                              'duration_on_ART': ss.lognorm_ex(mean=mean_duration_onART, stdev=5),
                                              'conc': conc,
                                              'return_sim': False,
                                              'save_as': calibration_name}

                                    to_run.append(kwargs)
                                    idx += 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--nruns", default=10, type=int, help="Number of seeds to run per scenario")
    parser.add_argument("--celery", default=False, type=bool, help="If True, use Celery for parallelization")

    args = parser.parse_args()
    thread_local = threading.local()

    if debug_mode:
        # Use debug mode to run the full sampling over seeds, but without Celery
        print('TODO')
    elif args.celery:
        futures = []
        result_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(to_run), desc=f"Total progress") as pbar:
            pbar.n = 0
            pbar.refresh()
            pbar.unpause()

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:

                for i, run_args in enumerate(to_run):
                    run_args['nruns'] = args.nruns
                    futures.append(executor.submit(run_scenario, run_args))
                    if i == 0:
                        time.sleep(5)

                while True:

                    done = [x for x in futures if x.done()]

                    for result in done:
                        if result.exception():
                            [x.cancel() for x in futures]
                            celery.control.purge()
                            celery.control.shutdown()
                            raise result.exception()

                    pbar.n = len(done)
                    pbar.refresh()
                    if len(done) == len(futures):
                        break
                    time.sleep(5)

        # Shut down the workers
        celery.control.shutdown()

    else:
        n_runs = 2
        if n_runs > 1:
            # outputs = sc.parallelize(run_sim, iterarg=[seed for seed in range(n_runs)], kwargs=kwargs)  # Run them in parallel
            outputs = sc.parallelize(run_sim, seed=0, iterkwargs=to_run)
            for idx, output in enumerate(outputs):
                sim_output = output[0]
                save = str(to_run[idx]["save_as"])
                sim_output.to_csv(location + "_calibration//" + save + ".csv")
                # plot_hiv(sim_output, location='zimbabwe', total_pop=9980999, save=save)
        else:
            with sc.Timer(label="Run model") as _:
                outputs = [run_sim(0, **kwargs)]

        # outputs[0][0].to_csv("HIV_output.csv")

        # s = MultiSim([x[-1] for x in outputs])
        # s.reduce(quantiles={"low": 0.25, "high": 0.75})
        # sim = s.base_sim

        print('Breakpoint')
