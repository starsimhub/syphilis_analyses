"""
Run hiv
"""

# %% Imports and settings
import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
from celery import group
from tqdm import tqdm
import numpy as np
import starsim as ss
from stisim.celery import run_sim
from stisim.multisim import MultiSim
from stisim.celery import celery
from starsim import Samples
import sciris as sc
from stisim.plotting import plot_hiv
import pandas as pd

quick_run = False
ss.options['multirng'] = False
debug_mode = False
result_dir = Path("results/Calibration")
to_run = []


def get_betas(mean_beta, std_beta, nruns=None):
    seeds = np.arange(nruns)
    # betas = np.random.normal(mean_beta, std_beta, nruns)
    return seeds  # , betas


def run_scenario(kwargs, filter=False):
    nruns = kwargs['nruns']
    del kwargs['nruns']

    seeds = get_betas(1, 1, nruns)
    description_baseline = 'calibration'

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

    result.forget()

    return True


if __name__ == '__main__':
    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]

    # Load calibration csv:
    calibration_scenarios = pd.read_csv("calibration_scenarios.csv")
    for row, scenario in calibration_scenarios.iterrows():
        if scenario['to_run'] == 'Y':
            init_prev = scenario['init_prev']
            ss_hiv_beta = scenario['ss_hiv_beta']
            maternal_hiv_beta = scenario['maternal_hiv_beta']
            mean_duration_onART = scenario['dur_on_ART_mean']
            mean_initial_cd4 = scenario['mean_initial_cd4']
            risk_groups_f_probs = np.array([scenario['f0_prob'], scenario['f1_prob'], scenario['f2_prob']])
            risk_groups_m_probs = np.array([scenario['m0_prob'], scenario['m1_prob'], scenario['m2_prob']])
            conc = {'f': [scenario['f0_conc'], scenario['f1_conc'], scenario['f2_conc']],
                    'm': [scenario['m0_conc'], scenario['m1_conc'], scenario['m2_conc']]}
            p_pair_form = scenario['p_pair_form']
            calibration_name = scenario['calibration_name']

            kwargs = {'location': location,
                      'total_pop': total_pop,
                      'dt': 1 / 12,
                      'n_agents': int(1e4),
                      'init_prev': ss.bernoulli(p=init_prev),
                      'cd4_start_dist': ss.lognorm_ex(mean=mean_initial_cd4, stdev=10),
                      'ss_hiv_beta': ss_hiv_beta,
                      'risk_groups_f': ss.choice(a=3, p=risk_groups_f_probs),
                      'risk_groups_m': ss.choice(a=3, p=risk_groups_m_probs),
                      'p_pair_form': ss.bernoulli(p=p_pair_form),
                      'maternal_hiv_beta': maternal_hiv_beta,
                      'duration_on_ART': ss.lognorm_ex(mean=mean_duration_onART, stdev=5),
                      'conc': conc,
                      'return_sim': True,
                      'save_as': calibration_name}

            to_run.append(kwargs)

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

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

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
                    time.sleep(1)

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
                plot_hiv(sim_output, location='zimbabwe', total_pop=9980999, save=save)
        else:
            with sc.Timer(label="Run model") as _:
                outputs = [run_sim(0, **kwargs)]

        # outputs[0][0].to_csv("HIV_output.csv")

        # s = MultiSim([x[-1] for x in outputs])
        # s.reduce(quantiles={"low": 0.25, "high": 0.75})
        # sim = s.base_sim

        print('Breakpoint')
