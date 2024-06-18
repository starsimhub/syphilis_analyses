"""
Test calibration
"""

#%% Imports and settings
import starsim as ss
import sciris as sc
import stisim as sti
import numpy as np
import pylab as pl
import pandas as pd

do_plot = 1
do_save = 0
n_agents = 2e3


#%% Define the tests
def make_sim():

    syph = sti.Syphilis(
        beta={'structuredsexual': [0.5, 0.25], 'maternal': [0.99, 0]},
        init_prev=0.05,
    )
    hiv = sti.HIV(
        beta={'structuredsexual': [1, 1], 'maternal': [1, 0]},
        beta_m2f=0.05,
        beta_f2m=0.025,
        beta_m2c=0.025,
        init_prev=0.15,
    )
    connector = sti.hiv_syph(hiv, syph, rel_sus_hiv_syph=2, rel_trans_hiv_syph=2)
    pregnancy = ss.Pregnancy(fertility_rate=20)
    death = ss.Deaths(death_rate=10)
    sexual = sti.StructuredSexual(prop_f1=0.2)
    maternal = ss.MaternalNet()

    sim = ss.Sim(
        dt=1,
        n_agents=n_agents,
        total_pop=9980999,
        start=1990,
        n_years=40,
        diseases=[syph, hiv],
        networks=[sexual, maternal],
        connectors=connector,
        demographics=[pregnancy, death],
    )

    return sim


def test_calibration(do_plot=True):

    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = dict(
        diseases = dict(
            hiv = dict(
                beta_m2f = [0.05, 0.01, 0.10],
                beta_f2m = [0.025, 0.005, 0.05],
            ),
            syphilis = dict(
                rel_trans_latent = [0.1, 0.05, 0.20],
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
    sim = make_sim()

    # Weight the data
    weights = {
        'n_alive': 1,
        'hiv.prevalence': 1,
        'hiv.n_infected': 1,
        'hiv.new_infections': 1,
        'hiv.new_deaths': 1,
        'syphilis.prevalence': 1
        }

    data = pd.read_csv(sti.root/'tests'/'test_data'/'zimbabwe_calib.csv')

    # Make the calibration
    calib = sti.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        data=data,
        weights=weights,
        total_trials=4, n_workers=2, die=True
    )

    calib.calibrate(confirm_fit=True)

    print(f'Fit with original pars: {calib.before_fit}')
    print(f'Fit with best-fit pars: {calib.after_fit}')
    if calib.after_fit <= calib.before_fit:
        print(f'✓ Calibration improved fit ({calib.after_fit} <= {calib.before_fit})')
    else:
        print(f"✗ Calibration did not improve fit, but this isn't guaranteed ({calib.after_fit} > {calib.before_fit})")

    return sim, calib


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    sim, calib = test_calibration()

    sc.toc(T)
    print('Done.')
