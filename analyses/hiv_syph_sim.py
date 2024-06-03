"""
Create a simulation with co-circulating HIV and syphilis
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import sciris as sc
import stisim as sti

from run_hiv import get_testing_products


def make_hiv():
    """
    Make a sim with HIV
    """
    ####################################################################################################################
    # HIV Params
    ####################################################################################################################
    hiv = HIV(
        beta={'structuredsexual': [0.05, 0.025], 'maternal': [0.05, 0.]},
        init_prev=0.07,
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
    sexual = StructuredSexual()
    maternal = ss.MaternalNet()

    ####################################################################################################################
    # Testing and treatment
    ####################################################################################################################
    n_art = pd.read_csv(sti.data/'zimbabwe_art.csv').set_index('year')
    fsw_testing, other_testing, low_cd4_testing = get_testing_products()
    art = sti.ART(
        ART_coverages_df=ART_coverages_df,
        dur_on_art=ss.lognorm_ex(7, 3),  # https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-021-10464-x
        art_efficacy=0.96
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
            art
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


if __name__ == '__main__':
    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]

    sim, output = run_hiv(location=location, total_pop=total_pop, dt=1 / 12, n_agents=int(1e4))
    # output.to_csv("HIV_output.csv")

    plot_hiv(output)

    sc.saveobj(f'sim_{location}.obj', sim)
