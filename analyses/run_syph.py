"""
Run syphilis
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc
# from stisim.networks import StructuredSexual
# from stisim.diseases.syphilis import Syphilis
import stisim as sti
from syph_tests import TestProb, ANCTesting, LinkedNewbornTesting, TreatNum


quick_run = True


def make_syph_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075):
    """ Make a sim with syphilis and genital ulcerative disease """
    syph = sti.Syphilis(
        beta={'structuredsexual': [0.5, 0.25], 'maternal': [0.99, 0]},
        init_prev_data=pd.read_csv('data/init_prev_syph.csv'),
        rel_trans_latent=latent_trans
    )
    gud = sti.GUD(
        beta={'structuredsexual': [0.5, 0.25], 'maternal': 0},
        init_prev_data=pd.read_csv('data/init_prev_gud.csv')
    )

    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(f'data/{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(f'data/{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(n_agents, age_data=pd.read_csv(f'data/{location}_age.csv'))
    sexual = sti.StructuredSexual()
    maternal = ss.MaternalNet()

    sim_kwargs = dict(
        dt=dt,
        total_pop=total_pop,
        start=1990,
        n_years=40,
        people=ppl,
        diseases=[syph, gud],
        networks=ss.ndict(sexual, maternal),
        demographics=[pregnancy, death],
    )

    return sim_kwargs


def make_testing_intvs():

    # Initialize interventions
    interventions = sc.autolist()

    # Read in testing probabilities and create an intervention representing the
    # initial visit/consult - this uses risk group/sex/year-specific testing rates
    # representing the probability of someone with symptoms seeking care
    symp_test_data = pd.read_csv('data/symp_test_prob.csv')
    symp_test = TestProb(
        product='symp_test_assigner',
        test_prob_data=symp_test_data,
        name='symp_test',
        label='symp_test',
    )

    # Funnel all symptomatic people into different management options
    # This is a way of representing the market share or product mix.
    synd_el = lambda sim: sim.get_intervention('symp_test').outcomes['syndromic']
    synd_mgmt = TestProb(product='syndromic', test_prob_data=1, eligibility=synd_el, name='synd_mgmt', label='synd_mgmt')
    dual_el = lambda sim: sim.get_intervention('symp_test').outcomes['dual']
    dual_test = TestProb(product='dual', test_prob_data=1, eligibility=dual_el, name='dual_test', label='dual_test')
    rst_el = lambda sim: sim.get_intervention('symp_test').outcomes['rst']
    rst = TestProb(product='rst', test_prob_data=1, eligibility=rst_el, name='rst', label='rst')

    # Add ANC testing
    anc_test_data = np.array([0.05]*31)
    test_years = np.arange(2000, 2031)
    anc = ANCTesting(product='rst', test_prob_data=anc_test_data, years=test_years, name='anc', label='anc')

    interventions += [symp_test, synd_mgmt, dual_test, rst, anc]

    # Some proportion of those who are probable syphilis cases will be given a confirmatory test
    def all_pos(sim):
        pos_list = sc.autolist()
        pos_list += sim.get_intervention('synd_mgmt').outcomes['positive'].tolist()
        pos_list += sim.get_intervention('dual_test').outcomes['positive'].tolist()
        pos_list += sim.get_intervention('rst').outcomes['positive'].tolist()
        pos_list += sim.get_intervention('anc').outcomes['positive'].tolist()
        return ss.uids(np.array(list(set(pos_list))))
    pos_mgmt = TestProb(product='pos_assigner', test_prob_data=.5, eligibility=all_pos, name='pos_mgmt', label='pos_mgmt')
    rpr_el = lambda sim: sim.get_intervention('pos_mgmt').outcomes['rpr']
    rpr = TestProb(product='rpr', test_prob_data=1, eligibility=rpr_el, name='rpr', label='rpr')

    interventions += [pos_mgmt, rpr]

    # Treatment
    def to_treat(sim):
        pos_list = sc.autolist()
        pos_list += sim.get_intervention('rpr').outcomes['positive'].tolist()
        pos_list += sim.get_intervention('pos_mgmt').outcomes['treat'].tolist()
        return ss.uids(np.array(list(set(pos_list))))

    treat_data = pd.read_csv('data/treat_prob.csv')
    max_capacity = np.array([50_000]*31)
    treat_years = np.arange(2000, 2031)
    treat = TreatNum(
        treat_prob_data=treat_data,
        max_capacity=max_capacity,
        years=treat_years,
        eligibility=to_treat,
        name='treat',
        label='treat',
    )
    interventions += treat

    # Newborn testing
    newborn_test = LinkedNewbornTesting(product='newborn_exam', test_prob_data=0.1)
    interventions += newborn_test

    # TODO
    # 1. consider sensitivity and specificity - build in prevalence of other GUD?
    # 2. SOC is to confirm with an RPR - construct an algorithm?
    # 3. Check that we whould be getting high positivity with reasonable numbers of false positives
    # 4. People who present with symptoms things might be:
    #       a. ‘diagnosed’ through syndromic management
    #       b. given a test (treponemal, nontreponemal, RPR/VDR, and dual HIV tests)

    return interventions


def run_syph(location='zimbabwe', total_pop=100e6, dt=1.0, n_agents=500, latent_trans=0.1):

    sim_kwargs = make_syph_sim(location=location, total_pop=total_pop, dt=dt, n_agents=n_agents, latent_trans=latent_trans)
    interventions = make_testing_intvs()
    sim = ss.Sim(interventions=interventions, **sim_kwargs)
    sim.run()

    return sim


def run_gud(location='zimbabwe', total_pop=100e6, dt=1.0, n_agents=500):

    sim_kwargs = make_syph_sim(location=location, total_pop=total_pop, dt=dt, n_agents=n_agents)
    gud = sti.GUD(
        beta={'structuredsexual': [0.5, 0.25], 'maternal': 0},
        init_prev_data=pd.read_csv('data/init_prev.csv')
    )
    sim_kwargs['diseases'] = gud
    sim = ss.Sim(**sim_kwargs)
    sim.run()

    return sim


def plot_mixing(sim):

    import matplotlib as mpl
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    nw = sim.networks['structuredsexual']
    mc = nw.contacts['age_p1']
    fc = nw.contacts['age_p2']
    h = ax.hist2d(fc, mc, bins=np.linspace(0, 75, 16), density=True, norm=mpl.colors.LogNorm())
    ax.set_xlabel('Age of female partner')
    ax.set_ylabel('Age of male partner')
    fig.colorbar(h[3], ax=ax)
    ax.set_title('Age mixing')
    fig.tight_layout()
    sc.savefig("figures/networks.png", dpi=100)
    return


def plot_syph(sim):
    # Check plots
    burnin = 10
    pi = int(burnin/sim.dt)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].stackplot(
        sim.yearvec[pi:],
        # sim.results.syphilis.n_susceptible[pi:],
        sim.results.syphilis.n_congenital[pi:],
        sim.results.syphilis.n_exposed[pi:],
        sim.results.syphilis.n_primary[pi:],
        sim.results.syphilis.n_secondary[pi:],
        sim.results.syphilis.n_latent[pi:],
        sim.results.syphilis.n_tertiary[pi:],
    )
    ax[0].legend(['Congenital', 'Exposed', 'Primary', 'Secondary', 'Latent', 'Tertiary'], loc='lower right')

    ax[1].plot(sim.yearvec[pi:], sim.results.syphilis.prevalence[pi:])
    ax[1].set_title('Syphilis prevalence')

    ax[2].plot(sim.yearvec[pi:], sim.results.n_alive[pi:])
    ax[2].set_title('Population')

    ax[3].plot(sim.yearvec[pi:], sim.results.syphilis.new_infections[pi:])
    ax[3].set_title('New infections')

    fig.tight_layout()
    plt.show()
    return


def plot_degree(sim):

    fig, axes = plt.subplots(2, 4, figsize=(9, 5), layout="tight")

    nw = sim.networks['structuredsexual']

    for ai, sex in enumerate(['female', 'male']):
        for rg in range(4):

            active = sim.people[sex] & (nw.active(sim.people))
            if rg < 3:
                # Get sexually active people of this sex and risk group, excluding FSW/clients
                group_bools = (nw.risk_group == rg) & active & (~nw.fsw) & (~nw.client)
            else:
                if sex == 'f':   group_bools = active & nw.fsw
                elif sex == 'f': group_bools = active & nw.client

            lp = nw.lifetime_partners[group_bools]

            if rg == 0:   bins = np.concatenate([np.arange(51), [100]])
            elif rg == 1: bins = np.concatenate([np.arange(51), [500]])
            elif rg == 2: bins = np.concatenate([np.arange(51), [1000]])
            elif rg == 3: bins = np.concatenate([np.arange(51), [1000]])

            counts, bins = np.histogram(lp, bins=bins)

            total = sum(counts)
            counts = counts/total

            axes[ai, rg].bar(bins[:-1], counts)
            axes[ai, rg].set_title(f'sex={sex}, risk={rg}')
            axes[ai, rg].set_ylim([0, 1])
            stats = f"Mean: {np.mean(lp):.1f}\n"
            stats += f"Median: {np.median(lp):.1f}\n"
            stats += f"Std: {np.std(lp):.1f}\n"
            stats += f"%>20: {np.count_nonzero(lp>=20)/total*100:.2f}\n"
            axes[ai, rg].text(1, 0.5, stats)

    sc.savefig("figures/partner_degree.png", dpi=300)

    plt.show()

    return


if __name__ == '__main__':

    location = 'zimbabwe'
    total_pop = dict(
        nigeria=93963392,
        zimbabwe=9980999,
    )[location]

    sim = run_syph(location=location, total_pop=total_pop, dt=1/12, n_agents=10_000, latent_trans=0.1)
    # sim = run_gud(location=location, total_pop=total_pop, dt=1/12, n_agents=10_000)
    import pylab as pl
    sim.plot('gud')
    pl.show()


    # sc.saveobj(f'sim_{location}.obj', sim)
    # plot_degree(sim)
    # plot_mixing(sim)
    # plot_syph(sim)

    # sim = sc.loadobj('sim.obj')

