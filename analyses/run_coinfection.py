"""
Run syphilis-HIV coinfection model
"""

# %% Imports and settings
import numpy as np
import starsim as ss
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc
from stisim.networks import StructuredSexual
from stisim.diseases.syphilis import Syphilis
from stisim.diseases.hiv import HIV

quick_run = False
ss.options['multirng']=False


def make_sim(location='zimbabwe', total_pop=100e6, dt=1, n_agents=500, latent_trans=0.075):
    """ Make a sim with syphilis """

    # Syphilis
    syph = Syphilis()
    syph.pars['beta'] = {'structuredsexual': [0.5, 0.25], 'maternal': [0.99, 0]}
    syph.pars['init_prev'] = ss.bernoulli(p=0.1)
    syph.pars['rel_trans']['latent'] = latent_trans

    # HIV
    hiv = HIV()
    hiv.pars['beta'] = {'structuredsexual': [0.5, 0.25], 'maternal': [0.99, 0]}
    hiv.pars['init_prev'] = ss.bernoulli(p=0.1)

    # Connector


    # Make demographic modules
    fertility_rates = {'fertility_rate': pd.read_csv(f'data/{location}_asfr.csv')}
    pregnancy = ss.Pregnancy(pars=fertility_rates)
    death_rates = {'death_rate': pd.read_csv(f'data/{location}_deaths.csv'), 'units': 1}
    death = ss.Deaths(death_rates)

    # Make people and networks
    ss.set_seed(1)
    ppl = ss.People(n_agents, age_data=pd.read_csv(f'data/{location}_age.csv'))
    sexual = StructuredSexual()
    maternal = ss.MaternalNet()

    sim_kwargs = dict(
        dt=dt,
        total_pop=total_pop,
        start=1990,
        n_years=40,
        people=ppl,
        diseases=syph,
        networks=ss.ndict(sexual, maternal),
        demographics=[pregnancy, death],
    )

    return sim_kwargs


def run_syph(location='zimbabwe', total_pop=100e6, dt=1.0, n_agents=500):

    sim_kwargs = make_syph_sim(location=location, total_pop=total_pop, dt=dt, n_agents=n_agents)
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

    for ai, sex in enumerate(['f', 'm']):
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
    sim = run_syph(location=location, total_pop=total_pop, dt=1/12, n_agents=int(10e3))
    sc.saveobj(f'sim_{location}.obj', sim)

    plot_degree(sim)
    plot_mixing(sim)
    plot_syph(sim)

    # sim = sc.loadobj('sim.obj')

