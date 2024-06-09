import starsim as ss
import stisim as sti
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc


class TrackValues(ss.Analyzer):
    # Track outputs for viral load and CD4 counts
    # Assumes no births; for diagnostic/debugging purposes only
    def init_pre(self, sim):
        super().init_pre(sim)
        self.n = len(sim.people)

        self.hiv_rel_sus = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)
        self.hiv_rel_trans = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)

        self.syph_rel_sus = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)
        self.syph_rel_trans = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)

        self.cd4 = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)

        self.care_seeking = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)

    @property
    def has_hiv(self):
        return 'hiv' in self.sim.diseases

    @property
    def has_syph(self):
        return isinstance(self.sim.diseases.get('syphilis'), sti.Syphilis)

    def apply(self, sim):

        if self.has_hiv:
            self.hiv_rel_sus[sim.ti, :self.n] = sim.diseases.hiv.rel_sus.values[:self.n]
            self.hiv_rel_trans[sim.ti, :self.n] = sim.diseases.hiv.rel_trans.values[:self.n]

        if self.has_syph:
            self.syph_rel_sus[sim.ti, :self.n] = sim.diseases.syphilis.rel_sus.values[:self.n]
            self.syph_rel_trans[sim.ti, :self.n] = sim.diseases.syphilis.rel_trans.values[:self.n]

        self.cd4[sim.ti, :self.n] = sim.diseases.hiv.cd4.values[:self.n]
        self.care_seeking[sim.ti, :self.n] = sim.diseases.hiv.care_seeking[:self.n]

    def plot(self, agents: dict):
        """
        :param agents: Dictionary of events per agent {'agent_description':[('event_type', ti),...]}
        :return: Matplotlib figure
        """

        def plot_with_events(ax, x, y, agents, title):
            h = ax.plot(x, y)
            x_ev = []
            y_ev = []
            for i, events in enumerate(agents.values()):
                for event in events:
                    x_ev.append(self.sim.yearvec[event[1]])
                    y_ev.append(y[event[1], i])
            ax.scatter(x_ev, y_ev, marker='*', color='yellow', edgecolor='red', s=100, linewidths=0.5, zorder=100)
            ax.set_title(title)
            return h

        if self.has_syph:
            fig, ax = plt.subplots(2, 4)
        else:
            fig, ax = plt.subplots(1, 4)

        ax = ax.ravel()

        h = plot_with_events(ax[0], self.sim.yearvec, self.cd4, agents, 'CD4')
        h = plot_with_events(ax[1], self.sim.yearvec, self.hiv_rel_sus, agents, 'HIV rel_sus')
        h = plot_with_events(ax[2], self.sim.yearvec, self.hiv_rel_trans, agents, 'HIV rel_trans')
        h = plot_with_events(ax[3], self.sim.yearvec, self.care_seeking, agents, 'HIV care seeking')

        if self.has_syph:
            h = plot_with_events(ax[4], self.sim.yearvec, self.syph_rel_sus, agents, 'Syphilis rel_sus')
            h = plot_with_events(ax[5], self.sim.yearvec, self.syph_rel_trans, agents, 'Syphilis rel_trans')

        fig.legend(h, agents.keys(), loc='upper right', bbox_to_anchor=(1.1, 1))

        return fig


class PerformTest(ss.Intervention):

    def __init__(self, events=None):
        """
        :param events: List of (uid, 'event', ti) to apply events to an agent
        """
        super().__init__()
        self.hiv_infections = defaultdict(list)
        self.syphilis_infections = defaultdict(list)
        self.art_start = defaultdict(list)
        self.art_stop = defaultdict(list)
        self.pregnant = defaultdict(list)

        if events:
            for uid, event, ti in events:
                if event == 'hiv_infection':
                    self.hiv_infections[ti].append(uid)
                elif event == 'syphilis_infection':
                    self.syphilis_infections[ti].append(uid)
                elif event == 'art_start':
                    self.art_start[ti].append(uid)
                elif event == 'art_stop':
                    self.art_stop[ti].append(uid)
                elif event == 'pregnant':
                    self.pregnant[ti].append(uid)
                else:
                    raise Exception(f'Unknown event "{event}"')

    def initiate_ART(self, uids):
        if len(uids):
            self.sim.diseases.hiv.start_art(ss.uids(uids))

    def end_ART(self, uids):
        if len(uids):
            self.sim.diseases.hiv.stop_art(ss.uids(uids))

    def set_pregnancy(self, uids):
        self.sim.demographics.pregnancy.pregnant[ss.uids(uids)] = True
        self.sim.demographics.pregnancy.ti_pregnant[ss.uids(uids)] = self.sim.ti

    def apply(self, sim):
        self.initiate_ART(self.art_start[sim.ti])
        self.end_ART(self.art_stop[sim.ti])
        if 'hiv' in sim.diseases:
            self.sim.diseases.hiv.set_prognoses(ss.uids(self.hiv_infections[sim.ti]))
        if 'syphilis' in sim.diseases:
            self.sim.diseases.syphilis.set_prognoses(ss.uids(self.syphilis_infections[sim.ti]))

        # Set pregnancies:
        self.set_pregnancy(self.pregnant[sim.ti])


def test_hiv():
    # AGENTS
    agents = sc.odict()
    agents['No infection'] = []
    agents['Infection without ART'] = [('hiv_infection', 1)]
    agents['Goes onto ART early (CD4 > 200) and stays on forever'] = [('hiv_infection', 1), ('art_start', 1 * 12)]
    agents['Goes onto ART late (CD4 < 200) and stays on forever'] = [('hiv_infection', 1), ('art_start', 10 * 12)]
    agents['Goes off ART with CD4 > 200'] = [('hiv_infection', 1), ('art_start', 5 * 12), ('art_stop', 12 * 12)]
    agents['Goes off ART with CD4 < 200'] = [('hiv_infection', 1), ('art_start', 9 * 12), ('art_stop', 12 * 12)]
    agents['pregnant'] = [('pregnant', 300), ('hiv_infection', 580)]

    events = []
    for i, x in enumerate(agents.values()):
        for y in x:
            events.append((i,) + y)

    pars = {}
    pars['n_agents'] = len(agents)
    pars['start'] = 2020
    pars['end'] = 2040
    pars['dt'] = 1 / 12
    hiv = sti.HIV(init_prev=0, p_hiv_death=0, dur_latent=5, include_aids_deaths=False, beta={'structuredsexual': [0, 0], 'maternal': [0, 0]})
    pars['diseases'] = [hiv]
    pars['networks'] = [sti.StructuredSexual(), ss.MaternalNet()]
    pars['demographics'] = [ss.Pregnancy(fertility_rate=0), ss.Deaths(death_rate=0)]
    pars['interventions'] = PerformTest(events)
    output = TrackValues()
    pars['analyzers'] = output

    sim = ss.Sim(pars, copy_inputs=False).run()
    fig = output.plot(agents)
    return sim


def test_hiv_syph():
    # AGENTS
    agents = sc.odict()
    agents['No infection'] = []
    agents['HIV only'] = [('hiv_infection', 1)]
    agents['HIV before syphilis'] = [('syphilis_infection', 1), ('hiv_infection', 12)]
    agents['HIV after syphilis'] = [('hiv_infection', 1), ('syphilis_infection', 12)]

    events = []
    for i, x in enumerate(agents.values()):
        for y in x:
            events.append((i,) + y)

    pars = {}
    pars['n_agents'] = len(agents)
    pars['start'] = 2020
    pars['end'] = 2040
    pars['dt'] = 1 / 12
    hiv = sti.HIV(init_prev=0, p_hiv_death=0, include_aids_deaths=False, beta={'structuredsexual': [0, 0], 'maternal': [0, 0]})
    syphilis = sti.SyphilisPlaceholder(prevalence=None)

    pars['diseases'] = [hiv, syphilis]
    pars['networks'] = [sti.StructuredSexual(), ss.MaternalNet()]
    pars['demographics'] = [ss.Pregnancy(fertility_rate=0), ss.Deaths(death_rate=0)]
    pars['interventions'] = PerformTest(events)
    output = TrackValues()
    pars['analyzers'] = output
    pars['connectors'] = sti.hiv_syph(hiv, syphilis, rel_sus_hiv_syph=100, rel_trans_hiv_syph=100)

    sim = ss.Sim(pars, copy_inputs=False).run()

    fig = output.plot(agents)

    return sim


if __name__ == '__main__':
    s0 = test_hiv()
    # s1 = test_hiv_syph()
