import starsim as ss
import stisim as sti
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc

np.seterr(all='raise')

class TrackValues(ss.Analyzer):
    # Track outputs for viral load and CD4 counts
    # Assumes no births; for diagnostic/debugging purposes only
    def initialize(self, sim):
        super().initialize(sim)
        self.n = len(sim.people)
        self.rel_sus = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)
        self.rel_trans = np.empty((sim.npts, self.n), dtype=ss.dtypes.float)
        self.cd4 =  np.empty((sim.npts, self.n), dtype=ss.dtypes.float)

    def apply(self, sim):
        self.rel_sus[sim.ti,:self.n] = sim.diseases.hiv.rel_sus.values[:self.n]
        self.rel_trans[sim.ti,:self.n] = sim.diseases.hiv.rel_trans.values[:self.n]
        self.cd4[sim.ti,:self.n] = sim.diseases.hiv.cd4.values[:self.n]

    def plot(self, agents:dict):
        """
        :param agents: Dictionary of events per agent {'agent_description':[('event_type', ti),...]}
        :return: Matplotlib figure
        """

        def plot_with_events(ax, x, y, agents, title):
            h = ax.plot(x,y)
            x_ev = []
            y_ev = []
            for i, events in enumerate(agents.values()):
                for event in events:
                    x_ev.append(sim.yearvec[event[1]])
                    y_ev.append(y[event[1],i])
            ax.scatter(x_ev, y_ev, marker='*', color='yellow', edgecolor='red',  s=100, linewidths=0.5, zorder=100)
            ax.set_title(title)
            return h

        fig, ax = plt.subplots(2,2)
        ax = ax.ravel()

        h = plot_with_events(ax[0], sim.yearvec, self.cd4, agents, 'CD4')
        h = plot_with_events(ax[1], sim.yearvec, self.rel_sus, agents, 'rel_sus')
        h = plot_with_events(ax[2], sim.yearvec, self.rel_trans, agents, 'rel_trans')

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

        if events:
            for uid, event, ti in events:
                if event == 'hiv_infection': self.hiv_infections[ti].append(uid)
                elif event == 'syphilis_infection': self.syphilis_infections[ti].append(uid)
                elif event == 'art_start': self.art_start[ti].append(uid)
                elif event == 'art_stop': self.art_stop[ti].append(uid)
                else: raise Exception(f'Unknown event "{event}"')


    def initiate_ART(self, uids):
        self.sim.diseases.hiv.on_art[ss.uids(uids)] = True
        self.sim.diseases.hiv.ti_art[ss.uids(uids)] = self.sim.ti
        self.sim.diseases.hiv.ti_stop_art[ss.uids(uids)] = np.nan

    def end_ART(self, uids):
        self.sim.diseases.hiv.on_art[ss.uids(uids)] = False
        self.sim.diseases.hiv.ti_stop_art[ss.uids(uids)] = self.sim.ti
        self.sim.diseases.hiv.ti_since_untreated[ss.uids(uids)] = self.sim.ti

    def apply(self, sim):
        self.initiate_ART(self.art_start[sim.ti])
        self.end_ART(self.art_stop[sim.ti])
        if 'hiv' in sim.diseases:
            self.sim.diseases.hiv.set_prognoses(ss.uids(self.hiv_infections[sim.ti]))
        if 'syphilis' in sim.diseases:
            self.sim.diseases.syphilis.set_prognoses(ss.uids(self.syphilis_infections[sim.ti]))


# AGENTS
agents = sc.odict()
agents['No infection'] = []
agents['Goes onto ART early (CD4 > 200) and stays on forever'] = [('hiv_infection', 1), ('art_start', 1*12)]
agents['Goes onto ART late (CD4 < 200) and stays on forever'] = [('hiv_infection', 1), ('art_start', 10*12)]
agents['Goes off ART with CD4 > 200'] = [('hiv_infection', 1), ('art_start', 10*12), ('art_stop', 15*12)]
agents['Goes off ART with CD4 < 200'] = [('hiv_infection', 1), ('art_start', 10*12), ('art_stop', 10*12+2)]

events = []
for i, x in enumerate(agents.values()):
    for y in x:
        events.append( (i,) + y )

pars = {}
pars['n_agents'] = 5
pars['start'] = 2020
pars['end'] = 2040
pars['dt'] = 1/12
hiv = sti.HIV(init_prev=0, p_death=0, beta={'structuredsexual': [0, 0], 'maternal': [0, 0]})
pars['diseases'] = [hiv]
pars['networks'] = [sti.StructuredSexual(), ss.MaternalNet()]
pars['demographics'] = [ss.Pregnancy(fertility_rate=0), ss.Deaths(death_rate=0)]
pars['interventions'] = PerformTest(events)
output = TrackValues()
pars['analyzers'] = output

sim = ss.Sim(pars,copy_inputs=False).run()

fig = output.plot(agents)
