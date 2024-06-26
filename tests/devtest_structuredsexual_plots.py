"""
Create plots illustrating the StructuredSexual network
"""


import numpy as np
import sciris as sc
import pylab as pl
import starsim as ss
import stisim as sti


class animate_network(ss.Analyzer):
    def __init__(self, layout='parallel'):
        super().__init__()
        self.ti = []
        self.data = []
        self.layout = layout
        return
    
    def init_pre(self, sim):
        self.count = 0
        self.yearvec = sim.yearvec
        return
        
    def apply(self, sim):
        self.ti.append(sim.ti)
        self.data.append(sim.networks[0].to_df())
        self.count += 1
        self.fem = sim.people.female.raw
        self.age = sim.people.age.raw
        self.order = np.argsort(self.age)
        
        self.rorder = np.zeros(len(self.order))
        self.rorder[self.order] = np.arange(len(self.order))

        
        n = len(sim.people)
        self.a = np.arange(n)
        if self.layout == 'square':
            sqn = np.floor(np.sqrt(n))
            self.x = self.rorder // sqn
            self.y = self.rorder % sqn
        elif self.layout == 'circle':
            const = 2*np.pi/n
            self.x = np.cos(self.rorder*const)
            self.y = np.sin(self.rorder*const)
        elif self.layout == 'parallel':
            self.x = self.fem + np.random.randn(n)*0.1
            self.y = self.rorder/self.rorder.max()*self.age.max()
        return
        
    def plot(self, fig=None, i=0):
        if fig is None:
            sc.options(dpi=200)
            fig = pl.figure(figsize=(8,10))
            pl.set_cmap('turbo')
        
        pl.title(f'Year = {self.yearvec[i]:0.1f}, step = {self.ti[i]}/{len(self.ti)}')
        pl.xticks([0,1], ['Male', 'Female'])
        pl.ylabel('Age')
        for tf,marker in enumerate(['s','o']):
            inds = sc.findinds(self.fem, tf)
            pl.scatter(self.x[inds], self.y[inds], color=sc.vectocolor(self.age[inds]), marker=marker)
        d = self.data[i]
        for p1,p2 in zip(d.p1, d.p2):
            x = [self.x[p1], self.x[p2]]
            y = [self.y[p1], self.y[p2]]
            pl.plot(x, y, lw=0.1, alpha=0.5, c='k')
        
        sc.figlayout()        
        return fig
    
    def animate(self):
        fig = None
        for i in range(self.count):
            if fig is not None: pl.cla()
            fig = self.plot(fig=fig, i=i)
            pl.pause(0.1)
        return
        

class count_partners(ss.Analyzer):
    def __init__(self):
        super().__init__()
        self.ti = []
        self.raw = []
        self.sets = []
        self.counts = []
        return
    
    def init_pre(self, sim):
        self.count = 0
        
        return
        
    def apply(self, sim):
        self.ti.append(sim.ti)
        self.raw.append(sim.networks[0].to_df())
        self.count += 1
        return
    
    def finalize(self):
        self.yearvec = sim.yearvec
        self.uids = sim.people.uid
        self.fem = sim.people.female.raw
        self.age = sim.people.age.raw
        self.n = len(self.uids)
        self.sets = [set() for u in self.uids]
        self.counts = np.zeros((self.n, self.count))
        for u in self.uids:
            for i, raw in enumerate(self.raw):
                contacts = ss.find_contacts(raw.p1.values, raw.p2.values, np.array([u]))
                self.sets[u] = self.sets[u] | contacts
                self.counts[u,i] = len(self.sets[u])
        
        return
        
    def plot_growth(self):
        sc.options(dpi=200)
        fig = pl.figure(figsize=(10,6))
        pl.title('Cumulative number of unique partners')
        pl.ylabel('Partner counts')
        pl.xlabel('Time')
        for u in self.uids:
            r = (np.random.rand(len(self.yearvec))-0.5)*1.0
            pl.plot(self.yearvec, self.counts[u,:]+r, alpha=0.3)
        pl.ylim(bottom=0)
        sc.figlayout()
        return fig
        

if __name__ == '__main__':
    
    kw = dict(
        n_agents = 2_000, # 10_000 is very slow, but matches the results
        n_years = 10,
        dt = 1/12,
    )
    
    torun = [
        # 'plot_pairs',
        # 'animate_pairs',
        'count_partners',
    ]
    
    if 'plot_pairs' in torun or 'animate_pairs' in torun:
        animnet = animate_network()
        nw = ['random', sti.StructuredSexual()][1] # To debug, use random here
        sim = ss.Sim(networks=nw, analyzers=animnet, copy_inputs=False, **kw)
        sim.run()
        if 'plot_pairs' in torun:
            animnet.plot()
        if 'animate_pairs' in torun:
            animnet.animate()

    if 'count_partners' in torun:
        # ckw = sc.mergedicts(kw, n_agents=100)
        countpart = count_partners()
        nw = sti.FastStructuredSexual()
        sim = ss.Sim(networks=nw, analyzers=countpart, copy_inputs=False, **kw)
        sim.run()
        # countpart.plot_stats()
        countpart.plot_growth()