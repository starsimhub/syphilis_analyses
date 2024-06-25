"""
Compare performance and results of regular & fast StructuredSexual.

Timings:
      10k  20k
rand  0.5  0.6
slow 14.8 94.3
fast  0.7  1.1
"""

import sciris as sc
import starsim as ss
import stisim as sti

kw = dict(n_agents=20e3)

s0 = ss.Sim(diseases='sis', networks='random', **kw)
s1 = ss.Sim(diseases='sis', networks=sti.StructuredSexual(), **kw)
s2 = ss.Sim(diseases='sis', networks=sti.FastStructuredSexual(), **kw)

with sc.timer('random'):
    s0.run()
    
with sc.timer('StructuredSexual'):
    s1.run()
    
with sc.timer('FastStructuredSexual'):
    s2.run()
    
    
for sim in [s0, s1, s2]:
    sim.plot()