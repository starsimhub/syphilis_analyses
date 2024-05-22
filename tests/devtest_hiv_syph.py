"""
Development test to demonstrate impact of including the HIV-Syphilis connector

One simulation is run with HIV and syphilis, but no connector. A second simulation
is run with the addition of the connector. Adding the connector should increase
transmission of HIV overall due to syphilis coinfection. The prevalance should
therefore be higher when the connector is included.
"""

import starsim as ss
import stisim as sti
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.seterr(all='raise')

location = 'zimbabwe'
fertility_rates = {'fertility_rate': pd.read_csv(sti.data / f'{location}_asfr.csv')}
death_rates = {'death_rate': pd.read_csv(sti.data / f'{location}_deaths.csv'), 'units': 1}

pars = {}
pars['n_agents'] = 1000

hiv = sti.HIV(init_prev=0.01, beta={'structuredsexual': [0.01, 0.01], 'maternal': [0.0, 0.0]})
syphilis = sti.SyphilisPlaceholder(prevalence=0.9)

pars['diseases'] = [hiv, syphilis]
# pars['connectors'] = sti.hiv_syph(hiv, syphilis)
pars['networks'] = [sti.StructuredSexual(
    # risk_groups_f=ss.choice(a=3, p=np.array([0.1, 0.8, 0.1])),
    # risk_groups_m=ss.choice(a=3, p=np.array([0.1, 0.8, 0.1])),
), ss.MaternalNet()]
pars['demographics'] = [ss.Pregnancy(fertility_rate=0), ss.Deaths(death_rate=0)]

s0 = ss.Sim(pars).run()

pars['connectors'] = sti.hiv_syph(hiv, syphilis, rel_sus_hiv_syph=2, rel_trans_hiv_syph=2)
s1 = ss.Sim(pars).run()

plt.plot(s0.yearvec, s0.results.hiv.prevalence, label='No connector')
plt.plot(s1.yearvec, s1.results.hiv.prevalence, label='With connector')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Prevalence')