import starsim as ss
import stisim as sti
import pandas as pd

import numpy as np

np.seterr(all='raise')

location = 'zimbabwe'
fertility_rates = {'fertility_rate': pd.read_csv(sti.data / f'{location}_asfr.csv')}
death_rates = {'death_rate': pd.read_csv(sti.data / f'{location}_deaths.csv'), 'units': 1}

pars = {}
pars['n_agents'] = 10000

hiv = sti.HIV(init_prev=0.15, beta={'structuredsexual': [0.3, 0.15], 'maternal': [0.95, 0.0]})
syphilis = sti.SyphilisPlaceholder(prevalence=0.9)

pars['diseases'] = [hiv, syphilis]
# pars['connectors'] = sti.hiv_syph(hiv, syphilis)
pars['networks'] = [sti.StructuredSexual(), ss.MaternalNet()]
pars['demographics'] = [ss.Pregnancy(pars=fertility_rates), ss.Deaths(death_rates)]

s0 = ss.Sim(pars).run()

pars['connectors'] = sti.hiv_syph(hiv, syphilis, rel_sus_hiv_syph=999, rel_trans_hiv_syph=999)
s1 = ss.Sim(pars).run()

print(s0.results.hiv.cum_infections)
print(s1.results.hiv.cum_infections)
