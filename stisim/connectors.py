"""
Syphilis-HIV connector for running coinfection analyses
"""

import starsim as ss
from stisim.diseases.syphilis import Syphilis
from stisim.diseases.hiv import HIV

__all__ = ['hiv_syph']


class hiv_syph(ss.Connector):

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='HIV-Syphilis', requires=[HIV, Syphilis])
        self.default_pars(
            rel_sus_syph_hiv=2,         # People with HIV are 2x more likely to acquire syphilis
            rel_sus_syph_aids=5,        # People with AIDS are 5x more likely to acquire syphilis
            rel_trans_syph_hiv=1.5,     # People with HIV are 1.5x more likely to transmit syphilis
            rel_trans_syph_aids=3,      # People with AIDS are 3x more likely to transmit syphilis
            rel_sus_hiv_syph=2.7,       # People with syphilis are 2.7x more likely to acquire HIV
            rel_trans_hiv_syph=2.7,     # People with syphilis are 2.7x more likely to transmit HIV
        )
        self.update_pars(pars, **kwargs)

        return

    def update(self, sim):
        """ Specify HIV-syphilis interactions """
        # People with HIV are more likely to acquire syphilis
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 500] = self.pars.rel_sus_syph_hiv
        sim.diseases.syphilis.rel_sus[sim.people.hiv.cd4 < 200] = self.pars.rel_sus_syph_aids

        # People with HIV are more likely to transmit syphilis
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 500] = self.pars.rel_trans_syph_hiv
        sim.diseases.syphilis.rel_trans[sim.people.hiv.cd4 < 200] = self.pars.rel_trans_syph_aids

        # People with syphilis are more likely to acquire HIV
        sim.diseases.hiv.rel_sus[sim.diseases.syphilis.active] = self.pars.rel_sus_hiv_syph

        # People with syphilis are more likely to transmit HIV
        sim.diseases.hiv.rel_trans[sim.diseases.syphilis.active] = self.pars.rel_trans_hiv_syph

        return

