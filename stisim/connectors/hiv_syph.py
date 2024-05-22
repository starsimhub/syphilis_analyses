"""
Syphilis-HIV connector for running coinfection analyses
"""

import starsim as ss

import stisim
import stisim as sti

__all__ = ['hiv_syph']


class hiv_syph(ss.Connector):

    def __init__(self, hiv_module, syphilis_module, pars=None, **kwargs):
        super().__init__()

        self.hiv = hiv_module
        self.syphilis = syphilis_module
        self.default_pars(
            # Changes to HIV due to syphilis coinfection
            rel_sus_hiv_syph=ss.normal(loc=1.5, scale=0.25),  # Relative increase in susceptibility to HIV due to syphilis
            rel_trans_hiv_syph=ss.normal(loc=1.2, scale=0.025),  # Relative increase in transmission due to syphilis

            # Changes to syphilis due to HIV coinfection
            rel_sus_syph_hiv=2,         # People with HIV are 2x more likely to acquire syphilis
            rel_sus_syph_aids=5,        # People with AIDS are 5x more likely to acquire syphilis
            rel_trans_syph_hiv=1.5,     # People with HIV are 1.5x more likely to transmit syphilis
            rel_trans_syph_aids=3,      # People with AIDS are 3x more likely to transmit syphilis
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.FloatArr('rel_sus_hiv_syph', default=self.pars.rel_sus_hiv_syph),
            ss.FloatArr('rel_trans_hiv_syph', default=self.pars.rel_trans_hiv_syph),
            ss.FloatArr('rel_sus_syph_hiv', default=self.pars.rel_sus_syph_hiv),
            ss.FloatArr('rel_sus_syph_aids', default=self.pars.rel_sus_syph_aids),
            ss.FloatArr('rel_trans_syph_hiv', default=self.pars.rel_trans_syph_hiv),
            ss.FloatArr('rel_trans_syph_aids', default=self.pars.rel_trans_syph_aids),
        )

        return

    def update(self):

        # HIV changes due to syphilis
        syphilis = self.syphilis.active
        self.hiv.rel_sus[syphilis] *= self.rel_sus_hiv_syph[syphilis]
        self.hiv.rel_trans[syphilis] *= self.rel_trans_hiv_syph[syphilis]

        # Syphilis changes due to HIV
        if isinstance(self.syphilis, stisim.SyphilisPlaceholder):
            return

        hiv = self.hiv.cd4 < 500
        self.syphilis.rel_sus[hiv] *= self.rel_sus_syph_hiv[hiv]
        self.syphilis.rel_trans[hiv] *= self.rel_trans_syph_hiv[hiv]

        aids = self.hiv.cd4 < 200
        self.syphilis.rel_sus[aids] *= self.rel_sus_syph_aids[aids]
        self.syphilis.rel_trans[aids] = self.rel_trans_syph_aids[aids]

        return

