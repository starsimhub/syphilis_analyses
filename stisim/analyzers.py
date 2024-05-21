"""
Analyzers
"""

import starsim as ss
import stisim as sti


__all__ = ['GroupResults']


class GroupResults(ss.Analyzer):
    """
    Group results for one module by attributes of another module
    Examples:
        syph_by_pregnancy = sti.GroupResults(module=('diseases', 'syphilis'), results=['prevalence', 'new_infections'], group_by=('demographics', 'pregnancy', 'pregnant'))
    """

    def __init__(self, module=None, results=None, group_by=None):
        super().__init__(label='GroupResults')
        self.module = module
        self.results = results
        self.group_by = group_by
        return

    def initialize(self, sim):
        return super().initialize(sim)

    def apply(self, sim):
        pass

    def finalize(self):
        return super().finalize()

