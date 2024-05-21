""" STIsim utilities """

import sciris as sc
import pandas as pd
import numpy as np


__all__ = ['make_init_prev_fn']


def make_init_prev_fn(module, sim, uids):
    """ Initialize prevalence by sex and risk group """

    if sc.isnumber(module.init_prev_data):
        init_prev = module.init_prev_data

    elif isinstance(module.init_prev_data, pd.DataFrame):

        init_prev = pd.Series(index=uids)
        df = module.init_prev_data

        n_risk_groups = sim.networks.structuredsexual.pars.n_risk_groups
        for rg in range(n_risk_groups):
            for sex in ['female', 'male']:
                for sw in [0, 1]:
                    thisdf = df.loc[(df.risk_group==rg) & (df.sex==sex) & (df.sw==sw)]
                    conditions = (sim.people[sex] & (sim.networks.structuredsexual.risk_group==rg))
                    if sw:
                        if sex == 'female': conditions = conditions & sim.networks.structuredsexual.fsw
                        if sex == 'male':   conditions = conditions & sim.networks.structuredsexual.client
                    init_prev[conditions[uids]] = thisdf.init_prev.values[0]

    else:
        errormsg = 'Format of init_prev_data must be float or dataframe.'
        raise ValueError(errormsg)

    # Scale and validate
    init_prev = init_prev * module.pars.rel_init_prev
    init_prev = np.clip(init_prev, a_min=0, a_max=1)

    return init_prev
