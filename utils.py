"""
Implements an ad hoc and very approximate solution to a linear assignment problem
"""

# %% Imports and settings
import numpy as np


def assign_partners(a1, a2, n_steps=None):
    """
    a1: first array
    a2: second array
    """

    return


if __name__ == '__main__':

    import numpy as np

    n1 = 10
    n2 = 12
    np.random.seed(42)
    f_ages = np.clip(np.random.normal(40, 12, n1), 0, 100)
    m_ages = np.clip(np.random.normal(40, 12, n2), 0, 100)
    age_gaps = np.random.normal(7, 2, n1)
    desired_m_ages = f_ages + age_gaps

    m_idx = np.argsort(m_ages)
    sorted_m_ages = m_ages[m_idx]
    f_idx = np.argsort(desired_m_ages)
    sorted_desired_ages = desired_m_ages[f_idx]


    a1 = np.sort(np.clip(np.random.normal(50, 12, n1), 0, 100))  # Ages of desired partners
    a2 = np.sort(np.clip(np.random.normal(40, 10, n2), 0, 100))  # Ages of men

    # Slow way
    import scipy.optimize as spo
    import scipy.spatial as spsp

    def scipy_lsa():
        cost = spsp.distance_matrix(a1[:, np.newaxis], a2[:, np.newaxis])
        row_ind, col_ind = spo.linear_sum_assignment(cost)
        return row_ind, col_ind

    import timeit
    print(timeit.timeit(scipy_lsa, number=100))


    # from timeit import default_timer as timer
    # start = timer()
    # scipy_lsa()
    # end = timer()
    # print(end - start)

    # print(cost[row_ind, col_ind].sum())

    assign_partners(a1, a2)
