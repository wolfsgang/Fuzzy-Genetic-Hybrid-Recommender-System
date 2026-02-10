import numpy as np


def genetic(cost_function):
    """Simple genetic optimizer returning the best chromosome found."""
    # [population, mutation_rate, generations, chromosome_length, winners_per_generation]
    params = [50, 0.05, 50, 21, 5]

    cur_pop = np.random.randint(2, size=(params[0], params[3]))
    next_pop = np.zeros((cur_pop.shape[0], cur_pop.shape[1]))
    fit_vec = np.zeros((params[0], 2))

    for _ in range(params[2]):
        fit_vec = np.array([np.array([x, cost_function()]) for x in range(params[0])])

        winners = np.zeros((params[4], params[3]))
        for n in range(len(winners)):
            selected = np.random.choice(range(len(fit_vec)), params[4] // 2, replace=False)
            winner_idx = np.argmin(fit_vec[selected, 1])
            winners[n] = cur_pop[int(fit_vec[selected[winner_idx]][0])]

        next_pop[: len(winners)] = winners

        repeats = (params[0] - len(winners)) // len(winners)
        next_pop[len(winners) :] = np.array(
            [
                np.array(np.random.permutation(np.repeat(winners[:, x], repeats, axis=0)))
                for x in range(winners.shape[1])
            ]
        ).T

        mutation_mask = [float(np.random.normal(0, 2, 1)) if np.random.random() < params[1] else 1.0 for _ in range(next_pop.size)]
        next_pop = np.multiply(next_pop, np.array(mutation_mask).reshape(next_pop.shape))
        cur_pop = next_pop

    best_soln = cur_pop[np.argmin(fit_vec[:, 1])]
    return best_soln
