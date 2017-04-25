import numpy
import math


def genetic(costFunction):
    # [Init pop (pop=50), mut rate (=5%), num generations (50), chromosome/solution length (19), # winners/per gen]
    params = [50, 0.05, 50, 21, 5]
    # generate initial population
    # curPop = np.random.choice(np.arange(-15,15,step=0.01),size=(params[0],params[3]),replace=False)
    curPop = np.random.randint(2, size=(params[0], params[3]))

    for i in range(params[2]):
        fitVec = np.array(
            [np.array([x, np.sum(costFunction(curPop[x]))]) for x in range(params[0])])
        # e.g. [0, 0.11] means that the 0th element in curPop (first solution) has an error of 0.11

        # create a winners array of size winner*solution
        winners = np.zeros((params[4], params[3]))
        for n in range(len(winners)):
            selected = np.random.choice(range(len(fitVec)), params[4] / 2,
                                        replace=False)  # select random indexes from pop
            wnr = np.argmin(fitVec[selected, 1])  # select one index with min fitness error
            winners[n] = curPop[int(fitVec[selected[wnr]][0])]  # add to winner population

        nextPop[:len(winners)] = winners  # populate new gen with winners
        # mating using crossover via permutation
        nextPop[len(winners):] = np.array(
            [np.array(
                np.random.permutation(np.repeat(winners[:, x], ((params[0] - len(winners)) / len(winners)), axis=0)))
             for x in range(winners.shape[1])]).T  # Populate the rest of the generation with offspring of mating pairs

        # mutation
        nextPop = np.multiply(nextPop, np.matrix(
            [np.float(np.random.normal(0, 2, 1)) if random.random() < params[1] else 1 for x in
             range(nextPop.size)]).reshape(nextPop.shape))  # randomly mutate part of the population
        curPop = nextPop

    best_soln = curPop[np.argmin(fitVec[:, 1])]
    return best_soln
