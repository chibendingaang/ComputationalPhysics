import numpy as np

class Energy:
    def __init__(self, L):
        self.L = L
        self.sigma = 2*np.random.randint(0, 2, (L, L)) - 1
        self.H = self._calculate_H()

    def _calculate_H(self):
        H = 0
        for i in range(0, L-1):
            for j in range(0, L-1):
                H += self.sigma[i, j] * self.sigma[i, j+1] + self.sigma[i, j] * self.sigma[i+1, j]
        H += np.sum([self.sigma[L-1, j] * self.sigma[L-1, j+1] for j in range(0, L-1)])
        H += np.sum([self.sigma[i, L-1] * self.sigma[i+1, L-1] for i in range(0, L-1)])
        return H

class MarkovChainMonteCarlo:
    def __init__(self, T):
        self.T = T
        self.energy = Energy(L)

    def run(self):
        #do the MonteCarlo process for a given temperature
        pass