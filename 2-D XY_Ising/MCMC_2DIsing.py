import numpy as np
import matplotlib.pyplot as plt
import random

class IsingModelSwendsenWang:
    def __init__(self, L, T, J=1.0):
        """
        Initialize the Ising Model with the Swendsen-Wang algorithm.
        
        Parameters:
        - L: Size of the lattice (L x L)
        - T: Temperature of the system
        - J: Interaction strength (default 1.0)
        """
        self.L = L  # Lattice size
        self.T = T  # Temperature
        self.J = J  # Interaction strength (default = 1.0)
        self.lattice = self.initialize_lattice()

    def initialize_lattice(self):
        """
        Initialize the lattice with random spins (+1 or -1).
        """
        return np.random.choice([-1, 1], size=(self.L, self.L))

    def calc_energy(self):
        """
        Calculate the energy of the current configuration of the lattice.
        """
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                S = self.lattice[i, j]
                # Nearest neighbors with periodic boundary conditions
                neighbors = self.lattice[(i+1)%self.L, j] + self.lattice[i, (j+1)%self.L] + self.lattice[(i-1)%self.L, j] + self.lattice[i, (j-1)%self.L]
                energy += -self.J * S * neighbors
        return energy / 2.0  # since each pair counted twice

    def create_clusters(self):
        """
        Create clusters of aligned spins using bond formation based on temperature.
        """
        bonds = np.zeros_like(self.lattice, dtype=bool)
        clusters = np.zeros_like(self.lattice, dtype=int)
        current_cluster = 0

        # Step 1: Identify bonds between neighbors with the bond probability
        for i in range(self.L):
            for j in range(self.L):
                # Check bonds to right and down neighbors only (to avoid double-counting)
                if self.lattice[i, j] == self.lattice[(i+1) % self.L, j]:  # Bond in vertical direction
                    if random.random() < 1 - np.exp(-2 * self.J / self.T):  # Bond probability
                        bonds[i, j] = True
                        bonds[(i+1) % self.L, j] = True
                if self.lattice[i, j] == self.lattice[i, (j+1) % self.L]:  # Bond in horizontal direction
                    if random.random() < 1 - np.exp(-2 * self.J / self.T):  # Bond probability
                        bonds[i, j] = True
                        bonds[i, (j+1) % self.L] = True

        # Step 2: Assign clusters using breadth-first search (BFS)
        def bfs(start_i, start_j):
            nonlocal current_cluster
            current_cluster += 1
            stack = [(start_i, start_j)]
            while stack:
                i, j = stack.pop()
                if clusters[i, j] == 0:  # If not already assigned to a cluster
                    clusters[i, j] = current_cluster
                    # Check neighbors and assign them to the cluster if connected by a bond
                    for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                        ni, nj = ni % self.L, nj % self.L  # Apply periodic boundary conditions
                        if bonds[i, j] and clusters[ni, nj] == 0 and self.lattice[ni, nj] == self.lattice[i, j]:
                            stack.append((ni, nj))

        for i in range(self.L):
            for j in range(self.L):
                if clusters[i, j] == 0:  # If this spin is not yet assigned to a cluster
                    bfs(i, j)

        return clusters

    def flip_clusters(self, clusters):
        """
        Flip the spins of entire clusters with 50% probability.
        """
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            if random.random() < 0.5:
                self.lattice[clusters == cluster] *= -1

    def swendsen_wang_step(self):
        """
        Perform one Monte Carlo step of the Swendsen-Wang algorithm.
        """
        clusters = self.create_clusters()
        self.flip_clusters(clusters)

    def simulate(self, steps=10000):
        """
        Run the Swendsen-Wang simulation for a given number of steps.
        
        Parameters:
        - steps: Number of Monte Carlo steps
        """
        energies = []
        for step in range(steps):
            self.swendsen_wang_step()
            if step % 100 == 0:  # Track energy every 100 steps
                energy = self.calc_energy()
                energies.append(energy)
        return energies

    def plot_lattice(self):
        """
        Plot the current configuration of the lattice.
        """
        plt.imshow(self.lattice, cmap='gray')
        plt.title(f"Ising Model at T = {self.T}")
        plt.show()

    def plot_energy(self, energies):
        """
        Plot the energy of the system over time.
        
        Parameters:
        - energies: List of energies recorded during the simulation
        """
        plt.plot(energies)
        plt.title("Energy vs Steps")
        plt.xlabel("Steps")
        plt.ylabel("Energy")
        plt.show()


# Example usage
L = 32  # Lattice size
T = 2.0  # Temperature (T < T_c for quenched system)
steps = 5000  # Number of Monte Carlo steps

ising = IsingModelSwendsenWang(L, T)
energies = ising.simulate(steps)

# Plot the final lattice configuration
ising.plot_lattice()

# Plot the energy over time
ising.plot_energy(energies)

