#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <set>
#include <mpi.h>

// ------------------------------
// Swendsen-Wang Ising Model
// Each MPI rank simulates one sublattice
// ------------------------------

class IsingModelSW {
public:
    int L;                       // lattice size
    double T;                    // temperature
    double J;                    // interaction strength
    std::vector<std::vector<int>> lattice;
    std::mt19937 gen;            // RNG engine based on Mersenne-Twister algorithm
    std::uniform_real_distribution<double> dist; // uniform [0,1)

    IsingModelSW(int L_in, double T_in, double J_in = 1.0, unsigned int seed = 42)
        : L(L_in), T(T_in), J(J_in), gen(seed), dist(0.0, 1.0) {
        initialize_lattice();
    }

    void initialize_lattice() {
        lattice.assign(L, std::vector<int>(L, 1));
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                lattice[i][j] = (dist(gen) < 0.5) ? -1 : 1;
            }
        }
    }

    double calc_energy() {
        double energy = 0.0;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int S = lattice[i][j];
                int right = lattice[i][(j + 1) % L];
                int down  = lattice[(i + 1) % L][j];
                energy += -J * S * (right + down);
            }
        }
        return energy;
    }

    // Union-Find (Disjoint Set Union) to manage clusters
    struct DSU {
        std::vector<int> parent, rank;
        DSU(int n) {
            parent.resize(n);
            rank.assign(n, 0);
            for (int i = 0; i < n; i++) parent[i] = i;
        }
        int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }
        void unite(int x, int y) {
            x = find(x); y = find(y);
            if (x == y) return;
            if (rank[x] < rank[y]) parent[x] = y;
            else if (rank[x] > rank[y]) parent[y] = x;
            else { parent[y] = x; rank[x]++; }
        }
    };

    void swendsen_wang_step() {
        double p = 1.0 - std::exp(-2.0 * J / T);
        DSU dsu(L * L);

        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int idx = i * L + j;
                int spin = lattice[i][j];
                int ni = i, nj = (j + 1) % L;
                if (lattice[ni][nj] == spin && dist(gen) < p) {
                    dsu.unite(idx, ni * L + nj);
                }
                ni = (i + 1) % L; nj = j;
                if (lattice[ni][nj] == spin && dist(gen) < p) {
                    dsu.unite(idx, ni * L + nj);
                }
            }
        }

        std::set<int> flippedClusters;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int root = dsu.find(i * L + j);
                if (flippedClusters.find(root) == flippedClusters.end()) {
                    if (dist(gen) < 0.5) {
                        flippedClusters.insert(root);
                    }
                }
            }
        }

        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                int root = dsu.find(i * L + j);
                if (flippedClusters.find(root) != flippedClusters.end()) {
                    lattice[i][j] *= -1;
                }
            }
        }
    }

    std::vector<double> simulate(int steps, int recordEvery = 100) {
        std::vector<double> energies;
        for (int step = 0; step < steps; step++) {
            swendsen_wang_step();
            if (step % recordEvery == 0) {
                double e = calc_energy();
                energies.push_back(e);
            }
        }
        return energies;
    }
};


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int L_global = 512;
    int L_sub = L_global / 4;   // 128
    int numSubs = 16;           
    double T = 2.0;
    int steps = 5000;

    if (size != numSubs) {
        if (rank == 0) {
            std::cerr << "Error: Please run with " << numSubs << " MPI processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Each rank simulates one sublattice
    IsingModelSW model(L_sub, T, 1.0, 1234 + rank);
    auto energies = model.simulate(steps);

    // Each rank writes its own output
    std::ofstream fout("energies_sub" + std::to_string(rank) + ".txt");
    for (size_t i = 0; i < energies.size(); i++) {
        fout << i << " " << energies[i] << "\n";
    }
    fout.close();

    std::ofstream flattice("lattice_sub" + std::to_string(rank) + ".txt");
    for (int i = 0; i < model.L; i++) {
        for (int j = 0; j < model.L; j++) {
            flattice << model.lattice[i][j] << " ";
        }
        flattice << "\n";
    }
    flattice.close();

    if (rank == 0) {
        std::cout << "Simulation finished with " << size << " ranks.\n";
    }

    MPI_Finalize();
    return 0;
}


