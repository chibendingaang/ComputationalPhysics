# Makefile for MPI-based Swendsen-Wang Ising model

CXX = mpicxx
CXXFLAGS = -O3 -std=c++17

TARGET = Ising2D_mpi
SRC = MCMC_2DIsing_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o energies_sub*.txt lattice_sub*.txt

