## TL;DR

This project is a **parallel C++ implementation** of the Swendsen–Wang Monte Carlo algorithm for the 2D Ising model. Instead of simulating the full 512×512 lattice as one piece, the system is divided into **16 sublattices** (128×128 each). Using MPI, each sublattice is assigned to a separate process, allowing the simulation to scale across multiple CPU cores or nodes. Each process outputs its own energy trace (`energies_sub*.txt`) and final lattice configuration (`lattice_sub*.txt`).

The setup is simple to build and run on an HPC cluster. A Makefile is provided for compilation with `mpicxx`, and a PBS job script shows how to launch the simulation with 16 ranks. This structure makes it beginner-friendly: the outputs are plain text, the code is heavily commented, and the design encourages you (or future-me) to extend it with further measurements or larger runs.

