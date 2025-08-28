#!/bin/bash
#PBS -N sw_ising_mpi
#PBS -l nodes=1:ppn=16
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q batch

cd $PBS_O_WORKDIR

# Load MPI module (adjust to your cluster)
module load mpi

# Compile
make clean && make

# Run with 16 MPI ranks (one per sublattice)
mpirun -np 16 ./sw_ising_mpi

