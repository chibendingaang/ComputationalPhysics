from math import log, pi
import numpy as np

def tau(n, L):
    mu = 1/2.; A = .05 #initial perturbation
    q= 2*pi*n/L
    g = 2*(mu*q - q**4)
    t = log((1 + g/A**2))/g
    return t

N = np.arange(1,17)
L = 128
Tau = np.zeros((len(N)))
for i,n in enumerate(N):
    Tau[i] = tau(n,L)

print Tau    
    