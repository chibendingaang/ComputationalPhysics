''' Dumps magnetization values evolving from the 
Model J stochastic PDE 
into a pickle file every 100 steps.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import time
import pickle as pkl
from scipy import signal
sin = np.sin
cos = np.cos
summ = np.sum
arr  = np.array
av = np.average
pi = np.pi
exp = np.exp

#import datetime 
#print(datetime.datetime().now().time())
start = time.clock()

L = 256
a = np.arange(L, dtype = np.float64)
#x = np.meshgrid(a)
sn = signal.square(2*np.pi*a//L, duty = 0.5 - 0.1) #keep the duty slightly less than 0.5 to keep the signal symmetric

dt = 0.001
Deltay = 0.06; Deltaz = 0.08
#norm = 1/(np.sqrt(1+Deltay**2))
Sx0 = np.ones(L); Sy0 = np.zeros(L) + Deltay*sn; Sz0 = np.zeros(L) + Deltaz*sn


dx = 1.
g = 0.5; mu = 0.5; Gamma = 1
alpha = 0.001
steps = 1000000 #Temperature dependent coefficient

avg = np.average
S2init = Sx0**2+ Sy0**2+ Sz0**2
S2tot = np.sum(S2init)
modphisqin = avg(S2init)
print('<S^2> initial value ', modphisqin)
print("Evolution with all terms, no noise; initial perturbation: square wave")
print('L = ', L, ', dt = ', dt, ', steps = ', steps, ', mu = ', mu, ', g = ', g, ', Deltay = ', Deltay, 
      ', Deltaz = ', Deltaz)



'''
The main function which integrates the Stochastic PDE with finite difference
Euler scheme.
'''


def dynamics(sx, sy, sz, dt, dx):
    dx2 = dx**2
    #1D-arrays for the spin bases
    Sx, Sy, Sz = np.zeros(L), np.zeros(L), np.zeros(L)
    s1new, s2new, s3new = 0,0,0
    noise = np.random.normal(0, np.sqrt(2*alpha*dt), (3,L))
    #2D-array for the noise terms, vector components
    
    hx = np.zeros(L)
    hy = np.zeros(L)
    hz = np.zeros(L)

    Lapsx = np.zeros(L)
    Lapsy = np.zeros(L)
    Lapsz = np.zeros(L)

    for i in range(L):
        #Periodic boundary conditions
        s1, s1xr, s1xl = sx[i], sx[(i+1)%L], sx[(i-1)%L]
        s2, s2xr, s2xl = sy[i], sy[(i+1)%L], sy[(i-1)%L]        
        s3, s3xr, s3xl = sz[i], sz[(i+1)%L], sz[(i-1)%L]
               
        #3-point stencil along each lattice direction
        sxnnf = arr([s1xr, s1xl])
        synnf = arr([s2xr, s2xl])
        sznnf = arr([s3xr, s3xl])

        Lap1sx = (summ(sxnnf) - 2*s1)/dx2
        Lap1sy = (summ(synnf) - 2*s2)/dx2
        Lap1sz = (summ(sznnf) - 2*s3)/dx2
        
        p = arr([s1**2, s2**2, s3**2])
        hx[i] = (summ(p) - 1)*s1
        hy[i] = (summ(p) - 1)*s2
        hz[i] = (summ(p) - 1)*s3

        Lapsx[i] = Lap1sx
        Lapsy[i] = Lap1sy
        Lapsz[i] = Lap1sz
                

    for i in range(L):
        lapsx = Lapsx[i]
        lapsy = Lapsy[i]
        lapsz = Lapsz[i]

        xnnf = arr([Lapsx[(i+1)%L], Lapsx[(i-1)%L]])
        ynnf = arr([Lapsy[(i+1)%L], Lapsy[(i-1)%L]])
        znnf = arr([Lapsz[(i+1)%L], Lapsz[(i-1)%L]])

        Lap2sx = (summ(xnnf) - 2*lapsx)/dx2
        Lap2sy = (summ(ynnf) - 2*lapsy)/dx2
        Lap2sz = (summ(znnf) - 2*lapsz)/dx2

        s1, s1xr, s1xl = sx[i], sx[(i+1)%L], sx[(i-1)%L]
        s2, s2xr, s2xl = sy[i], sy[(i+1)%L], sy[(i-1)%L]        
        s3, s3xr, s3xl = sz[i], sz[(i+1)%L], sz[(i-1)%L]
                
        h1nbrs = arr([hx[(i+1)%L], hx[(i-1)%L]]) #h1xr, h1xl, h1yr, h1yl, h1zr, h1zl =
        h2nbrs = arr([hy[(i+1)%L], hy[(i-1)%L]]) #, h2xr, h2xl, h2yr, h2yl, h2zr, h2zl
        h3nbrs = arr([hz[(i+1)%L], hz[(i-1)%L]]) # h3xr, h3xl, h3yr, h3yl, h3zr, h3zl

        h1, h2, h3 = hx[i], hy[i], hz[i]
        
        Laphx = (summ(h1nbrs) - 2*h1)/(dx2)
        Laphy = (summ(h2nbrs) - 2*h2)/(dx2)
        Laphz = (summ(h3nbrs) - 2*h3)/(dx2)
        
        Gx = g*dt*(s2*lapsz - s3*lapsy)
        Gy = g*dt*(s3*lapsx - s1*lapsz)
        Gz = g*dt*(s1*lapsy - s2*lapsx)
        
        #anisotropy direction is chosen to be x
        #central difference
        Px = dt*mu*0.5*(s2*(s3xr - s3xl) - s3*(s2xr - s2xl)) #sy.gsradx_sz - sz.gradx_sy
        Py = dt*mu*0.5*(s3*(s1xr - s1xl) - s1*(s3xr - s3xl))
        Pz = dt*mu*0.5*(s1*(s2xr - s2xl) - s2*(s1xr - s1xl))
        
        Ax = s1
        Ay = s2
        Az = s3
        
        #B1x = lapsx; #Bx = dt*(B1x)
        Cx = dt*Lap2sx
        Dx = dt*Laphx
        #B1y = lapsy; #By = dt*(B1y)
        Cy = dt*Lap2sy
        Dy = dt*Laphy
        #B1z = lapsz; #Bz = dt*(B1z)
        Cz = dt*Lap2sz
        Dz = dt*Laphz
        '''
                Asq = Ax**2 + Ay**2 + Az**2
                Csq = Cx**2 + Cy**2 + Cz**2
                Dsq = Dx**2 + Dy**2 + Dz**2
                Gsq = Gx**2 + Gy**2 + Gz**2
                Psq = Px**2 + Py**2 + Pz**2
                GP = 2*Gx*Px + 2*Gy*Py + 2*Gz*Pz
                AC = 2*Ax*Cx + 2*Ay*Cy + 2*Az*Cz
                AD = 2*Ax*Dx + 2*Az*Dz + 2*Ay*Dy
                CD = 2*Cx*Dx + 2*Cy*Dy + 2*Cz*Dz
                norm = 1/(np.sqrt(Asq + Csq+ Dsq + Gsq+ Psq+ AC + AD + CD + GP))
        '''
        Ex = 0.5*(noise[0][(i+1)%L] - noise[0][(i-1)%L] )/dx #3*noise[0][i][j][k]
        Ey = 0.5*(noise[1][(i+1)%L] - noise[1][(i-1)%L] )/dx #3*noise[1][i][j][k]
        Ez = 0.5*(noise[2][(i+1)%L] - noise[2][(i-1)%L] )/dx #3*noise[2][i][j][k]
        
        s1new = (Ax + Dx - Cx + Gx + Px)  #s1new = Ax - Bx - Cx + Ex
        s2new = (Ay + Dy - Cy + Gy + Py)  #s2new = Ay - By - Cy + Ey
        s3new = (Az + Dz - Cz + Gz + Pz ) #s3new = Az - Bz - Cz + Ez

        Sx[i], Sy[i], Sz[i] = s1new, s2new, s3new
    return Sx, Sy, Sz

#def timeiteration(sxini, syini, szini, steps):
sx, sy, sz = Sx0, Sy0, Sz0 #initialzing
#Writing 4D-arrays to store the Spin values at each time step is MEMORY EXPENSIVE!!!
Modspinsq = np.zeros([steps, L])
    
for t in range(steps):
    sx, sy, sz = dynamics(sx, sy, sz, dt, dx)
    if t%100 == 0:
        f = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/mu_{}/init4/sx{}_mu_{}_allterms_dt_{}.npy'.format(L,  mu, t, mu, dt), 'wb'); np.save(f, sx); f.close()
        f = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/mu_{}/init4/sy{}_mu_{}_allterms_dt_{}.npy'.format(L,  mu, t, mu, dt), 'wb'); np.save(f, sy); f.close()
        f = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/mu_{}/init4/sz{}_mu_{}_allterms_dt_{}.npy'.format(L,  mu, t, mu, dt), 'wb'); np.save(f, sz); f.close()
    Modspinsq[t] = sx**2 + sy**2 + sz**2

f = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/mu_{}/init4/sx{}_mu_{}_allterms_dt_{}.npy'.format(L,  mu, t, mu, dt), 'wb'); np.save(f, sx); f.close()
f = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/mu_{}/init4/sy{}_mu_{}_allterms_dt_{}.npy'.format(L,  mu, t, mu, dt), 'wb'); np.save(f, sy); f.close()
f = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/mu_{}/init4/sz{}_mu_{}_allterms_dt_{}.npy'.format(L,  mu, t, mu, dt), 'wb'); np.save(f, sz); f.close()

Sxftot = np.sum(sx)
Syftot = np.sum(sy)
Szftot = np.sum(sz)
print ('Final values')
print ('Sxtot, Sytot, Sztot')
print(Sxftot, ', ', Syftot, ', ',  Szftot)

avmodspin = av(Modspinsq, 1)
modphisq_avg = avmodspin/S2tot #see definition of S2tot = np.sum(S2init)

output = open('../ordered_then_perturbed/1Dchain/L_{}/squarewave/allterms/init4_modphisq_avg_mu_{}_weirdqx_dt_{}_{}.npy'.format(L, mu, dt,steps ), 'wb')
np.save(output, modphisq_avg , allow_pickle = False)
output.close()

stop = time.clock()
print ("\n", stop - start, 'time for simulation run')