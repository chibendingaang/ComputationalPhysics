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
#import pickle as pkl
from scipy import signal
sin = np.sin
cos = np.cos
summ = np.sum
arr  = np.array
av = np.average
pi = np.pi
exp = np.exp


start = time.clock() #python 2
#start = time.perf_counter() #python3
L = 128
g = 0.1; mu = 0.5; Gamma =  1
steps = 500000
dt = 0.01; dx = 1.
Deltay = 0.05; Deltaz = 0

a = np.arange(L, dtype = np.float64)
qn = 5; sn = np.sin(2*np.pi*qn*a/L)
#sn = signal.square(2*np.pi*a//L, duty = 0.5 - 0.1) #keep the duty slightly less than 0.5 to keep the signal symmetric
Sx0 = np.ones(L); Sy0 = np.zeros(L) + Deltay*sn; Sz0 = np.zeros(L) + Deltaz*sn
#RK = 'RK2'
print ('Signal\n' , sn) 
 #Total number of steps


avg = np.average
S2init = Sx0**2+ Sy0**2+ Sz0**2
S2tot = np.sum(S2init)
modphisqin = avg(S2init)
print('<S^2> initial value ', modphisqin)
print("Evolution with all terms, no noise; initial perturbation: A sine wave signal")
print('L = ', L, ', dt = ', dt, ', steps = ', steps, ', mu = ', mu, ', g = ', g, ', Deltay = ', Deltay, 
      ', Deltaz = ', Deltaz , ', qn = ', qn)



'''
The main function which integrates the Stochastic PDE with finite difference
2nd order Runge-Kutta scheme.
'''

##The discretized equation
def RK2(s_1, s_2, s_3, dx, dt):
    dx2 = dx**2;
    #2D-array for the noise terms, vector components
    #noise = np.random.normal(0, np.sqrt(2*alpha*dt), (3,L))
    #1D-arrays to store the spin component values in the intermediate steps

##K1
    S_1, S_2, S_3 = np.zeros(L), np.zeros(L), np.zeros(L)
    s1new, s2new, s3new = 0,0,0
    
    #arrays for the nonlinear term (S.S -1)S_alpha
    Nl1 = np.zeros(L)
    Nl2 = np.zeros(L)
    Nl3 = np.zeros(L)
    
    Laps1 = np.zeros(L)
    Laps2 = np.zeros(L)
    Laps3 = np.zeros(L)
    
    for i in range(L):
        #Periodic boundary conditions
        s1, s1xr, s1xl = s_1[i], s_1[(i+1)%L], s_1[(i-1)%L]
        s2, s2xr, s2xl = s_2[i], s_2[(i+1)%L], s_2[(i-1)%L]        
        s3, s3xr, s3xl = s_3[i], s_3[(i+1)%L], s_3[(i-1)%L]
               
        #3-point stencil along each lattice direction
        s1nnf = arr([s1xr, s1xl])
        s2nnf = arr([s2xr, s2xl])
        s3nnf = arr([s3xr, s3xl])
        Lap1s1 = (summ(s1nnf) - 2*s1)/dx2
        Lap1s2 = (summ(s2nnf) - 2*s2)/dx2
        Lap1s3 = (summ(s3nnf) - 2*s3)/dx2
        
        p = arr([s1**2, s2**2, s3**2])
        #Following arrays are required to be stored; since we want to find their Laplacian in the next loop
        #nonlinear term
        Nl1[i] = (summ(p) - 1)*s1
        Nl2[i] = (summ(p) - 1)*s2
        Nl3[i] = (summ(p) - 1)*s3
        
        Laps1[i] = Lap1s1
        Laps2[i] = Lap1s2
        Laps3[i] = Lap1s3
                

    for i in range(L):         
        #s1, s1xr, s1xl = s_1[i], s_1[(i+1)%L], s_1[(i-1)%L]
        #s2, s2xr, s2xl = s_2[i], s_2[(i+1)%L], s_2[(i-1)%L]        
        #s3, s3xr, s3xl = s_3[i], s_3[(i+1)%L], s_3[(i-1)%L]
        
        laps1 = Laps1[i]; laps2 = Laps2[i]; laps3 = Laps3[i]

        xnnf = arr([Laps1[(i+1)%L], Laps1[(i-1)%L]])
        ynnf = arr([Laps2[(i+1)%L], Laps2[(i-1)%L]])
        znnf = arr([Laps3[(i+1)%L], Laps3[(i-1)%L]])
##This is Laplacian of Laplacian of S[i]; have I nonlinearized it in the process?
        Lap2s1 = (summ(xnnf) - 2*laps1)/dx2
        Lap2s2 = (summ(ynnf) - 2*laps2)/dx2
        Lap2s3 = (summ(znnf) - 2*laps3)/dx2
        
        nl1, nl2, nl3 = Nl1[i], Nl2[i], Nl3[i]
        
        nl1nbrs = arr([Nl1[(i+1)%L], Nl1[(i-1)%L]]) #h1xr, h1xl, h1yr, h1yl, h1zr, h1zl
        nl2nbrs = arr([Nl2[(i+1)%L], Nl2[(i-1)%L]]) #, h2xr, h2xl, h2yr, h2yl, h2zr, h2zl
        nl3nbrs = arr([Nl3[(i+1)%L], Nl3[(i-1)%L]]) # h3xr, h3xl, h3yr, h3yl, h3zr, h3zl
              
        Laph1 = (summ(nl1nbrs) - 2*nl1)/(dx2)
        Laph2 = (summ(nl2nbrs) - 2*nl2)/(dx2)
        Laph3 = (summ(nl3nbrs) - 2*nl3)/(dx2)

        Cx = Lap2s1; Dx = Laph1
        Cy = Lap2s2; Dy = Laph2
        Cz = Lap2s3; Dz = Laph3

        Gx = g*(s2*laps3 - s3*laps2)
        Gy = g*(s3*laps1 - s1*laps3)
        Gz = g*(s1*laps2 - s2*laps1)
        
        Px = mu*0.5*(s2*(s3xr - s3xl) - s3*(s2xr - s2xl)) #sy.gsradx_sz - sz.gradx_sy
        Py = mu*0.5*(s3*(s1xr - s1xl) - s1*(s3xr - s3xl))
        Pz = mu*0.5*(s1*(s2xr - s2xl) - s2*(s1xr - s1xl))
        #Ex = 0.5*(noise[0][(i+1)%L] - noise[0][(i-1)%L] )/dx #3*noise[0][i][j][k]
        #Ey = 0.5*(noise[1][(i+1)%L] - noise[1][(i-1)%L] )/dx #3*noise[1][i][j][k]
        #Ez = 0.5*(noise[2][(i+1)%L] - noise[2][(i-1)%L] )/dx #3*noise[2][i][j][k]

        s1new = dt*(Dx - Cx + Gx + Px)  #s1new = Ax - Bx - Cx + Ex
        s2new = dt*(Dy - Cy + Gy + Py)  
        s3new = dt*(Dz - Cz + Gz + Pz) 

        S_1[i], S_2[i], S_3[i] = s1new, s2new, s3new
    
##K2     
    S__1, S__2, S__3 = np.zeros(L), np.zeros(L), np.zeros(L)
    s1new, s2new, s3new = 0,0,0
    
    #arrays for the nonlinear term (S.S -1)S_alpha
    Nl1 = np.zeros(L)
    Nl2 = np.zeros(L)
    Nl3 = np.zeros(L)
    
    Laps1 = np.zeros(L)
    Laps2 = np.zeros(L)
    Laps3 = np.zeros(L)
  
    for i in range(L):
        #Periodic boundary conditions
        s1, s1xr, s1xl = s_1[i]+ S_1[i]/2, s_1[(i+1)%L] + S_1[(i+1)%L]/2, s_1[(i-1)%L] + S_1[(i-1)%L]/2
        s2, s2xr, s2xl = s_2[i]+ S_2[i]/2, s_2[(i+1)%L] + S_2[(i+1)%L]/2, s_2[(i-1)%L] + S_2[(i-1)%L]/2        
        s3, s3xr, s3xl = s_3[i]+ S_3[i]/2, s_3[(i+1)%L] + S_3[(i+1)%L]/2, s_3[(i-1)%L] + S_3[(i-1)%L]/2
               
        #3-point stencil along each lattice direction
        s1nnf = arr([s1xr, s1xl])
        s2nnf = arr([s2xr, s2xl])
        s3nnf = arr([s3xr, s3xl])
        
        Lap1s1 = (summ(s1nnf) - 2*s1)/dx2
        Lap1s2 = (summ(s2nnf) - 2*s2)/dx2
        Lap1s3 = (summ(s3nnf) - 2*s3)/dx2
        
        p = arr([s1**2, s2**2, s3**2])
        #Following arrays are required to be stored; since we want to find their Laplacian in the next loop
        #nonlinear term
        Nl1[i] = (summ(p) - 1)*s1
        Nl2[i] = (summ(p) - 1)*s2
        Nl3[i] = (summ(p) - 1)*s3
        
        Laps1[i] = Lap1s1
        Laps2[i] = Lap1s2
        Laps3[i] = Lap1s3
                

    for i in range(L):         
        #s1, s1xr, s1xl = s_1[i]+ S_1[i]/2, s_1[(i+1)%L] + S_1[(i+1)%L]/2, s_1[(i-1)%L] + S_1[(i-1)%L]/2
        #s2, s2xr, s2xl = s_2[i]+ S_2[i]/2, s_2[(i+1)%L] + S_2[(i+1)%L]/2, s_2[(i-1)%L] + S_2[(i-1)%L]/2        
        #s3, s3xr, s3xl = s_3[i]+ S_3[i]/2, s_3[(i+1)%L] + S_3[(i+1)%L]/2, s_3[(i-1)%L] + S_3[(i-1)%L]/2
        
        laps1 = Laps1[i]; laps2 = Laps2[i]; laps3 = Laps3[i]

        xnnf = arr([Laps1[(i+1)%L], Laps1[(i-1)%L]])
        ynnf = arr([Laps2[(i+1)%L], Laps2[(i-1)%L]])
        znnf = arr([Laps3[(i+1)%L], Laps3[(i-1)%L]])
##This is Laplacian of Laplacian of S[i]; have I nonlinearized it in the process?
        Lap2s1 = (summ(xnnf) - 2*laps1)/dx2
        Lap2s2 = (summ(ynnf) - 2*laps2)/dx2
        Lap2s3 = (summ(znnf) - 2*laps3)/dx2
        
        nl1, nl2, nl3 = Nl1[i], Nl2[i], Nl3[i]
        
        nl1nbrs = arr([Nl1[(i+1)%L], Nl1[(i-1)%L]]) #h1xr, h1xl, h1yr, h1yl, h1zr, h1zl
        nl2nbrs = arr([Nl2[(i+1)%L], Nl2[(i-1)%L]]) #, h2xr, h2xl, h2yr, h2yl, h2zr, h2zl
        nl3nbrs = arr([Nl3[(i+1)%L], Nl3[(i-1)%L]]) # h3xr, h3xl, h3yr, h3yl, h3zr, h3zl
              
        Laph1 = (summ(nl1nbrs) - 2*nl1)/(dx2)
        Laph2 = (summ(nl2nbrs) - 2*nl2)/(dx2)
        Laph3 = (summ(nl3nbrs) - 2*nl3)/(dx2)

        Cx = Lap2s1; Dx = Laph1
        Cy = Lap2s2; Dy = Laph2
        Cz = Lap2s3; Dz = Laph3

        Gx = g*(s2*laps3 - s3*laps2)
        Gy = g*(s3*laps1 - s1*laps3)
        Gz = g*(s1*laps2 - s2*laps1)
        
        Px = mu*0.5*(s2*(s3xr - s3xl) - s3*(s2xr - s2xl)) #sy.gsradx_sz - sz.gradx_sy
        Py = mu*0.5*(s3*(s1xr - s1xl) - s1*(s3xr - s3xl))
        Pz = mu*0.5*(s1*(s2xr - s2xl) - s2*(s1xr - s1xl))
        #Ex = 0.5*(noise[0][(i+1)%L] - noise[0][(i-1)%L] )/dx #3*noise[0][i][j][k]
        #Ey = 0.5*(noise[1][(i+1)%L] - noise[1][(i-1)%L] )/dx #3*noise[1][i][j][k]
        #Ez = 0.5*(noise[2][(i+1)%L] - noise[2][(i-1)%L] )/dx #3*noise[2][i][j][k]

        s1new = dt*(Dx - Cx + Gx + Px)  #s1new = Ax - Bx - Cx + Ex
        s2new = dt*(Dy - Cy + Gy + Py)  
        s3new = dt*(Dz - Cz + Gz + Pz) 

        S__1[i], S__2[i], S__3[i] = s_1[i] + s1new, s_2[i] + s2new, s_3[i]+ s3new
    
    return S__1, S__2, S__3

#dt used here is half of dt; since Operator splitting method is used       

#def timeiteration(sxini, syini, szini, steps):
s_1, s_2, s_3 = Sx0, Sy0, Sz0 #initialzing
#Writing 2D-arrays to store the Spin values at each time step is MEMORY EXPENSIVE!!!
Modspinsq = np.zeros([steps, L])

for t in range(steps):
    s_1, s_2, s_3 = RK2(s_1, s_2, s_3, dx, dt)
    if t%100 == 0: 
        if t == 1000: print ('average S1, S2, S3 values: ', av(s_1), av(s_2), av(s_3))
        f = open('../../qmodes/q{}/s_1_{}_Gamma_{}_mu_{}_allterms_dt_{}.npy'.format( qn, t, Gamma, mu, dt), 'wb'); np.save(f, s_1); f.close()
        f = open('../../qmodes/q{}/s_2_{}_Gamma_{}_mu_{}_allterms_dt_{}.npy'.format( qn, t, Gamma, mu, dt), 'wb'); np.save(f, s_2); f.close()
        f = open('../../qmodes/q{}/s_3_{}_Gamma_{}_mu_{}_allterms_dt_{}.npy'.format( qn, t, Gamma, mu, dt), 'wb'); np.save(f, s_3); f.close()
        Modspinsq[t] = s_1**2 + s_2**2 + s_3**2

f = open('../../qmodes/q{}/s_1_{}_Gamma_{}_mu_{}_allterms_dt_{}.npy'.format( qn, t, Gamma, mu, dt), 'wb'); np.save(f, s_1); f.close()
f = open('../../qmodes/q{}/s_2_{}_Gamma_{}_mu_{}_allterms_dt_{}.npy'.format( qn, t, Gamma, mu, dt), 'wb'); np.save(f, s_2); f.close()
f = open('../../qmodes/q{}/s_3_{}_Gamma_{}_mu_{}_allterms_dt_{}.npy'.format( qn, t, Gamma, mu, dt), 'wb'); np.save(f, s_3); f.close()

S_1ftot = np.sum(s_1)
S_2ftot = np.sum(s_2)
S_3ftot = np.sum(s_3)
print ('Final values')
print ('Sxtot, Sytot, Sztot')
print(S_1ftot, ', ', S_2ftot, ', ',  S_3ftot)

avmodspin = av(Modspinsq, 1)
S2init = Sx0**2+ Sy0**2+ Sz0**2; S2tot = np.sum(S2init)/L
modphisq_avg = avmodspin/S2tot #see definition of S2tot = np.sum(S2init)

output = open('../../qmodes/modphisq_avg_q{}_Gamma_{}_mu_{}_weirdqx_dt_{}_{}.npy'.format(L, qn, Gamma, Gamma, mu, dt,steps ), 'wb')
np.save(output, modphisq_avg , allow_pickle = False)
output.close()

stop = time.clock() #python2
#stop = time.perf_counter()  #python3
print ("\n", stop - start, 'time for simulation run')
