import numpy as np
import matplotlib.pyplot as plt

av = np.average

L = 128 ; mu = 0.5;  dt = 0.05
Gamma = 0;  g = 0.5; pi = np.pi

Deltay = 0.1
#x,y,z = np.meshgrid(a,a,a)
#t = np.shape(M54q1phisq)[0];  
t = 200000

S2 = np.load('../L_128/squarewave/allterms/trial_Gamma_0_modphisq_avg_mu_0.5_weirdqx_dt_0.05_200000.npy', 'r')


time = np.arange(0,t)
plt.figure()
plt.plot(time, S2)
plt.grid()
plt.xlim((0, 200000)); plt.ylim((0.9, 2));
#plt.title('|S|^2 with time; L = {}, mu = {}'.format(L, mu))
plt.xlabel('steps'); plt.ylabel('|S|^2')
#plt.savefig('./L_{}/positgsign/squarewaveplots/init4/Modphisqavg_{}_{}_q{}_dt_{}.png'.format(L, qn, L, dynamictype, qn, dt))
plt.show()
