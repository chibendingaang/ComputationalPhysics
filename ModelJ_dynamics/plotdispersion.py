import numpy as np
import matplotlib.pyplot as plt
import plotmag

pi = np.pi; exp = np.exp; mean = np.mean
L = 54; dt = 0.02
mu = 0.5; Gam = 1; g = 0.5
#qn = np.arange(8)
qn = 7


a = np.arange(L);
Deltay = 0.05
x,y,z = np.meshgrid(a,a,a)
#t = np.shape(M54q1phisq)[0];  
#phisq = M54q1phisq[::10]

t = 20
#logsperpt = plotmag.logsperp(qn, L, t)
time = np.arange(0,t,10, dtype=np.float64)
qn = np.arange(1,11) 
slope = np.zeros(10)
qx = 2*pi*qn/L #used for plotting
disp = mu*qx - 9*Gam*qx**4


def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    return m

for i,q in enumerate(qn):
    slope[i] = best_fit_slope(time,plotmag.logsperp(q, L, t))

dynamictype = 'allterms'

plt.figure()
#plt.plot(qn, disp, 'r', label = 'expected curve')
plt.plot(qn, slope, 'g', label = 'calculated curve')
plt.legend()
plt.grid()
plt.xlabel('qn'); plt.ylabel('d ln(Sperp) /dt')
plt.title('slope of ln(Sperp) vs t; L = {}, mu = {}, qn = {}'.format(L, mu, qn))
plt.savefig('./L_{}/positgsign/squarewaveplots/init4/expectedvscalculated_{}_{}_q{}_dt_{}.png'.format(L, L, dynamictype, qn, dt))
plt.show()

