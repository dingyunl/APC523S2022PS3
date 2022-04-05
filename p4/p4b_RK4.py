import numpy as np
import matplotlib.pyplot as plt

# Set the parameters and the initial conditions
N = 80000
T = 40
dt= T / N

kp = 2.0e3
km = 3.0e-12
kp2 = 2.0e1

t = np.linspace(0, T, N+1)

# Define the RHS function 
def func(x):
    k = np.zeros([5])
    k[0] =   kp*x[2]*x[1] - km*x[4]*x[0] - kp2*x[0]*x[3]
    k[1] = - kp*x[2]*x[1] + km*x[4]*x[0] + kp2*x[0]*x[3]
    k[2] = - kp*x[2]*x[1] + km*x[4]*x[0]
    k[3] =                               - kp2*x[0]*x[3]
    k[4] =   kp*x[2]*x[1] - km*x[4]*x[0] + kp2*x[0]*x[3]
    return k

# RK4
x = np.array([0.01, 0.01, 0.75, 0.23, 0.00])
x_save = np.zeros([5, N+1])
x_save[:,0] = x

for i in range(N):
    k1 = func(x)
    k2 = func(x+k1*dt/2)
    k3 = func(x+k2*dt/2)
    k4 = func(x+k3*dt)
    
    x += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    x_save[:, i+1] = x
    
chem = ['N', 'O', 'N2', 'O2', 'NO']
for k in range(5):
    plt.loglog(t, x_save[k], label=chem[k])
plt.xlabel('time(s)')
plt.ylabel('Mole fraction')
plt.title('RK4')
plt.legend(loc='lower left')
plt.savefig('p4b-i.png')
