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

# Define the Jacobian
def Jacobian(x):
    return np.matrix([[-20.0*x[3] - 3.0e-12*x[4], 2000.0*x[2], 2000.0*x[1], -20.0*x[0], -3.0e-12*x[0]], 
                      [20.0*x[3] + 3.0e-12*x[4], -2000.0*x[2], -2000.0*x[1], 20.0*x[0], 3.0e-12*x[0]], 
                      [3.0e-12*x[4], -2000.0*x[2], -2000.0*x[1], 0, 3.0e-12*x[0]], 
                      [-20.0*x[3], 0, 0, -20.0*x[0], 0], 
                      [20.0*x[3] - 3.0e-12*x[4], 2000.0*x[2], 2000.0*x[1], 20.0*x[0], -3.0e-12*x[0]]])

# Define the Newton-Raphson linearized implicit solver
def NR(x, dt):
    Jx = Jacobian(x)
    Inv = np.linalg.inv(np.identity(5)-Jx*dt)
    return x + np.matmul(Inv, func(x))*dt

# DIRK2
x = np.array([0.01, 0.01, 0.75, 0.23, 0.00])
x_save = np.zeros([5, N+1])
x_save[:,0] = x

for i in range(N):
    x1 = NR(x, dt/3)
    k1 = func(np.transpose(x1))
    x2 = NR(x+k1*0.75*dt, dt/4)
    k2 = func(np.transpose(x2))
    x += (3*k1 + k2) * dt / 4
    x_save[:, i+1] = x
    
chem = ['N', 'O', 'N2', 'O2', 'NO']
for k in range(5):
    plt.loglog(t, x_save[k], label=chem[k])
plt.xlabel('time(s)')
plt.ylabel('Mole fraction')
plt.title('DIRK')
plt.legend(loc='lower left')
plt.savefig('p4b-ii.png')
