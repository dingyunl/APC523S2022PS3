import numpy as np
import matplotlib.pyplot as plt

# Define the Jacobian
def Jacobian(x):
    return np.matrix([[-20.0*x[3] - 3.0e-12*x[4], 2000.0*x[2], 2000.0*x[1], -20.0*x[0], -3.0e-12*x[0]], 
                      [20.0*x[3] + 3.0e-12*x[4], -2000.0*x[2], -2000.0*x[1], 20.0*x[0], 3.0e-12*x[0]], 
                      [3.0e-12*x[4], -2000.0*x[2], -2000.0*x[1], 0, 3.0e-12*x[0]], 
                      [-20.0*x[3], 0, 0, -20.0*x[0], 0], 
                      [20.0*x[3] - 3.0e-12*x[4], 2000.0*x[2], 2000.0*x[1], 20.0*x[0], -3.0e-12*x[0]]])


# Define the RHS function 
def func(x):
    k = np.zeros([5])
    k[0] =   kp*x[2]*x[1] - km*x[4]*x[0] - kp2*x[0]*x[3]
    k[1] = - kp*x[2]*x[1] + km*x[4]*x[0] + kp2*x[0]*x[3]
    k[2] = - kp*x[2]*x[1] + km*x[4]*x[0]
    k[3] =                               - kp2*x[0]*x[3]
    k[4] =   kp*x[2]*x[1] - km*x[4]*x[0] + kp2*x[0]*x[3]
    return k

# Define the Newton-Raphson linearized implicit solver
def NR(x, dt):
    Jx = Jacobian(x)
    Inv = np.linalg.inv(np.identity(5)-Jx*dt)
    return x + np.matmul(Inv, func(x))*dt


# Set the parameters and the initial conditions
kp = 2.0e3
km = 3.0e-12
kp2 = 2.0e1
x = np.array([0.01, 0.01, 0.75, 0.23, 0.00])
J = Jacobian(x)
vl, vc = np.linalg.eig(J)
dt = 1 / np.amax(abs(vl))
T = 40
N = int(T/dt) + 1

# BDF2
time = np.zeros([N+1])
time[0] = 0
time[1] = dt

x_save = np.zeros([5, N+1])
x_save[:,0] = x
# Use RK4 to initialize
k1 = func(x)
k2 = func(x+k1*dt/2)
k3 = func(x+k2*dt/2)
k4 = func(x+k3*dt)   
x_new = x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
x_save[:,1]= x_new

i = 2
t = dt*2
while t < T:
    x_com = np.array(x_new*4/3 - x/3)
    x = x_new
    x_new = NR(x_com, dt*2/3)
    x_new = np.array([x_new[0,0], x_new[0,1], x_new[0,2], x_new[0,3], x_new[0,4]])
    x_save[:,i] = x_new
    time[i] = t
    
    err = np.max(np.divide(abs(x_new-x), x))
    if (err < 0.1):
        dt = 2*dt
    
    t += dt
    i += 1

chem = ['N', 'O', 'N2', 'O2', 'NO']
for k in range(5):
    plt.loglog(time[0:i-1], x_save[k, 0:i-1], label=chem[k])
plt.xlabel('time(s)')
plt.ylabel('Mole fraction')
plt.title('BDF2')
plt.legend(loc='lower left')
plt.savefig('p4b-iii.png')
