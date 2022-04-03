import numpy as np
import matplotlib.pyplot as plt

# Set the parameters
N     = 1000      # number of time steps
dt    = 100 / N   # time step
omg   = 5.0       # natural frequency \omega = \sqrt{k/m}
f     = 1.0       # F/m
omg_f = 0.1       # frequency of the external force \omega_f

# Set the initial conditions
t = np.linspace(0, 100, N+1)
x = np.zeros([N+1])
v = np.zeros([N+1])

# Analytical solution
x0 = f/(omg**2-omg_f**2) * np.cos(omg_f*t)

# Forward Euler 
for i in range(N):
    x[i+1] += dt * v[i]
    v[i+1] += dt * (f*np.cos(omg_f*t[i]) - omg**2*x[i])

# L2 error
err = abs(x0-x)

# System response
plt.figure()
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig('response_p2a.png')
# Phase portrait
plt.figure()
plt.plot(x, v)
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.savefig('phase_p2a.png')
# L2 error compared to the analytical solution
plt.figure()
plt.semilogy(t, err)
plt.xlabel('t')
plt.ylabel(r'$\Vert e(t) \Vert_{L^2}$')
plt.savefig('error_p2a.png')
