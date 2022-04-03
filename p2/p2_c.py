import numpy as np
import matplotlib.pyplot as plt

# Set the parameters
N     = 100000       # number of time steps
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

# RK4
for i in range(N):
    k1x = v[i]
    k1v = f*np.cos(omg_f*t[i]) - omg**2*x[i]
    
    k2x = v[i] + k1v*dt/2
    k2v = f*np.cos(omg_f*(t[i]+dt/2)) - omg**2*(x[i]+k1x*dt/2)
    
    k3x = v[i] + k2v*dt/2
    k3v = f*np.cos(omg_f*(t[i]+dt/2)) - omg**2*(x[i]+k2x*dt/2)
    
    k4x = v[i] + k3v*dt
    k4v = f*np.cos(omg_f*(t[i]+dt)) - omg**2*(x[i]+k3x*dt)
    
    x[i+1] += dt * (k1x+2*k2x+2*k3x+k4x) / 6
    v[i+1] += dt * (k1v+2*k2v+2*k3v+k4v) / 6

# L2 error
err = abs(x0-x)

# System response
plt.figure()
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.savefig('response_p2c.png')
# Phase portrait
plt.figure()
plt.plot(x, v)
plt.xlabel('x(t)')
plt.ylabel('v(t)')
plt.savefig('phase_p2c.png')
# L2 error compared to the analytical solution
plt.figure()
plt.semilogy(t, err)
plt.xlabel('t')
plt.ylabel(r'$\Vert e(t) \Vert_{L^2}$')
plt.savefig('error_p2c.png')
