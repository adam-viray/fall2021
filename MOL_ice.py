#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:44:09 2021

@author: user
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv, spsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
from scipy.interpolate import interp1d
import scipy.optimize as opt

data = np.array([
       [ 801.       ,   -1.4404231],
       [ 741.       ,   -3.3368846],
       [ 721.       ,   -4.3305769],
       [ 701.       ,   -5.3550385],
       [ 681.       ,   -6.4179615],
       [ 661.       ,   -7.3731923],
       [ 641.       ,   -8.3976538],
       [ 621.       ,   -9.3298077],
       [ 601.       ,   -9.9696538],
       [ 581.       ,  -10.947962 ],
       [ 561.       ,  -11.572423 ],
       [ 541.       ,  -12.012269 ],
       [ 521.       ,  -12.652115 ],
       [ 501.       ,  -12.630423 ],
       [ 481.       ,  -12.654885 ],
       [ 461.       ,  -13.817808 ],
       [ 441.       ,  -13.834577 ],
       [ 421.       ,  -12.9975   ],
       [ 401.       ,  -13.429654 ],
       [ 381.       ,  -13.261808 ],
       [ 361.       ,  -13.863192 ],
       [ 341.       ,  -13.472269 ],
       [ 321.       ,  -13.565962 ],
       [ 301.       ,  -13.782731 ],
       [ 281.       ,  -13.168731 ],
       [ 261.       ,  -13.131654 ],
       [ 241.       ,  -12.963808 ],
       [ 221.       ,  -12.095962 ],
       [ 201.       ,  -12.0435   ],
       [ 181.       ,  -12.152577 ],
       [ 161.       ,  -11.577038 ],
       [ 141.       ,  -11.3015   ],
       [ 121.       ,  -11.279808 ],
       [ 101.       ,  -10.5735   ],
       [  81.       ,  -10.074885 ],
       [  61.       ,   -9.6993462],
       [  41.       ,   -9.5238077],
       [  21.       ,   -9.4867308],
       [   1.       ,   -9.6111923]])

data[:,0] = data[:,0] - 1

spy = 31556926                               # Seconds per year                      s a-1
k = 2.1                                      # Thermal diffusivity of ice            W m-1 degK-1
rho = 911                                    # Density of ice                        kg m-3
C_p = 2009                                   # Heat capacity                         J kg-1 degK-1
dzs_dx = np.radians(0.7)                     # Surface slope of ice                  radians
lamda = 7e-3                                 # Temperature lapse rate                degK m-1
dtheta_dx = lamda*dzs_dx                     # Horizontal temperature gradient       degK m-1
u_s = 90/spy                                 # Horizontal surface velocity           m s-1
u_b = 40/spy                                 # Horizontal basal velocity             m s-1
a_dot = -1/spy                               # Surface mass balance                  m s-1
g = 9.81                                     # Acceleration due to gravity           m s-2
z_s = 800                                    # Ice top or surface elevation          m
z_b = 0                                      # Ice bottom or bed elevation           m
theta_s = 273.15 - 9.6111923                 # Surface temperature                   degK
Q_geo = 32e-3                                # Geothermal heat flow                  W m-2
beta = 9.8e-8                                # Pressure dependence of melting point  degK Pa-1
theta_pmp = 273.15 - beta*rho*g*(z_s - z_b)  # Pressure melting point of ice at bed  degK
Q_net = Q_geo + rho*g*(z_s - z_b)*dzs_dx*u_b # Basal thermal flow including friction W m-2

t0 = 0
tf = 500*31556926

nz = 100
dz = (z_s-z_b)/nz
z_space = np.linspace(z_b,z_s,nz)

theta_0 = theta_s*np.ones_like(z_space)
#f = interp1d(x=1-data[::-1,0], y=data[::-1,1]+273.15)
#theta_0 = f(z_space)

def sigma(z): # Re-scaled vertical coordinate
    return (z_s - z) / (z_s - z_b)

def du_dz(z): # Vertical shear
    return 4*(u_s - u_b)*sigma(z)**3 / (z_s - z_b)

def phi(z): # Heat sources from deformation of ice
    return rho*g*(z - z_b) * du_dz(z) * dzs_dx

def u(z): # Horizontal ice velocity at depth
    return (u_s - u_b)*(1 - sigma(z)**4) + u_b

def w(z): # Vertical ice velocity
    return -a_dot*(1-sigma(5/4 - sigma(z)**4/4))

def enforce_BCs(theta): # enforce boundary conditions on theta array
    # use upwind estimation for dtheta/dz to find lower BC
    theta[-1] = theta_s
    theta[0] = ((2*dz*Q_net/k + 4*theta[1] - theta[2])/3
                 if theta[0] < theta_pmp else theta_pmp)
    return theta

w_ = w(z_space)
s = -u(z_space)*dtheta_dx + phi(z_space)/rho/C_p
theta = enforce_BCs(theta_0)

def RHS(t,theta):
    d0 = np.array(-w_/2/dz)[:-2]
    d1 = np.array(k/rho/C_p/dz**2 + 2*w_/dz)[:-1]
    d2 = np.array(-2*k/rho/C_p/dz**2 - 3*w_/2/dz)
    d3 = np.array(k/rho/C_p/dz**2)
    theta = enforce_BCs(theta)
    return sp.diags([d0, d1, d2, d3], [-2, -1, 0, 1], (nz,nz), 'csr')@theta + s
    

def f(x):
    a_dot = x[1]/spy
    u_b = x[0]/spy
    print(x)
    i = solve_ivp(RHS, (t0,tf), theta_0)
    theta_f = interp1d(x = z_space, y = i.y[:,-1])
    soln = theta_f(data[:,0])
    return np.sum(np.square(np.flip(data, axis=0)[:,1] - soln))

opt_params = opt.minimize(f, (40, -1), method='Nelder-Mead', bounds=((0,90),(-10,0)))

a_dot = opt_params.x[1]/spy
u_b = opt_params.x[0]/spy

solution = solve_ivp(RHS, (t0,tf), theta_0)

plt.plot(solution.y[:,-1],z_space,c='r')
plt.scatter(data[:,1]+273.3, 801-data[:,0], c='k')
plt.show()
