#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 12:01:36 2021

@author: user
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import inv, spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA

def area(b):
    # get area under curve, assuming IC 0 outside step function
    return np.sum(b)

def step_dot(b, t, bc="periodic"):
    # solve for the next time step
    soln = A.dot(b)
    if bc == "periodic":
        soln[0], soln[-1], soln[1], soln[-2] = soln[-2], soln[1], soln[-1], soln[0]
    elif bc == "dirichlet":
        soln[0], soln[-1] = 0,0
    u[:,t+1] = soln
    return soln

def step_solv(b, t, bc="periodic"):
    soln = spsolve(L,R.dot(b))
    if bc == "periodic":
        soln[0], soln[-1] = soln[-1], soln[0]
    elif bc == "dirichlet":
        soln[0], soln[-1] = 0,0
    u[:,t+1] = soln
    return soln

# LHS coefficient sparse matrix, theta method
def LHS_theta():
    d0_l = 1 + 2*theta*D*dt/dx**2
    d1_l = -theta*D*dt/dx**2
    return sp.diags([d1_l, d0_l, d1_l], [-1, 0, 1], (nx,nx), 'csc')

# RHS coefficient sparse matrix, second-order upwinded
def RHS_upwind_2():
    d0_r = -v*dt/2/dx
    d1_r = (1-theta)*D*dt/dx**2 + 4*v*dt/2/dx
    d2_r = 1 - 2*(1-theta)*D*dt/dx**2 - 3*v*dt/2/dx
    d3_r = (1-theta)*D*dt/dx**2
    return sp.diags([d0_r, d1_r, d2_r, d3_r], [-2, -1, 0, 1], (nx,nx), 'csc')

# RHS coefficient sparse matrix, first-order upwinded
def RHS_upwind_1():
    d0_r = (1-theta)*D*dt/dx**2 + v*dt/dx
    d1_r = 1 - 2*(1-theta)*D*dt/dx**2 - v*dt/dx
    d2_r = (1-theta)*D*dt/dx**2
    return sp.diags([d0_r, d1_r, d2_r], [-1, 0, 1], (nx,nx), 'csc')
    
# RHS coefficient sparse matrix, fourth-order centered
def RHS_centered_4():
    d0_r = -v*dt/12/dx
    d1_r = (1-theta)*D*dt/dx**2 + 8*v*dt/12/dx
    d2_r = 1 - 2*(1-theta)*D*dt/dx**2
    d3_r = (1-theta)*D*dt/dx**2 - 8*v*dt/12/dx
    d4_r = -d0_r
    return sp.diags([d0_r, d1_r, d2_r, d3_r, d4_r], [-2, -1, 0, 1, 2], (nx,nx), 'csc')

# establish spatial domain
nx = 100
x_space = np.linspace(0,1,nx)
dx = 1/nx

# establish time domain
dt = .00001
tf = .1
nt = int(tf/dt)
t_space = np.linspace(0, tf, nt)

# establish initial condition
b = np.zeros(nx)
b[int(.45*nx):int(.55*nx)] = 10
u = np.zeros((nx,nt))
u[:,0] = b

# parameters
D = .01
v = 5
theta = 1

L = LHS_theta()
R = RHS_upwind_2()

# with D, v, theta, dx, and dt constant, matrix A is also constant
A = inv(L)@R

for t in range(len(t_space)-1):
    #b = step_dot(b,t)
    b = step_solv(b,t)
    if t%125 == 0:
        plt.plot(b)
        plt.title("area under curve: {:e}".format(area(b)))
        plt.show()
        
plt.imshow(u)
plt.show()