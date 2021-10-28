#!/usr/bin/env python
from pylab import *
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import ode
from matplotlib import rc
from math import sin, pi

# initial values :
n     = 500                     # num of x-positions
xs    = 0                        # left end point
xe    = 100.                     # right end point
dt    = .1                       # time-step for animation
t0    = 0.0                      # begin time
tf    = 15.0                      # end-time
x     = linspace(xs, xe, n)      # z-coordinate corresponding to theta
dx    = x[1] - x[0]              # vertical step
u0    = zeros(n)                 # initial values
u0[n//5-n//20:n//5+n//20] = 1.

k = 0.0                          # diffusion coefficient
v = 2                           # velocity

# diffusion matrix :
#
# using   d^2T   u_j-1 - 2u_j + u_j+1
#         ---- = --------------------
#         dx^2          dx^2
D = sparse.lil_matrix((n, n))
D.setdiag(-2*ones(n))
D.setdiag(ones(n-1), k=1)
D.setdiag(ones(n-1), k=-1)
D[0,:] = zeros(n)
D[-1,:] = zeros(n)
D = D / dx**2 * k
D = D.tocsr()

# advection (upwinded) :
#
# using   dT   u_j-2 - 4u_j-1 + 3u_j
#         -- = ---------------------
#         dx           2dx
A = sparse.lil_matrix((n, n))
A.setdiag(3*ones(n))
A.setdiag(-4*ones(n-1), k=-1)
A.setdiag(ones(n-2), k=-2)
A = A/(2*dx)
A[0,:] = zeros(n)
A[-1,:] = zeros(n)
A[0,0] = 1.0
A[-1,-1] = 1.0
A = A.tocsr()

def fix_boundary(y, t):
  y[0] = 0.
  y[-1] = 0.
  return y

# right-hand-side function :
def rhs(t, y, v, D, A):
  y = fix_boundary(y,t)
  return D * y - v * (A * y)


# Create the ODE Machinery :
i = ode(rhs)
i.set_integrator('vode', method='bdf',atol=1e-12)
i.set_f_params(v, D, A)
i.set_initial_value(u0,t0)

# Animation parameters :
xmin = xs
xmax = xe
ymin = 0
ymax = 1.1
ion()
fig = figure()
axis([xmin, xmax, ymin, ymax])
ph,   = plot(x, u0, 'go-')
fig_text = figtext(.70,.75,'Time = 0.0')
title('Advection-diffusion results (u)')
xlabel('Position')
ylabel('Concentration')

# Loop to solve linear system for each time step :
while i.t <= tf:
  fig_text.set_text('Time = %.2f yr' % (i.t) )
  u = i.integrate(i.t+dt)
  u = fix_boundary(i.y, i.t)
  ph.set_ydata(u)
  fig.canvas.draw()
  fig.canvas.flush_events()
