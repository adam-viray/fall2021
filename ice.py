#!/usr/bin/env python
from pylab import *
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import ode
from matplotlib import rc
from math import sin, pi

#rc('text', usetex=True)
#rc('font', family='serif')

# constants :
g     = 9.81                     # gravitational acceleration
spy   = 31556926                 # seconds per year
rho   = 911                      # density of ice
cp    = 2009                     # heat capacity of ice
beta  = 9.8e-8                   # pressure dependence of melting point
k     = 2.1                      # thermal diffusivity of ice
dzsdx = 1.5e-2                   # surface slope of ice
dTdx  = 7.0e-3                   # horizontal temp. gradient
qGeo  = -32.0e-3                 # geothermal heat-flux


# initial values :
n     = 30                       # num of z-positions
zs    = 0                        # surface start
zb    = -800                    # depth
dt    = 100. * spy               # time-step
t0    = 0.0                      # begin time
tf    = 10000*spy                # end-time
z     = linspace(0, zb, n)       # z-coordinate corresponding to theta
dz    = z[1] - z[0]              # vertical step
theta = zeros(n)                 # temperature
sigma = zeros(n)                 # rescaled vertical component
u     = zeros(n)                 # horizontal ice velocity
w     = zeros(n)                 # vertical ice velocity
tPmp  = beta*rho*g*(z[0]-z[-1])  # Pressure melting point of ice at bed

for i in range(n):
  sigma[i] = (z[i] - z[-1]) / (z[0] - z[-1])
  u[i]     = 90./spy*sigma[i]**4     # meters per second
  w[i]     = -1.0 /spy*sigma[i]        # meters per second


# diffusion matrix :
#
# using   d^2T   u_j-1 - 2u_j + u_j+1
#         ---- = --------------------
#         dz^2          dz^2
A = sparse.lil_matrix((n, n))
A.setdiag(-2*ones(n))
A.setdiag(ones(n-1), k=1)
A.setdiag(ones(n-1), k=-1)
A[0,:] = zeros(n)
A[-1,:] = zeros(n)
A = A / dz**2 * k/(rho*cp)  # k/(rho*cp) = m^2/s
A = A.tocsr()

# vertical ice advection (upwinded) :
#
# using   dT   u_j-2 - 4u_j-1 + 3u_j
#         -- = ---------------------
#         dz           2dz
B = sparse.lil_matrix((n, n))
B.setdiag(3*ones(n))
B.setdiag(-4*ones(n-1), k=1)
B.setdiag(ones(n-2), k=2)
B = B/(2*dz)
B[0,:] = zeros(n)
B[-1,:] = zeros(n)
B[0,0] = 1.0
B[-1,-1] = 1.0
B = B.tocsr()

# horizontal ice advection :
C = sparse.lil_matrix((n, n))
C.setdiag(dTdx*ones(n))
C = C.tocsr()


# surface temperature function :
def surface(t):
  #return -10 + 5*sin(2*pi*t/spy)
  return -10

# solved for u_j (zb upwinded) :
#
# with    dT |
#       k -- |     = qGeo
#         dz |z=zb
#
# and     dT   u_j-2 - 4u_j-1 + 3u_j
#         -- = ---------------------
#         dz           2dz
def fix_boundary(y, t):
  y[0] = surface(t)
  y[-1] = (qGeo*2*dz/k + 4*y[-2] - y[-3])/3.0
  # set max temp on bottom :
  # This should be a column calc.
  if y[-1] >= tPmp :
    y[-1] = tPmp
  return y

# right-hand-side function :
def rhs(t, y, w, u, A, B, C):
  y = fix_boundary(y,t)
  return A * y - w * (B * y)


# Create the ODE Machinery :
i = ode(rhs)
i.set_integrator('vode', method='bdf')
i.set_f_params(w, u, A, B, C)
i.set_initial_value(theta,t0)

# Animation parameters :
xmax = 15
xmin = -20
ymax = 0
ymin = zb
ion()
fig = figure()
axis([xmin, xmax, ymin, ymax])
ph,  = plot(theta, z, 'go-')
fig_text = figtext(.70,.75,'Time = 0.0 yr')
title(r'Heat Distribution in Ice ($\theta$)')
xlabel(r'$\theta$ ($\degree$C)')
ylabel(r'$z$ (m)')

# Loop to solve linear system for each time step :
while i.t <= tf:
  fig_text.set_text('Time = %.2f yr' % (i.t / spy) )
  theta = i.integrate(i.t+dt)
  theta = fix_boundary(i.y, i.t)
  ph.set_xdata(theta)
  fig.canvas.draw()
  fig.canvas.flush_events()

