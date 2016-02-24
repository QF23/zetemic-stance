
'''
program: population_evolution.py
created: 2016-02-24 -- 15 CEST

Performs the evolution of a set of points (initialized on a 2D grid).
'''


import numpy
import matplotlib.pyplot as plt


def velocity(x, y, alpha, beta, s, J):
    vx = - x + 1.0 / (1.0 + numpy.exp(- beta * (alpha * s - J * y)))
    vy = - y + 1.0 / (1.0 + numpy.exp(- beta * ((1.0 - alpha) * s - J * x)))
    return vx, vy


# parameters
beta = 8.0
s = 1.0
J = 0.5
alpha = 0.5

# time parameters
dt = 1e-3
tmax = 20.0
itmax = int(tmax / dt + 0.5)

# grid initialization
ngrid = 20
xx = numpy.linspace(0.0, 1.0, num=ngrid)
yy = numpy.linspace(0.0, 1.0, num=ngrid)
x, y = numpy.meshgrid(xx, yy)
x = x.flatten()
y = y.flatten()

# evolution loop
data_x = []
data_y = []
for it in xrange(itmax):
    # store some values
    if it % 100 == 0:
        data_x.append(x.copy())
        data_y.append(y.copy())
    # evolution step
    vx, vy = velocity(x, y, alpha, beta, s, J)
    x += dt * vx
    y += dt * vy

data_x = numpy.array(data_x)
data_y = numpy.array(data_y)
title = '$\\beta=%s$, $s=%s$, $J=%s$, $\\alpha=%s$ |' % (beta, s, J, alpha)
title += ' $\\delta_t=%s$, $t_\\mathrm{max}=%s$' % (dt, tmax)


# plot 1
fig, (ax1, ax2) = plt.subplots(2, 1)
for i in xrange(ngrid ** 2):
    ax1.plot(data_x[:, i])
    ax2.plot(data_y[:, i])
for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.set_xlabel('$t$', fontsize=18)
ax1.set_ylabel('$x(t)$', fontsize=18)
ax2.set_ylabel('$y(t)$', fontsize=18)
plt.tight_layout()
plt.suptitle(title, y=1.01)
plt.show()
##plt.savefig('fig_1.png', bbox_inches='tight', dpi=128)
##plt.close()

# plot 2
fig, ax = plt.subplots(1, 1)
for i in xrange(ngrid ** 2):
    plt.plot(data_x[:, i], data_y[:, i])
ax.set_aspect(1)
plt.title(title)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
plt.show()
##plt.savefig('fig_2.png', bbox_inches='tight', dpi=128)
