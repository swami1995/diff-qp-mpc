#https://github.com/AdamMOD/ndulum/blob/main/rigid-n-dulum-free-control.py

import sympy
import numpy as np
import scipy
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt
from sympy.physics import vector
from sympy.physics import mechanics
from matplotlib import animation
import matplotlib


plt.style.use('seaborn')
matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
sympy.init_printing(use_latex=True)

n = 1 # number of links 
g = sympy.symbols("g")
x = mechanics.dynamicsymbols("x")
xd = mechanics.dynamicsymbols("x", 1)
xdd = mechanics.dynamicsymbols("x", 2)
theta = mechanics.dynamicsymbols("theta:"+ str(n))
thetad = mechanics.dynamicsymbols("theta:" + str(n), 1)
thetadd = mechanics.dynamicsymbols("theta:" + str(n), 2)

lengths = sympy.symbols("l:" + str(n))
masses = sympy.symbols("m:" + str(n))
m_top = sympy.symbols("m_{top}")
radii = sympy.symbols("r:" + str(n))
inertia_vals = sympy.symbols("I:" + str(n))

values = {g: 9.81, m_top: 2}  # what is m_top?
o_point = {x: 0, xd: 0, xdd: 0}
# Absolute positions of the joints
x0 = np.zeros(n * 2 + 2) 
x0[0] = .0
x0[1:n+1] = 0*np.pi + 0*np.linspace(-0.1, 0.1, n)
x0 = np.array([0.0,np.pi/2,0, 0, 0, 0])

print(x0)

for i in range(n):
    o_point.update({theta[i]: np.pi, thetad[i]: 0, thetadd[i]: 0})
    values.update({lengths[i]: 1, masses[i]: 1})
    values.update({radii[i]: .5, inertia_vals[i]: 1})
lengthsum = sum([values[i] for i in lengths])

u = mechanics.dynamicsymbols("u")

T = []
U = []

N = mechanics.ReferenceFrame("N")
P = mechanics.Point("P")
P.set_vel(N, 0)

PIv = N.y * xd
PIp = N.y * x
PI = P.locatenew("PI",PIp)
PI.set_vel(N, PIv) 
top_pivot = mechanics.Particle('top_pivot', PI, m_top)
T.append(top_pivot.kinetic_energy(N))

pivot0_frame = mechanics.ReferenceFrame("pivot0_f")
pivot0_frame.orient(N, "Axis", [ theta[0], N.z])
pivot0_frame.set_ang_vel(N, ( thetad[0]* N.z))


pivot0 = PI.locatenew("pivot0", lengths[0] * pivot0_frame.x)
pivot0.v2pt_theory(PI, N, pivot0_frame)

com0 = PI.locatenew("com0", radii[0] * pivot0_frame.x)
com0.v2pt_theory(PI, N, pivot0_frame)

inertai_dyad = vector.outer(pivot0_frame.z, pivot0_frame.z) * inertia_vals[0]
body = mechanics.RigidBody("B", com0, pivot0_frame, masses[0], (inertai_dyad, com0))

U.append(com0.pos_from(P).dot(N.x) * masses[0] * -g)
T.append(body.kinetic_energy(N))


pivot_prev = pivot0

for i in range(1,n):
    P_f = mechanics.ReferenceFrame("P_f")
    P_f.orient(N, "Axis", [theta[i], N.z])
    P_f.set_ang_vel(N, thetad[i] * N.z)

    pivot = mechanics.Point("p")
    pivot.set_pos(pivot_prev, lengths[i] * P_f.x)
    pivot.v2pt_theory(pivot_prev, N, P_f)

    com = mechanics.Point("com")
    com.set_pos(pivot_prev, radii[i] * P_f.x)
    com.v2pt_theory(pivot_prev, N, P_f)

    inertai_dyad = vector.outer(P_f.z, P_f.z) * inertia_vals[i]
    body = mechanics.RigidBody("B", com, P_f, masses[i], (inertai_dyad, com))

    U.append(com.pos_from(P).dot(N.x) * masses[i] * -g)
    T.append(body.kinetic_energy(N))

    pivot_prev = pivot

L = sum(T) - sum(U) + u * x

Lagrangian = mechanics.LagrangesMethod(L, [x] + theta)
Lagrangian.form_lagranges_equations()

M = Lagrangian.mass_matrix_full
K = Lagrangian.forcing_full
import ipdb; ipdb.set_trace()
M = M.subs(values)
K = K.subs(values)


print("State vector length is: {}".format(2 * n + 2))

M_lambd, K_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad + [u]), M), sympy.lambdify(tuple([x] + theta + [xd] + thetad + [u]), K)

A, B, inp = Lagrangian.linearize(q_ind=[x]+theta, qd_ind=[xd]+thetad, A_and_B=True, op_point = o_point)

print("A matrix free vars: {} ".format(A.free_symbols))
print("B matrix free vars: {}".format(B.free_symbols))
print("Input is {}".format(inp))

A = A.subs(values)
B = B.subs(values)

print("A matrix is: {} \n B matrix is: {}".format(A,B))

A = np.array(A,dtype=np.float64)
B = np.array(B,dtype=np.float64).T[0]
A_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad), A)

err_cost = np.eye(2 * n + 2)
err_cost[0,0] = 2
act_cost = 1e-1
Bt = np.reshape(B,(2 * n + 2,1))

riccati = linalg.solve_continuous_are(A, Bt, err_cost, act_cost)
gains = 1/act_cost * B @ riccati
ref = np.array([o_point[x]] + [o_point[theta_v] for theta_v in theta] + [o_point[xd]] + [o_point[thetad_v] for thetad_v in thetad])

print("Gains are: {}".format(gains))
print("Reference value : {}, linearized about: {}".format(ref, o_point))


cmat = np.array([A**i @ Bt for i in range(2 *n + 2)],dtype=np.float64)
cmat = np.reshape(cmat, (2 * n + 2, 2 * n + 2)).T
ctrlblty = np.linalg.matrix_rank(cmat)

print("Controllability is {}".format(ctrlblty))

##################
# Dynamics functions
##################

# Nonlinear cont. dynamics
def nonlin_dynamics(x, u):
    # import ipdb; ipdb.set_trace()
    dxdt = np.linalg.solve(M_lambd(*x, u), K_lambd(*x, u))
    return dxdt.T[0]     

# Linearized cont. dynamics
def lin_dynamics(x, u):
    x_der = A_lambd(*x) @ x + B.dot(u)
    return x_der

def rk4(f, x, u, dt):
    k1 = f(x, u)
    # import pdb; pdb.set_trace()
    k2 = f(x + k1 * dt/2, u)
    k3 = f(x + k2 * dt/2, u)
    k4 = f(x + k3 * dt, u)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

##################
# Simulate the controlled system
##################

time = 10
dt = .05
t = np.arange(0,time,dt)

x = 1*x0
soltn = np.zeros((int(time/dt), 2 * n + 2))
for i in range(len(t)):
    u = 0*np.dot(gains,(ref-x))  # LQR control law
    # import pdb; pdb.set_trace()
    x = rk4(nonlin_dynamics, x, u, .05)
    soltn[i] = 1*x
soltnf = soltn

##################
# Plot the results
##################

#linear
xl, yl = [], []
for i in range(n+1):
    if(i != 0):
        xl.append(xl[i-1] + values[lengths[i-1]] * np.sin(soltn[:,i]))
        yl.append(yl[i-1] - values[lengths[i-1]] * np.cos(soltn[:,i]))
    else:
        xl.append(soltn[:,i])
        yl.append(np.zeros(soltn.shape[0]))      

xl, yl = np.array(xl),  np.array(yl)
plt.plot(t, soltn[:,:int(len(soltn.T)/2)],ls='--')

#nonlinear
xf, yf = [], []
for i in range(n+1):
    if(i != 0):
        xf.append(xf[i-1] + values[lengths[i-1]] * np.sin(soltnf[:,i]))
        yf.append(yf[i-1] - values[lengths[i-1]] * np.cos(soltnf[:,i]))
    else:
        xf.append(soltnf[:,i])
        yf.append(np.zeros(soltnf.shape[0]))      

xf, yf = np.array(xf),  np.array(yf),
ax = plt.axes()
plt.plot(t, soltnf[:,:int(len(soltnf.T)/2)])
ax.legend([x] + theta + [x] + theta)

fig,ax = plt.subplots()
plt.xlim(-lengthsum*1.5, lengthsum*1.5)
plt.ylim(-lengthsum*1.1,lengthsum*1.1)

line, = ax.plot([], [],ls='--', linewidth=4)
linef, = ax.plot([], [],ls='-', linewidth=4)
# rectangle marker
cart, = ax.plot([], [], marker='s', markersize=20, color='black')

def init():
    #line.set_data(xl[:,0], yl[:,0])
    linef.set_data(xf[:,0], yf[:,0])
    return line,linef
def animate(i):
    #line.set_data(xl[:,i], yl[:,i])
    linef.set_data(xf[:,i], yf[:,i])
    # cart.set_data(xf[0,i], yf[0,i])
    # draw the cart at xf[0,i] and yf[0,i]
    return line, linef

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = int(time/dt), interval = 1000*dt)
#anim.save('free-n-dulum.gif', writer='imagemagick')
#print(sympy.pretty(A.eigenvals()))
[print("\n{}".format(i)) for i in np.linalg.eigvals(A-B @ gains)]
plt.show()