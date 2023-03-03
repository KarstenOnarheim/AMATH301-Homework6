import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

##################### Coding Problem 1 ##########################
## Part a - Solve the ODE with forward Euler
# It will be helpful to define the anonymous function for the right-hand side
P = lambda p: p*(1-p)*(p-1/2)
dt = .5
tspan = np.arange(0, 10 + dt, dt)

p = np.zeros(len(tspan))
p[0] = .9
for k in range(len(tspan)-1):
    p[k+1] = p[k] + dt*P(p[k])
A1 = p

## Part b - Solve using backward Euler

p = np.zeros(len(tspan))
p[0] = .9
for k in range(len(tspan)-1):
    g = lambda z: z - p[k] - dt*P(z)
    p[k+1] = scipy.optimize.fsolve(g, p[k])
A2 = p

## Part c - Solve using the midpoint method

p = np.zeros(len(tspan))
p[0] = .9
for k in range(len(tspan)-1):
    k1 = p[k] + .5*dt*P(p[k])
    p[k+1] = p[k] + dt*P(k1)
A3 = p

## Part d - Solve with RK4


p = np.zeros(len(tspan))
p[0] = .9
for k in range(len(tspan)-1):
    k1 = P(p[k])
    k2 = P(p[k] + .5*dt*k1)
    k3 = P(p[k] + .5*dt*k2)
    k4 = P(p[k] + dt*k3)
    p[k+1] = p[k] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
A4 = p

######################### Coding problem 2 ###################
## Part (a)
# To solve with solve_ivp we need to define an anonymous function. I've
# defined the two ODEs for R'(t) and J'(t) below. You put them together using
# an adapter function. 
a = 1/2
Rprime = lambda R, J: a*R + J
Jprime = lambda R, J: -R - a*J

odefun = lambda t, v: np.array([Rprime(v[0], v[1]), Jprime(v[0], v[1])])

# Initial Condition
x0 = np.array([2, 1])


sol = scipy.integrate.solve_ivp(odefun, [0, 20], x0)


t = sol.t
R = sol.y[0]
J = sol.y[1]

A5 = R
A6 = J
## (b) 
A7 = np.array([R[-1], J[-1]])

## (c) 
dt = 0.1
trange = np.arange(0, 20+dt, dt)

R = np.zeros(len(trange))
J = np.zeros(len(trange))
R[0] = 2
J[0] = 1

for k in range(len(trange)-1):
    R[k+1] = R[k] + dt * Rprime(R[k], J[k])
    J[k+1] = J[k] + dt * Jprime(R[k], J[k])
A8 = R
A9 = J

## (d) 
A10 = np.array([R[-1], J[-1]])

## (e) 
A11 = np.linalg.norm(A10-A7)

##################### Coding Problem 3 ##########################
# Define the parameters we are going to use:
g = 9.8
L = 11
sigma = 0.12

## Part a
# We now have two ODEs. Define them below as anonymous functions!

thetaPrime = lambda v: v
vPrime = lambda theta, v: -g/L * np.sin(theta)  -  sigma * v

theta0 = -1 * np.pi / 8
v0 = -0.1

## Part b
odefun = lambda t, p: np.array([thetaPrime(p[1]), vPrime(p[0], p[1])])

A12 = odefun(1, [2,3])

## Part c
sol = scipy.integrate.solve_ivp(odefun, [0, 50], [theta0, v0])
A13 = sol.y

