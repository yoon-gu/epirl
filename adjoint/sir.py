import sympy as sp
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from tqdm import tqdm

# State
S, I = sp.symbols('S I')
state = [S, I]

# Control
u = sp.symbols('u')

# Parmaeters
beta, gamma, w1, w2, w3 = sp.symbols('beta gamma w_1 w_2 w_3')
params = [beta, gamma, w1, w2, w3]

# Adjoints
l_S, l_I = sp.symbols('lambda_S lambda_I')
adjoint = [l_S, l_I]
l = sp.Matrix([l_S, l_I])

f = w1 * I + w2 * u*S + w3 * u**2
g = sp.Matrix([-beta*S*I -u*S, beta*S*I - gamma*I])
H = sp.Matrix([f + l.dot(g)])


# Automation
dHdx = H.jacobian(state)
dHdl = H.jacobian(adjoint)
dHdu = H.jacobian([u])

dHdx_fn = sp.lambdify([*adjoint, *state, *params, u], dHdx)
dHdl_fn = sp.lambdify([*adjoint, *state, *params, u], dHdl)
dHdu_fn = sp.lambdify([*adjoint, *state, *params, u], dHdu)

# ODE Systems
def state_de(y, t, beta_, gamma_, w1, w2, w3, u_interp):
    S_, I_ = y
    u_ = u_interp(t)
    return dHdl_fn(0, 0, S_, I_, beta_, gamma_, w1, w2, w3, u_)[0]

def adjoint_de(y, t, x_interp, beta_, gamma_, w1, w2, w3, u_interp):
    l_S_, l_I_ = y
    S_, I_ = x_interp(t)
    u_ = u_interp(t)
    val = -dHdx_fn(l_S_, l_I_, S_, I_, beta_, gamma_, w1, w2, w3, u_)[0]
    return val

t0 = 0
tf = 30
beta = 0.002
gamma = 0.5
S0 = 990
I0 = 10
w1, w2, w3 = 1, 1, 1

params = [beta, gamma, w1, w2, w3]

# Initial
y0 = np.array([S0, I0])
t = np.linspace(t0,tf, 301)
dt = t[1] - t[0]
u0 = np.ones_like(t)

MaxIter = 1000 + 1
alpha = 1E-1
old_cost = 1E8
for it in tqdm(range(MaxIter)):
    # State
    u_intp = lambda tc: np.interp(tc, t, u0)
    sol = odeint(state_de, y0, t, args=(*params, u_intp))

    # Cost
    S, I = np.hsplit(sol, 2)
    S_mid = (S[1:] + S[:-1]) / 2.
    I_mid = (I[1:] + I[:-1]) / 2.
    u_mid = (u0[1:] + u0[:-1]) / 2.
    cost1 = dt * w1 * np.sum(I_mid.flatten())
    cost2 = dt * w2 * np.sum(S_mid.flatten() * u_mid)
    cost3 = dt * w3 * np.sum(u_mid.flatten() ** 2)
    cost = cost1 + cost2 + cost3

    # Adjoint
    u_intp = lambda tc: np.interp(tf - tc, t, u0)
    x_intp = lambda tc: np.array([np.interp(tf - tc, t, sol[:, 0]), np.interp(tf - tc, t, sol[:, 1])])
    y_T = np.array([0,0])
    l_sol = odeint(adjoint_de, y_T, t, args=(x_intp, *params, u_intp))
    l_sol = np.flipud(l_sol)

    # Simple Gradient
    # Hu = w2 * sol[:, 0] + 2 * w3 * u0 - l_sol[:,0] * sol[:, 0]
    Hu = dHdu_fn(l_sol[:,0], l_sol[:,1], sol[:, 0], sol[:, 1], beta, gamma, w1, w2, w3, u0)[0][0]
    u1 = np.clip(u0 - alpha * Hu , 0, 1)
    if old_cost < cost:
        alpha = alpha / 1.1 # simple adaptive learning rate

    # Convergence
    if np.abs(old_cost - cost) / alpha  <= 1E-7:
        break

    old_cost = cost
    u0 = u1

plt.plot(t, sol)
plt.grid()
plt.show()

plt.plot(t, u0)
plt.grid()
plt.show()