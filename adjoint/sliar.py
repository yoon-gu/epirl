import sympy as sp
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config_sliar")
def run(conf: DictConfig):
    # State
    S, L, I, A = sp.symbols('S L I A')
    state = [S, L, I, A]

    # Control
    u = sp.symbols('u')

    # Parmaeters
    alpha, beta, sigma, epsilon, q, delta, kappa, p, tau, eta, P, Q = sp.symbols('alpha beta sigma epsilon q delta kappa p tau eta P Q')
    params = [alpha, beta, sigma, epsilon, q, delta, kappa, p, tau, eta, P, Q]

    # Adjoints
    l_S, l_L, l_I, l_A = sp.symbols('lambda_S lambda_L lambda_I lambda_A')
    adjoint = [l_S, l_L, l_I, l_A]
    l = sp.Matrix([l_S, l_L, l_I, l_A])

    f = P*I + Q*u**2
    g = sp.Matrix(
        [
        - beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - u * S,
        beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - kappa * L,
        p * kappa * L - alpha * I - tau * I,
        (1 - p) * kappa * L  - eta * A
        ]
    )
    H = sp.Matrix([f + l.dot(g)])

    dHdx = H.jacobian(state)
    dHdl = H.jacobian(adjoint)
    dHdu = H.jacobian([u])

    # Automation
    cost_fn = sp.lambdify([*state, *params, u], f)
    dHdx_fn = sp.lambdify([*adjoint, *state, *params, u], dHdx)
    dHdl_fn = sp.lambdify([*adjoint, *state, *params, u], dHdl)
    dHdu_fn = sp.lambdify([*adjoint, *state, *params, u], dHdu)

    # ODE Systems
    def state_de(y, t, params, u_interp):
        return dHdl_fn(*(y*0), *y, *params, u_interp(t))[0]

    def adjoint_de(y, t, x_interp, params, u_interp):
        val = -dHdx_fn(*y, *x_interp(t), *params, u_interp(t))[0]
        return val

    t0 = conf.t0
    tf = conf.tf

    params = [  conf.alpha, conf.beta, conf.sigma, conf.epsilon, \
                conf.q, conf.delta, conf.kappa, conf.p, conf.tau, \
                conf.eta, conf.P, conf.Q
             ]

    # Initial
    y0 = np.array([conf.S0, conf.L0, conf.I0, conf.A0])
    state_dim = len(y0)
    t = np.linspace(t0,tf, 301)
    dt = t[1] - t[0]
    u0 = np.ones_like(t)

    MaxIter = conf.MaxIter
    update_weight = conf.update_weight
    old_cost = 1E8
    for it in tqdm(range(MaxIter+1)):
        # State
        u_intp = lambda tc: np.interp(tc, t, u0)
        sol = odeint(state_de, y0, t, args=(params, u_intp))

        # Cost
        state_mid = [(ss[1:] + ss[:-1]) / 2. for ss in np.hsplit(sol, state_dim)]
        u_mid = (u0[1:] + u0[:-1]) / 2.
        cost = np.sum ( dt * cost_fn(*state_mid, *params, np.expand_dims(u_mid, 1)) )

        # Adjoint
        u_intp = lambda tc: np.interp(tf - tc, t, u0)
        x_intp = lambda tc: np.array([np.interp(tf - tc, t, sol[:, k]) for k in range(state_dim)])
        y_T = y0 * 0
        l_sol = odeint(adjoint_de, y_T, t, args=(x_intp, params, u_intp))
        l_sol = np.flipud(l_sol)

        # Simple Gradient
        l_sols = [l_sol[:, k] for k in range(state_dim)]
        sols = [sol[:, k] for k in range(state_dim)]

        Hu = dHdu_fn(*l_sols, *sols, *params, u0)[0][0]
        u1 = np.clip(u0 - update_weight * Hu , 0, 1)
        if old_cost < cost:
            update_weight = update_weight / 1.0001 # simple adaptive learning rate

        # Convergence
        if np.abs(old_cost - cost) / update_weight  <= conf.tolerance:
            break

        old_cost = cost
        u0 = u1

    plt.clf()
    plt.plot(t, sol)
    plt.grid()
    plt.savefig('state.png')

    plt.clf()
    plt.plot(t, u0)
    plt.grid()
    plt.savefig('control.png')

if __name__ == '__main__':
    run()