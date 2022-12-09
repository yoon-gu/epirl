import sympy as sp
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(conf: DictConfig):
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
    beta = conf.beta
    gamma = conf.gamma
    S0 = conf.S0
    I0 = conf.I0
    w1, w2, w3 = conf.w1, conf.w2, conf.w3

    params = [beta, gamma, w1, w2, w3]

    # Initial
    y0 = np.array([S0, I0])
    state_dim = len(y0)
    t = np.linspace(t0,tf, 301)
    dt = t[1] - t[0]
    u0 = np.ones_like(t)

    MaxIter = conf.MaxIter
    alpha = conf.alpha
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
        y_T = np.array([0,0])
        l_sol = odeint(adjoint_de, y_T, t, args=(x_intp, params, u_intp))
        l_sol = np.flipud(l_sol)

        # Simple Gradient
        l_sols = [l_sol[:, k] for k in range(state_dim)]
        sols = [sol[:, k] for k in range(state_dim)]

        Hu = dHdu_fn(*l_sols, *sols, *params, u0)[0][0]
        u1 = np.clip(u0 - alpha * Hu , 0, 1)
        if old_cost < cost:
            alpha = alpha / 1.1 # simple adaptive learning rate

        # Convergence
        if np.abs(old_cost - cost) / alpha  <= 1E-7:
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