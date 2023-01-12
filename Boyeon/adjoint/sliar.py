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
    beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, P, Q, R, W = sp.symbols('beta sigma kappa alpha tau p eta epsilon q delta P Q R W')
    params = [beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, P, Q, R, W]

    # Adjoints
    l_S, l_L, l_I, l_A = sp.symbols('lambda_S lambda_L lambda_I lambda_A')
    adjoint = [l_S, l_L, l_I, l_A]
    l = sp.Matrix([l_S, l_L, l_I, l_A])

    f = P*I + Q*u**2 + R*tau**2 + W*sigma**2
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

    # parameters
    t0 = 0
    tf = 300
    S0 = 1E6
    L0 = 0
    I0 = 1
    A0 = 0
    sigma = 0 
    kappa = 0.526
    alpha = 0.244
    tau = 0
    p = 0.667
    eta = 0.244
    epsilon = 0
    q = 0.5
    delta = 1
    R0 = 1.9847
    beta = R0/(S0 * ((epsilon / kappa) + ((1 - q)*p/alpha) + (delta*(1-p)/eta)))

    params = [beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, conf.P, conf.Q, conf.R, conf.W]

    # Initial
    y0 = np.array([S0, L0, I0, A0])
    state_dim = len(y0)
    t = np.linspace(t0,tf, 301)
    dt = t[1] - t[0]
    u0 = np.ones_like(t)

    MaxIter = conf.MaxIter
    update_weight = conf.update_weight
    old_cost = 1E8
    costs = [old_cost]
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
        costs.append(old_cost)
        u0 = u1

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(t, sol[:,0], '.-b', label = 'S')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(t, sol[:,1], '.-y', label = 'L')
    ax2.plot(t, sol[:,2], '.-r', label = 'I')
    ax2.plot(t, sol[:,3], '.-g', label = 'A')
    ax2.legend(loc='lower right')
    plt.grid()
    plt.legend()
    plt.title('SLIAR model with control'+' cost : '+str(old_cost))
    plt.xlabel('day')
    plt.savefig('SLIAR_w_control_adj.png', dpi=300)
    plt.show(block=False)

    plt.plot(t, u0)
    plt.grid()
    plt.title(' cost : '+str(old_cost))
    plt.savefig('control_nu.png', dpi=300)
    plt.show()

    plt.clf()
    plt.plot(costs)
    plt.grid()
    plt.savefig('cost.png')

if __name__ == '__main__':
    run()