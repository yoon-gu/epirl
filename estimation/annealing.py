import numpy as np
import hydra
from scipy.optimize import dual_annealing
from scipy.integrate import odeint
import matplotlib.pylab as plt
from omegaconf import DictConfig
from pprint import pprint

@hydra.main(version_base=None, config_path='.', config_name='anneal')
def main(conf: DictConfig) -> None:
    beta = 0.01
    N = 100
    gamma = 0.5

    def sir(y, t, beta, gamma, u):
       S, I = y
       dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
       return dydt

    t = np.linspace(0, 21, 101)
    initial_state = np.array([90, 10])
    sol = odeint(sir, initial_state, t, args=(beta, gamma, 0.0))

    def cost(x, data):
       beta, gamma, s0, i0 = x
       t = np.linspace(0, 21, 101)
       initial_state = np.array([s0, i0])
       sol_ = odeint(sir, initial_state, t, args=(beta, gamma, 0.0))
       cost = np.linalg.norm(sol_[:, 1] - data)
       return cost

    func = lambda x: cost(x, sol[:, 1])
    lw = [0.0, 0.0, 0.0, 0.0]
    up = [1.0, 1.0, 100, 50]

    pprint(conf)
    ret = dual_annealing(func,
                        bounds=list(zip(lw, up)),
                        maxiter=conf.maxiter,
                        minimizer_kwargs=conf.minimizer_kwargs,
                        initial_temp=conf.initial_temp,
                        restart_temp_ratio=conf.restart_temp_ratio,
                        visit=conf.visit,
                        accept=conf.accept,
                        maxfun=conf.maxfun,
                        seed=conf.seed,
                        no_local_search=conf.no_local_search,
                        x0=conf.x0,
                        local_search_options=conf.local_search_options)

    print(ret)

    t = np.linspace(0, 21, 101)
    initial_state = np.array([90, 10])
    sol = odeint(sir, initial_state, t, args=(beta, gamma, 0.0))

    plt.plot(t, sol[:, 1], '.-', label='GROUND TRUTH')

    t = np.linspace(0, 21, 101)
    initial_state = np.array([ret.x[2], ret.x[3]])
    sol = odeint(sir, initial_state, t, args=(ret.x[0], ret.x[1], 0.0))

    plt.plot(t, sol[:, 1], label='Estimated')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()