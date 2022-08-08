import time
from temp_parameters import *
from model import *
from compute_incidence import *


if __name__ == '__main__':

    # Human agent
    n_region = 2
    region_h = np.arange(0, n_region)
    state_h = np.arange(0, 4)

    # Mosquito agent
    region_v = np.arange(0, n_region)
    state_v = np.arange(0, 4)

    # Set options
    year_init = 2013
    year_termi = 2018
    day_init = 121
    vsupply_day = np.arange(127, 131)
    dt = 0.1

    # Result path
    result_path = '../../result/malaria3_by'
    suffix = ''

    # Time span
    year_span = np.arange(year_init, year_termi + 1)
    n_year = len(year_span)
    tspan = np.arange(day_init, 365 * n_year + dt, dt)
    n_step = len(tspan) - 1

    # Susceptible mosquito supply
    tspan_vsupply = np.zeros((n_year, len(vsupply_day)), int)
    for i in range(len(vsupply_day)):
        tspan_vsupply[:, i] = np.arange(vsupply_day[i], tspan[-1], 365)
    tspan_vsupply = np.reshape(tspan_vsupply, np.prod(np.shape(tspan_vsupply)))

    # Load human demographic data
    demographic_data = pd.read_csv('../data/demographic_2013-2018_avg.csv',
                                   names=['rate'])

    N_h0 = demographic_data.loc[0].values[0]
    birth_rate = demographic_data.loc[1].values[0]
    death_rate = demographic_data.loc[2].values[0]

    # Load climate data
    climate_data = pd.read_csv('../data/climate_2013-2018.csv',
                               names=['Date', 'Temperature', 'Rainfall'])

    # Temperature dependent parameters
    temperature = np.array(climate_data['Temperature'])
    rainfall_avg = 3

    b = biting(temperature)
    mu_a = deposition(temperature)
    delta_a = immature_death(temperature)
    mu_v = maturation(temperature, rainfall_avg)
    delta_v = mature_death(temperature)

    # Parameters
    params = {'region_h': region_h,
              'state_h': state_h,
              'region_v': region_v,
              'state_v': state_v,

              'year_init': year_init,
              'year_termi': year_termi,
              'day_init': day_init,
              'tspan': tspan,
              'tspan_vsupply': tspan_vsupply,
              'dt': dt,

              'N_h0': N_h0,

              'mu_h': birth_rate / 365,
              'delta_h': death_rate / 365,
              'c_init': 650,
              'c': 165,
              'p': 0.429522905768833,
              'beta_hv': 0.097912363499787,
              'beta_vh': 0.035469351096884,
              'tau_s': 14,
              'tau_l': 314,
              'tau_r': 207,
              'nu_v': 1 / 10,
              'gamma_h': 1 / 4,
              'q': 0.04,
              'rho_h': 1 / 35,

              'b': b,
              'mu_a': mu_a,
              'delta_a': delta_a,
              'mu_v': mu_v,
              'delta_v': delta_v,

              'result_path': result_path,
              'suffix': suffix}

    # Check computing time
    t_start = time.time()

    # Run the model
    model = MalariaModel(params=params)
    for i in range(n_step):
        model.step()

    t_end = time.time()
    print('Running the model', t_end - t_start)

    # Compute incidence
    incidence = compute_incidence(model=model)

    # Plot incidence
    incidence.plot_incidence()