import numpy as np


def biting(temperature):

    n_temp = temperature.shape[0]

    # Initialize
    biting_rate = np.zeros((n_temp,))

    for i in range(n_temp):

        temp_i = temperature[i]

        if temp_i >= 0:
            biting_rate[i] = max(0.000203 * temp_i * (temp_i - 11.7) * np.sqrt(42.3 - temp_i), 0)

    return biting_rate


def deposition(temperature):

    n_temp = temperature.shape[0]

    # Initialize
    deposition_rate = np.zeros((n_temp,))

    for i in range(n_temp):

        temp_i = temperature[i]

        deposition_rate[i] = max(-0.153 * (temp_i ** 2) + 8.61 * temp_i - 97.7, 0)

    return deposition_rate


def immature_death(temperature):

    n_temp = temperature.shape[0]

    # Initialize
    death_rate = np.zeros((n_temp,))

    for i in range(n_temp):

        temp_i = temperature[i]

        death_rate[i] = min(0.002 * np.exp(((temp_i - 23) / 6.05) ** 2), 1)

    return death_rate


def mature_death(temperature):

    n_temp = temperature.shape[0]

    # Initialize
    death_rate = np.zeros((n_temp,))

    for i in range(n_temp):

        temp_i = temperature[i]

        if temp_i > 15:
            if temp_i < 32:
                death_rate[i] = 1 / 30
            else:
                death_rate[i] = -29 / 30 / 19 * (-temp_i + 51) + 1
        elif temp_i < -4:
            death_rate[i] = 1
        else:
            death_rate[i] = -29 / 30 / 19 * (temp_i + 4) + 1

    return death_rate


def maturation(temperature, rainfall_avg):

    n_temp = temperature.shape[0]

    R_L = 76
    R = rainfall_avg

    # Initialize
    maturation_rate = np.zeros((n_temp,))

    for i in range(n_temp):

        temp_i = temperature[i]

        if (temp_i >= 16.5) and (temp_i <= 35.6):
            f = -0.153 * (temp_i ** 2) + 8.61 * temp_i - 97.7
            e = f / mature_death(np.array([temp_i]))
            p_E = 3.6 * R * (R_L - R) / (R_L ** 2)
            if p_E < 0:
                p_E = 0
            p_L = np.exp(-0.00554 * temp_i + 0.06737) * R * (R_L - R) / (R_L ** 2)
            p_P = 3 * R * (R_L - R) / (R_L ** 2)
            tau_EA = 1 / (-0.00094 * (temp_i ** 2) + 0.049 * temp_i - 0.552)
            maturation_rate[i] = e * p_E * p_L * p_P / tau_EA

    return maturation_rate