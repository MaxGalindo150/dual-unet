from types import SimpleNamespace

import numpy as np
from scipy.stats import norm


def generate_standard_absorption_profiles(
    profile_nr: int, model: SimpleNamespace
) -> np.ndarray[float, float]:
    """
    Generate standard absorption profiles for the SSM model.
    """
    mu = np.zeros(model.Nd)  # Initialize mu as a zero array of size self.Nd
    # Depending on the value of profile_Nr, generate different absorption profiles
    if profile_nr == 1:
        # Add random normal values to mu
        mu += 8e2 * np.random.normal(20, 2.0, model.Nd)
        mu += 8e2 * np.random.normal(30, 2, model.Nd)
        mu += 6e2 * np.random.normal(25, 1.7, model.Nd)
    elif profile_nr == 2:
        # Add normal distribution values to mu
        mu += 10e4 * norm.pdf(np.arange(0, model.Nd), 5, 1.5)
        mu += 10e4 * norm.pdf(np.arange(0, model.Nd), 15, 1.5)
        mu += 3e4 * norm.pdf(np.arange(0, model.Nd), 10, 1.0)
    elif profile_nr == 3:
        # Add random normal values to mu
        mu += 20e3 * np.random.normal(model.Nd / 4, model.Nd / 15, model.Nd)
        mu += 20e3 * np.random.normal(3 * model.Nd / 4, model.Nd / 15, model.Nd)
        mu += 8e3 * np.random.normal(model.Nd / 2, model.Nd / 15, model.Nd)
    elif profile_nr == 4:
        # Add constant values to specific ranges in mu
        mu[int(np.floor(model.Nd / 4)) : int(np.floor(3 * model.Nd / 4)) + 1] += 1e1
        mu[
            int(np.floor(4 * model.Nd / 10)) : int(np.floor(6 * model.Nd / 10)) + 1
        ] += 1e1
    else:
        # If profile_Nr is not 1, 2, 3, or 4, raise an error
        raise ValueError("Absorption Profile Number unknown!")
    return mu  # Return the generated absorption profile
