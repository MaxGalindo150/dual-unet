from types import SimpleNamespace

import numpy as np

from SSM.standart_absorption_profiles import generate_standard_absorption_profiles
from SSM.standart_input_signal import generate_standard_input_signal


def set_settings(
    example: int,
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    if example == 1:
        model = SimpleNamespace(
            **{
                "dt": 1e-9,
                "dz": 30e-7,
                "Nz": 20,
                "idx_l": 0,
                "idx_r": 20,
                "Nd": 20,
                "Ny": 100,
                "sigma_q2": 1e-28,
                "sigma_w2": 1e-10,
                "regularization_term": 0,
            }
        )

        model.idx_l += 1
        model.idx_r += 1
        profile_nr = 2
        probe = SimpleNamespace(
            **{
                "tau": 77e-12,
                "chi": 3e-2,
                "c0": 1500,
                "beta": 1,
                "Cp": 1,
                "mu": generate_standard_absorption_profiles(profile_nr, model),
            }
        )
        signal_nr = 5
        signal = SimpleNamespace(
            **{"Ni": 100, "Omega_min": 0.1 * np.pi, "Omega_max": 0.25 * np.pi}
        )
        signal.i = generate_standard_input_signal(signal_nr, signal, model)
    elif example == 2:
        model = SimpleNamespace(
            **{
                "dt": 1e-9,
                "dz": 2e-5,
                "Nz": 200,
                "idx_l": 0,
                "idx_r": 200,
                "Nd": 200,
                "Ny": 2000,
                "sigma_q2": 1e-28,
                "sigma_w2": 1e-10,
                "regularization_term": 1.5e-11,
            }
        )

        model.idx_l += 1
        model.idx_r += 1

        profile_nr = 3
        probe = SimpleNamespace(
            **{
                "tau": 77e-12,
                "chi": 3e-2,
                "c0": 1500,
                "beta": 1,
                "Cp": 1,
                "mu": generate_standard_absorption_profiles(profile_nr, model),
            }
        )
        signal_nr = 5
        signal = SimpleNamespace(
            **{
                "Ni": 20,
                "Omega_min": 0.7 * 3.141592653589793,
                "Omega_max": 0.99 * 3.141592653589793,
            }
        )
        signal.i = generate_standard_input_signal(signal_nr, signal, model)
    else:
        model = SimpleNamespace()
        probe = SimpleNamespace()
        signal = SimpleNamespace()

    return model, probe, signal
