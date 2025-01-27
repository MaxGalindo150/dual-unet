from types import SimpleNamespace
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm

def generate_standard_input_signal(signal_Nr: int, signal: SimpleNamespace, model: SimpleNamespace) -> np.ndarray:
        if signal_Nr == 1:  # Dirac
            i[0] = 1
        elif signal_Nr == 2:  # Chirp
            t = np.arange(0, signal.Ni*model.dt, model.dt)
            i = -np.sin(2 * np.pi * (signal.omega_min * t - 0.5 * (signal.omega_max - signal.omega_max) * t**2))
            i -= min(i)
            MinIdx, _ = find_peaks(-i)
            i[MinIdx[-1]:] = 0
        elif signal_Nr == 3:  # Secuencia aleatoria
            i = np.random.rand(signal.Ni, 1)
        elif signal_Nr == 4:  # Secuencia binaria aleatoria
            i = np.round(np.random.rand(signal.Ni, 1))
        elif signal_Nr == 5:  # Gaussiana
            i = norm.pdf(np.arange(0,signal.Ni), 6, 2).reshape(-1, 1)
        elif signal_Nr == 6:  # Suma de gaussianas
            i = np.exp(-(np.arange(signal.Ni) - 5)**2 / (2 * 1.5**2))
            i += np.exp(-(np.arange(signal.Ni) - 13)**2 / (2 * 1.5**2))
            i += np.exp(-(np.arange(signal.Ni) - 21)**2 / (2 * 1.5**2))
            i += np.exp(-(np.arange(signal.Ni) - 29)**2 / (2 * 1.5**2))
        else:
            raise ValueError('Input Signal Number unknown!')

        # Normalización
        i /= np.sqrt(np.dot(i.T, i))
        return i
