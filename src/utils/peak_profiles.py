# src/utils/peak_profiles.py
import numpy as np


def gaussian(x, amplitude, center, sigma):
    """
    Гауссов пик.

    amplitude : высота/масштаб
    center    : положение максимума
    sigma     : параметр ширины
    """
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def lorentzian(x, amplitude, center, gamma):
    """
    Лоренцев пик.

    gamma : половина ширины на полувысоте (HWHM).
    """
    return amplitude * (gamma**2) / ((x - center) ** 2 + gamma**2)


def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """
    Псевдо-Войт — линейная комбинация гаусса и лоренца.

    eta in [0, 1] — вес лоренцевой компоненты.
    """
    g = gaussian(x, 1.0, center, sigma)
    l = lorentzian(x, 1.0, center, gamma)
    profile = eta * l + (1 - eta) * g
    return amplitude * profile
