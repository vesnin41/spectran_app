# src/utils/spectrum_math.py
import numpy as np


# ======= Базовые вспомогательные функции =======

def deg2rad(deg: float) -> float:
    """Перевод градусов в радианы."""
    return np.radians(deg)


def rad2deg(rad: float) -> float:
    """Перевод радиан в градусы."""
    return np.degrees(rad)


def normalize_data(intens):
    """
    Нормировка массива интенсивностей к диапазону [0, 1].

    Если все значения одинаковы, возвращает нулевой массив.
    """
    intens = np.asarray(intens, dtype=float)
    if intens.max() == intens.min():
        return np.zeros_like(intens)
    return (intens - intens.min()) / (intens.max() - intens.min())


# ======= Связь d_hkl и 2θ (закон Брегга) =======

def get_2theta(row, wavelength: float = 1.54060) -> float:
    """
    Вычисление 2θ (град) по d_hkl (Å) по закону Брегга.

        nλ = 2 d_hkl sin θ, здесь n = 1,
        2θ = 2 arcsin(λ / (2 d_hkl)).
    """
    d_hkl = row["d_hkl"]
    return 2 * rad2deg(np.arcsin(wavelength / (2 * d_hkl)))


def bragg_2theta_from_d(d_hkl: float, wavelength: float = 1.54060) -> float:
    """То же самое, но без DataFrame: 2θ (град) по d_hkl (Å)."""
    return 2 * rad2deg(np.arcsin(wavelength / (2 * d_hkl)))


def bragg_d_from_2theta(two_theta: float, wavelength: float = 1.54060) -> float:
    """Расчёт d_hkl (Å) по 2θ (град)."""
    theta = deg2rad(two_theta / 2)
    return wavelength / (2 * np.sin(theta))


def get_d_hkl(row, wavelength: float = 1.54060) -> float:
    """
    Расчёт d_hkl (Å) по положению пика center (2θ, град).

    Удобно использовать на результатах подгонки пиков.
    """
    angle_deg = row["center"] / 2
    theta = deg2rad(angle_deg)
    return wavelength / (2 * np.sin(theta))


# ======= FWHM для разных профилей =======

def gaussian_fwhm(sigma: float) -> float:
    """FWHM гауссова пика: 2 * sqrt(2 ln 2) * σ."""
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def lorentzian_fwhm(gamma: float) -> float:
    """FWHM лоренцева пика: 2 * γ."""
    return 2 * gamma


def pseudo_voigt_fwhm(sigma: float, gamma: float) -> float:
    """
    Приближённая FWHM псевдо-Войта.

    Используется стандартная аппроксимация:
    FWHM ≈ 0.5346 * (2γ) + sqrt(0.2166*(2γ)^2 + (2*sqrt(2 ln 2)*σ)^2)
    """
    g = 2 * gamma
    gs = gaussian_fwhm(sigma)
    return 0.5346 * g + np.sqrt(0.2166 * g**2 + gs**2)


def get_FWHM(row) -> float:
    """
    Обёртка для DataFrame: FWHM из σ для гауссовой модели.

    Если ты используешь GaussianModel с параметром 'sigma',
    эта функция даёт FWHM в тех же единицах, что и ось 2θ.
    """
    sigma = row["sigma"]
    return np.round(gaussian_fwhm(sigma), 3)


# ======= Размер кристаллитов (Шеррер) =======

def scherrer_size(two_theta: float, fwhm_deg: float,
                  wavelength: float = 0.154060, k: float = 0.9) -> float:
    """
    Формула Шеррера в общем виде.

    two_theta : положение пика (градусы)
    fwhm_deg  : ширина пика на полувысоте (градусы)
    wavelength: длина волны (нм), по умолчанию CuKα = 0.154060 нм
    k         : коэффициент формы (обычно ~0.9)
    """
    theta = deg2rad(two_theta / 2)
    beta = deg2rad(fwhm_deg)
    if beta == 0:
        return np.nan
    return k * wavelength / (beta * np.cos(theta))


def get_cristal_size(row, wavelength: float = 0.154060, k: float = 0.9) -> float:
    """
    Размер кристаллита (нм) из строки таблицы с данными пика.

    Ожидает наличие полей:
        'center' — положение пика (2θ, град),
        'fwhm'   — FWHM (град).
    """
    two_theta = row["center"]
    fwhm_deg = row["fwhm"]
    return np.round(scherrer_size(two_theta, fwhm_deg, wavelength=wavelength, k=k), 3)


def get_crystal_size(row, wavelength: float = 0.154060, k: float = 0.9) -> float:
    """
    Alias for get_cristal_size with corrected spelling.
    """
    return get_cristal_size(row, wavelength=wavelength, k=k)
