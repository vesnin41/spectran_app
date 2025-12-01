from dataclasses import dataclass


@dataclass
class XRDConfig:
    theta_min: float = 20.0
    theta_max: float = 60.0
    savgol_window: int = 31
    savgol_poly: int = 3
    ci_fwhm_min: float = 0.1
    ci_fwhm_max: float = 2.0
    rel_height_min: float = 0.05
    prominence_rel: float = 0.02
    delta_2theta_max: float = 0.2


CONFIG = XRDConfig()
