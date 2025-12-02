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
    amorph_sigma_scale: float = 4.0  # how many times wider amorphous components are vs median sigma
    amorph_sigma_min_mult: float = 2.0  # lower bound multiplier for amorphous sigma
    amorph_height_fraction: float = 0.2  # fraction of max intensity for amorphous start height
    extra_height_fraction: float = 0.1  # fraction of max intensity for extra narrow components


CONFIG = XRDConfig()
