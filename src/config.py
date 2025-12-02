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
    amorph_width_factor: float = 3.0  # how many times wider than median crystalline FWHM counts as amorphous
    amorph_fwhm_min: float = 1.5  # absolute FWHM threshold (deg 2θ) to treat as amorphous
    min_crystal_size_nm: float = 2.0  # below this size treat as non-crystalline
    amorph_components_default: int = 1  # default amorphous components for interactive mode


CONFIG = XRDConfig()
