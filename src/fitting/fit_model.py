# src/fitting/fit_model.py

from lmfit import models
import numpy as np
import pandas as pd
import random


class Model:
    """
    Wrapper around lmfit.Model to build a composite peak model
    from a list of peak definitions in spec['model'].

    spec structure:
        spec = {
            'x': np.ndarray  # two_theta
            'y': np.ndarray  # intensity
            'model': [
                {
                    'type': 'GaussianModel',
                    'params': { ... initial params ... }
                },
                ...
            ]
        }
    """

    def __init__(self, spec):
        self.spec = spec
        self._output = None

    # ------------- PUBLIC API -----------------

    def fit(self):
        """
        Fit composite model to the spectrum.

        Returns
        -------
        output : lmfit.model.ModelResult
        """
        model, params = self.generate_model(self.spec)
        output = model.fit(
            self.spec["y"],
            params,
            x=self.spec["x"],
        )
        self._output = output
        return output

    def print_best_values(self):
        """
        Extract fitted peak parameters into a DataFrame.

        Returns
        -------
        df : pandas.DataFrame
            Columns:
                prefix, center, model, amplitude, height,
                sigma, gamma, fwhm
        """
        if self._output is None:
            raise RuntimeError("print_best_values called before fit()")

        params = self._output.params
        rows = []

        for i, model_dict in enumerate(self.spec["model"]):
            prefix = f"m{i}_"
            model_type = model_dict.get("type", "")

            def _get(name):
                par = params.get(prefix + name, None)
                return float(par.value) if par is not None else np.nan

            center = _get("center")
            amplitude = _get("amplitude")
            sigma = _get("sigma")
            gamma = _get("gamma")

            fwhm = np.nan
            height = np.nan

            if model_type == "GaussianModel":
                if np.isfinite(sigma) and sigma > 0:
                    fwhm = 2.3548 * abs(sigma)
                    if np.isfinite(amplitude) and amplitude != 0:
                        height = amplitude / (abs(sigma) * np.sqrt(2 * np.pi))
            elif model_type == "LorentzianModel":
                if np.isfinite(gamma) and gamma > 0:
                    fwhm = 2.0 * abs(gamma)
                    if np.isfinite(amplitude) and amplitude != 0:
                        height = amplitude / (np.pi * abs(gamma))
            elif model_type == "VoigtModel":
                sigma_eff = abs(sigma) if np.isfinite(sigma) else 0.0
                gamma_eff = abs(gamma) if np.isfinite(gamma) else 0.0
                fwhm_g = 2.3548 * sigma_eff
                fwhm_l = 2.0 * gamma_eff
                if fwhm_g > 0 or fwhm_l > 0:
                    fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
                height = np.nan

            rows.append(
                dict(
                    prefix=prefix,
                    model=model_type,
                    center=center,
                    amplitude=amplitude,
                    height=height,
                    sigma=sigma,
                    gamma=gamma,
                    fwhm=fwhm,
                )
            )

        data = pd.DataFrame(rows)
        for col in ["center", "amplitude", "height", "sigma", "gamma", "fwhm"]:
            data[col] = data[col].round(3)

        return data

    # ------------- INTERNAL MODEL BUILDING -----------------

    @classmethod
    def generate_model(cls, spec):
        """
        Build a composite lmfit model from spec['model'] definitions.

        Returns
        -------
        composite_model : lmfit.Model
        params : lmfit.Parameters
        """
        composite_model = None
        params = None

        x = spec["x"]
        y = spec["y"]
        rng = random.Random(0)  # deterministic starting guesses

        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        y_max = np.max(y)

        # Construct each peak model
        for i, model_def in enumerate(spec["model"]):
            prefix = f"m{i}_"

            model_type = model_def["type"]
            try:
                model = getattr(models, model_type)(prefix=prefix)
            except AttributeError:
                raise NotImplementedError(f"lmfit does not support model: {model_type}")

            # Parameter bounds
            model.set_param_hint("sigma", min=1e-6, max=x_range)
            model.set_param_hint("center", min=x_min, max=x_max)
            model.set_param_hint("height", min=1e-6, max=1.1 * y_max)
            model.set_param_hint("amplitude", min=1e-6)

            # Default random starting values
            default_params = {
                prefix + "center": x_min + x_range * rng.random(),
                prefix + "height": y_max * rng.random(),
                prefix + "sigma": x_range * 0.01,  # more gentle default than random
            }

            # If user provided params from update_spec_from_peaks → override
            model_params = model.make_params(
                **default_params,
                **model_def.get("params", {}),
            )

            # Merge parameters into global set
            if params is None:
                params = model_params
            else:
                params.update(model_params)

            # Merge models into composite
            composite_model = model if composite_model is None else composite_model + model

        return composite_model, params
