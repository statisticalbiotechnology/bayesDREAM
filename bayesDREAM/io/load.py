"""
Load methods for bayesDREAM fitted parameters.
"""

import os
import torch

class ModelLoader:
    """Handles loading fitted parameters."""

    def __init__(self, model):
        """
        Initialize model loader.

        Parameters
        ----------
        model : bayesDREAM
            The parent model instance
        """
        self.model = model

    def load_technical_fit(self, input_dir: str = None, use_posterior: bool = True,
                          modalities: list = None, load_model_level: bool = True):
        """
        Load fitted technical parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        use_posterior : bool
            If True, loads full posterior samples. If False, uses posterior mean as point estimates.
        modalities : list of str, optional
            List of modality names to load. If None, attempts to load all existing modalities.
            Example: ['gene', 'atac']
        load_model_level : bool, optional
            If True, loads model-level alpha_x_prefit, alpha_y_prefit, and
            posterior_samples_technical (default: True).

        Returns
        -------
        dict
            Loaded parameters
        """
        if input_dir is None:
            input_dir = self.model.output_dir

        loaded = {}

        # Determine which modalities to load
        if modalities is None:
            modalities_to_load = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_load = modalities

        # Load model-level parameters
        if load_model_level:
            # Load alpha_x_prefit
            alpha_x_path = os.path.join(input_dir, 'alpha_x_prefit.pt')
            if os.path.exists(alpha_x_path):
                alpha_x = torch.load(alpha_x_path)
                if use_posterior:
                    self.model.alpha_x_prefit = alpha_x
                    self.model.alpha_x_type = 'posterior'
                else:
                    self.model.alpha_x_prefit = alpha_x.mean(dim=0)
                    self.model.alpha_x_type = 'point'
                loaded['alpha_x_prefit'] = self.model.alpha_x_prefit
                print(f"[LOAD] alpha_x_prefit ({self.model.alpha_x_type}) ← {alpha_x_path}")

            # Load alpha_y_prefit
            alpha_y_path = os.path.join(input_dir, 'alpha_y_prefit.pt')
            if os.path.exists(alpha_y_path):
                alpha_y = torch.load(alpha_y_path)
                if use_posterior:
                    self.model.alpha_y_prefit = alpha_y
                    self.model.alpha_y_type = 'posterior'
                else:
                    self.model.alpha_y_prefit = alpha_y.mean(dim=0)
                    self.model.alpha_y_type = 'point'
                loaded['alpha_y_prefit'] = self.model.alpha_y_prefit
                print(f"[LOAD] alpha_y_prefit ({self.model.alpha_y_type}) ← {alpha_y_path}")

            # Load posterior samples
            posterior_path = os.path.join(input_dir, 'posterior_samples_technical.pt')
            if os.path.exists(posterior_path):
                self.model.posterior_samples_technical = torch.load(posterior_path)
                loaded['posterior_samples_technical'] = self.model.posterior_samples_technical
                print(f"[LOAD] posterior_samples_technical ← {posterior_path}")

        # Load per-modality alpha_y_prefit
        for mod_name in modalities_to_load:
            mod_path = os.path.join(input_dir, f'alpha_y_prefit_{mod_name}.pt')
            if os.path.exists(mod_path):
                alpha_y_mod = torch.load(mod_path)
                if use_posterior:
                    self.model.modalities[mod_name].alpha_y_prefit = alpha_y_mod
                else:
                    self.model.modalities[mod_name].alpha_y_prefit = alpha_y_mod.mean(dim=0)
                loaded[f'alpha_y_prefit_{mod_name}'] = self.model.modalities[mod_name].alpha_y_prefit
                print(f"[LOAD] {mod_name}.alpha_y_prefit ← {mod_path}")

        print(f"[LOAD] Technical fit loaded from {input_dir}")
        print(f"[LOAD] Modalities loaded: {modalities_to_load}")
        return loaded

    def load_cis_fit(self, input_dir: str = None, use_posterior: bool = True):
        """
        Load fitted cis parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        use_posterior : bool
            If True, loads full posterior samples. If False, uses posterior mean as point estimate.

        Returns
        -------
        dict
            Loaded parameters
        """
        if input_dir is None:
            input_dir = self.model.output_dir

        loaded = {}

        # Load x_true
        x_true_path = os.path.join(input_dir, 'x_true.pt')
        if os.path.exists(x_true_path):
            x_true = torch.load(x_true_path)
            if use_posterior:
                self.model.x_true = x_true
                self.model.x_true_type = 'posterior'
            else:
                self.model.x_true = x_true.mean(dim=0)
                self.model.x_true_type = 'point'
            loaded['x_true'] = self.model.x_true
            print(f"[LOAD] x_true ({self.model.x_true_type}) ← {x_true_path}")

        # Load posterior samples
        posterior_path = os.path.join(input_dir, 'posterior_samples_cis.pt')
        if os.path.exists(posterior_path):
            self.model.posterior_samples_cis = torch.load(posterior_path)
            loaded['posterior_samples_cis'] = self.model.posterior_samples_cis
            print(f"[LOAD] posterior_samples_cis ← {posterior_path}")

        print(f"[LOAD] Cis fit loaded from {input_dir}")
        return loaded


    def load_trans_fit(self, input_dir: str = None, modalities: list = None,
                      load_model_level: bool = True):
        """
        Load fitted trans parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to load. If None, attempts to load all existing modalities.
            Example: ['gene', 'atac']
        load_model_level : bool, optional
            If True, loads model-level posterior_samples_trans (default: True).

        Returns
        -------
        dict
            Loaded parameters
        """
        if input_dir is None:
            input_dir = self.model.output_dir

        loaded = {}

        # Determine which modalities to load
        if modalities is None:
            modalities_to_load = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_load = modalities

        # Load model-level posterior samples
        if load_model_level:
            posterior_path = os.path.join(input_dir, 'posterior_samples_trans.pt')
            if os.path.exists(posterior_path):
                self.model.posterior_samples_trans = torch.load(posterior_path)
                loaded['posterior_samples_trans'] = self.model.posterior_samples_trans
                print(f"[LOAD] posterior_samples_trans ← {posterior_path}")

        # Load per-modality posterior samples
        for mod_name in modalities_to_load:
            mod_path = os.path.join(input_dir, f'posterior_samples_trans_{mod_name}.pt')
            if os.path.exists(mod_path):
                self.model.modalities[mod_name].posterior_samples_trans = torch.load(mod_path)
                loaded[f'posterior_samples_trans_{mod_name}'] = self.model.modalities[mod_name].posterior_samples_trans
                print(f"[LOAD] {mod_name}.posterior_samples_trans ← {mod_path}")

        print(f"[LOAD] Trans fit loaded from {input_dir}")
        print(f"[LOAD] Modalities loaded: {modalities_to_load}")
        return loaded
