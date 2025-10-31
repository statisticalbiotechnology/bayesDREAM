"""
Load methods for bayesDREAM fitted parameters.
"""

import os
import torch
import pandas as pd

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
                          modalities: list = None):
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

        Returns
        -------
        dict
            Loaded parameters
        """
        if input_dir is None:
            input_dir = os.path.join(self.model.output_dir, self.model.label)

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

        # Automatically load model-level parameters if primary modality is included
        should_load_model_level = self.model.primary_modality in modalities_to_load

        # Load model-level parameters (when primary modality is being loaded)
        if should_load_model_level:
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

        # Load per-modality alpha_y_prefit and posterior_samples_technical
        for mod_name in modalities_to_load:
            mod = self.model.modalities[mod_name]

            mod_path = os.path.join(input_dir, f'alpha_y_prefit_{mod_name}.pt')
            if os.path.exists(mod_path):
                alpha_y_mod = torch.load(mod_path)
                if use_posterior:
                    alpha_y_to_set = alpha_y_mod
                else:
                    alpha_y_to_set = alpha_y_mod.mean(dim=0)

                # Set generic alpha_y_prefit
                mod.alpha_y_prefit = alpha_y_to_set

                # Also set distribution-specific attribute
                if mod.distribution == 'negbinom':
                    # For negbinom, saved value is multiplicative
                    mod.alpha_y_prefit_mult = alpha_y_to_set
                else:
                    # For normal, binomial, mvnormal, multinomial: additive
                    mod.alpha_y_prefit_add = alpha_y_to_set

                loaded[f'alpha_y_prefit_{mod_name}'] = mod.alpha_y_prefit
                print(f"[LOAD] {mod_name}.alpha_y_prefit ← {mod_path}")

            # Load modality-specific posterior_samples_technical
            posterior_path = os.path.join(input_dir, f'posterior_samples_technical_{mod_name}.pt')
            if os.path.exists(posterior_path):
                loaded_data = torch.load(posterior_path)

                # Check if new format (with metadata) or old format (just dict)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    # New format with metadata
                    mod.posterior_samples_technical = loaded_data['posterior_samples']
                    loaded[f'posterior_samples_technical_{mod_name}'] = mod.posterior_samples_technical

                    # Reconstruct feature_meta DataFrame if present
                    feature_meta_df = None
                    if loaded_data.get('feature_meta') is not None:
                        feature_meta_df = pd.DataFrame(loaded_data['feature_meta'])

                    loaded[f'posterior_samples_technical_{mod_name}_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'feature_names': loaded_data.get('feature_names'),
                        'n_features': loaded_data.get('n_features'),
                        'feature_meta': feature_meta_df
                    }
                    print(f"[LOAD] {mod_name}.posterior_samples_technical ({loaded_data.get('n_features')} features) ← {posterior_path}")
                else:
                    # Old format (backward compatibility)
                    mod.posterior_samples_technical = loaded_data
                    loaded[f'posterior_samples_technical_{mod_name}'] = mod.posterior_samples_technical
                    print(f"[LOAD] {mod_name}.posterior_samples_technical (legacy format) ← {posterior_path}")

                # Also extract and set specific alpha attributes from posterior_samples
                # This ensures backward compatibility even if files were saved without the specific attributes
                if 'alpha_y_add' in mod.posterior_samples_technical:
                    if not hasattr(mod, 'alpha_y_prefit_add') or mod.alpha_y_prefit_add is None:
                        mod.alpha_y_prefit_add = mod.posterior_samples_technical['alpha_y_add']
                        print(f"[LOAD] {mod_name}.alpha_y_prefit_add ← extracted from posterior_samples_technical")

                if 'alpha_y_mult' in mod.posterior_samples_technical or 'alpha_y' in mod.posterior_samples_technical:
                    alpha_y_mult_key = 'alpha_y_mult' if 'alpha_y_mult' in mod.posterior_samples_technical else 'alpha_y'
                    if not hasattr(mod, 'alpha_y_prefit_mult') or mod.alpha_y_prefit_mult is None:
                        mod.alpha_y_prefit_mult = mod.posterior_samples_technical[alpha_y_mult_key]
                        print(f"[LOAD] {mod_name}.alpha_y_prefit_mult ← extracted from posterior_samples_technical")

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
            input_dir = os.path.join(self.model.output_dir, self.model.label)

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
            loaded_data = torch.load(posterior_path)

            # Check if new format (with metadata) or old format (just dict)
            if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                # New format with metadata
                self.model.posterior_samples_cis = loaded_data['posterior_samples']
                loaded['posterior_samples_cis'] = self.model.posterior_samples_cis

                # Reconstruct feature_meta DataFrame if present
                feature_meta_df = None
                if loaded_data.get('feature_meta') is not None:
                    feature_meta_df = pd.DataFrame(loaded_data['feature_meta'])

                loaded['posterior_samples_cis_metadata'] = {
                    'cis_gene': loaded_data.get('cis_gene'),
                    'modality_name': loaded_data.get('modality_name'),
                    'feature_meta': feature_meta_df
                }
                print(f"[LOAD] posterior_samples_cis (cis_gene: {loaded_data.get('cis_gene')}) ← {posterior_path}")
            else:
                # Old format (backward compatibility)
                self.model.posterior_samples_cis = loaded_data
                loaded['posterior_samples_cis'] = self.model.posterior_samples_cis
                print(f"[LOAD] posterior_samples_cis (legacy format) ← {posterior_path}")

        print(f"[LOAD] Cis fit loaded from {input_dir}")
        return loaded


    def load_trans_fit(self, input_dir: str = None, modalities: list = None):
        """
        Load fitted trans parameters.

        Parameters
        ----------
        input_dir : str, optional
            Directory to load from. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to load. If None, attempts to load all existing modalities.
            Example: ['gene', 'atac']

        Returns
        -------
        dict
            Loaded parameters
        """
        if input_dir is None:
            input_dir = os.path.join(self.model.output_dir, self.model.label)

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

        # Automatically load model-level parameters if primary modality is included
        should_load_model_level = self.model.primary_modality in modalities_to_load

        # Load model-level posterior samples (when primary modality is being loaded)
        if should_load_model_level:
            # Try new filename pattern first (includes modality name)
            posterior_path = os.path.join(input_dir, f'posterior_samples_trans_{self.model.primary_modality}.pt')
            if not os.path.exists(posterior_path):
                # Fall back to old filename pattern (backward compatibility)
                posterior_path = os.path.join(input_dir, 'posterior_samples_trans.pt')

            if os.path.exists(posterior_path):
                loaded_data = torch.load(posterior_path)
                # Check if new format (with metadata) or old format (just dict)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    # New format with metadata
                    self.model.posterior_samples_trans = loaded_data['posterior_samples']
                    loaded['posterior_samples_trans'] = self.model.posterior_samples_trans

                    # Reconstruct feature_meta DataFrame if present
                    feature_meta_df = None
                    if loaded_data.get('feature_meta') is not None:
                        feature_meta_df = pd.DataFrame(loaded_data['feature_meta'])

                    loaded['posterior_samples_trans_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'feature_names': loaded_data.get('feature_names'),
                        'n_features': loaded_data.get('n_features'),
                        'cis_gene': loaded_data.get('cis_gene'),
                        'feature_meta': feature_meta_df
                    }
                    print(f"[LOAD] posterior_samples_trans (modality: {loaded_data.get('modality_name')}, {loaded_data.get('n_features')} features) ← {posterior_path}")
                else:
                    # Old format (backward compatibility)
                    self.model.posterior_samples_trans = loaded_data
                    loaded['posterior_samples_trans'] = self.model.posterior_samples_trans
                    print(f"[LOAD] posterior_samples_trans (legacy format) ← {posterior_path}")

        # Load per-modality posterior samples
        for mod_name in modalities_to_load:
            mod_path = os.path.join(input_dir, f'posterior_samples_trans_{mod_name}.pt')
            if os.path.exists(mod_path):
                loaded_data = torch.load(mod_path)
                # Check if new format (with metadata) or old format (just dict)
                if isinstance(loaded_data, dict) and 'posterior_samples' in loaded_data:
                    # New format with metadata
                    self.model.modalities[mod_name].posterior_samples_trans = loaded_data['posterior_samples']
                    loaded[f'posterior_samples_trans_{mod_name}'] = self.model.modalities[mod_name].posterior_samples_trans

                    # Reconstruct feature_meta DataFrame if present
                    feature_meta_df = None
                    if loaded_data.get('feature_meta') is not None:
                        feature_meta_df = pd.DataFrame(loaded_data['feature_meta'])

                    loaded[f'posterior_samples_trans_{mod_name}_metadata'] = {
                        'modality_name': loaded_data.get('modality_name'),
                        'distribution': loaded_data.get('distribution'),
                        'feature_names': loaded_data.get('feature_names'),
                        'n_features': loaded_data.get('n_features'),
                        'cis_gene': loaded_data.get('cis_gene'),
                        'feature_meta': feature_meta_df
                    }
                    print(f"[LOAD] {mod_name}.posterior_samples_trans (distribution: {loaded_data.get('distribution')}, {loaded_data.get('n_features')} features) ← {mod_path}")
                else:
                    # Old format (backward compatibility)
                    self.model.modalities[mod_name].posterior_samples_trans = loaded_data
                    loaded[f'posterior_samples_trans_{mod_name}'] = self.model.modalities[mod_name].posterior_samples_trans
                    print(f"[LOAD] {mod_name}.posterior_samples_trans (legacy format) ← {mod_path}")

        print(f"[LOAD] Trans fit loaded from {input_dir}")
        print(f"[LOAD] Modalities loaded: {modalities_to_load}")
        return loaded
