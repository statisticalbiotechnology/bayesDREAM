"""
Save methods for bayesDREAM fitted parameters.
"""

import os
import torch

class ModelSaver:
    """Handles saving fitted parameters."""

    def __init__(self, model):
        """
        Initialize model saver.

        Parameters
        ----------
        model : bayesDREAM
            The parent model instance
        """
        self.model = model

    ########################################################
    # Save/Load fitted parameters
    ########################################################

    def save_technical_fit(self, output_dir: str = None, modalities: list = None):
        """
        Save fitted technical parameters from fit_technical().

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to save. If None, saves all modalities.
            Example: ['gene', 'atac']

        Returns
        -------
        dict
            Paths to saved files

        Notes
        -----
        Saves per-modality:
        - alpha_y_prefit_{modality}.pt: Appropriate alpha_y for distribution (add or mult)
        - posterior_samples_technical_{modality}.pt: Full posterior samples

        Saves model-level (automatically when primary modality is included):
        - alpha_x_prefit.pt: Cis gene overdispersion (if set)
        - alpha_y_prefit.pt: Trans gene overdispersion (from primary modality)
        """
        if output_dir is None:
            output_dir =  os.path.join(self.model.output_dir, self.model.label)

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Determine which modalities to save
        if modalities is None:
            modalities_to_save = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_save = modalities

        # Automatically save model-level parameters if primary modality is included
        should_save_model_level = self.model.primary_modality in modalities_to_save

        # Save model-level parameters (when primary modality is being saved)
        if should_save_model_level:
            if hasattr(self.model, 'alpha_x_prefit') and self.model.alpha_x_prefit is not None:
                path = os.path.join(output_dir, 'alpha_x_prefit.pt')
                torch.save(self.model.alpha_x_prefit, path)
                saved_files['alpha_x_prefit'] = path
                saved_files['alpha_x_type'] = self.model.alpha_x_type
                print(f"[SAVE] alpha_x_prefit ({self.model.alpha_x_type}) → {path}")

            if hasattr(self.model, 'alpha_y_prefit') and self.model.alpha_y_prefit is not None:
                path = os.path.join(output_dir, 'alpha_y_prefit.pt')
                torch.save(self.model.alpha_y_prefit, path)
                saved_files['alpha_y_prefit'] = path
                saved_files['alpha_y_type'] = self.model.alpha_y_type
                print(f"[SAVE] alpha_y_prefit ({self.model.alpha_y_type}) → {path}")

        # Save per-modality alpha_y_prefit and posterior_samples_technical
        for mod_name in modalities_to_save:
            mod = self.model.modalities[mod_name]

            # Save distribution-appropriate alpha_y
            if hasattr(mod, 'alpha_y_prefit') and mod.alpha_y_prefit is not None:
                # Determine which alpha_y to save based on distribution
                if mod.distribution == 'negbinom':
                    # Save multiplicative for negbinom
                    if hasattr(mod, 'alpha_y_prefit_mult') and mod.alpha_y_prefit_mult is not None:
                        alpha_to_save = mod.alpha_y_prefit_mult
                        alpha_type = 'mult'
                    else:
                        alpha_to_save = mod.alpha_y_prefit
                        alpha_type = 'mult (default)'
                else:
                    # Save additive for normal, binomial, mvnormal, multinomial
                    if hasattr(mod, 'alpha_y_prefit_add') and mod.alpha_y_prefit_add is not None:
                        alpha_to_save = mod.alpha_y_prefit_add
                        alpha_type = 'add'
                    else:
                        alpha_to_save = mod.alpha_y_prefit
                        alpha_type = 'add (default)'

                path = os.path.join(output_dir, f'alpha_y_prefit_{mod_name}.pt')
                torch.save(alpha_to_save, path)
                saved_files[f'alpha_y_prefit_{mod_name}'] = path
                saved_files[f'alpha_y_prefit_{mod_name}_type'] = alpha_type
                print(f"[SAVE] {mod_name}.alpha_y_prefit ({alpha_type}) → {path}")

            # Save modality-specific posterior_samples_technical
            if hasattr(mod, 'posterior_samples_technical') and mod.posterior_samples_technical is not None:
                # Remove large observation arrays before saving
                # Also ensure that alpha_y_add and alpha_y_mult are included for backward compatibility
                posterior_clean = {k: v for k, v in mod.posterior_samples_technical.items()
                                 if k not in ['y_obs_ntc', 'y_obs']}

                # Verify critical keys are present for downstream loading
                if mod.distribution != 'negbinom' and 'alpha_y_add' not in posterior_clean:
                    if hasattr(mod, 'alpha_y_prefit_add') and mod.alpha_y_prefit_add is not None:
                        posterior_clean['alpha_y_add'] = mod.alpha_y_prefit_add
                        print(f"[SAVE] {mod_name}.posterior_samples_technical: added alpha_y_add for completeness")

                if mod.distribution == 'negbinom' and 'alpha_y_mult' not in posterior_clean and 'alpha_y' not in posterior_clean:
                    if hasattr(mod, 'alpha_y_prefit_mult') and mod.alpha_y_prefit_mult is not None:
                        posterior_clean['alpha_y_mult'] = mod.alpha_y_prefit_mult
                        print(f"[SAVE] {mod_name}.posterior_samples_technical: added alpha_y_mult for completeness")

                # Add feature metadata (including full DataFrame for excluded features tracking)
                posterior_with_meta = {
                    'posterior_samples': posterior_clean,
                    'modality_name': mod_name,
                    'distribution': mod.distribution,
                    'feature_names': mod.feature_names if hasattr(mod, 'feature_names') else None,
                    'n_features': mod.dims.get('n_features', None),
                    'feature_meta': mod.feature_meta.to_dict('records') if hasattr(mod, 'feature_meta') and mod.feature_meta is not None else None
                }

                path = os.path.join(output_dir, f'posterior_samples_technical_{mod_name}.pt')
                torch.save(posterior_with_meta, path)
                saved_files[f'posterior_samples_technical_{mod_name}'] = path
                print(f"[SAVE] {mod_name}.posterior_samples_technical ({mod.dims.get('n_features')} features) → {path}")

        print(f"[SAVE] Technical fit saved to {output_dir}")
        print(f"[SAVE] Modalities saved: {modalities_to_save}")
        return saved_files


    def save_cis_fit(self, output_dir: str = None):
        """
        Save fitted cis parameters from fit_cis().

        Saves:
        - x_true: True cis gene expression (posterior samples or point estimate)
        - x_true_type: Type ('posterior' or 'point')
        - posterior_samples_cis: Full posterior samples

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.

        Returns
        -------
        dict
            Paths to saved files
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Save x_true
        if hasattr(self.model, 'x_true') and self.model.x_true is not None:
            path = os.path.join(output_dir, 'x_true.pt')
            torch.save(self.model.x_true, path)
            saved_files['x_true'] = path
            x_type = getattr(self.model, 'x_true_type', 'posterior')
            saved_files['x_true_type'] = x_type
            print(f"[SAVE] x_true ({x_type}) → {path}")

        # Save posterior samples
        if hasattr(self.model, 'posterior_samples_cis') and self.model.posterior_samples_cis is not None:
            # Remove large observation arrays
            posterior_clean = {k: v for k, v in self.model.posterior_samples_cis.items()
                             if k not in ['x_obs', 'y_obs']}

            # Add cis gene metadata (including feature_meta from cis modality)
            cis_mod = self.model.get_modality('cis')
            posterior_with_meta = {
                'posterior_samples': posterior_clean,
                'cis_gene': self.model.cis_gene,
                'modality_name': 'cis',  # Cis always uses 'cis' modality
                'feature_meta': cis_mod.feature_meta.to_dict('records') if hasattr(cis_mod, 'feature_meta') and cis_mod.feature_meta is not None else None
            }

            path = os.path.join(output_dir, 'posterior_samples_cis.pt')
            torch.save(posterior_with_meta, path)
            saved_files['posterior_samples_cis'] = path
            print(f"[SAVE] posterior_samples_cis (cis_gene: {self.model.cis_gene}) → {path}")

        print(f"[SAVE] Cis fit saved to {output_dir}")
        return saved_files


    def save_trans_fit(self, output_dir: str = None, modalities: list = None):
        """
        Save fitted trans parameters from fit_trans().

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to save. If None, saves all modalities.
            Example: ['gene', 'atac']

        Returns
        -------
        dict
            Paths to saved files

        Notes
        -----
        Saves per-modality:
        - posterior_samples_trans_{modality}.pt: Full posterior samples for each modality

        Saves model-level (automatically when primary modality is included):
        - posterior_samples_trans.pt: Model-level posterior samples
        """
        if output_dir is None:
            output_dir = os.path.join(self.model.output_dir, self.model.label)

        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Determine which modalities to save
        if modalities is None:
            modalities_to_save = list(self.model.modalities.keys())
        else:
            # Validate requested modalities
            invalid = set(modalities) - set(self.model.modalities.keys())
            if invalid:
                raise ValueError(f"Unknown modalities: {invalid}. Available: {list(self.model.modalities.keys())}")
            modalities_to_save = modalities

        # Automatically save model-level parameters if primary modality is included
        should_save_model_level = self.model.primary_modality in modalities_to_save

        # Save model-level posterior samples (when primary modality is being saved)
        if should_save_model_level:
            if hasattr(self.model, 'posterior_samples_trans') and self.model.posterior_samples_trans is not None:
                # Remove large observation arrays and add metadata
                posterior_clean = {k: v for k, v in self.model.posterior_samples_trans.items()
                                 if k not in ['y_obs', 'x_obs']}

                # Get modality for feature info
                primary_mod = self.model.get_modality(self.model.primary_modality)

                # Add modality and feature metadata (including full feature_meta DataFrame)
                posterior_with_meta = {
                    'posterior_samples': posterior_clean,
                    'modality_name': self.model.primary_modality,
                    'distribution': primary_mod.distribution,
                    'feature_names': primary_mod.feature_names if hasattr(primary_mod, 'feature_names') else None,
                    'n_features': primary_mod.dims.get('n_features', None),
                    'feature_meta': primary_mod.feature_meta.to_dict('records') if hasattr(primary_mod, 'feature_meta') and primary_mod.feature_meta is not None else None,
                    'cis_gene': self.model.cis_gene
                }

                # Include modality name in filename to prevent overwrites
                path = os.path.join(output_dir, f'posterior_samples_trans_{self.model.primary_modality}.pt')
                torch.save(posterior_with_meta, path)
                saved_files['posterior_samples_trans'] = path
                print(f"[SAVE] posterior_samples_trans (modality: {self.model.primary_modality}, {primary_mod.dims.get('n_features')} features) → {path}")

        # Save per-modality posterior samples
        for mod_name in modalities_to_save:
            mod = self.model.modalities[mod_name]
            if hasattr(mod, 'posterior_samples_trans') and mod.posterior_samples_trans is not None:
                posterior_clean = {k: v for k, v in mod.posterior_samples_trans.items()
                                 if k not in ['y_obs', 'x_obs']}

                # Add modality and feature metadata (including full feature_meta DataFrame)
                posterior_with_meta = {
                    'posterior_samples': posterior_clean,
                    'modality_name': mod_name,
                    'distribution': mod.distribution,
                    'feature_names': mod.feature_names if hasattr(mod, 'feature_names') else None,
                    'n_features': mod.dims.get('n_features', None),
                    'feature_meta': mod.feature_meta.to_dict('records') if hasattr(mod, 'feature_meta') and mod.feature_meta is not None else None,
                    'cis_gene': self.model.cis_gene
                }

                path = os.path.join(output_dir, f'posterior_samples_trans_{mod_name}.pt')
                torch.save(posterior_with_meta, path)
                saved_files[f'posterior_samples_trans_{mod_name}'] = path
                print(f"[SAVE] {mod_name}.posterior_samples_trans (distribution: {mod.distribution}, {mod.dims.get('n_features')} features) → {path}")

        print(f"[SAVE] Trans fit saved to {output_dir}")
        print(f"[SAVE] Modalities saved: {modalities_to_save}")
        return saved_files
