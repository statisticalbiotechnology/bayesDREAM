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

    def save_technical_fit(self, output_dir: str = None, modalities: list = None,
                          save_model_level: bool = True):
        """
        Save fitted technical parameters from fit_technical().

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to save. If None, saves all modalities.
            Example: ['gene', 'atac']
        save_model_level : bool, optional
            If True, saves model-level alpha_x_prefit, alpha_y_prefit, and
            posterior_samples_technical (default: True).

        Returns
        -------
        dict
            Paths to saved files

        Notes
        -----
        Saves per-modality:
        - alpha_y_prefit_{modality}.pt: Overdispersion parameters for each modality

        Saves model-level (if save_model_level=True):
        - alpha_x_prefit.pt: Cis gene overdispersion (if set)
        - alpha_y_prefit.pt: Trans gene overdispersion
        - posterior_samples_technical.pt: Full posterior samples
        """
        if output_dir is None:
            output_dir = self.model.output_dir

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

        # Save model-level parameters (backward compatibility)
        if save_model_level:
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

            if hasattr(self.model, 'posterior_samples_technical') and self.model.posterior_samples_technical is not None:
                # Remove large observation arrays before saving
                posterior_clean = {k: v for k, v in self.model.posterior_samples_technical.items()
                                 if k not in ['y_obs_ntc', 'y_obs']}
                path = os.path.join(output_dir, 'posterior_samples_technical.pt')
                torch.save(posterior_clean, path)
                saved_files['posterior_samples_technical'] = path
                print(f"[SAVE] posterior_samples_technical → {path}")

        # Save per-modality alpha_y_prefit
        for mod_name in modalities_to_save:
            mod = self.model.modalities[mod_name]
            if hasattr(mod, 'alpha_y_prefit') and mod.alpha_y_prefit is not None:
                path = os.path.join(output_dir, f'alpha_y_prefit_{mod_name}.pt')
                torch.save(mod.alpha_y_prefit, path)
                saved_files[f'alpha_y_prefit_{mod_name}'] = path
                print(f"[SAVE] {mod_name}.alpha_y_prefit → {path}")

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
            output_dir = self.model.output_dir

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
            path = os.path.join(output_dir, 'posterior_samples_cis.pt')
            torch.save(posterior_clean, path)
            saved_files['posterior_samples_cis'] = path
            print(f"[SAVE] posterior_samples_cis → {path}")

        print(f"[SAVE] Cis fit saved to {output_dir}")
        return saved_files


    def save_trans_fit(self, output_dir: str = None, modalities: list = None,
                      save_model_level: bool = True):
        """
        Save fitted trans parameters from fit_trans().

        Parameters
        ----------
        output_dir : str, optional
            Directory to save to. If None, uses self.model.output_dir.
        modalities : list of str, optional
            List of modality names to save. If None, saves all modalities.
            Example: ['gene', 'atac']
        save_model_level : bool, optional
            If True, saves model-level posterior_samples_trans (default: True).

        Returns
        -------
        dict
            Paths to saved files

        Notes
        -----
        Saves per-modality:
        - posterior_samples_trans_{modality}.pt: Full posterior samples for each modality

        Saves model-level (if save_model_level=True):
        - posterior_samples_trans.pt: Model-level posterior samples
        """
        if output_dir is None:
            output_dir = self.model.output_dir

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

        # Save model-level posterior samples
        if save_model_level:
            if hasattr(self.model, 'posterior_samples_trans') and self.model.posterior_samples_trans is not None:
                # Remove large observation arrays
                posterior_clean = {k: v for k, v in self.model.posterior_samples_trans.items()
                                 if k not in ['y_obs', 'x_obs']}
                path = os.path.join(output_dir, 'posterior_samples_trans.pt')
                torch.save(posterior_clean, path)
                saved_files['posterior_samples_trans'] = path
                print(f"[SAVE] posterior_samples_trans → {path}")

        # Save per-modality posterior samples
        for mod_name in modalities_to_save:
            mod = self.model.modalities[mod_name]
            if hasattr(mod, 'posterior_samples_trans') and mod.posterior_samples_trans is not None:
                posterior_clean = {k: v for k, v in mod.posterior_samples_trans.items()
                                 if k not in ['y_obs', 'x_obs']}
                path = os.path.join(output_dir, f'posterior_samples_trans_{mod_name}.pt')
                torch.save(posterior_clean, path)
                saved_files[f'posterior_samples_trans_{mod_name}'] = path
                print(f"[SAVE] {mod_name}.posterior_samples_trans → {path}")

        print(f"[SAVE] Trans fit saved to {output_dir}")
        print(f"[SAVE] Modalities saved: {modalities_to_save}")
        return saved_files
