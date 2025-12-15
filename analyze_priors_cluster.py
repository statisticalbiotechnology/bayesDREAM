"""
Analyze prior vs posterior distributions across different fitting conditions.

This script loads saved model states and generates diagnostic plots comparing
priors to posteriors for different conditions (CRISPRa/i, uniform priors, etc.).

Usage:
    python analyze_priors_cluster.py --base_dir ./testing/output --sj_id chr6:34236964:34237203:+
"""

import argparse
import os
import sys
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from plot_binomial_priors import plot_additive_hill_priors, compare_one_vs_two_groups


def load_model_state(state_path):
    """
    Load a saved model state.

    Parameters
    ----------
    state_path : str
        Path to saved model pickle file

    Returns
    -------
    model : bayesDREAM
        Loaded model
    """
    print(f"Loading model from {state_path}")
    with open(state_path, 'rb') as f:
        model = pickle.load(f)
    return model


def find_model_files(base_dir):
    """
    Find all saved model files in base directory.

    Parameters
    ----------
    base_dir : str
        Base directory to search

    Returns
    -------
    dict
        Dictionary mapping condition names to file paths
    """
    base_path = Path(base_dir)
    model_files = {}

    # Common patterns
    patterns = {
        'crispra_only': '*crispra*model*.pkl',
        'crispri_only': '*crispri*model*.pkl',
        'both_groups': '*both*model*.pkl',
        'uniform_priors': '*uniform*model*.pkl',
        'min_denom_3': '*min*denom*3*model*.pkl',
        'min_denom_0': '*min*denom*0*model*.pkl',
    }

    for condition, pattern in patterns.items():
        matches = list(base_path.glob(pattern))
        if matches:
            # Take the first match (or most recent if multiple)
            model_files[condition] = str(sorted(matches)[-1])
            print(f"Found {condition}: {model_files[condition]}")

    return model_files


def analyze_single_model(model, sj_id, condition_name, output_dir, modality_name='splicing_sj'):
    """
    Generate prior vs posterior plots for a single model.

    Parameters
    ----------
    model : bayesDREAM
        Fitted model
    sj_id : str
        Splice junction ID
    condition_name : str
        Name of condition (for file naming)
    output_dir : str
        Output directory for plots
    modality_name : str
        Modality name
    """
    print(f"\n{'='*80}")
    print(f"Analyzing {condition_name}")
    print(f"{'='*80}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate prior vs posterior plot
    save_path = output_dir / f'priors_{condition_name}_{sj_id.replace(":", "_")}.png'

    try:
        fig = plot_additive_hill_priors(
            model,
            sj_id=sj_id,
            modality_name=modality_name,
            save_path=str(save_path)
        )
        plt.close(fig)
        print(f"✓ Saved plot to {save_path}")
    except Exception as e:
        print(f"✗ Failed to plot {condition_name}: {e}")
        import traceback
        traceback.print_exc()


def compare_conditions(models_dict, sj_id, output_dir, modality_name='splicing_sj'):
    """
    Generate comparison plots between different conditions.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping condition names to model objects
    sj_id : str
        Splice junction ID
    output_dir : str
        Output directory
    modality_name : str
        Modality name
    """
    print(f"\n{'='*80}")
    print(f"Comparing conditions")
    print(f"{'='*80}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparison 1: CRISPRa only vs both groups
    if 'crispra_only' in models_dict and 'both_groups' in models_dict:
        print("\nComparing: CRISPRa only vs Both groups")
        save_path = output_dir / f'compare_crispra_vs_both_{sj_id.replace(":", "_")}.png'
        try:
            fig = compare_one_vs_two_groups(
                model_1grp=models_dict['crispra_only'],
                model_2grp=models_dict['both_groups'],
                sj_id=sj_id,
                modality_name=modality_name,
                save_path=str(save_path)
            )
            plt.close(fig)
            print(f"✓ Saved comparison to {save_path}")
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Comparison 2: Data-driven priors vs uniform priors
    if 'both_groups' in models_dict and 'uniform_priors' in models_dict:
        print("\nComparing: Data-driven priors vs Uniform priors")
        save_path = output_dir / f'compare_datadriven_vs_uniform_{sj_id.replace(":", "_")}.png'
        try:
            fig = compare_one_vs_two_groups(
                model_1grp=models_dict['both_groups'],
                model_2grp=models_dict['uniform_priors'],
                sj_id=sj_id,
                modality_name=modality_name,
                save_path=str(save_path)
            )
            plt.close(fig)
            print(f"✓ Saved comparison to {save_path}")
        except Exception as e:
            print(f"✗ Failed: {e}")

    # Comparison 3: min_denom 0 vs 3
    if 'min_denom_0' in models_dict and 'min_denom_3' in models_dict:
        print("\nComparing: min_denom=0 vs min_denom=3")
        save_path = output_dir / f'compare_mindenom0_vs_3_{sj_id.replace(":", "_")}.png'
        try:
            fig = compare_one_vs_two_groups(
                model_1grp=models_dict['min_denom_0'],
                model_2grp=models_dict['min_denom_3'],
                sj_id=sj_id,
                modality_name=modality_name,
                save_path=str(save_path)
            )
            plt.close(fig)
            print(f"✓ Saved comparison to {save_path}")
        except Exception as e:
            print(f"✗ Failed: {e}")


def create_summary_table(models_dict, sj_id, output_dir, modality_name='splicing_sj'):
    """
    Create a summary table of posterior means and credible intervals.

    Parameters
    ----------
    models_dict : dict
        Dictionary of models
    sj_id : str
        Splice junction ID
    output_dir : str
        Output directory
    modality_name : str
        Modality name
    """
    print(f"\n{'='*80}")
    print(f"Creating summary table")
    print(f"{'='*80}")

    output_dir = Path(output_dir)

    rows = []

    for condition, model in models_dict.items():
        try:
            modality = model.get_modality(modality_name)

            # Find SJ index
            if 'coord.intron' in modality.feature_meta.columns:
                sj_idx = modality.feature_meta[modality.feature_meta['coord.intron'] == sj_id].index[0]
            else:
                sj_idx = sj_id

            sj_position = list(modality.feature_meta.index).index(sj_idx)

            # Extract posteriors
            post = modality.posterior_samples_trans

            def get_stats(param_name):
                if param_name not in post:
                    return None, None, None
                vals = post[param_name][:, sj_position]
                if isinstance(vals, torch.Tensor):
                    vals = vals.cpu().numpy()
                return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

            # Get statistics for each parameter
            params = ['A', 'Vmax_a', 'Vmax_b', 'K_a', 'K_b', 'n_a', 'n_b']

            row = {'condition': condition}
            for param in params:
                mean, lower, upper = get_stats(param)
                if mean is not None:
                    row[f'{param}_mean'] = mean
                    row[f'{param}_lower'] = lower
                    row[f'{param}_upper'] = upper
                    row[f'{param}_CI_width'] = upper - lower
                    row[f'{param}_includes_zero'] = (lower <= 0 <= upper)

            rows.append(row)

        except Exception as e:
            print(f"✗ Failed to extract stats for {condition}: {e}")

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = output_dir / f'summary_table_{sj_id.replace(":", "_")}.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary table to {csv_path}")

    # Print table
    print("\nSummary Table:")
    print(df.to_string())

    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze prior vs posterior for multiple conditions')
    parser.add_argument('--base_dir', type=str, default='./testing/output',
                        help='Base directory containing saved models')
    parser.add_argument('--sj_id', type=str, default='chr6:34236964:34237203:+',
                        help='Splice junction ID to analyze')
    parser.add_argument('--modality_name', type=str, default='splicing_sj',
                        help='Modality name')
    parser.add_argument('--output_dir', type=str, default='./testing/output/prior_analysis',
                        help='Output directory for plots')
    parser.add_argument('--model_files', type=str, nargs='+', default=None,
                        help='Specific model files to analyze (overrides auto-detection)')
    parser.add_argument('--condition_names', type=str, nargs='+', default=None,
                        help='Condition names for each model file (must match --model_files)')

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"Prior vs Posterior Analysis")
    print(f"{'='*80}")
    print(f"Base directory: {args.base_dir}")
    print(f"Splice junction: {args.sj_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Load models
    models_dict = {}

    if args.model_files is not None:
        # Use specified model files
        if args.condition_names is None or len(args.condition_names) != len(args.model_files):
            raise ValueError("--condition_names must be provided and match --model_files length")

        for condition, file_path in zip(args.condition_names, args.model_files):
            models_dict[condition] = load_model_state(file_path)
    else:
        # Auto-detect model files
        model_files = find_model_files(args.base_dir)

        if not model_files:
            print(f"No model files found in {args.base_dir}")
            print("Please specify --model_files and --condition_names manually")
            return

        # Load each model
        for condition, file_path in model_files.items():
            models_dict[condition] = load_model_state(file_path)

    print(f"\nLoaded {len(models_dict)} models: {list(models_dict.keys())}")

    # Analyze each model individually
    for condition, model in models_dict.items():
        analyze_single_model(
            model=model,
            sj_id=args.sj_id,
            condition_name=condition,
            output_dir=args.output_dir,
            modality_name=args.modality_name
        )

    # Compare conditions
    if len(models_dict) > 1:
        compare_conditions(
            models_dict=models_dict,
            sj_id=args.sj_id,
            output_dir=args.output_dir,
            modality_name=args.modality_name
        )

    # Create summary table
    create_summary_table(
        models_dict=models_dict,
        sj_id=args.sj_id,
        output_dir=args.output_dir,
        modality_name=args.modality_name
    )

    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
