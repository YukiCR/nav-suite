#!/usr/bin/env python3
"""
Script to analyze action space coverage.

Calculates:
- Ratio of actions with negative linear velocity (backward movement)
- Distribution of actions across different velocity ranges
"""

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def load_all_actions(hdf5_path):
    """Load all actions from the dataset."""
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            raise ValueError(f"No 'data' group found in {hdf5_path}")

        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                          key=lambda x: int(x.split('_')[1]))

        all_actions = []
        for key in demo_keys:
            demo = data_group[key]
            if 'actions' in demo:
                actions = demo['actions'][:]
                all_actions.append(actions)

        return np.vstack(all_actions) if all_actions else np.array([])


def analyze_action_coverage(actions, dataset_name):
    """Analyze action space coverage."""
    print(f"\n{'='*60}")
    print(f"Action Space Coverage Analysis: {dataset_name}")
    print(f"{'='*60}")

    # Total actions
    total_actions = len(actions)
    print(f"Total state-action pairs: {total_actions:,}")

    # Linear velocity (vx) statistics
    vx = actions[:, 0]
    vy = actions[:, 1]
    yaw_rate = actions[:, 2]

    # Negative linear velocity ratio
    negative_vx = np.sum(vx < 0)
    negative_vx_ratio = negative_vx / total_actions

    print(f"\nLinear Velocity (vx) Analysis:")
    print(f"  Min: {vx.min():.4f}")
    print(f"  Max: {vx.max():.4f}")
    print(f"  Mean: {vx.mean():.4f}")
    print(f"  Std: {vx.std():.4f}")
    print(f"  Negative vx count: {negative_vx:,}")
    print(f"  Negative vx ratio: {negative_vx_ratio:.4f} ({negative_vx_ratio*100:.2f}%)")

    # Distribution bins
    print(f"\nVelocity Distribution:")
    bins = [(-np.inf, -0.5), (-0.5, 0), (0, 0.5), (0.5, np.inf)]
    bin_labels = ['Strong Backward (< -0.5)', 'Backward (-0.5 to 0)', 'Forward (0 to 0.5)', 'Strong Forward (> 0.5)']

    for (low, high), label in zip(bins, bin_labels):
        if low == -np.inf:
            count = np.sum(vx < high)
        elif high == np.inf:
            count = np.sum(vx >= low)
        else:
            count = np.sum((vx >= low) & (vx < high))
        ratio = count / total_actions
        print(f"  {label:<25}: {count:>8,} ({ratio*100:>6.2f}%)")

    # Other action dimensions
    print(f"\nLateral Velocity (vy):")
    print(f"  Min: {vy.min():.4f}, Max: {vy.max():.4f}")

    print(f"\nYaw Rate:")
    print(f"  Min: {yaw_rate.min():.4f}, Max: {yaw_rate.max():.4f}")

    return {
        'total_actions': total_actions,
        'negative_vx_ratio': negative_vx_ratio,
        'vx_stats': {'min': vx.min(), 'max': vx.max(), 'mean': vx.mean(), 'std': vx.std()},
        'actions': actions
    }


def plot_action_distributions(ref_actions, sub_actions, output_path=None):
    """Plot action distributions for both datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    dims = ['Linear Velocity (vx)', 'Lateral Velocity (vy)', 'Yaw Rate']
    colors = ['blue', 'green']

    for row, (actions, label) in enumerate([(ref_actions, 'Reference'), (sub_actions, 'Subsampled')]):
        for col in range(3):
            ax = axes[row, col]
            data = actions[:, col]

            ax.hist(data, bins=50, color=colors[row], alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
            ax.set_xlabel(dims[col])
            ax.set_ylabel('Count')
            ax.set_title(f'{label}: {dims[col]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved distribution plot to: {output_path}")

    return fig


def compare_datasets(ref_stats, sub_stats):
    """Print comparison between datasets."""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")

    print(f"\n{'Metric':<30} {'Reference':>15} {'Subsampled':>15}")
    print("-" * 60)

    metrics = [
        ('Total Actions', f"{ref_stats['total_actions']:,}", f"{sub_stats['total_actions']:,}"),
        ('Negative vx Ratio', f"{ref_stats['negative_vx_ratio']:.4f}", f"{sub_stats['negative_vx_ratio']:.4f}"),
        ('Negative vx %', f"{ref_stats['negative_vx_ratio']*100:.2f}%", f"{sub_stats['negative_vx_ratio']*100:.2f}%"),
        ('vx Mean', f"{ref_stats['vx_stats']['mean']:.4f}", f"{sub_stats['vx_stats']['mean']:.4f}"),
        ('vx Std', f"{ref_stats['vx_stats']['std']:.4f}", f"{sub_stats['vx_stats']['std']:.4f}"),
    ]

    for metric, ref_val, sub_val in metrics:
        print(f"{metric:<30} {ref_val:>15} {sub_val:>15}")

    # Coverage assessment
    print(f"\n{'='*60}")
    print("COVERAGE ASSESSMENT")
    print(f"{'='*60}")

    ref_neg = ref_stats['negative_vx_ratio']
    sub_neg = sub_stats['negative_vx_ratio']

    if ref_neg < 0.01:
        ref_assessment = "LIMITED - Very few backward actions"
    elif ref_neg < 0.05:
        ref_assessment = "MODERATE - Some backward coverage"
    else:
        ref_assessment = "GOOD - Adequate backward coverage"

    if sub_neg < 0.01:
        sub_assessment = "LIMITED - Very few backward actions"
    elif sub_neg < 0.05:
        sub_assessment = "MODERATE - Some backward coverage"
    else:
        sub_assessment = "GOOD - Adequate backward coverage"

    print(f"\nReference Dataset:")
    print(f"  Backward action ratio: {ref_neg*100:.2f}%")
    print(f"  Assessment: {ref_assessment}")

    print(f"\nSubsampled Dataset:")
    print(f"  Backward action ratio: {sub_neg*100:.2f}%")
    print(f"  Assessment: {sub_assessment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze action space coverage in IL datasets'
    )
    parser.add_argument(
        '--reference-dataset',
        type=str,
        default='/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_go1_nav_latent_lader/dagger_finetuned_3/merged_dataset.hdf5',
        help='Path to reference HDF5 dataset'
    )
    parser.add_argument(
        '--subsampled-dataset',
        type=str,
        default='datasets/go1_nav_dataset_subsampled.hdf5',
        help='Path to subsampled HDF5 dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/action_coverage',
        help='Directory to save plots'
    )

    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ACTION SPACE COVERAGE ANALYSIS")
    print("=" * 60)

    # Load datasets
    print("\nLoading reference dataset...")
    ref_actions = load_all_actions(args.reference_dataset)

    print("Loading subsampled dataset...")
    sub_actions = load_all_actions(args.subsampled_dataset)

    # Analyze each dataset
    ref_stats = analyze_action_coverage(ref_actions, "Reference (DAgger)")
    sub_stats = analyze_action_coverage(sub_actions, "Subsampled (Expert RL)")

    # Compare
    compare_datasets(ref_stats, sub_stats)

    # Plot distributions
    plot_path = os.path.join(args.output_dir, 'action_distributions.png')
    plot_action_distributions(ref_actions, sub_actions, plot_path)

    print(f"\n✓ Analysis complete.")
