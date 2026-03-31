#!/usr/bin/env python3
"""
Script to compare action smoothness between two IL datasets.

Compares:
1. Visual: Plot actions from multiple episodes side-by-side
2. Metrics: Compute smoothness/continuity metrics
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from scipy import signal
from scipy.stats import entropy


def load_episode_actions(hdf5_path, num_episodes=5, seed=42):
    """Load actions from random episodes in the dataset."""
    np.random.seed(seed)

    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            raise ValueError(f"No 'data' group found in {hdf5_path}")

        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                          key=lambda x: int(x.split('_')[1]))

        # Randomly select episodes
        selected_keys = np.random.choice(demo_keys, size=min(num_episodes, len(demo_keys)), replace=False)

        episodes = []
        for key in selected_keys:
            demo = data_group[key]
            if 'actions' in demo:
                actions = demo['actions'][:]
                episodes.append({
                    'key': key,
                    'actions': actions,
                    'length': actions.shape[0]
                })

        return episodes


def compute_smoothness_metrics(actions):
    """
    Compute various smoothness metrics for action trajectories.

    Returns dict with:
    - velocity_variance: Variance of action velocity (1st derivative)
    - acceleration_variance: Variance of action acceleration (2nd derivative)
    - jerk_variance: Variance of action jerk (3rd derivative)
    - direction_changes: Number of direction changes per dimension
    - spectral_flatness: Measure of how "noise-like" the signal is (lower = smoother)
    - total_variation: Total variation of the signal (lower = smoother)
    - continuity_score: Overall smoothness score (higher = smoother)
    """
    metrics = {}

    if len(actions) < 4:
        return {k: np.nan for k in ['velocity_variance', 'acceleration_variance', 'jerk_variance',
                                     'direction_changes', 'spectral_flatness', 'total_variation', 'continuity_score']}

    # Compute derivatives (assuming 50Hz or unit timestep)
    velocity = np.diff(actions, axis=0)  # 1st derivative
    acceleration = np.diff(velocity, axis=0)  # 2nd derivative
    jerk = np.diff(acceleration, axis=0)  # 3rd derivative

    # 1. Velocity variance (lower = smoother)
    metrics['velocity_variance'] = np.mean(np.var(velocity, axis=0))

    # 2. Acceleration variance (lower = smoother)
    metrics['acceleration_variance'] = np.mean(np.var(acceleration, axis=0))

    # 3. Jerk variance (lower = smoother)
    metrics['jerk_variance'] = np.mean(np.var(jerk, axis=0))

    # 4. Direction changes per dimension
    direction_changes = []
    for dim in range(actions.shape[1]):
        vel_dim = velocity[:, dim]
        sign_changes = np.sum(np.diff(np.sign(vel_dim)) != 0)
        direction_changes.append(sign_changes)
    metrics['direction_changes'] = np.mean(direction_changes)

    # 5. Total variation (sum of absolute differences, lower = smoother)
    total_variation = np.sum(np.abs(velocity), axis=0)
    metrics['total_variation'] = np.mean(total_variation)

    # 6. Spectral analysis - compute high-frequency content
    spectral_flatness_vals = []
    for dim in range(actions.shape[1]):
        # Compute power spectral density
        f, psd = signal.welch(actions[:, dim], fs=50.0, nperseg=min(256, len(actions)))
        # Spectral flatness (geometric mean / arithmetic mean of PSD)
        # Lower flatness = more tonal/smooth, higher = more noise-like
        # Manual calculation since scipy.signal.spectral is deprecated
        psd_nonzero = psd[psd > 0]  # Avoid log(0)
        if len(psd_nonzero) > 0:
            geometric_mean = np.exp(np.mean(np.log(psd_nonzero)))
            arithmetic_mean = np.mean(psd_nonzero)
            sf = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1.0
        else:
            sf = 1.0
        spectral_flatness_vals.append(sf)
    metrics['spectral_flatness'] = np.mean(spectral_flatness_vals)

    # 7. Overall continuity score (higher = smoother)
    # Normalize and combine metrics (inverse relationship)
    # Using weighted combination where lower derivatives variance = smoother
    continuity_score = (
        1.0 / (1.0 + metrics['velocity_variance']) * 0.3 +
        1.0 / (1.0 + metrics['acceleration_variance']) * 0.3 +
        1.0 / (1.0 + metrics['jerk_variance']) * 0.2 +
        1.0 / (1.0 + metrics['total_variation'] / 100) * 0.2
    )
    metrics['continuity_score'] = continuity_score

    return metrics


def compute_ema(data, alpha=0.1):
    """Compute exponential moving average."""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def plot_episode_comparison(ref_episodes, sub_episodes, output_path=None, ema_alpha=0.1):
    """
    Plot actions from reference and subsampled datasets side-by-side.
    Reference on left, subsampled on right.
    Also plots EMA (exponential moving average) as dashed lines.
    """
    num_episodes = min(len(ref_episodes), len(sub_episodes))

    # Action dimension names (assuming SE2: vx, vy, yaw_rate)
    action_names = ['vx (m/s)', 'vy (m/s)', 'yaw_rate (rad/s)']

    fig, axes = plt.subplots(num_episodes, 6, figsize=(18, 3 * num_episodes))

    if num_episodes == 1:
        axes = axes.reshape(1, -1)

    for ep_idx in range(num_episodes):
        ref_ep = ref_episodes[ep_idx]
        sub_ep = sub_episodes[ep_idx]

        # Plot reference episodes (left 3 columns)
        for dim in range(3):
            ax = axes[ep_idx, dim]
            ax.plot(ref_ep['actions'][:, dim], linewidth=1.5, color='blue', alpha=0.7, label='Action')
            # Add EMA as dashed line
            ema = compute_ema(ref_ep['actions'][:, dim], alpha=ema_alpha)
            ax.plot(ema, linewidth=2.0, color='darkblue', linestyle='--', label='EMA')
            ax.set_ylabel(action_names[dim])
            if ep_idx == 0:
                ax.set_title(f'Reference: {ref_ep["key"]}\n({ref_ep["length"]} steps)')
                ax.legend(loc='upper right', fontsize=7)
            elif ep_idx == num_episodes - 1:
                ax.set_xlabel('Timestep')
            ax.grid(True, alpha=0.3)

        # Plot subsampled episodes (right 3 columns)
        for dim in range(3):
            ax = axes[ep_idx, dim + 3]
            ax.plot(sub_ep['actions'][:, dim], linewidth=1.5, color='green', alpha=0.7, label='Action')
            # Add EMA as dashed line
            ema = compute_ema(sub_ep['actions'][:, dim], alpha=ema_alpha)
            ax.plot(ema, linewidth=2.0, color='darkgreen', linestyle='--', label='EMA')
            ax.set_ylabel(action_names[dim])
            if ep_idx == 0:
                ax.set_title(f'Subsampled: {sub_ep["key"]}\n({sub_ep["length"]} steps)')
                ax.legend(loc='upper right', fontsize=7)
            elif ep_idx == num_episodes - 1:
                ax.set_xlabel('Timestep')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")

    return fig


def plot_metric_comparison(ref_metrics_list, sub_metrics_list, output_path=None):
    """Plot bar charts comparing metrics between datasets."""
    # Aggregate metrics
    ref_metrics = aggregate_metrics(ref_metrics_list)
    sub_metrics = aggregate_metrics(sub_metrics_list)

    metric_names = ['velocity_variance', 'acceleration_variance', 'jerk_variance',
                   'direction_changes', 'total_variation', 'spectral_flatness',
                   'continuity_score']

    display_names = ['Velocity Var', 'Acceleration Var', 'Jerk Var',
                    'Dir Changes', 'Total Variation', 'Spectral Flatness',
                    'Continuity Score']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, display_name) in enumerate(zip(metric_names, display_names)):
        ax = axes[idx]

        ref_val = ref_metrics[name]['mean']
        ref_std = ref_metrics[name]['std']
        sub_val = sub_metrics[name]['mean']
        sub_std = sub_metrics[name]['std']

        x = [0, 1]
        vals = [ref_val, sub_val]
        stds = [ref_std, sub_std]
        colors = ['blue', 'green']

        bars = ax.bar(x, vals, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Reference', 'Subsampled'])
        ax.set_ylabel(display_name)
        ax.set_title(display_name)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=8)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to: {output_path}")

    return fig


def aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple episodes."""
    aggregated = {}
    keys = metrics_list[0].keys()

    for key in keys:
        values = [m[key] for m in metrics_list if not np.isnan(m[key])]
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'all': values
            }
        else:
            aggregated[key] = {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'all': []
            }

    return aggregated


def print_comparison_report(ref_metrics, sub_metrics):
    """Print a detailed comparison report."""
    print("\n" + "=" * 80)
    print("DATASET SMOOTHNESS COMPARISON REPORT")
    print("=" * 80)

    print("\n{:<25} {:>20} {:>20} {:>10}".format(
        "Metric", "Reference", "Subsampled", "Better"))
    print("-" * 80)

    # Metrics where lower is better
    lower_is_better = ['velocity_variance', 'acceleration_variance', 'jerk_variance',
                       'direction_changes', 'total_variation', 'spectral_flatness']

    # Metrics where higher is better
    higher_is_better = ['continuity_score']

    for key in lower_is_better + higher_is_better:
        ref_val = ref_metrics[key]['mean']
        sub_val = sub_metrics[key]['mean']

        if key in lower_is_better:
            better = "Sub" if sub_val < ref_val else "Ref"
        else:
            better = "Sub" if sub_val > ref_val else "Ref"

        print("{:<25} {:>20.6f} {:>20.6f} {:>10}".format(
            key, ref_val, sub_val, better))

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("-" * 80)
    print("Lower velocity/acceleration/jerk variance  = smoother actions")
    print("Fewer direction changes                    = less oscillation")
    print("Lower spectral flatness                    = more tonal, less noise-like")
    print("Higher continuity score                    = overall smoother trajectory")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare action smoothness between IL datasets'
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
        '--num-episodes',
        type=int,
        default=5,
        help='Number of episodes to plot'
    )
    parser.add_argument(
        '--num-metrics-episodes',
        type=int,
        default=50,
        help='Number of episodes to use for metric computation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/smoothness_analysis',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for episode selection'
    )
    parser.add_argument(
        '--ema-alpha',
        type=float,
        default=0.3,
        help='Alpha parameter for EMA smoothing (0.0-1.0, higher = more responsive)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("DATASET SMOOTHNESS ANALYSIS")
    print("=" * 80)

    # Load episodes for plotting
    print(f"\nLoading {args.num_episodes} episodes from each dataset for visualization...")
    ref_episodes = load_episode_actions(args.reference_dataset, args.num_episodes, args.seed)
    sub_episodes = load_episode_actions(args.subsampled_dataset, args.num_episodes, args.seed)

    print(f"  Reference: {len(ref_episodes)} episodes loaded")
    print(f"  Subsampled: {len(sub_episodes)} episodes loaded")

    # Plot comparison
    print("\nGenerating episode comparison plots...")
    plot_path = os.path.join(args.output_dir, 'episode_comparison.png')
    plot_episode_comparison(ref_episodes, sub_episodes, plot_path, ema_alpha=args.ema_alpha)

    # Compute metrics for more episodes
    print(f"\nComputing smoothness metrics using {args.num_metrics_episodes} episodes...")
    ref_metrics_episodes = load_episode_actions(args.reference_dataset, args.num_metrics_episodes, args.seed)
    sub_metrics_episodes = load_episode_actions(args.subsampled_dataset, args.num_metrics_episodes, args.seed + 1)

    ref_metrics_list = [compute_smoothness_metrics(ep['actions']) for ep in ref_metrics_episodes]
    sub_metrics_list = [compute_smoothness_metrics(ep['actions']) for ep in sub_metrics_episodes]

    ref_metrics = aggregate_metrics(ref_metrics_list)
    sub_metrics = aggregate_metrics(sub_metrics_list)

    # Print report
    print_comparison_report(ref_metrics, sub_metrics)

    # Plot metrics comparison
    print("\nGenerating metrics comparison plot...")
    metrics_plot_path = os.path.join(args.output_dir, 'metrics_comparison.png')
    plot_metric_comparison(ref_metrics_list, sub_metrics_list, metrics_plot_path)

    print(f"\n✓ Analysis complete. Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
