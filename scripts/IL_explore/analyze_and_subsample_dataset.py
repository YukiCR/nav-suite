#!/usr/bin/env python3
"""
Script to:
1. Compute data amount (state-action pairs) in datasets
2. Create a subsampled dataset matching a target data amount

The subsampling is done by selecting whole episodes (complete trajectories).
"""

import h5py
import numpy as np
import os
import argparse
from pathlib import Path


def count_state_action_pairs(hdf5_path):
    """Count total state-action pairs in an HDF5 dataset."""
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            raise ValueError(f"No 'data' group found in {hdf5_path}")

        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')],
                          key=lambda x: int(x.split('_')[1]))

        total_pairs = 0
        episode_lengths = []

        for demo_key in demo_keys:
            demo = data_group[demo_key]
            if 'actions' in demo:
                length = demo['actions'].shape[0]
                total_pairs += length
                episode_lengths.append(length)

        return {
            'total_pairs': total_pairs,
            'num_episodes': len(demo_keys),
            'episode_lengths': episode_lengths,
            'avg_length': np.mean(episode_lengths) if episode_lengths else 0,
            'min_length': min(episode_lengths) if episode_lengths else 0,
            'max_length': max(episode_lengths) if episode_lengths else 0
        }


def copy_hdf5_structure(src_path, tgt_path, selected_episodes=None):
    """Create a new HDF5 file with only selected episodes from source."""
    with h5py.File(src_path, 'r') as src_f:
        with h5py.File(tgt_path, 'w') as tgt_f:
            # Copy root attributes
            for attr_name, attr_value in src_f.attrs.items():
                tgt_f.attrs[attr_name] = attr_value

            # Copy all top-level groups
            for key in src_f.keys():
                item = src_f[key]
                if isinstance(item, h5py.Group):
                    if key == 'data' and selected_episodes is not None:
                        # Handle data group specially - only copy selected episodes
                        tgt_data = tgt_f.create_group('data')
                        for attr_name, attr_value in item.attrs.items():
                            tgt_data.attrs[attr_name] = attr_value

                        for i, episode_key in enumerate(selected_episodes):
                            if episode_key not in item:
                                continue
                            new_key = f"demo_{i}"
                            _copy_group_recursive(item[episode_key], tgt_data, new_key)
                    else:
                        # Copy other groups normally
                        _copy_group_recursive(item, tgt_f, key)
                elif isinstance(item, h5py.Dataset):
                    # Copy datasets at root level
                    tgt_f.create_dataset(key, data=item[:], dtype=item.dtype)


def _copy_group_recursive(src_group, tgt_parent, new_name):
    """Recursively copy a group and all its contents."""
    tgt_group = tgt_parent.create_group(new_name)
    # Copy attributes
    for attr_name, attr_value in src_group.attrs.items():
        tgt_group.attrs[attr_name] = attr_value

    # Copy datasets and subgroups
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            _copy_group_recursive(item, tgt_group, key)
        elif isinstance(item, h5py.Dataset):
            tgt_group.create_dataset(key, data=item[:], dtype=item.dtype)
            # Copy attributes
            for attr_name, attr_value in item.attrs.items():
                tgt_group[key].attrs[attr_name] = attr_value


def create_subsampled_dataset(source_path, target_path, target_pairs, seed=42):
    """
    Create a new dataset by selecting episodes from source to match target_pairs.
    Goes slightly over to ensure we have at least target_pairs.
    Whole episodes are selected - trajectories are never broken.
    """
    np.random.seed(seed)

    with h5py.File(source_path, 'r') as src_f:
        src_data = src_f['data']
        demo_keys = sorted([k for k in src_data.keys() if k.startswith('demo_')],
                          key=lambda x: int(x.split('_')[1]))

        # Get lengths of all episodes
        episode_info = []
        for demo_key in demo_keys:
            demo = src_data[demo_key]
            if 'actions' in demo:
                length = demo['actions'].shape[0]
                episode_info.append((demo_key, length))

        # Shuffle episodes and select until we reach target
        np.random.shuffle(episode_info)

        selected_episodes = []
        current_pairs = 0

        for demo_key, length in episode_info:
            selected_episodes.append(demo_key)
            current_pairs += length
            if current_pairs >= target_pairs:
                break

        print(f"Selected {len(selected_episodes)} episodes to get {current_pairs} state-action pairs")
        print(f"Target was {target_pairs} pairs")
        print(f"Note: Whole episodes are selected - trajectories are never broken")

        # Create new dataset by copying only selected episodes
        if os.path.exists(target_path):
            os.remove(target_path)

        # Copy HDF5 structure with only selected episodes
        copy_hdf5_structure(source_path, target_path, selected_episodes)

        return len(selected_episodes), current_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and subsample IL datasets to match target data amount'
    )
    parser.add_argument(
        '--current-dataset',
        type=str,
        default='datasets/go1_nav_dataset.hdf5',
        help='Path to current/source HDF5 dataset'
    )
    parser.add_argument(
        '--reference-dataset',
        type=str,
        default='/home/chengrui/wk/ILBL_isaac/go1_mimic/go1_mimic/logs/robomimic/ILBL-Go1-Mimic-Rough-v0/bc_rnn_go1_nav_latent_lader/dagger_finetuned_3/merged_dataset.hdf5',
        help='Path to reference HDF5 dataset (to match data amount)'
    )
    parser.add_argument(
        '--output-dataset',
        type=str,
        default='datasets/go1_nav_dataset_subsampled.hdf5',
        help='Path for output subsampled dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for episode selection'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze datasets without creating subsampled version'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Dataset Analysis and Subsampling")
    print("=" * 80)

    # 1. Compute current dataset data amount
    print("\n1. Analyzing current dataset:")
    print(f"   Path: {args.current_dataset}")

    if os.path.exists(args.current_dataset):
        stats_current = count_state_action_pairs(args.current_dataset)
        print(f"   - Total state-action pairs: {stats_current['total_pairs']:,}")
        print(f"   - Number of episodes: {stats_current['num_episodes']:,}")
        print(f"   - Average episode length: {stats_current['avg_length']:.2f}")
        print(f"   - Min episode length: {stats_current['min_length']}")
        print(f"   - Max episode length: {stats_current['max_length']}")
    else:
        print(f"   ERROR: File not found: {args.current_dataset}")
        return

    # 2. Check reference dataset
    print("\n2. Analyzing reference dataset (DAgger final):")
    print(f"   Path: {args.reference_dataset}")

    if os.path.exists(args.reference_dataset):
        stats_ref = count_state_action_pairs(args.reference_dataset)
        print(f"   - Total state-action pairs: {stats_ref['total_pairs']:,}")
        print(f"   - Number of episodes: {stats_ref['num_episodes']:,}")
        print(f"   - Average episode length: {stats_ref['avg_length']:.2f}")
        print(f"   - Min episode length: {stats_ref['min_length']}")
        print(f"   - Max episode length: {stats_ref['max_length']}")
    else:
        print(f"   WARNING: Reference file not found, using value from report: 63,054")
        stats_ref = {'total_pairs': 63054}

    if args.analyze_only:
        print("\n3. Analysis complete (--analyze-only specified, skipping subsampling)")
        return

    # 3. Create subsampled dataset
    print("\n3. Creating subsampled dataset:")
    target_pairs = stats_ref['total_pairs']
    print(f"   Target state-action pairs: {target_pairs:,}")
    print(f"   Output path: {args.output_dataset}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_dataset)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_episodes, actual_pairs = create_subsampled_dataset(
        args.current_dataset,
        args.output_dataset,
        target_pairs,
        seed=args.seed
    )

    # 4. Verify the new dataset
    print("\n4. Verifying new dataset:")
    stats_new = count_state_action_pairs(args.output_dataset)
    print(f"   - Total state-action pairs: {stats_new['total_pairs']:,}")
    print(f"   - Number of episodes: {stats_new['num_episodes']:,}")
    print(f"   - Average episode length: {stats_new['avg_length']:.2f}")

    # Calculate ratio
    ratio = stats_new['total_pairs'] / stats_current['total_pairs']
    print(f"\n5. Summary:")
    print(f"   - Original dataset: {stats_current['total_pairs']:,} pairs")
    print(f"   - Reference dataset: {stats_ref['total_pairs']:,} pairs")
    print(f"   - New dataset: {stats_new['total_pairs']:,} pairs")
    print(f"   - Subsampling ratio: {ratio:.2%} of original")
    print(f"   - Difference from target: {stats_new['total_pairs'] - target_pairs:+,} pairs")

    print(f"\n✓ Subsampled dataset saved to: {args.output_dataset}")


if __name__ == "__main__":
    main()
