"""
Batch script to process PCA models for all experiments or check status.
"""

import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple


def parse_experiment_name(folder_name: str) -> Tuple[float, float]:
    """Parse phase and jitter noise levels from experiment folder name."""
    match = re.search(r'p([\d.]+)_j([\d.]+)', folder_name)
    if match:
        phase = float(match.group(1))
        jitter = float(match.group(2))
        return phase, jitter
    return None, None


def check_pca_status(base_dir: Path) -> Dict:
    """
    Check which experiments have PCA models computed.

    Args:
        base_dir: Base directory containing experiments

    Returns:
        Status dictionary with experiment info
    """
    status = {
        'total': 0,
        'completed': 0,
        'missing': 0,
        'experiments': []
    }

    # Find all experiment directories
    exp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and '_p' in d.name and '_j' in d.name])
    status['total'] = len(exp_dirs)

    for exp_dir in exp_dirs:
        phase, jitter = parse_experiment_name(exp_dir.name)
        pca_dir = exp_dir / 'pca_models'

        exp_info = {
            'name': exp_dir.name,
            'phase': phase,
            'jitter': jitter,
            'has_pca': pca_dir.exists(),
            'n_pca_models': 0
        }

        if pca_dir.exists():
            pca_files = list(pca_dir.glob('epoch_*_pca_h.pkl'))
            exp_info['n_pca_models'] = len(pca_files)
            if len(pca_files) > 0:
                status['completed'] += 1
            else:
                status['missing'] += 1
        else:
            status['missing'] += 1

        status['experiments'].append(exp_info)

    return status


def print_status(status: Dict) -> None:
    """Print formatted status of PCA processing."""
    print("\n" + "=" * 60)
    print("PCA Processing Status")
    print("=" * 60)
    print(f"Total experiments: {status['total']}")
    print(f"Completed: {status['completed']}")
    print(f"Missing/Incomplete: {status['missing']}")
    print("\nDetailed Status:")
    print("-" * 60)

    # Group by status
    completed = []
    missing = []

    for exp in status['experiments']:
        if exp['n_pca_models'] > 0:
            completed.append(exp)
        else:
            missing.append(exp)

    if completed:
        print("\n✓ Completed:")
        for exp in completed:
            print(f"  - {exp['name']} (p={exp['phase']}, j={exp['jitter']}): {exp['n_pca_models']} epochs")

    if missing:
        print("\n✗ Missing/Incomplete:")
        for exp in missing:
            print(f"  - {exp['name']} (p={exp['phase']}, j={exp['jitter']})")

    print("\n" + "=" * 60)


def create_batch_script(base_dir: Path, missing_only: bool = True) -> None:
    """
    Create a batch script to process all experiments.

    Args:
        base_dir: Base directory containing experiments
        missing_only: Only process experiments without PCA models
    """
    status = check_pca_status(base_dir)

    script_lines = ["#!/bin/bash", "# Batch script to process PCA models", ""]

    for exp in status['experiments']:
        if missing_only and exp['n_pca_models'] > 0:
            continue

        cmd = f"python analysis/save_pca_models.py --base_dir {base_dir} --specific_experiment {exp['name']}"
        script_lines.append(f"echo 'Processing {exp['name']}...'")
        script_lines.append(cmd)
        script_lines.append("")

    script_path = Path('process_pca_batch.sh')
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))

    print(f"Created batch script: {script_path}")
    print(f"Run with: bash {script_path}")


def main():
    parser = argparse.ArgumentParser(description='Batch process PCA models')
    parser.add_argument('--base_dir', type=str, default='experiments/gaussian_noise',
                        help='Base directory containing experiments')
    parser.add_argument('--check', action='store_true',
                        help='Check status of PCA processing')
    parser.add_argument('--create_script', action='store_true',
                        help='Create batch processing script')
    parser.add_argument('--missing_only', action='store_true',
                        help='Only process missing experiments in batch script')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return

    if args.check or args.create_script:
        status = check_pca_status(base_dir)

        if args.check:
            print_status(status)

        if args.create_script:
            create_batch_script(base_dir, args.missing_only)
    else:
        print("Please specify --check or --create_script")
        print("Use --help for more information")


if __name__ == '__main__':
    main()
