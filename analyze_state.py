"""
Script to analyze hidden and cell states across training checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

from analysis import StateAnalyzer
from utils.logging import setup_logger


def main():
    """
    Main entry point for state analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze LSTM hidden and cell states')
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to experiment directory')
    parser.add_argument('--analysis_config', type=str, required=True,
                        help='Path to analysis config file')

    args = parser.parse_args()

    # Check experiment path exists
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        print(f"Error: Experiment path does not exist: {experiment_path}")
        sys.exit(1)

    # Load analysis config
    with open(args.analysis_config, 'r') as f:
        analysis_config = json.load(f)

    # Setup logging
    logger = setup_logger(
        name='state_analysis',
        log_file=str(experiment_path / 'logs' / 'analysis.log')
    )

    logger.info(f"Starting state analysis for: {experiment_path}")
    logger.info(f"Analysis config: {json.dumps(analysis_config, indent=2)}")

    # Create analyzer and run analysis
    analyzer = StateAnalyzer(
        experiment_path=str(experiment_path),
        analysis_config=analysis_config,
        logger=logger
    )

    analyzer.analyze_all_checkpoints()

    logger.info("State analysis complete")


if __name__ == '__main__':
    main()
