#!/usr/bin/env python3
"""
Plot Optuna hyperparameter optimization results using Matplotlib.

This script loads an existing Optuna study and generates visualization plots:
- Optimization history
- Parameter importance
- Parallel coordinate plot (optional)

The script is idempotent and can be safely run multiple times.
"""

import argparse
import logging
from pathlib import Path

import optuna
import optuna.visualization.matplotlib as optuna_matplotlib
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_study(study_db: Path) -> optuna.Study:
    """Load existing Optuna study from SQLite database."""
    if not study_db.exists():
        raise FileNotFoundError(
            f"Study database not found: {study_db}\n"
            "Please run hyperparameter optimization first using:\n"
            "python scripts/tune_type_detector.py"
        )

    study = optuna.load_study(
        study_name="cutlery_classifier_optimization", storage=f"sqlite:///{study_db}"
    )

    logger.info(f"Loaded study with {len(study.trials)} trials")
    logger.info(f"Best trial value: {study.best_trial.value:.2f}%")

    return study


def save_plot(plot_func, output_path: Path, plot_name: str) -> None:
    """Generate and save a single plot with proper error handling.

    Args:
        plot_func: Function that returns a matplotlib.axes.Axes object
        output_path: Path where to save the plot
        plot_name: Name of the plot for logging
    """
    try:
        logger.info(f"Generating {plot_name}...")
        ax = plot_func()
        fig = ax.figure
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {output_path.name}")
    except Exception as e:
        logger.warning(f"Failed to save {plot_name}: {e}")


def save_plots(study: optuna.Study, output_dir: Path) -> None:
    """Generate and save visualization plots one at a time."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot optimization history
    history_path = output_dir / "optimization_history.png"
    save_plot(
        lambda: optuna_matplotlib.plot_optimization_history(study),
        history_path,
        "optimization history plot",
    )

    # Plot parameter importance
    importance_path = output_dir / "param_importances.png"
    save_plot(
        lambda: optuna_matplotlib.plot_param_importances(study),
        importance_path,
        "parameter importance plot",
    )

    # Plot parallel coordinates (optional)
    try:
        parallel_path = output_dir / "parallel_coordinates.png"
        save_plot(
            lambda: optuna_matplotlib.plot_parallel_coordinate(study),
            parallel_path,
            "parallel coordinates plot",
        )
    except Exception as e:
        logger.warning(f"Could not generate parallel coordinates plot: {e}")


def main():
    """Main function to load study and generate plots."""
    parser = argparse.ArgumentParser(
        description="Plot Optuna optimization results using Matplotlib"
    )
    parser.add_argument(
        "--study_db",
        type=Path,
        default=Path("results/optuna_study.db"),
        help="Path to Optuna study database (default: results/optuna_study.db)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/optuna_plots"),
        help="Directory to save plots (default: results/optuna_plots)",
    )
    args = parser.parse_args()

    logger.info("Note — First time Kaleido may take 1–3 minutes to initialize.")

    try:
        # Load study and generate plots
        study = load_study(args.study_db)
        save_plots(study, args.output_dir)
        logger.info("Successfully completed plot generation!")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise


if __name__ == "__main__":
    main()
