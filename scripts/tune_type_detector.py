#!/usr/bin/env python3
"""
Hyperparameter tuning script for cutlery classifier using Optuna.

This script performs hyperparameter optimization for the cutlery classifier
using Optuna. It tunes the following parameters:
- Learning rate
- Weight decay
- Dropout rate
- Optimizer choice
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.training.trainer import CutleryTrainer
from src.models.factory import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_base_config() -> Dict[str, Any]:
    """Load base configuration from config file."""
    config_path = project_root / "config" / "train_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Clone base config to avoid modifying the original
    config = dict(base_config)

    # Define hyperparameters to tune
    config["training"] = config.get("training", {})
    config["training"].update(
        {
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "num_epochs": 10,  # Reduced epochs for faster tuning
        }
    )

    if config["training"]["optimizer"] == "sgd":
        config["training"]["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)

    # Update model config
    config["model"] = config.get("model", {})
    config["model"].update(
        {"dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5)}
    )

    # Create trainer with tuning config
    trainer = CutleryTrainer(config, model_name="type_detector_tuning")

    # Create model and setup training
    trainer.create_model()
    trainer.setup_training()

    # Create dataloaders
    train_loader, val_loader, test_loader = trainer.create_dataloaders()

    # Train for specified epochs
    history = trainer.train(train_loader, val_loader)

    # Return best validation accuracy as the objective value
    return trainer.best_val_acc


def save_best_model(study: optuna.Study, base_config: Dict[str, Any]) -> None:
    """Save the best model configuration and create a new training run."""
    best_params = study.best_params
    best_config = dict(base_config)

    # Update config with best parameters
    best_config["training"] = best_config.get("training", {})
    best_config["training"].update(
        {
            "optimizer": best_params["optimizer"],
            "learning_rate": best_params["learning_rate"],
            "weight_decay": best_params["weight_decay"],
        }
    )

    if best_params["optimizer"] == "sgd":
        best_config["training"]["momentum"] = best_params["momentum"]

    best_config["model"] = best_config.get("model", {})
    best_config["model"]["dropout_rate"] = best_params["dropout_rate"]

    # Create trainer with best config
    trainer = CutleryTrainer(best_config, model_name="type_detector")

    # Create and train model with best parameters
    trainer.create_model()
    trainer.setup_training()
    train_loader, val_loader, test_loader = trainer.create_dataloaders()
    trainer.train(train_loader, val_loader)

    # Save best model
    best_model_path = (
        project_root / "models" / "checkpoints" / "type_detector_best_tuned.pth"
    )
    torch.save(trainer.model.state_dict(), best_model_path)
    logger.info(f"Best tuned model saved to: {best_model_path}")

    # Save best configuration
    best_config_path = project_root / "results" / "best_tuning_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info(f"Best configuration saved to: {best_config_path}")


def save_optuna_plots(study: optuna.Study) -> None:
    """Save Optuna visualization plots."""
    plots_dir = project_root / "results" / "optuna_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot optimization history
    history_fig = plot_optimization_history(study)
    history_path = plots_dir / "optimization_history.png"
    history_fig.write_image(str(history_path))
    logger.info(f"Saved optimization history plot to: {history_path}")

    # Plot parameter importances
    importance_fig = plot_param_importances(study)
    importance_path = plots_dir / "param_importances.png"
    importance_fig.write_image(str(importance_path))
    logger.info(f"Saved parameter importance plot to: {importance_path}")

    # Plot parallel coordinates (optional)
    try:
        parallel_fig = plot_parallel_coordinate(study)
        parallel_path = plots_dir / "parallel_coordinates.png"
        parallel_fig.write_image(str(parallel_path))
        logger.info(f"Saved parallel coordinates plot to: {parallel_path}")
    except Exception as e:
        logger.warning(f"Could not generate parallel coordinates plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune cutlery classifier hyperparameters"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    args = parser.parse_args()

    # Load base configuration
    base_config = load_base_config()

    # Create study
    db_path = project_root / "results" / "optuna_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="cutlery_classifier_optimization",
        direction="maximize",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    # Run optimization
    logger.info(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)

    # Log best results
    logger.info("Optimization completed!")
    logger.info(
        f"Best trial value (validation accuracy): {study.best_trial.value:.2f}%"
    )
    logger.info("Best hyperparameters:")
    for param, value in study.best_params.items():
        logger.info(f"  {param}: {value}")

    # Save best model and config
    save_best_model(study, base_config)

    # Generate and save plots
    save_optuna_plots(study)
    logger.info("Done!")


if __name__ == "__main__":
    main()
