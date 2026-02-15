#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS - TRAINING UTILITIES
==============================================================================

Training loops, early stopping, cross-validation, and experiment tracking
for both GAT SC→FC prediction and contrastive learning pipelines.

Author: SARS-CoV-2 Neuroimaging Study
Date: February 2026
==============================================================================
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

from .config import GNNMultimodalConfig, GATConfig, ContrastiveConfig
from .gat_sc_fc import GATSCFC, SCFCPredictionLoss, compute_cohort_decoupling
from .contrastive import (
    MultimodalContrastiveModel,
    NTXentLoss,
    NodeContrastiveLoss,
    augment_graph,
    compute_regional_coherence,
)
from .data_loader import BrainConnectomeDataset, MultimodalPairedDataset

logger = logging.getLogger(__name__)


# ==============================================================================
# EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric and stops training when no improvement
    is observed for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False

        improved = (
            (self.mode == "min" and value < self.best_value - self.min_delta)
            or (self.mode == "max" and value > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ==============================================================================
# GAT SC→FC TRAINING
# ==============================================================================

def train_gat_sc_fc(
    config: GNNMultimodalConfig,
    atlas: str,
    sc_metric: str = "sift2",
) -> Dict:
    """
    Train the GAT SC→FC prediction model using leave-one-out cross-validation.

    For each fold, one subject is held out as test, and the model is
    trained on the remaining subjects. This allows computing individual-level
    prediction errors (decoupling indices) for each subject.

    Parameters
    ----------
    config : GNNMultimodalConfig
        Full configuration.
    atlas : str
        Atlas to use.
    sc_metric : str
        SC edge weight metric.

    Returns
    -------
    dict with:
        'fold_results': per-fold training histories and test metrics
        'cohort_decoupling': group-level decoupling maps
        'model_state': best model weights
    """
    device = torch.device(config.gat.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training GAT SC→FC on {device} for {atlas}")

    # Load dataset
    dataset = BrainConnectomeDataset(config, atlas, sc_metric)
    n_subjects = len(dataset)

    if n_subjects < 3:
        raise ValueError(f"Need at least 3 subjects, got {n_subjects}")

    n_rois = dataset.n_rois

    # Leave-one-out cross-validation
    fold_results = []
    all_predictions = {}

    for test_idx in range(n_subjects):
        test_subject = dataset.valid_subjects[test_idx]
        train_indices = [i for i in range(n_subjects) if i != test_idx]

        logger.info(f"Fold {test_idx+1}/{n_subjects}: test={test_subject}")

        # Initialize model
        model = GATSCFC(n_rois, config.gat).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.gat.learning_rate,
            weight_decay=config.gat.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.gat.n_epochs, eta_min=1e-6
        )
        criterion = SCFCPredictionLoss(lambda_corr=0.5, lambda_sparse=0.01)
        early_stop = EarlyStopping(patience=config.gat.patience, mode="min")

        # Training loop
        history = {"train_loss": [], "train_corr": []}
        best_model_state = None

        for epoch in range(config.gat.n_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_corr = 0.0

            for idx in train_indices:
                data = dataset[idx].to(device)
                optimizer.zero_grad()

                output = model(data, return_attention=True)
                losses = criterion(
                    output["fc_pred"],
                    data.y,
                    output.get("attention_weights"),
                )

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += losses["total"].item()
                epoch_corr += losses["mean_corr"].item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_indices)
            avg_corr = epoch_corr / len(train_indices)
            history["train_loss"].append(avg_loss)
            history["train_corr"].append(avg_corr)

            if early_stop.step(avg_loss, epoch):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

            if early_stop.counter == 0:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(device)

        # Test on held-out subject
        model.eval()
        test_data = dataset[test_idx].to(device)
        with torch.no_grad():
            test_output = model(test_data, return_attention=True)
            test_losses = criterion(test_output["fc_pred"], test_data.y)

        # Per-region metrics for test subject
        fc_pred = test_output["fc_pred"].cpu().numpy()
        fc_actual = test_data.y.cpu().numpy()

        # Regional prediction correlation
        regional_corr = np.zeros(n_rois)
        for i in range(n_rois):
            mask = np.ones(n_rois, dtype=bool)
            mask[i] = False
            p, a = fc_pred[i, mask], fc_actual[i, mask]
            if np.std(p) > 1e-8 and np.std(a) > 1e-8:
                regional_corr[i] = np.corrcoef(p, a)[0, 1]

        fold_result = {
            "test_subject": test_subject,
            "test_loss": test_losses["total"].item(),
            "test_corr": test_losses["mean_corr"].item(),
            "regional_corr": regional_corr,
            "history": history,
            "best_epoch": early_stop.best_epoch,
        }
        fold_results.append(fold_result)
        all_predictions[test_subject] = {
            "fc_pred": fc_pred,
            "fc_actual": fc_actual,
            "regional_corr": regional_corr,
        }

        logger.info(
            f"  Test loss: {test_losses['total'].item():.4f}, "
            f"corr: {test_losses['mean_corr'].item():.4f}"
        )

    # Train final model on all data for attention analysis
    logger.info("Training final model on all subjects...")
    final_model = GATSCFC(n_rois, config.gat).to(device)
    final_optimizer = optim.AdamW(
        final_model.parameters(),
        lr=config.gat.learning_rate,
        weight_decay=config.gat.weight_decay,
    )

    for epoch in range(config.gat.n_epochs):
        final_model.train()
        total_loss = 0.0
        for idx in range(n_subjects):
            data = dataset[idx].to(device)
            final_optimizer.zero_grad()
            output = final_model(data, return_attention=True)
            losses = criterion(output["fc_pred"], data.y, output.get("attention_weights"))
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_optimizer.step()
            total_loss += losses["total"].item()

    # Compute cohort-level decoupling
    cohort_results = compute_cohort_decoupling(final_model, dataset, device)

    # Save results
    output_dir = config.output_dir / "gat_sc_fc" / atlas
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "cohort_decoupling.npz",
        mean_error=cohort_results["mean_error"],
        std_error=cohort_results["std_error"],
        mean_corr=cohort_results["mean_corr"],
        std_corr=cohort_results["std_corr"],
        all_errors=cohort_results["all_errors"],
        all_corrs=cohort_results["all_corrs"],
        subjects=np.array(cohort_results["subjects"]),
    )

    if "mean_attention" in cohort_results:
        np.save(output_dir / "mean_attention_matrix.npy", cohort_results["mean_attention"])

    torch.save(final_model.state_dict(), output_dir / "final_model.pt")

    logger.info(f"GAT SC→FC results saved to {output_dir}")

    return {
        "fold_results": fold_results,
        "cohort_decoupling": cohort_results,
        "all_predictions": all_predictions,
        "final_model_state": final_model.state_dict(),
        "atlas": atlas,
        "n_rois": n_rois,
    }


# ==============================================================================
# CONTRASTIVE LEARNING TRAINING
# ==============================================================================

def train_contrastive(
    config: GNNMultimodalConfig,
    atlas: str,
    sc_metric: str = "sift2",
) -> Dict:
    """
    Train the multimodal contrastive model.

    All subjects are used for training (self-supervised, no labels needed).
    The model learns aligned SC-FC representations through the NT-Xent
    contrastive objective.

    Parameters
    ----------
    config : GNNMultimodalConfig
        Full configuration.
    atlas : str
        Atlas to use.
    sc_metric : str
        SC edge weight metric.

    Returns
    -------
    dict with training results and learned representations.
    """
    device = torch.device(
        config.contrastive.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Training contrastive model on {device} for {atlas}")

    # Load paired dataset
    paired_dataset = MultimodalPairedDataset(config, atlas, sc_metric)
    n_subjects = len(paired_dataset)

    if n_subjects < 4:
        raise ValueError(f"Need at least 4 subjects for contrastive learning, got {n_subjects}")

    n_rois = paired_dataset._pairs[0][0].x.size(0) if paired_dataset._pairs else 100

    # Collect all graphs
    sc_graphs = [paired_dataset[i][0] for i in range(n_subjects)]
    fc_graphs = [paired_dataset[i][1] for i in range(n_subjects)]

    # Initialize model
    model = MultimodalContrastiveModel(n_rois, config.contrastive).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.contrastive.learning_rate,
        weight_decay=config.contrastive.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )

    graph_loss_fn = NTXentLoss(
        temperature=config.contrastive.temperature,
        symmetric=config.contrastive.symmetric_loss,
    )
    node_loss_fn = NodeContrastiveLoss(temperature=config.contrastive.temperature)
    early_stop = EarlyStopping(patience=config.contrastive.patience, mode="min")

    # Training loop
    history = {"total_loss": [], "graph_loss": [], "node_loss": []}

    for epoch in range(config.contrastive.n_epochs):
        model.train()

        # Optionally augment graphs
        if config.contrastive.use_augmentation:
            sc_aug = [
                augment_graph(g, config.contrastive.augment_edge_drop, config.contrastive.augment_feat_mask)
                for g in sc_graphs
            ]
            fc_aug = [
                augment_graph(g, config.contrastive.augment_edge_drop, config.contrastive.augment_feat_mask)
                for g in fc_graphs
            ]
        else:
            sc_aug = sc_graphs
            fc_aug = fc_graphs

        optimizer.zero_grad()

        output = model(sc_aug, fc_aug)

        # Graph-level contrastive loss
        g_loss = graph_loss_fn(output["z_sc"], output["z_fc"])

        # Node-level contrastive loss
        n_loss = node_loss_fn(output["sc_node_projs"], output["fc_node_projs"])

        # Combined loss
        total_loss = g_loss["total"] + 0.5 * n_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["total_loss"].append(total_loss.item())
        history["graph_loss"].append(g_loss["total"].item())
        history["node_loss"].append(n_loss.item())

        if epoch % 50 == 0:
            logger.info(
                f"  Epoch {epoch}: total={total_loss.item():.4f} "
                f"graph={g_loss['total'].item():.4f} node={n_loss.item():.4f}"
            )

        if early_stop.step(total_loss.item(), epoch):
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Compute regional coherence
    coherence = compute_regional_coherence(model, sc_graphs, fc_graphs, device)

    # Discover subgroups
    combined_emb = np.concatenate(
        [coherence["graph_embeddings_sc"], coherence["graph_embeddings_fc"]],
        axis=1,
    )

    from .contrastive import discover_subgroups
    subgroups = discover_subgroups(
        combined_emb,
        method=config.contrastive.cluster_method,
        n_clusters_range=config.contrastive.n_clusters_range,
    )

    # Save results
    output_dir = config.output_dir / "contrastive" / atlas
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "coherence_results.npz",
        coherence_per_subject=coherence["coherence_per_subject"],
        mean_coherence=coherence["mean_coherence"],
        std_coherence=coherence["std_coherence"],
        graph_embeddings_sc=coherence["graph_embeddings_sc"],
        graph_embeddings_fc=coherence["graph_embeddings_fc"],
        subjects=np.array(paired_dataset.valid_subjects),
    )

    np.savez(
        output_dir / "subgroup_results.npz",
        labels=subgroups["labels"],
        embeddings_2d=subgroups["embeddings_2d"],
        best_k=subgroups["best_k"],
    )

    torch.save(model.state_dict(), output_dir / "contrastive_model.pt")

    logger.info(f"Contrastive results saved to {output_dir}")

    return {
        "coherence": coherence,
        "subgroups": subgroups,
        "history": history,
        "model_state": model.state_dict(),
        "atlas": atlas,
        "n_rois": n_rois,
        "subjects": paired_dataset.valid_subjects,
    }


# ==============================================================================
# FULL PIPELINE
# ==============================================================================

def run_full_pipeline(
    config: Optional[GNNMultimodalConfig] = None,
    atlases: Optional[List[str]] = None,
) -> Dict:
    """
    Run the complete GNN multimodal analysis pipeline.

    Executes both GAT SC→FC prediction and contrastive learning
    across all specified atlases.

    Parameters
    ----------
    config : GNNMultimodalConfig, optional
        Configuration. Uses defaults if None.
    atlases : list, optional
        Override atlas list from config.

    Returns
    -------
    dict with all results organized by atlas and method.
    """
    if config is None:
        config = GNNMultimodalConfig()

    # Create output directories now (not at config instantiation)
    config.prepare_output_dirs()

    atlases = atlases or config.atlases

    # Set random seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    results = {}

    for atlas in atlases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing atlas: {atlas}")
        logger.info(f"{'='*60}")

        results[atlas] = {}

        # GAT SC→FC prediction
        try:
            logger.info("--- GAT SC→FC Prediction ---")
            gat_results = train_gat_sc_fc(config, atlas)
            results[atlas]["gat_sc_fc"] = gat_results
        except Exception as e:
            logger.error(f"GAT training failed for {atlas}: {e}")
            results[atlas]["gat_sc_fc"] = {"error": str(e)}

        # Contrastive learning
        try:
            logger.info("--- Contrastive Multimodal Learning ---")
            contrastive_results = train_contrastive(config, atlas)
            results[atlas]["contrastive"] = contrastive_results
        except Exception as e:
            logger.error(f"Contrastive training failed for {atlas}: {e}")
            results[atlas]["contrastive"] = {"error": str(e)}

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "atlases": atlases,
        "n_subjects": len(config.subjects),
        "config": {
            "gat": {
                "n_heads": config.gat.n_heads,
                "hidden_dim": config.gat.hidden_dim,
                "n_layers": config.gat.n_layers,
                "n_epochs": config.gat.n_epochs,
            },
            "contrastive": {
                "encoder_type": config.contrastive.encoder_type,
                "hidden_dim": config.contrastive.hidden_dim,
                "temperature": config.contrastive.temperature,
                "n_epochs": config.contrastive.n_epochs,
            },
        },
    }

    with open(config.output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPipeline complete. Results in {config.output_dir}")
    return results
