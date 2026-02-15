#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS - VISUALIZATION
==============================================================================

Publication-quality visualization for GNN multimodal SC-FC analyses.

Figure panels aligned with high-impact journal standards (Brain,
NeuroImage, Human Brain Mapping).

Author: SARS-CoV-2 Neuroimaging Study
Date: February 2026
==============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ==============================================================================
# GAT SC→FC FIGURES
# ==============================================================================

def plot_scfc_prediction_overview(
    fold_results: List[Dict],
    cohort_decoupling: Dict,
    atlas: str,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Main figure for GAT SC→FC prediction results.

    Panel layout (2×3):
    A: Training curves (loss across folds)
    B: LOO-CV prediction accuracy per subject
    C: Regional prediction correlation map
    D: Predicted vs actual FC (example subject)
    E: Regional decoupling index (sorted bar plot)
    F: Attention matrix (learned effective connectivity)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Panel A: Training curves
    ax_a = fig.add_subplot(gs[0, 0])
    for i, fold in enumerate(fold_results):
        ax_a.plot(fold["history"]["train_loss"], alpha=0.3, color="steelblue", linewidth=0.8)
    # Plot mean
    max_len = max(len(f["history"]["train_loss"]) for f in fold_results)
    mean_loss = np.zeros(max_len)
    count = np.zeros(max_len)
    for f in fold_results:
        L = len(f["history"]["train_loss"])
        mean_loss[:L] += f["history"]["train_loss"]
        count[:L] += 1
    count[count == 0] = 1
    ax_a.plot(mean_loss / count, color="navy", linewidth=2, label="Mean")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Training Loss")
    ax_a.set_title("A. Training Convergence (LOO-CV)", fontweight="bold", loc="left")
    ax_a.legend(frameon=False)

    # Panel B: Per-subject prediction accuracy
    ax_b = fig.add_subplot(gs[0, 1])
    subjects = [f["test_subject"] for f in fold_results]
    test_corrs = [f["test_corr"] for f in fold_results]
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in test_corrs]
    bars = ax_b.bar(range(len(subjects)), test_corrs, color=colors, alpha=0.8, edgecolor="white")
    ax_b.axhline(y=np.mean(test_corrs), color="navy", linestyle="--", linewidth=1.5,
                  label=f"Mean: {np.mean(test_corrs):.3f}")
    ax_b.set_xlabel("Subject")
    ax_b.set_ylabel("Prediction Correlation (r)")
    ax_b.set_title("B. LOO-CV Prediction Accuracy", fontweight="bold", loc="left")
    ax_b.set_xticks(range(len(subjects)))
    ax_b.set_xticklabels([s.replace("sub-", "") for s in subjects], rotation=45, fontsize=7)
    ax_b.legend(frameon=False)

    # Panel C: Regional prediction correlation (brain map proxy)
    ax_c = fig.add_subplot(gs[0, 2])
    mean_corr = cohort_decoupling.get("mean_corr", np.array([]))
    if len(mean_corr) > 0:
        sorted_idx = np.argsort(mean_corr)
        colors_c = plt.cm.RdYlGn(plt.Normalize(vmin=mean_corr.min(), vmax=mean_corr.max())(mean_corr[sorted_idx]))
        ax_c.barh(range(len(mean_corr)), mean_corr[sorted_idx], color=colors_c, edgecolor="none")
        ax_c.set_xlabel("Prediction Correlation (r)")
        ax_c.set_ylabel("Brain Region (sorted)")
        ax_c.set_yticks([])
    ax_c.set_title("C. Regional SC→FC Coupling", fontweight="bold", loc="left")

    # Panel D: Example predicted vs actual FC
    ax_d = fig.add_subplot(gs[1, 0])
    if fold_results:
        # Use the first fold's test prediction
        example = fold_results[0]
        rc = example.get("regional_corr", np.array([]))
        if len(rc) > 0:
            ax_d.scatter(range(len(rc)), rc, s=10, c=rc,
                        cmap="RdYlGn", alpha=0.7, edgecolors="none")
            ax_d.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
            ax_d.set_xlabel("Brain Region")
            ax_d.set_ylabel("Prediction Correlation (r)")
            ax_d.set_title(
                f"D. Example: {example['test_subject']}",
                fontweight="bold", loc="left",
            )

    # Panel E: Decoupling index (prediction error)
    ax_e = fig.add_subplot(gs[1, 1])
    mean_error = cohort_decoupling.get("mean_error", np.array([]))
    std_error = cohort_decoupling.get("std_error", np.array([]))
    if len(mean_error) > 0:
        sorted_idx = np.argsort(mean_error)[::-1]
        n_show = min(30, len(mean_error))
        top_idx = sorted_idx[:n_show]
        ax_e.barh(
            range(n_show), mean_error[top_idx],
            xerr=std_error[top_idx],
            color="salmon", alpha=0.8, edgecolor="none",
            error_kw={"linewidth": 0.5},
        )
        if labels is not None and len(labels) >= len(mean_error):
            ylabels = [str(labels[i])[:20] for i in top_idx]
            ax_e.set_yticks(range(n_show))
            ax_e.set_yticklabels(ylabels, fontsize=6)
        else:
            ax_e.set_yticks([])
        ax_e.set_xlabel("SC-FC Decoupling (MSE)")
        ax_e.invert_yaxis()
    ax_e.set_title("E. Most Decoupled Regions", fontweight="bold", loc="left")

    # Panel F: Mean attention matrix
    ax_f = fig.add_subplot(gs[1, 2])
    mean_attn = cohort_decoupling.get("mean_attention")
    if mean_attn is not None:
        im = ax_f.imshow(mean_attn, cmap="hot", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax_f, fraction=0.046, pad=0.04, label="Attention Weight")
        ax_f.set_xlabel("Target Region")
        ax_f.set_ylabel("Source Region")
    ax_f.set_title("F. Learned Attention (Effective SC)", fontweight="bold", loc="left")

    fig.suptitle(
        f"GAT SC→FC Prediction — {atlas.replace('_', ' ').title()} Atlas",
        fontsize=14, fontweight="bold", y=1.02,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    return fig


# ==============================================================================
# CONTRASTIVE LEARNING FIGURES
# ==============================================================================

def plot_contrastive_overview(
    coherence: Dict,
    subgroups: Dict,
    history: Dict,
    atlas: str,
    subjects: List[str],
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Main figure for contrastive multimodal learning results.

    Panel layout (2×3):
    A: Training loss curves (graph + node)
    B: Regional coherence map (sorted)
    C: Subject embedding space (t-SNE) with subgroup coloring
    D: Coherence per subject (heatmap)
    E: SC vs FC embedding alignment (scatter)
    F: Subgroup silhouette analysis
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Panel A: Training loss
    ax_a = fig.add_subplot(gs[0, 0])
    epochs = range(len(history.get("total_loss", [])))
    ax_a.plot(epochs, history.get("total_loss", []), color="navy", linewidth=1.5, label="Total")
    ax_a.plot(epochs, history.get("graph_loss", []), color="steelblue", linewidth=1, alpha=0.7, label="Graph")
    ax_a.plot(epochs, history.get("node_loss", []), color="coral", linewidth=1, alpha=0.7, label="Node")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Contrastive Loss")
    ax_a.set_title("A. Training Convergence", fontweight="bold", loc="left")
    ax_a.legend(frameon=False)

    # Panel B: Regional coherence (sorted)
    ax_b = fig.add_subplot(gs[0, 1])
    mean_coh = coherence.get("mean_coherence", np.array([]))
    std_coh = coherence.get("std_coherence", np.array([]))
    if len(mean_coh) > 0:
        sorted_idx = np.argsort(mean_coh)
        colors_b = plt.cm.RdYlBu(plt.Normalize(vmin=0, vmax=1)(mean_coh[sorted_idx]))
        ax_b.barh(range(len(mean_coh)), mean_coh[sorted_idx], color=colors_b, edgecolor="none")
        ax_b.set_xlabel("SC-FC Coherence (cosine similarity)")
        ax_b.set_ylabel("Brain Region (sorted)")
        ax_b.set_yticks([])
        ax_b.axvline(x=np.mean(mean_coh), color="navy", linestyle="--", linewidth=1)
    ax_b.set_title("B. Regional Multimodal Coherence", fontweight="bold", loc="left")

    # Panel C: Subject embedding space
    ax_c = fig.add_subplot(gs[0, 2])
    emb_2d = subgroups.get("embeddings_2d", np.array([]))
    cluster_labels = subgroups.get("labels", np.array([]))
    if len(emb_2d) > 0 and emb_2d.shape[0] > 0:
        scatter = ax_c.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=cluster_labels, cmap="Set2",
            s=80, edgecolors="black", linewidths=0.5,
            alpha=0.8,
        )
        for i, subj in enumerate(subjects[:len(emb_2d)]):
            ax_c.annotate(
                subj.replace("sub-", ""),
                (emb_2d[i, 0], emb_2d[i, 1]),
                fontsize=6, ha="center", va="bottom",
            )
        ax_c.set_xlabel("t-SNE 1")
        ax_c.set_ylabel("t-SNE 2")
        best_k = subgroups.get("best_k", "?")
        ax_c.set_title(f"C. Subject Subgroups (k={best_k})", fontweight="bold", loc="left")

    # Panel D: Coherence per subject (heatmap)
    ax_d = fig.add_subplot(gs[1, 0])
    coh_per_sub = coherence.get("coherence_per_subject", np.array([]))
    if coh_per_sub.ndim == 2 and coh_per_sub.size > 0:
        im = ax_d.imshow(
            coh_per_sub, aspect="auto", cmap="RdYlBu",
            vmin=0, vmax=1, interpolation="nearest",
        )
        ax_d.set_xlabel("Brain Region")
        ax_d.set_ylabel("Subject")
        ylabels = [s.replace("sub-", "") for s in subjects[:coh_per_sub.shape[0]]]
        ax_d.set_yticks(range(len(ylabels)))
        ax_d.set_yticklabels(ylabels, fontsize=7)
        plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04, label="Coherence")
    ax_d.set_title("D. Subject × Region Coherence", fontweight="bold", loc="left")

    # Panel E: SC vs FC embedding alignment
    ax_e = fig.add_subplot(gs[1, 1])
    sc_emb = coherence.get("graph_embeddings_sc", np.array([]))
    fc_emb = coherence.get("graph_embeddings_fc", np.array([]))
    if sc_emb.size > 0 and fc_emb.size > 0:
        # Cosine similarity between each subject's SC and FC embedding
        from numpy.linalg import norm
        cosines = np.array([
            np.dot(sc_emb[i], fc_emb[i]) / (norm(sc_emb[i]) * norm(fc_emb[i]) + 1e-8)
            for i in range(len(sc_emb))
        ])
        colors_e = plt.cm.RdYlGn(plt.Normalize(vmin=0, vmax=1)(cosines))
        ax_e.bar(range(len(cosines)), cosines, color=colors_e, alpha=0.8, edgecolor="white")
        ax_e.axhline(y=np.mean(cosines), color="navy", linestyle="--", linewidth=1.5)
        ax_e.set_xlabel("Subject")
        ax_e.set_ylabel("SC-FC Embedding Alignment")
        ax_e.set_xticks(range(len(subjects[:len(cosines)])))
        ax_e.set_xticklabels([s.replace("sub-", "") for s in subjects[:len(cosines)]],
                             rotation=45, fontsize=7)
    ax_e.set_title("E. Subject-level SC↔FC Alignment", fontweight="bold", loc="left")

    # Panel F: Silhouette scores
    ax_f = fig.add_subplot(gs[1, 2])
    sil_scores = subgroups.get("silhouette_scores", {})
    if sil_scores:
        ks = sorted(sil_scores.keys())
        scores = [sil_scores[k] for k in ks]
        ax_f.bar(ks, scores, color="steelblue", alpha=0.8, edgecolor="white")
        best_k = subgroups.get("best_k", ks[0])
        best_idx = ks.index(best_k) if best_k in ks else 0
        ax_f.bar(best_k, scores[best_idx], color="coral", alpha=0.9, edgecolor="white")
        ax_f.set_xlabel("Number of Clusters (k)")
        ax_f.set_ylabel("Silhouette Score")
    ax_f.set_title("F. Subgroup Validation", fontweight="bold", loc="left")

    fig.suptitle(
        f"Contrastive SC↔FC Analysis — {atlas.replace('_', ' ').title()} Atlas",
        fontsize=14, fontweight="bold", y=1.02,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    return fig


# ==============================================================================
# CONVERGENCE FIGURE
# ==============================================================================

def plot_convergence(
    conv_results: Dict,
    atlas: str,
    labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot convergence between GAT and contrastive findings.

    Shows spatial correlation between the two methods' regional
    metrics, highlighting regions where both methods agree.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gat_z = conv_results.get("gat_z", np.array([]))
    cont_z = conv_results.get("contrastive_z", np.array([]))
    agreement = conv_results.get("agreement_map", np.array([]))

    if len(gat_z) > 0:
        # Panel A: Scatter of GAT vs Contrastive
        ax = axes[0]
        sc = ax.scatter(gat_z, cont_z, c=agreement, cmap="RdYlGn",
                       s=30, edgecolors="gray", linewidths=0.3, alpha=0.8)
        rho = conv_results["spatial_correlation"]["rho"]
        p = conv_results["spatial_correlation"]["p_value"]
        ax.set_xlabel("GAT Prediction Quality (z)")
        ax.set_ylabel("Contrastive Coherence (z)")
        ax.set_title(f"A. Method Convergence (ρ={rho:.3f}, p={p:.1e})",
                     fontweight="bold", loc="left")
        # Regression line
        z = np.polyfit(gat_z, cont_z, 1)
        x_line = np.linspace(gat_z.min(), gat_z.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "k--", linewidth=1, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Agreement")

        # Panel B: Agreement map (sorted)
        ax = axes[1]
        sorted_idx = np.argsort(agreement)
        colors = plt.cm.RdYlGn(plt.Normalize(vmin=agreement.min(), vmax=agreement.max())(agreement[sorted_idx]))
        ax.barh(range(len(agreement)), agreement[sorted_idx], color=colors, edgecolor="none")
        ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_xlabel("Agreement Score")
        ax.set_ylabel("Brain Region (sorted)")
        ax.set_yticks([])
        ax.set_title("B. Regional Agreement Map", fontweight="bold", loc="left")

        # Panel C: Top/bottom overlap
        ax = axes[2]
        overlap_data = [
            conv_results.get("top20_overlap", 0),
            conv_results.get("bottom20_overlap", 0),
        ]
        bars = ax.bar(
            ["Top 20% (coupled)", "Bottom 20% (decoupled)"],
            overlap_data, color=["#2ecc71", "#e74c3c"], alpha=0.8,
        )
        ax.set_ylabel("Fraction Overlap")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.2, color="gray", linestyle=":", linewidth=1, label="Chance")
        ax.set_title("C. Cross-Method Overlap", fontweight="bold", loc="left")
        ax.legend(frameon=False)

    fig.suptitle(
        f"GAT × Contrastive Convergence — {atlas.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


# ==============================================================================
# HIERARCHICAL GRADIENT FIGURE
# ==============================================================================

def plot_hierarchical_gradient(
    gradient_results: Dict,
    metric_name: str = "SC-FC Coupling",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the SC-FC coupling hierarchy across Yeo networks.

    Compares observed hierarchy against the expected normative
    gradient (visual/somatomotor → default/frontoparietal).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    network_means = gradient_results.get("network_means", {})
    network_stds = gradient_results.get("network_stds", {})
    hierarchy = gradient_results.get("expected_hierarchy", [])

    if not network_means:
        ax.text(0.5, 0.5, "No network data available", ha="center", va="center")
        return fig

    # Color palette for Yeo networks
    YEO_COLORS = {
        "Vis": "#781286",
        "SomMot": "#4682B4",
        "DorsAttn": "#00760E",
        "SalVentAttn": "#C43AFA",
        "Limbic": "#DCF8A4",
        "Cont": "#E69422",
        "Default": "#CD3E4E",
    }

    networks = [n for n in hierarchy if n in network_means]
    means = [network_means[n] for n in networks]
    stds = [network_stds.get(n, 0) for n in networks]
    colors = [YEO_COLORS.get(n, "gray") for n in networks]

    bars = ax.bar(
        range(len(networks)), means, yerr=stds,
        color=colors, alpha=0.85, edgecolor="white",
        error_kw={"linewidth": 1, "capsize": 4},
    )
    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels(networks, rotation=30, ha="right")
    ax.set_ylabel(metric_name)

    # Add hierarchy arrow
    ax.annotate(
        "Unimodal → Transmodal",
        xy=(0.5, -0.15), xycoords="axes fraction",
        fontsize=9, ha="center", style="italic", color="gray",
    )

    # Report hierarchy correlation
    hc = gradient_results.get("hierarchy_correlation")
    if hc:
        ax.set_title(
            f"SC-FC Hierarchy (Spearman ρ = {hc['rho']:.3f}, p = {hc['p_value']:.3f})",
            fontweight="bold",
        )
    else:
        ax.set_title("SC-FC Coupling by Network", fontweight="bold")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
