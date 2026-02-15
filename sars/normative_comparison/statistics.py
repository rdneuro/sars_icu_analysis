"""
╔══════════════════════════════════════════════════════════════════════╗
║  Normative Comparison - Statistics Module                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Análises estatísticas para comparação de grupos:                    ║
║  - Testes de permutação (não-paramétricos)                           ║
║  - Effect sizes (Cohen's d, Hedges' g)                               ║
║  - Correção para múltiplas comparações (FDR, Bonferroni)             ║
║  - Bootstrap confidence intervals                                     ║
║  - Análise de outliers e normatividade                               ║
║                                                                      ║
║  REFERÊNCIAS:                                                        ║
║  - Nichols & Holmes (2002) - Permutation tests in neuroimaging       ║
║  - Hedges (1981) - Effect size correction for small samples          ║
║  - Benjamini & Hochberg (1995) - FDR correction                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from scipy import stats
from collections import defaultdict
import warnings

from .config import ROBUST_METRICS, STATS_CONFIG


# =============================================================================
# EFFECT SIZE CALCULATIONS
# =============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calcular Cohen's d (effect size não corrigido).
    
    d = (mean1 - mean2) / pooled_std
    
    Parameters
    ----------
    group1, group2 : np.ndarray
        Arrays com valores dos dois grupos
        
    Returns
    -------
    float
        Cohen's d (positive = group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calcular Hedges' g (Cohen's d corrigido para amostras pequenas).
    
    Hedges' g aplica uma correção que reduz o viés de Cohen's d
    quando n < 50.
    
    Parameters
    ----------
    group1, group2 : np.ndarray
        Arrays com valores dos dois grupos
        
    Returns
    -------
    float
        Hedges' g
    """
    d = cohens_d(group1, group2)
    
    n1, n2 = len(group1), len(group2)
    df = n1 + n2 - 2
    
    # Correction factor (Hedges, 1981)
    # J ≈ 1 - 3/(4*df - 1)
    correction = 1 - (3 / (4 * df - 1))
    
    return d * correction


def interpret_effect_size(effect: float) -> str:
    """
    Interpretar magnitude do effect size (Cohen's conventions).
    
    Parameters
    ----------
    effect : float
        Effect size (d ou g)
        
    Returns
    -------
    str
        Interpretação ('negligible', 'small', 'medium', 'large')
    """
    abs_effect = abs(effect)
    
    if abs_effect < 0.2:
        return 'negligible'
    elif abs_effect < 0.5:
        return 'small'
    elif abs_effect < 0.8:
        return 'medium'
    else:
        return 'large'


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    stat_func: callable = hedges_g,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calcular confidence interval via bootstrap.
    
    Parameters
    ----------
    group1, group2 : np.ndarray
        Arrays com valores dos grupos
    stat_func : callable
        Função estatística (ex: hedges_g)
    n_bootstrap : int
        Número de amostras bootstrap
    ci_level : float
        Nível de confiança (0.95 = 95%)
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[float, float, float]
        (statistic, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    
    n1, n2 = len(group1), len(group2)
    observed = stat_func(group1, group2)
    
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx1 = np.random.choice(n1, size=n1, replace=True)
        idx2 = np.random.choice(n2, size=n2, replace=True)
        
        boot_stat = stat_func(group1[idx1], group2[idx2])
        bootstrap_stats.append(boot_stat)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return observed, ci_lower, ci_upper


# =============================================================================
# PERMUTATION TESTS
# =============================================================================

def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    stat_func: callable = None,
    n_permutations: int = 5000,
    alternative: str = 'two-sided',
    seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Teste de permutação não-paramétrico.
    
    Parameters
    ----------
    group1, group2 : np.ndarray
        Arrays com valores dos grupos
    stat_func : callable, optional
        Função estatística. Se None, usa diferença de médias
    n_permutations : int
        Número de permutações
    alternative : str
        'two-sided', 'greater', 'less'
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[float, float, np.ndarray]
        (observed_stat, p_value, null_distribution)
    """
    np.random.seed(seed)
    
    if stat_func is None:
        stat_func = lambda x, y: np.mean(x) - np.mean(y)
    
    observed = stat_func(group1, group2)
    
    # Combine groups
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    null_dist = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        perm_g1 = combined[:n1]
        perm_g2 = combined[n1:]
        
        null_dist[i] = stat_func(perm_g1, perm_g2)
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(null_dist) >= np.abs(observed))
    elif alternative == 'greater':
        p_value = np.mean(null_dist >= observed)
    elif alternative == 'less':
        p_value = np.mean(null_dist <= observed)
    else:
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less'")
    
    return observed, p_value, null_dist


def permutation_test_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42
) -> Tuple[float, float, np.ndarray]:
    """
    Teste de permutação usando Hedges' g como estatística.
    """
    return permutation_test(
        group1, group2,
        stat_func=hedges_g,
        n_permutations=n_permutations,
        alternative='two-sided',
        seed=seed
    )


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================

def correct_pvalues(
    pvalues: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrigir p-values para múltiplas comparações.
    
    Parameters
    ----------
    pvalues : np.ndarray
        Array de p-values
    method : str
        'bonferroni', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by' (Benjamini-Yekutieli)
    alpha : float
        Nível de significância
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (corrected_pvalues, significant_mask)
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    
    if method == 'bonferroni':
        corrected = np.minimum(pvalues * n, 1.0)
        significant = corrected < alpha
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        # Critical values
        ranks = np.arange(1, n + 1)
        critical = (ranks / n) * alpha
        
        # Find largest k where p(k) <= critical(k)
        below_critical = sorted_pvals <= critical
        
        if below_critical.any():
            max_k = np.where(below_critical)[0][-1] + 1
            threshold = sorted_pvals[max_k - 1]
        else:
            threshold = 0
        
        significant = pvalues <= threshold
        
        # Adjusted p-values
        corrected = np.zeros(n)
        corrected[sorted_idx] = np.minimum.accumulate(
            (sorted_pvals * n / ranks)[::-1]
        )[::-1]
        corrected = np.minimum(corrected, 1.0)
        
    elif method == 'fdr_by':
        # Benjamini-Yekutieli (more conservative)
        c = np.sum(1 / np.arange(1, n + 1))  # Constant for dependency
        
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        ranks = np.arange(1, n + 1)
        corrected = np.zeros(n)
        corrected[sorted_idx] = np.minimum.accumulate(
            (sorted_pvals * n * c / ranks)[::-1]
        )[::-1]
        corrected = np.minimum(corrected, 1.0)
        
        significant = corrected < alpha
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corrected, significant


# =============================================================================
# GROUP COMPARISON (MAIN FUNCTION)
# =============================================================================

def compare_groups(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics: List[str] = None,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    correction_method: str = 'fdr_bh',
    effect_size_method: str = 'hedges_g',
    bootstrap_ci_n: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Comparação estatística completa entre grupos COVID e controles.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas dos pacientes COVID
    control_df : pd.DataFrame
        Métricas dos controles
    metrics : List[str], optional
        Métricas para comparar. Se None, usa ROBUST_METRICS
    n_permutations : int
        Número de permutações
    alpha : float
        Nível de significância
    correction_method : str
        Método de correção ('bonferroni', 'fdr_bh', 'fdr_by')
    effect_size_method : str
        'cohens_d' ou 'hedges_g'
    bootstrap_ci_n : int
        Número de amostras para CI
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Resultados completos da comparação
    """
    if metrics is None:
        available = set(covid_df.columns) & set(control_df.columns)
        metrics = [m for m in ROBUST_METRICS if m in available]
    
    if not metrics:
        raise ValueError("No common metrics found for comparison")
    
    print(f"\n{'='*60}")
    print("STATISTICAL COMPARISON: COVID vs Controls")
    print('='*60)
    print(f"COVID n={len(covid_df)}, Control n={len(control_df)}")
    print(f"Metrics: {len(metrics)}")
    print(f"Permutations: {n_permutations}")
    print(f"Correction: {correction_method}")
    
    effect_func = hedges_g if effect_size_method == 'hedges_g' else cohens_d
    
    results = []
    pvalues_perm = []
    pvalues_param = []
    
    for metric in metrics:
        covid_values = covid_df[metric].dropna().values
        ctrl_values = control_df[metric].dropna().values
        
        if len(covid_values) < 5 or len(ctrl_values) < 5:
            warnings.warn(f"Skipping {metric}: insufficient data")
            continue
        
        # Descriptive statistics
        covid_mean = np.mean(covid_values)
        covid_std = np.std(covid_values, ddof=1)
        ctrl_mean = np.mean(ctrl_values)
        ctrl_std = np.std(ctrl_values, ddof=1)
        
        # Parametric test (t-test)
        t_stat, p_param = stats.ttest_ind(covid_values, ctrl_values)
        pvalues_param.append(p_param)
        
        # Permutation test
        _, p_perm, null_dist = permutation_test(
            covid_values, ctrl_values,
            n_permutations=n_permutations,
            seed=seed
        )
        pvalues_perm.append(p_perm)
        
        # Effect size
        effect = effect_func(covid_values, ctrl_values)
        effect_interp = interpret_effect_size(effect)
        
        # Bootstrap CI for effect size
        _, ci_lower, ci_upper = bootstrap_ci(
            covid_values, ctrl_values,
            stat_func=effect_func,
            n_bootstrap=bootstrap_ci_n,
            seed=seed
        )
        
        # Additional tests
        # Mann-Whitney U (non-parametric)
        u_stat, p_mann = stats.mannwhitneyu(
            covid_values, ctrl_values, alternative='two-sided'
        )
        
        # Normality tests
        _, p_norm_covid = stats.shapiro(covid_values) if len(covid_values) <= 50 else (0, 0)
        _, p_norm_ctrl = stats.shapiro(ctrl_values) if len(ctrl_values) <= 50 else (0, 0)
        
        results.append({
            'metric': metric,
            # Descriptive
            'covid_mean': covid_mean,
            'covid_std': covid_std,
            'covid_n': len(covid_values),
            'control_mean': ctrl_mean,
            'control_std': ctrl_std,
            'control_n': len(ctrl_values),
            'mean_diff': covid_mean - ctrl_mean,
            'percent_diff': 100 * (covid_mean - ctrl_mean) / ctrl_mean if ctrl_mean != 0 else 0,
            # Effect size
            f'{effect_size_method}': effect,
            'effect_ci_lower': ci_lower,
            'effect_ci_upper': ci_upper,
            'effect_interpretation': effect_interp,
            # P-values
            'p_ttest': p_param,
            'p_permutation': p_perm,
            'p_mannwhitney': p_mann,
            't_statistic': t_stat,
            'u_statistic': u_stat,
            # Normality
            'p_shapiro_covid': p_norm_covid,
            'p_shapiro_control': p_norm_ctrl,
        })
    
    df_results = pd.DataFrame(results)
    
    # Apply multiple comparison correction
    if len(pvalues_perm) > 1:
        # Correct permutation p-values
        corrected_perm, sig_perm = correct_pvalues(
            np.array(pvalues_perm), method=correction_method, alpha=alpha
        )
        df_results['p_perm_corrected'] = corrected_perm
        df_results['significant_perm'] = sig_perm
        
        # Correct parametric p-values
        corrected_param, sig_param = correct_pvalues(
            np.array(pvalues_param), method=correction_method, alpha=alpha
        )
        df_results['p_ttest_corrected'] = corrected_param
        df_results['significant_ttest'] = sig_param
    else:
        df_results['p_perm_corrected'] = df_results['p_permutation']
        df_results['significant_perm'] = df_results['p_permutation'] < alpha
        df_results['p_ttest_corrected'] = df_results['p_ttest']
        df_results['significant_ttest'] = df_results['p_ttest'] < alpha
    
    # Sort by effect size
    df_results = df_results.sort_values(
        f'{effect_size_method}', key=abs, ascending=False
    ).reset_index(drop=True)
    
    # Summary
    n_sig = df_results['significant_perm'].sum()
    print(f"\n✓ Significant findings (permutation, {correction_method}): {n_sig}/{len(metrics)}")
    
    if n_sig > 0:
        print("\nSignificant metrics:")
        sig_metrics = df_results[df_results['significant_perm']]
        for _, row in sig_metrics.iterrows():
            direction = '↑' if row['mean_diff'] > 0 else '↓'
            print(f"  {row['metric']}: {direction} "
                  f"g={row[effect_size_method]:.3f} [{row['effect_ci_lower']:.3f}, {row['effect_ci_upper']:.3f}], "
                  f"p={row['p_perm_corrected']:.4f}")
    
    return df_results


# =============================================================================
# NORMATIVE SCORING
# =============================================================================

def compute_z_scores(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Calcular z-scores normativos para cada paciente COVID.
    
    Z-score indica quantos desvios-padrão o valor está da média de controles.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas COVID
    control_df : pd.DataFrame
        Métricas controles (referência normativa)
    metrics : List[str], optional
        Métricas para calcular z-scores
        
    Returns
    -------
    pd.DataFrame
        DataFrame com z-scores para cada paciente e métrica
    """
    if metrics is None:
        available = set(covid_df.columns) & set(control_df.columns)
        metrics = [m for m in ROBUST_METRICS if m in available]
    
    result = covid_df[['subject_id']].copy() if 'subject_id' in covid_df.columns else pd.DataFrame()
    
    for metric in metrics:
        ctrl_mean = control_df[metric].mean()
        ctrl_std = control_df[metric].std()
        
        if ctrl_std == 0:
            z_scores = np.zeros(len(covid_df))
        else:
            z_scores = (covid_df[metric].values - ctrl_mean) / ctrl_std
        
        result[f'{metric}_zscore'] = z_scores
    
    # Compute composite z-score (mean of all z-scores)
    z_cols = [c for c in result.columns if c.endswith('_zscore')]
    if z_cols:
        result['composite_zscore'] = result[z_cols].mean(axis=1)
    
    return result


def identify_extreme_subjects(
    z_scores_df: pd.DataFrame,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Identificar pacientes com valores extremos (|z| > threshold).
    
    Parameters
    ----------
    z_scores_df : pd.DataFrame
        DataFrame com z-scores
    threshold : float
        Threshold para considerar extremo
        
    Returns
    -------
    pd.DataFrame
        Pacientes com valores extremos
    """
    z_cols = [c for c in z_scores_df.columns if c.endswith('_zscore')]
    
    results = []
    
    for idx, row in z_scores_df.iterrows():
        subject = row.get('subject_id', f'subj_{idx}')
        
        extreme_metrics = []
        for col in z_cols:
            z = row[col]
            if abs(z) > threshold:
                metric = col.replace('_zscore', '')
                direction = 'high' if z > 0 else 'low'
                extreme_metrics.append({
                    'metric': metric,
                    'z_score': z,
                    'direction': direction
                })
        
        if extreme_metrics:
            results.append({
                'subject_id': subject,
                'n_extreme': len(extreme_metrics),
                'extreme_metrics': extreme_metrics,
                'composite_zscore': row.get('composite_zscore', np.nan)
            })
    
    return pd.DataFrame(results)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_summary_table(
    stats_df: pd.DataFrame,
    format_type: str = 'publication'
) -> pd.DataFrame:
    """
    Gerar tabela resumo formatada para publicação.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Resultado de compare_groups()
    format_type : str
        'publication' ou 'detailed'
        
    Returns
    -------
    pd.DataFrame
        Tabela formatada
    """
    if format_type == 'publication':
        # Formato para publicação científica
        summary = stats_df[['metric', 'covid_mean', 'covid_std', 
                           'control_mean', 'control_std']].copy()
        
        # Formatar como mean ± std
        summary['COVID'] = summary.apply(
            lambda r: f"{r['covid_mean']:.3f} ± {r['covid_std']:.3f}", axis=1
        )
        summary['Control'] = summary.apply(
            lambda r: f"{r['control_mean']:.3f} ± {r['control_std']:.3f}", axis=1
        )
        
        # Effect size com CI
        effect_col = 'hedges_g' if 'hedges_g' in stats_df.columns else 'cohens_d'
        summary['Effect size (95% CI)'] = stats_df.apply(
            lambda r: f"{r[effect_col]:.2f} [{r['effect_ci_lower']:.2f}, {r['effect_ci_upper']:.2f}]",
            axis=1
        )
        
        # P-value corrigido
        summary['p-value'] = stats_df['p_perm_corrected'].apply(
            lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'
        )
        
        # Significância
        summary['Sig.'] = stats_df['significant_perm'].apply(
            lambda x: '*' if x else ''
        )
        
        summary = summary[['metric', 'COVID', 'Control', 'Effect size (95% CI)', 
                          'p-value', 'Sig.']]
        summary.columns = ['Metric', 'COVID (mean ± SD)', 'Control (mean ± SD)',
                          "Hedges' g (95% CI)", 'p-value (FDR)', 'Sig.']
        
    else:  # detailed
        summary = stats_df.copy()
    
    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("Statistics Module - Group Comparisons")
    print(f"\nConfiguration:")
    for k, v in STATS_CONFIG.items():
        print(f"  {k}: {v}")
