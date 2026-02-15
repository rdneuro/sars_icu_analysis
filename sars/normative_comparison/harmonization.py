"""
╔══════════════════════════════════════════════════════════════════════╗
║  Normative Comparison - Harmonization Module                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ComBat harmonization no nível de MÉTRICAS DERIVADAS.                ║
║                                                                      ║
║  JUSTIFICATIVA METODOLÓGICA:                                         ║
║  Harmonizar métricas topológicas (não matrizes brutas) é mais        ║
║  defensável porque:                                                  ║
║  1. Métricas como modularity, efficiency, clustering dependem da     ║
║     topologia RELATIVA da rede, não dos valores absolutos            ║
║  2. Menos vulnerável a diferenças de protocolo de aquisição          ║
║  3. Mais publicável e menos vulnerável a críticas de revisor         ║
║  4. ComBat preserva efeitos biológicos enquanto remove batch effects ║
║                                                                      ║
║  REFERÊNCIAS:                                                        ║
║  - Johnson et al. (2007) - Original ComBat                           ║
║  - Fortin et al. (2017) - neuroCombat para neuroimagem               ║
║  - Fortin et al. (2018) - Harmonization of cortical thickness        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import warnings
from scipy import stats

# Tentar importar neuroCombat
try:
    from neuroCombat import neuroCombat
    HAS_NEUROCOMBAT = True
except ImportError:
    HAS_NEUROCOMBAT = False
    warnings.warn(
        "neuroCombat not installed. Install with: pip install neuroCombat. "
        "Falling back to basic z-score normalization."
    )

from .config import ROBUST_METRICS, COMBAT_CONFIG


# =============================================================================
# COMBAT HARMONIZATION
# =============================================================================

def combat_harmonize_metrics(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics_to_harmonize: List[str] = None,
    covariates: List[str] = None,
    parametric: bool = True,
    eb: bool = True,
    ref_batch: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplicar ComBat harmonization no nível de métricas.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas dos pacientes COVID (deve ter coluna 'site')
    control_df : pd.DataFrame
        Métricas dos controles (deve ter coluna 'site')
    metrics_to_harmonize : List[str], optional
        Lista de métricas para harmonizar. Se None, usa ROBUST_METRICS
    covariates : List[str], optional
        Covariáveis a preservar (ex: ['age', 'sex'])
    parametric : bool
        Use parametric adjustment
    eb : bool
        Use empirical Bayes estimation
    ref_batch : str, optional
        Batch de referência (site). Se None, todos são ajustados
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (covid_harmonized, control_harmonized)
    """
    if metrics_to_harmonize is None:
        # Usar métricas robustas que existem em ambos os dataframes
        available = set(covid_df.columns) & set(control_df.columns)
        metrics_to_harmonize = [m for m in ROBUST_METRICS if m in available]
    
    if not metrics_to_harmonize:
        raise ValueError("No common metrics found to harmonize")
    
    print(f"\n{'='*60}")
    print("ComBat HARMONIZATION - Metric Level")
    print('='*60)
    print(f"Metrics to harmonize ({len(metrics_to_harmonize)}): {', '.join(metrics_to_harmonize)}")
    
    # Combinar dataframes
    covid_copy = covid_df.copy()
    control_copy = control_df.copy()
    
    # Garantir coluna site
    if 'site' not in covid_copy.columns:
        covid_copy['site'] = 'covid_site'
    if 'site' not in control_copy.columns:
        control_copy['site'] = 'control_site'
    
    # Combinar para harmonização
    combined = pd.concat([covid_copy, control_copy], ignore_index=True)
    
    print(f"COVID subjects: {len(covid_copy)}")
    print(f"Control subjects: {len(control_copy)}")
    print(f"Sites: {combined['site'].unique()}")
    
    if HAS_NEUROCOMBAT:
        # Usar neuroCombat
        harmonized = _combat_with_neurocombat(
            combined, 
            metrics_to_harmonize,
            covariates,
            parametric,
            eb,
            ref_batch
        )
    else:
        # Fallback: z-score normalization por site
        warnings.warn("Using z-score normalization as ComBat fallback")
        harmonized = _zscore_harmonization(combined, metrics_to_harmonize)
    
    # Separar de volta
    n_covid = len(covid_copy)
    covid_harm = harmonized.iloc[:n_covid].copy()
    control_harm = harmonized.iloc[n_covid:].copy()
    
    # Validar
    _validate_harmonization(covid_df, covid_harm, control_df, control_harm, metrics_to_harmonize)
    
    print(f"\n✓ Harmonization complete")
    
    return covid_harm, control_harm


def _combat_with_neurocombat(
    df: pd.DataFrame,
    metrics: List[str],
    covariates: List[str] = None,
    parametric: bool = True,
    eb: bool = True,
    ref_batch: str = None
) -> pd.DataFrame:
    """
    ComBat usando neuroCombat package.
    """
    # Preparar dados para neuroCombat
    # neuroCombat espera: data (features x samples), batch (array)
    
    data = df[metrics].values.T  # (n_metrics, n_subjects)
    batch = df['site'].values
    
    # Covariáveis
    if covariates:
        available_covs = [c for c in covariates if c in df.columns]
        if available_covs:
            # neuroCombat espera design matrix
            covar_df = df[available_covs].copy()
            # Encode categorical
            for col in covar_df.columns:
                if covar_df[col].dtype == 'object':
                    covar_df[col] = pd.factorize(covar_df[col])[0]
            covars = covar_df.values
        else:
            covars = None
    else:
        covars = None
    
    print(f"  Running neuroCombat (parametric={parametric}, eb={eb})...")
    
    # Run ComBat
    data_combat = neuroCombat(
        dat=data,
        batch=batch,
        mod=covars,
        parametric=parametric,
        eb=eb,
        ref_batch=ref_batch
    )
    
    # Reconstruir dataframe
    result = df.copy()
    result[metrics] = data_combat['data'].T
    
    return result


def _zscore_harmonization(
    df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Fallback: Z-score normalization por site.
    
    Menos sofisticado que ComBat, mas útil quando neuroCombat não disponível.
    """
    result = df.copy()
    
    for metric in metrics:
        # Z-score por site
        result[metric] = result.groupby('site')[metric].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        
        # Re-escalar para range original (preservar interpretabilidade)
        global_mean = df[metric].mean()
        global_std = df[metric].std()
        result[metric] = result[metric] * global_std + global_mean
    
    return result


def _validate_harmonization(
    covid_orig: pd.DataFrame,
    covid_harm: pd.DataFrame,
    control_orig: pd.DataFrame,
    control_harm: pd.DataFrame,
    metrics: List[str]
) -> None:
    """
    Validar que a harmonização preservou estrutura dos dados.
    """
    print("\n  Harmonization Validation:")
    
    for metric in metrics[:3]:  # Primeiras 3 métricas
        if metric not in covid_orig.columns:
            continue
            
        # Antes
        covid_orig_mean = covid_orig[metric].mean()
        ctrl_orig_mean = control_orig[metric].mean()
        diff_orig = covid_orig_mean - ctrl_orig_mean
        
        # Depois
        covid_harm_mean = covid_harm[metric].mean()
        ctrl_harm_mean = control_harm[metric].mean()
        diff_harm = covid_harm_mean - ctrl_harm_mean
        
        print(f"    {metric}:")
        print(f"      Original: COVID={covid_orig_mean:.3f}, Ctrl={ctrl_orig_mean:.3f}, diff={diff_orig:.3f}")
        print(f"      Harmonized: COVID={covid_harm_mean:.3f}, Ctrl={ctrl_harm_mean:.3f}, diff={diff_harm:.3f}")


# =============================================================================
# ALTERNATIVE HARMONIZATION METHODS
# =============================================================================

def residualize_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    covariates: List[str] = ['age', 'sex'],
    site_as_covariate: bool = True
) -> pd.DataFrame:
    """
    Harmonização por residualização (regressão linear).
    
    Remove efeitos de site e covariáveis via regressão.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com métricas
    metrics : List[str]
        Métricas para residualizar
    covariates : List[str]
        Covariáveis a remover
    site_as_covariate : bool
        Se True, inclui site como covariável
        
    Returns
    -------
    pd.DataFrame
        DataFrame com métricas residualizadas
    """
    from sklearn.linear_model import LinearRegression
    
    result = df.copy()
    
    # Preparar covariáveis
    cov_cols = []
    
    if site_as_covariate and 'site' in df.columns:
        site_dummies = pd.get_dummies(df['site'], prefix='site', drop_first=True)
        cov_cols.extend(site_dummies.columns)
        result = pd.concat([result, site_dummies], axis=1)
    
    for cov in covariates:
        if cov in df.columns:
            if df[cov].dtype == 'object':
                dummies = pd.get_dummies(df[cov], prefix=cov, drop_first=True)
                cov_cols.extend(dummies.columns)
                result = pd.concat([result, dummies], axis=1)
            else:
                cov_cols.append(cov)
    
    if not cov_cols:
        warnings.warn("No covariates available for residualization")
        return result
    
    X = result[cov_cols].values
    
    for metric in metrics:
        if metric not in result.columns:
            continue
        
        y = result[metric].values
        mask = ~np.isnan(y)
        
        if mask.sum() < len(cov_cols) + 2:
            continue
        
        model = LinearRegression()
        model.fit(X[mask], y[mask])
        
        # Residuais + média global
        predicted = model.predict(X)
        residuals = y - predicted
        result[metric] = residuals + np.nanmean(y)
    
    # Remover colunas dummy
    result = result.drop(columns=[c for c in cov_cols if c in result.columns], errors='ignore')
    
    return result


def rank_normalize_metrics(
    df: pd.DataFrame,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Normalização por ranks (inverse normal transform).
    
    Transforma dados para distribuição normal padrão via ranks.
    Robusto a outliers e diferenças de escala entre sites.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com métricas
    metrics : List[str]
        Métricas para normalizar
        
    Returns
    -------
    pd.DataFrame
        DataFrame com métricas normalizadas por rank
    """
    result = df.copy()
    
    for metric in metrics:
        if metric not in result.columns:
            continue
        
        values = result[metric].values
        n = len(values)
        
        # Rank
        ranks = stats.rankdata(values, method='average')
        
        # Inverse normal transform
        # Blom transformation: (r - 3/8) / (n + 1/4)
        normalized = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
        
        result[metric] = normalized
    
    return result


def percentile_normalize(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalização percentílica usando distribuição de controles.
    
    Cada valor COVID é expresso como percentil da distribuição de controles.
    Útil para z-scores normativos.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas COVID
    control_df : pd.DataFrame
        Métricas controles (referência)
    metrics : List[str]
        Métricas para normalizar
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (covid_percentiles, control_percentiles)
    """
    covid_result = covid_df.copy()
    control_result = control_df.copy()
    
    for metric in metrics:
        if metric not in covid_df.columns or metric not in control_df.columns:
            continue
        
        control_values = control_df[metric].dropna().values
        
        # COVID percentiles
        covid_values = covid_df[metric].values
        covid_percentiles = np.array([
            stats.percentileofscore(control_values, v, kind='weak')
            for v in covid_values
        ])
        covid_result[f'{metric}_percentile'] = covid_percentiles
        
        # Control percentiles (within-group)
        control_values_all = control_df[metric].values
        control_percentiles = np.array([
            stats.percentileofscore(control_values, v, kind='weak')
            for v in control_values_all
        ])
        control_result[f'{metric}_percentile'] = control_percentiles
    
    return covid_result, control_result


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnose_batch_effects(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Diagnosticar presença de batch effects antes da harmonização.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas COVID
    control_df : pd.DataFrame
        Métricas controles
    metrics : List[str], optional
        Métricas para testar
        
    Returns
    -------
    pd.DataFrame
        Resultados dos testes de batch effect
    """
    if metrics is None:
        metrics = [m for m in ROBUST_METRICS 
                  if m in covid_df.columns and m in control_df.columns]
    
    results = []
    
    for metric in metrics:
        covid_values = covid_df[metric].dropna().values
        ctrl_values = control_df[metric].dropna().values
        
        # Levene test (homogeneity of variance)
        stat_levene, p_levene = stats.levene(covid_values, ctrl_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(covid_values)-1)*np.var(covid_values) + 
             (len(ctrl_values)-1)*np.var(ctrl_values)) / 
            (len(covid_values) + len(ctrl_values) - 2)
        )
        cohens_d = (np.mean(covid_values) - np.mean(ctrl_values)) / pooled_std
        
        results.append({
            'metric': metric,
            'covid_mean': np.mean(covid_values),
            'covid_std': np.std(covid_values),
            'control_mean': np.mean(ctrl_values),
            'control_std': np.std(ctrl_values),
            'levene_stat': stat_levene,
            'levene_p': p_levene,
            'variance_ratio': np.var(covid_values) / np.var(ctrl_values),
            'cohens_d': cohens_d
        })
    
    return pd.DataFrame(results)


def compare_before_after(
    covid_orig: pd.DataFrame,
    covid_harm: pd.DataFrame,
    control_orig: pd.DataFrame,
    control_harm: pd.DataFrame,
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Comparar métricas antes e depois da harmonização.
    
    Parameters
    ----------
    covid_orig, covid_harm : pd.DataFrame
        Métricas COVID original e harmonizada
    control_orig, control_harm : pd.DataFrame
        Métricas controles original e harmonizada
    metrics : List[str], optional
        Métricas para comparar
        
    Returns
    -------
    pd.DataFrame
        Comparação
    """
    if metrics is None:
        metrics = [m for m in ROBUST_METRICS if m in covid_orig.columns]
    
    results = []
    
    for metric in metrics:
        # Antes
        diff_orig = covid_orig[metric].mean() - control_orig[metric].mean()
        t_orig, p_orig = stats.ttest_ind(
            covid_orig[metric].dropna(), 
            control_orig[metric].dropna()
        )
        
        # Depois
        diff_harm = covid_harm[metric].mean() - control_harm[metric].mean()
        t_harm, p_harm = stats.ttest_ind(
            covid_harm[metric].dropna(), 
            control_harm[metric].dropna()
        )
        
        results.append({
            'metric': metric,
            'diff_original': diff_orig,
            't_original': t_orig,
            'p_original': p_orig,
            'diff_harmonized': diff_harm,
            't_harmonized': t_harm,
            'p_harmonized': p_harm,
            'diff_preserved': np.sign(diff_orig) == np.sign(diff_harm)
        })
    
    return pd.DataFrame(results)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("ComBat Harmonization Module")
    print(f"neuroCombat available: {HAS_NEUROCOMBAT}")
    
    if not HAS_NEUROCOMBAT:
        print("\nTo install neuroCombat:")
        print("  pip install neuroCombat")
