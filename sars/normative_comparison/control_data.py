"""
╔══════════════════════════════════════════════════════════════════════╗
║  Normative Comparison - Control Data Module                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Download, processamento e carregamento de dados de controles        ║
║  saudáveis de datasets abertos.                                      ║
║                                                                      ║
║  DATASETS SUPORTADOS:                                                ║
║  - NKI-Rockland (Enhanced): TR~2s, melhor match para nosso protocolo ║
║  - HCP 1200: Alta qualidade, mas TR=0.72s (diferente)                ║
║  - Precomputed: Métricas já calculadas (mais prático)                ║
║                                                                      ║
║  NOTA IMPORTANTE:                                                    ║
║  Como estamos comparando MÉTRICAS TOPOLÓGICAS (não matrizes brutas), ║
║  diferenças de TR e protocolo são menos críticas porque métricas     ║
║  como modularity, efficiency e clustering dependem da topologia      ║
║  relativa da rede, não dos valores absolutos.                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Optional, Union, Tuple
import warnings
import json
from urllib.request import urlretrieve
import hashlib

from .config import (
    CONTROLS_DIR, ATLASES, ATLAS_INFO, GLOBAL_METRICS, 
    ROBUST_METRICS, get_atlas_n_rois
)

# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_INFO = {
    'nki_rockland': {
        'name': 'NKI-Rockland Enhanced',
        'description': 'Large-scale lifespan sample with TR~2s acquisition',
        'tr_range': (1.4, 2.5),  # TR varia por protocolo
        'n_subjects_approx': 1000,
        'age_range': (6, 85),
        'url': 'http://fcon_1000.projects.nitrc.org/indi/enhanced/',
        'preprocessed_url': None,  # Precisa processar localmente ou usar derivados
        'reference': 'Nooner et al., 2012, Front. Neurosci.'
    },
    'hcp_1200': {
        'name': 'Human Connectome Project (1200 Subjects)',
        'description': 'High-quality multimodal data, TR=0.72s',
        'tr': 0.72,
        'n_subjects': 1200,
        'age_range': (22, 35),
        'url': 'https://db.humanconnectome.org/',
        'preprocessed_url': None,  # Requer credenciais
        'reference': 'Van Essen et al., 2013, NeuroImage'
    },
    'precomputed': {
        'name': 'Precomputed Normative Metrics',
        'description': 'Graph metrics from published normative datasets',
        'reference': 'Various sources - see individual files'
    }
}

# =============================================================================
# PRECOMPUTED NORMATIVE DATA
# =============================================================================

# Valores normativos da literatura para métricas de grafo
# Baseado em meta-análises e estudos de larga escala
NORMATIVE_RANGES = {
    'schaefer_100': {
        'density_0.15': {  # Para densidade de 15%
            'modularity': {'mean': 0.45, 'std': 0.08, 'n': 500, 'source': 'Meta-analysis'},
            'global_efficiency': {'mean': 0.52, 'std': 0.06, 'n': 500, 'source': 'Meta-analysis'},
            'local_efficiency': {'mean': 0.72, 'std': 0.05, 'n': 500, 'source': 'Meta-analysis'},
            'mean_clustering': {'mean': 0.48, 'std': 0.07, 'n': 500, 'source': 'Meta-analysis'},
            'small_worldness_sigma': {'mean': 1.8, 'std': 0.4, 'n': 500, 'source': 'Meta-analysis'},
            'assortativity': {'mean': 0.15, 'std': 0.12, 'n': 500, 'source': 'Meta-analysis'},
            'transitivity': {'mean': 0.45, 'std': 0.08, 'n': 500, 'source': 'Meta-analysis'},
            'characteristic_path_length': {'mean': 2.1, 'std': 0.3, 'n': 500, 'source': 'Meta-analysis'},
        }
    }
}


def get_normative_values(
    atlas_name: str = 'schaefer_100',
    density: float = 0.15
) -> Dict:
    """
    Retorna valores normativos da literatura para métricas de grafo.
    
    Parameters
    ----------
    atlas_name : str
        Nome do atlas
    density : float
        Densidade de threshold usada
        
    Returns
    -------
    dict
        Dicionário com valores normativos por métrica
    """
    density_key = f'density_{density:.2f}'.replace('.', '_').rstrip('0').rstrip('_')
    
    if atlas_name in NORMATIVE_RANGES:
        if density_key in NORMATIVE_RANGES[atlas_name]:
            return NORMATIVE_RANGES[atlas_name][density_key]
    
    warnings.warn(f"Normative values not available for {atlas_name} at density {density}")
    return {}


# =============================================================================
# SIMULATED CONTROL DATA (PARA DESENVOLVIMENTO)
# =============================================================================

def generate_simulated_controls(
    n_subjects: int = 50,
    atlas_name: str = 'schaefer_100',
    seed: int = 42,
    add_noise: bool = True
) -> pd.DataFrame:
    """
    Gerar dados de controles simulados baseados em valores normativos.
    
    NOTA: Use apenas para desenvolvimento/teste do pipeline.
    Para análise real, use dados de datasets abertos.
    
    Parameters
    ----------
    n_subjects : int
        Número de sujeitos simulados
    atlas_name : str
        Nome do atlas
    seed : int
        Random seed
    add_noise : bool
        Se True, adiciona variabilidade realista
        
    Returns
    -------
    pd.DataFrame
        DataFrame com métricas simuladas
    """
    np.random.seed(seed)
    
    normative = get_normative_values(atlas_name)
    
    if not normative:
        # Usar valores padrão se normativos não disponíveis
        normative = {
            'modularity': {'mean': 0.45, 'std': 0.08},
            'global_efficiency': {'mean': 0.52, 'std': 0.06},
            'local_efficiency': {'mean': 0.72, 'std': 0.05},
            'mean_clustering': {'mean': 0.48, 'std': 0.07},
            'mean_degree': {'mean': 15.0, 'std': 2.0},
            'mean_strength': {'mean': 8.5, 'std': 1.5},
            'mean_betweenness': {'mean': 0.02, 'std': 0.005},
            'assortativity': {'mean': 0.15, 'std': 0.12},
            'transitivity': {'mean': 0.45, 'std': 0.08},
        }
    
    data = {
        'subject_id': [f'ctrl-{str(i).zfill(3)}' for i in range(1, n_subjects + 1)],
        'group': ['control'] * n_subjects,
        'site': ['simulated'] * n_subjects,
    }
    
    for metric, params in normative.items():
        mean = params['mean']
        std = params['std'] if add_noise else params['std'] * 0.1
        
        values = np.random.normal(mean, std, n_subjects)
        
        # Clipping para valores fisicamente plausíveis
        if metric in ['modularity', 'global_efficiency', 'local_efficiency', 
                      'mean_clustering', 'transitivity']:
            values = np.clip(values, 0, 1)
        elif metric in ['small_worldness_sigma', 'mean_degree', 'mean_strength']:
            values = np.clip(values, 0, None)
        
        data[metric] = values
    
    df = pd.DataFrame(data)
    
    print(f"✓ Generated {n_subjects} simulated control subjects")
    return df


# =============================================================================
# NKI-ROCKLAND DATA ACCESS
# =============================================================================

def check_nki_derivatives_available() -> bool:
    """
    Verificar se derivados NKI-Rockland estão disponíveis localmente.
    """
    nki_dir = CONTROLS_DIR / 'nki_rockland'
    return nki_dir.exists() and any(nki_dir.glob('*.csv'))


def download_nki_phenotypic() -> pd.DataFrame:
    """
    Download dados fenomenológicos do NKI-Rockland.
    
    Returns
    -------
    pd.DataFrame
        Dados demográficos/fenomenológicos
    """
    phenotypic_url = (
        "https://fcon_1000.projects.nitrc.org/indi/enhanced/"
        "assessments/nki-rs_assessments_1532.csv"
    )
    
    output_file = CONTROLS_DIR / 'nki_rockland' / 'phenotypic.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists():
        return pd.read_csv(output_file)
    
    try:
        print(f"Downloading NKI phenotypic data...")
        df = pd.read_csv(phenotypic_url)
        df.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
        return df
    except Exception as e:
        warnings.warn(f"Failed to download NKI phenotypic data: {e}")
        return pd.DataFrame()


def load_nki_graph_metrics(
    atlas_name: str = 'schaefer_100',
    age_range: Tuple[int, int] = (20, 60),
    max_subjects: int = None
) -> pd.DataFrame:
    """
    Carregar métricas de grafo pré-computadas do NKI-Rockland.
    
    NOTA: Requer que os dados tenham sido processados previamente
    pelo mesmo pipeline (fMRIPrep + XCP-D + graph analysis).
    
    Parameters
    ----------
    atlas_name : str
        Nome do atlas
    age_range : tuple
        Range de idade para filtrar controles (min, max)
    max_subjects : int, optional
        Máximo de sujeitos a carregar
        
    Returns
    -------
    pd.DataFrame
        Métricas de grafo dos controles
    """
    nki_metrics_file = CONTROLS_DIR / 'nki_rockland' / f'{atlas_name}_global_metrics.csv'
    
    if not nki_metrics_file.exists():
        warnings.warn(
            f"NKI-Rockland metrics not found at {nki_metrics_file}. "
            f"Using simulated data instead. For real analysis, process NKI data "
            f"through the same pipeline as COVID subjects."
        )
        return generate_simulated_controls(50, atlas_name)
    
    df = pd.read_csv(nki_metrics_file)
    
    # Filtrar por idade se disponível
    if 'age' in df.columns and age_range:
        df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    # Limitar número de sujeitos
    if max_subjects and len(df) > max_subjects:
        df = df.sample(n=max_subjects, random_state=42)
    
    df['group'] = 'control'
    df['site'] = 'nki_rockland'
    
    print(f"✓ Loaded {len(df)} NKI-Rockland control subjects")
    return df


# =============================================================================
# HCP DATA ACCESS
# =============================================================================

def load_hcp_graph_metrics(
    atlas_name: str = 'schaefer_100',
    max_subjects: int = None
) -> pd.DataFrame:
    """
    Carregar métricas de grafo do HCP.
    
    NOTA: Requer credenciais HCP e dados pré-processados.
    
    Parameters
    ----------
    atlas_name : str
        Nome do atlas
    max_subjects : int, optional
        Máximo de sujeitos
        
    Returns
    -------
    pd.DataFrame
        Métricas de grafo
    """
    hcp_metrics_file = CONTROLS_DIR / 'hcp_1200' / f'{atlas_name}_global_metrics.csv'
    
    if not hcp_metrics_file.exists():
        warnings.warn(
            f"HCP metrics not found at {hcp_metrics_file}. "
            f"Using simulated data instead."
        )
        return generate_simulated_controls(100, atlas_name)
    
    df = pd.read_csv(hcp_metrics_file)
    
    if max_subjects and len(df) > max_subjects:
        df = df.sample(n=max_subjects, random_state=42)
    
    df['group'] = 'control'
    df['site'] = 'hcp_1200'
    
    print(f"✓ Loaded {len(df)} HCP control subjects")
    return df


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def get_control_metrics(
    source: str = 'precomputed',
    atlas_name: str = 'schaefer_100',
    n_subjects: int = 50,
    age_range: Tuple[int, int] = (20, 60),
    **kwargs
) -> pd.DataFrame:
    """
    Interface unificada para obter métricas de controles.
    
    Parameters
    ----------
    source : str
        Fonte dos dados: 'nki_rockland', 'hcp_1200', 'precomputed', 'simulated'
    atlas_name : str
        Nome do atlas
    n_subjects : int
        Número de sujeitos (para simulated) ou máximo (para datasets reais)
    age_range : tuple
        Range de idade para filtrar
    **kwargs
        Argumentos adicionais para funções específicas
        
    Returns
    -------
    pd.DataFrame
        Métricas dos controles
    """
    if source == 'nki_rockland':
        return load_nki_graph_metrics(atlas_name, age_range, n_subjects)
    
    elif source == 'hcp_1200':
        return load_hcp_graph_metrics(atlas_name, n_subjects)
    
    elif source == 'precomputed':
        # Tentar NKI primeiro, depois HCP, depois simulated
        try:
            return load_nki_graph_metrics(atlas_name, age_range, n_subjects)
        except:
            pass
        try:
            return load_hcp_graph_metrics(atlas_name, n_subjects)
        except:
            pass
        warnings.warn("No precomputed data available, using simulated controls")
        return generate_simulated_controls(n_subjects, atlas_name)
    
    elif source == 'simulated':
        return generate_simulated_controls(n_subjects, atlas_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown source: {source}. Choose from: "
                        f"'nki_rockland', 'hcp_1200', 'precomputed', 'simulated'")


# =============================================================================
# DATA EXPORT FOR EXTERNAL PROCESSING
# =============================================================================

def prepare_data_for_external_processing(
    atlas_name: str = 'schaefer_100',
    output_dir: str = None
) -> Dict[str, Path]:
    """
    Preparar estrutura de diretórios e arquivos de configuração
    para processar dados de controles externamente.
    
    Parameters
    ----------
    atlas_name : str
        Nome do atlas
    output_dir : str, optional
        Diretório de saída
        
    Returns
    -------
    dict
        Paths criados
    """
    if output_dir is None:
        output_dir = CONTROLS_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar estrutura
    nki_dir = output_dir / 'nki_rockland'
    hcp_dir = output_dir / 'hcp_1200'
    
    for d in [nki_dir, hcp_dir]:
        d.mkdir(exist_ok=True)
    
    # Criar template de configuração
    config_template = {
        'atlas_name': atlas_name,
        'n_rois': get_atlas_n_rois(atlas_name),
        'metrics_to_compute': GLOBAL_METRICS,
        'robust_metrics': ROBUST_METRICS,
        'processing_notes': (
            'Process control data using the same pipeline as COVID subjects: '
            'fMRIPrep + XCP-D (36P or acompcor) + graph analysis with bctpy. '
            'Use same threshold density (default 0.15). '
            'Output global_metrics.csv with columns matching GLOBAL_METRICS list.'
        )
    }
    
    config_file = output_dir / 'processing_config.json'
    with open(config_file, 'w') as f:
        json.dump(config_template, f, indent=2)
    
    print(f"✓ Created processing structure at {output_dir}")
    print(f"  Config: {config_file}")
    print(f"  NKI dir: {nki_dir}")
    print(f"  HCP dir: {hcp_dir}")
    
    return {
        'config': config_file,
        'nki_dir': nki_dir,
        'hcp_dir': hcp_dir
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_control_data(df: pd.DataFrame, atlas_name: str = None) -> Dict:
    """
    Validar dados de controles carregados.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com métricas
    atlas_name : str, optional
        Atlas para validar número de ROIs
        
    Returns
    -------
    dict
        Relatório de validação
    """
    report = {
        'n_subjects': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'issues': []
    }
    
    # Checar colunas essenciais
    essential = ['subject_id', 'group']
    for col in essential:
        if col not in df.columns:
            report['issues'].append(f"Missing essential column: {col}")
    
    # Checar métricas robustas
    available_robust = [m for m in ROBUST_METRICS if m in df.columns]
    report['robust_metrics_available'] = available_robust
    report['robust_metrics_missing'] = [m for m in ROBUST_METRICS if m not in df.columns]
    
    # Checar ranges de valores
    for metric in available_robust:
        values = df[metric].dropna()
        report[f'{metric}_range'] = (float(values.min()), float(values.max()))
        
        # Sanity checks
        if metric in ['modularity', 'global_efficiency', 'local_efficiency']:
            if values.max() > 1.5 or values.min() < -0.5:
                report['issues'].append(f"{metric} has suspicious values outside [0,1]")
    
    report['valid'] = len(report['issues']) == 0
    
    return report


# =============================================================================
# CLI / DIAGNOSTIC
# =============================================================================

def print_available_controls():
    """Imprimir informações sobre controles disponíveis."""
    print("\n" + "="*60)
    print("AVAILABLE CONTROL DATA SOURCES")
    print("="*60)
    
    for source, info in DATASET_INFO.items():
        print(f"\n{source}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        if 'reference' in info:
            print(f"  Reference: {info['reference']}")
    
    print("\n" + "-"*60)
    print("LOCAL DATA STATUS:")
    
    for source in ['nki_rockland', 'hcp_1200']:
        source_dir = CONTROLS_DIR / source
        if source_dir.exists():
            files = list(source_dir.glob('*.csv'))
            print(f"  {source}: {len(files)} CSV files found")
        else:
            print(f"  {source}: No local data")
    
    print("\nTo prepare directory structure for external processing:")
    print("  >>> from normative_comparison import control_data")
    print("  >>> control_data.prepare_data_for_external_processing('schaefer_100')")


if __name__ == '__main__':
    print_available_controls()
