"""
Configuração - Paths para o projeto SARS.
"""

from pathlib import Path
import numpy as np
from typing import List

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path('/mnt/nvme1n1p1/sars_cov_2_project')

# Caminho para conectividade rs-fMRI:
# data/output/rsfmri/connectivity/{atlas}/{strategy}/{subject}/{file}.npy
CONNECTIVITY_DIR = PROJECT_ROOT / 'data' / 'output' / 'rsfmri' / 'connectivity'

# Onde salvar outputs
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'proc'
COMPARISON_DIR = OUTPUT_DIR / 'normative_comparison'

# =============================================================================
# ATLAS
# =============================================================================

ATLASES = ['schaefer_100', 'synthseg_86', 'aal3_170', 'brainnetome_246']

ATLAS_ROIS = {
    'schaefer_100': 100,
    'synthseg_86': 86,
    'aal3_170': 170,
    'brainnetome_246': 246,
}

def get_atlas_n_rois(name: str) -> int:
    return ATLAS_ROIS.get(name, 100)

# =============================================================================
# SUJEITOS
# =============================================================================

COVID_SUBJECTS = [f'sub-{str(i).zfill(2)}' for i in range(1, 25) if i != 21]

def get_available_subjects(
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor'
) -> List[str]:
    """Retorna sujeitos com dados disponíveis."""
    available = []
    for sub in COVID_SUBJECTS:
        path = CONNECTIVITY_DIR / atlas_name / strategy / sub / 'connectivity_correlation_fisherz.npy'
        if path.exists():
            available.append(sub)
    return available

# =============================================================================
# CARREGAMENTO
# =============================================================================

def load_connectivity_matrix(
    subject_id: str,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    conn_type: str = 'correlation_fisherz'
) -> np.ndarray:
    """Carrega matriz de conectividade FC."""
    path = CONNECTIVITY_DIR / atlas_name / strategy / subject_id / f'connectivity_{conn_type}.npy'
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return np.load(path)


def load_adjacency_matrix(
    subject_id: str,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    threshold_percentile: float = 85
) -> np.ndarray:
    """Carrega FC e converte para adjacência binária."""
    matrix = load_connectivity_matrix(subject_id, atlas_name, strategy)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    triu_idx = np.triu_indices_from(matrix, k=1)
    values = matrix[triu_idx]
    threshold = np.percentile(values, threshold_percentile)
    adj = (matrix >= threshold).astype(float)
    np.fill_diagonal(adj, 0)
    return adj


# =============================================================================
# DEBUG
# =============================================================================

def print_config():
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"CONNECTIVITY_DIR: {CONNECTIVITY_DIR}")
    print(f"  exists: {CONNECTIVITY_DIR.exists()}")
    print("\nSubjects disponíveis:")
    for atlas in ATLASES:
        subs = get_available_subjects(atlas)
        print(f"  {atlas}: {len(subs)}")


if __name__ == '__main__':
    print_config()
