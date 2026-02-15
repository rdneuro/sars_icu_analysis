"""
Utilitários
===========

Funções auxiliares para gerenciamento de atlas e visualização.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


class AtlasManager:
    """
    Gerencia múltiplos atlas de parcellamento cerebral.
    
    Suporta: synthseg, schaefer_100, aal3, brainnetome
    
    Parameters
    ----------
    atlas_dir : Path
        Diretório contendo dados dos atlas
    """
    
    def __init__(self, atlas_dir: Optional[Path] = None):
        self.atlas_dir = atlas_dir
        self.available_atlases = ['synthseg', 'schaefer_100', 'aal3', 'brainnetome']
        self.atlas_info = self._load_atlas_info()
        
    def _load_atlas_info(self) -> Dict:
        """Carrega informações sobre cada atlas"""
        return {
            'synthseg': {
                'n_regions': 99,
                'type': 'anatomical',
                'includes_subcortical': True
            },
            'schaefer_100': {
                'n_regions': 100,
                'type': 'functional',
                'includes_subcortical': False,
                'networks': 7  # Yeo networks
            },
            'aal3': {
                'n_regions': 166,
                'type': 'anatomical',
                'includes_subcortical': True
            },
            'brainnetome': {
                'n_regions': 246,
                'type': 'anatomical',
                'includes_subcortical': True
            }
        }
    
    def get_atlas_size(self, atlas_name: str) -> int:
        """Retorna número de regiões do atlas"""
        return self.atlas_info[atlas_name]['n_regions']
    
    def get_network_assignments(
        self,
        atlas_name: str
    ) -> Optional[np.ndarray]:
        """
        Retorna assignments de rede (para atlas funcionais).
        
        Parameters
        ----------
        atlas_name : str
            Nome do atlas
            
        Returns
        -------
        assignments : np.ndarray or None
            Assignment de cada ROI a uma rede
        """
        if atlas_name == 'schaefer_100':
            # Schaefer_100 7-network (Yeo)
            # Aproximação: ~14 ROIs por rede
            assignments = np.repeat(np.arange(7), 14)[:100]
            return assignments
        
        return None
    
    def compare_atlases(
        self,
        data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Compara resultados através de múltiplos atlas.
        
        Parameters
        ----------
        data_dict : Dict[str, np.ndarray]
            Dados por atlas {'atlas_name': data}
            
        Returns
        -------
        comparison : Dict
            Estatísticas comparativas
        """
        comparison = {}
        
        for atlas_name, data in data_dict.items():
            comparison[atlas_name] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'n_regions': len(data)
            }
        
        return comparison
    
    def harmonize_across_atlases(
        self,
        data_dict: Dict[str, np.ndarray],
        method: str = 'z-score'
    ) -> Dict[str, np.ndarray]:
        """
        Harmoniza dados através de atlas (normalização).
        
        Parameters
        ----------
        data_dict : Dict[str, np.ndarray]
            Dados por atlas
        method : str
            Método de harmonização ('z-score', 'minmax')
            
        Returns
        -------
        harmonized : Dict[str, np.ndarray]
            Dados harmonizados
        """
        harmonized = {}
        
        for atlas_name, data in data_dict.items():
            if method == 'z-score':
                harmonized[atlas_name] = (data - np.mean(data)) / (np.std(data) + 1e-10)
            elif method == 'minmax':
                harmonized[atlas_name] = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
        
        return harmonized


def load_subject_data(
    subject_id: str,
    data_dir: Path,
    atlas_name: str,
    modalities: List[str] = ['sc', 'fc']
) -> Dict[str, np.ndarray]:
    """
    Carrega dados de um sujeito.
    
    Parameters
    ----------
    subject_id : str
        ID do sujeito
    data_dir : Path
        Diretório de dados
    atlas_name : str
        Nome do atlas
    modalities : List[str]
        Modalidades a carregar ('sc', 'fc', 'timeseries')
        
    Returns
    -------
    data : Dict[str, np.ndarray]
        Dados carregados
    """
    data = {}
    
    for modality in modalities:
        filepath = data_dir / atlas_name / subject_id / f"{modality}.npy"
        
        if filepath.exists():
            data[modality] = np.load(filepath)
        else:
            print(f"Warning: {filepath} não encontrado")
    
    return data


def batch_load_subjects(
    subject_ids: List[str],
    data_dir: Path,
    atlas_name: str,
    modalities: List[str] = ['fc']
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Carrega dados de múltiplos sujeitos.
    
    Parameters
    ----------
    subject_ids : List[str]
        Lista de IDs
    data_dir : Path
        Diretório de dados
    atlas_name : str
        Atlas
    modalities : List[str]
        Modalidades
        
    Returns
    -------
    batch_data : Dict
        Dados por sujeito
    """
    batch_data = {}
    
    for subject_id in subject_ids:
        try:
            batch_data[subject_id] = load_subject_data(
                subject_id, data_dir, atlas_name, modalities
            )
        except Exception as e:
            print(f"Erro ao carregar {subject_id}: {e}")
    
    return batch_data


def save_results(
    results: Dict,
    output_path: Path,
    format: str = 'npz'
):
    """
    Salva resultados de análise.
    
    Parameters
    ----------
    results : Dict
        Resultados a salvar
    output_path : Path
        Caminho de saída
    format : str
        Formato ('npz', 'json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'npz':
        np.savez_compressed(output_path, **results)
    elif format == 'json':
        # Converte arrays para listas
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)


def create_analysis_report(
    results: Dict,
    output_path: Path
):
    """
    Cria relatório markdown dos resultados.
    
    Parameters
    ----------
    results : Dict
        Resultados da análise
    output_path : Path
        Caminho do relatório
    """
    with open(output_path, 'w') as f:
        f.write("# Neuro Criticality Analysis Report\n\n")
        
        for section, data in results.items():
            f.write(f"## {section}\n\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- **{key}**: {value:.4f}\n")
                    elif isinstance(value, np.ndarray):
                        f.write(f"- **{key}**: array shape {value.shape}\n")
            
            f.write("\n")
