"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradients Core Module                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Computação de gradientes de conectividade com múltiplos métodos.            ║
║                                                                              ║
║  CONCEITO:                                                                   ║
║  Gradientes de conectividade (Margulies et al., 2016) descrevem a           ║
║  organização macroscópica do córtex ao longo de eixos contínuos.            ║
║  O primeiro gradiente (G1) tipicamente captura a transição entre            ║
║  regiões sensoriais primárias e áreas de associação de alto nível.          ║
║                                                                              ║
║  MÉTODOS DISPONÍVEIS:                                                        ║
║  1. Diffusion Map (DM) - Captura estrutura não-linear, robusto a ruído      ║
║  2. PCA - Rápido, interpretável, linear                                      ║
║  3. Laplacian Eigenmaps (LE) - Preserva estrutura local                     ║
║  4. UMAP - Preserva estrutura global e local (requer umap-learn)            ║
║                                                                              ║
║  KERNELS PARA AFINIDADE:                                                     ║
║  - normalized_angle: Ângulo normalizado (recomendado para FC)               ║
║  - cosine: Similaridade cosseno                                              ║
║  - pearson/spearman: Correlação                                              ║
║  - gaussian: Kernel RBF                                                       ║
║                                                                              ║
║  REFERÊNCIAS:                                                                ║
║  - Margulies et al. (2016) PNAS - Principal gradient                        ║
║  - Vos de Wael et al. (2020) CommBio - BrainSpace                           ║
║  - Huntenburg et al. (2018) NeuroImage - Gradients review                   ║
║  - Coifman & Lafon (2006) - Diffusion Maps                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

# =============================================================================
# CHECK OPTIONAL DEPENDENCIES
# =============================================================================

try:
    from brainspace.gradient import GradientMaps
    from brainspace.gradient.alignment import procrustes_alignment
    HAS_BRAINSPACE = True
except ImportError:
    HAS_BRAINSPACE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GradientResult:
    """
    Resultado de uma análise de gradiente.
    
    Attributes
    ----------
    gradients : np.ndarray
        Array de gradientes, shape (n_rois, n_components).
        Cada coluna é um gradiente ordenado por variância explicada.
    lambdas : np.ndarray
        Eigenvalues associados a cada componente.
    explained_variance : np.ndarray
        Proporção da variância explicada por cada componente.
    n_components : int
        Número de componentes extraídos.
    approach : str
        Método usado ('dm', 'pca', 'le', 'umap').
    kernel : str
        Kernel usado para construção de afinidade.
    affinity_matrix : np.ndarray, optional
        Matriz de afinidade computada (se retain_affinity=True).
    subject_id : str, optional
        Identificador do sujeito.
    """
    gradients: np.ndarray
    lambdas: np.ndarray
    explained_variance: np.ndarray
    n_components: int
    approach: str
    kernel: str
    affinity_matrix: np.ndarray = field(default=None, repr=False)
    subject_id: str = None
    
    @property
    def G1(self) -> np.ndarray:
        """Primeiro gradiente (hierarquia principal)."""
        return self.gradients[:, 0]
    
    @property
    def G2(self) -> np.ndarray:
        """Segundo gradiente."""
        return self.gradients[:, 1] if self.n_components > 1 else None
    
    @property
    def G3(self) -> np.ndarray:
        """Terceiro gradiente."""
        return self.gradients[:, 2] if self.n_components > 2 else None
    
    def get_gradient(self, n: int) -> np.ndarray:
        """Retorna o n-ésimo gradiente (1-indexed)."""
        if n < 1 or n > self.n_components:
            raise ValueError(f"Gradiente {n} não existe. Disponíveis: 1-{self.n_components}")
        return self.gradients[:, n-1]
    
    def to_dict(self) -> Dict:
        """Converter para dicionário."""
        return {
            'gradients': self.gradients,
            'lambdas': self.lambdas,
            'explained_variance': self.explained_variance,
            'n_components': self.n_components,
            'approach': self.approach,
            'kernel': self.kernel,
            'subject_id': self.subject_id,
        }


@dataclass 
class GroupGradientResult:
    """
    Resultado de análise de gradientes de grupo (alinhados).
    
    Attributes
    ----------
    individual_gradients : Dict[str, GradientResult]
        Gradientes individuais por sujeito.
    aligned_gradients : Dict[str, np.ndarray]
        Gradientes alinhados por sujeito.
    template : np.ndarray
        Template de referência usado para alinhamento.
    group_mean : np.ndarray
        Média dos gradientes alinhados.
    group_std : np.ndarray
        Desvio padrão dos gradientes alinhados.
    """
    individual_gradients: Dict[str, GradientResult]
    aligned_gradients: Dict[str, np.ndarray]
    template: np.ndarray
    group_mean: np.ndarray
    group_std: np.ndarray
    n_subjects: int
    n_components: int


# =============================================================================
# KERNEL / AFFINITY FUNCTIONS
# =============================================================================

def compute_affinity(
    connectivity: np.ndarray,
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    gamma: float = None
) -> np.ndarray:
    """
    Construir matriz de afinidade a partir de matriz de conectividade.
    
    A matriz de afinidade quantifica a similaridade entre padrões de
    conectividade de diferentes regiões. Regiões com padrões similares
    terão alta afinidade.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade funcional (n_rois, n_rois)
    kernel : str
        Tipo de kernel:
        - 'normalized_angle': Ângulo normalizado (recomendado)
        - 'cosine': Similaridade cosseno
        - 'pearson': Correlação de Pearson
        - 'spearman': Correlação de Spearman  
        - 'gaussian': Kernel RBF/Gaussiano
        - 'none': Usar conectividade diretamente
    sparsity : float
        Proporção de conexões mais fracas a zerar (0-1).
        0.9 = manter apenas top 10% conexões mais fortes.
        Use 0 para manter todas.
    gamma : float, optional
        Parâmetro para kernel gaussiano. Se None, usa mediana das distâncias.
        
    Returns
    -------
    np.ndarray
        Matriz de afinidade (n_rois, n_rois)
    """
    # Garantir simetria e remover diagonal
    conn = connectivity.copy()
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    
    n_rois = conn.shape[0]
    
    # Aplicar sparsity threshold
    if sparsity > 0:
        threshold = np.percentile(conn[conn > 0], sparsity * 100)
        conn[conn < threshold] = 0
    
    # Calcular afinidade baseada no kernel
    if kernel == 'none':
        # Usar conectividade diretamente
        affinity = conn.copy()
        
    elif kernel == 'cosine':
        # Similaridade cosseno entre linhas
        # Cada linha representa o perfil de conectividade de uma ROI
        norms = np.linalg.norm(conn, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Evitar divisão por zero
        conn_norm = conn / norms
        affinity = conn_norm @ conn_norm.T
        
    elif kernel == 'normalized_angle':
        # Ângulo normalizado (BrainSpace default)
        # Baseado em cosseno, mas normalizado para [0, 1]
        norms = np.linalg.norm(conn, axis=1, keepdims=True)
        norms[norms == 0] = 1
        conn_norm = conn / norms
        cos_sim = conn_norm @ conn_norm.T
        # Clamp para evitar problemas numéricos
        cos_sim = np.clip(cos_sim, -1, 1)
        # Converter para ângulo normalizado
        affinity = 1 - np.arccos(cos_sim) / np.pi
        
    elif kernel == 'pearson':
        # Correlação de Pearson entre linhas
        conn_centered = conn - conn.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(conn_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1
        conn_norm = conn_centered / norms
        affinity = conn_norm @ conn_norm.T
        
    elif kernel == 'spearman':
        # Correlação de Spearman (rank-based)
        from scipy.stats import rankdata
        conn_ranked = np.apply_along_axis(rankdata, 1, conn)
        conn_centered = conn_ranked - conn_ranked.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(conn_centered, axis=1, keepdims=True)
        norms[norms == 0] = 1
        conn_norm = conn_centered / norms
        affinity = conn_norm @ conn_norm.T
        
    elif kernel == 'gaussian':
        # Kernel RBF/Gaussiano
        # Primeiro calcular distância euclidiana
        dists = squareform(pdist(conn, metric='euclidean'))
        
        # Determinar gamma se não fornecido
        if gamma is None:
            gamma = 1 / (2 * np.median(dists[dists > 0])**2)
        
        affinity = np.exp(-gamma * dists**2)
        
    else:
        raise ValueError(f"Kernel desconhecido: {kernel}. "
                        f"Use: normalized_angle, cosine, pearson, spearman, gaussian, none")
    
    # Garantir simetria e valores não-negativos
    affinity = (affinity + affinity.T) / 2
    affinity = np.maximum(affinity, 0)
    np.fill_diagonal(affinity, 0)
    
    return affinity


# =============================================================================
# GRADIENT COMPUTATION METHODS
# =============================================================================

def compute_diffusion_map(
    affinity: np.ndarray,
    n_components: int = 10,
    alpha: float = 0.5,
    diffusion_time: int = 0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computar Diffusion Map embedding.
    
    Diffusion Maps capturam a estrutura geométrica dos dados através
    de um processo de difusão na matriz de afinidade. São robustos
    a ruído e capturam relações não-lineares.
    
    Parameters
    ----------
    affinity : np.ndarray
        Matriz de afinidade (n_rois, n_rois)
    n_components : int
        Número de componentes a extrair
    alpha : float
        Parâmetro de normalização (0 = graph Laplacian, 1 = Laplace-Beltrami)
        0.5 é um bom default que balanceia densidade e geometria.
    diffusion_time : int
        Tempo de difusão. 0 = usar eigenvalues diretamente.
        Valores maiores = maior escala de difusão.
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (embeddings, eigenvalues)
    """
    np.random.seed(random_state)
    n = affinity.shape[0]
    
    # Normalização por densidade (anisotropic diffusion)
    if alpha > 0:
        # Densidade: soma das linhas
        d = np.sum(affinity, axis=1)
        d[d == 0] = 1  # Evitar divisão por zero
        
        # Normalizar pela densidade^alpha
        d_alpha = np.power(d, -alpha)
        affinity = affinity * np.outer(d_alpha, d_alpha)
    
    # Construir matriz de transição (row-stochastic)
    d = np.sum(affinity, axis=1)
    d[d == 0] = 1
    d_inv = 1.0 / d
    
    # P = D^(-1) * W
    transition = affinity * d_inv[:, np.newaxis]
    
    # Simetrizar para eigendecomposition
    d_sqrt = np.sqrt(d)
    d_inv_sqrt = 1.0 / d_sqrt
    
    # M = D^(-1/2) * W * D^(-1/2)
    M = affinity * np.outer(d_inv_sqrt, d_inv_sqrt)
    M = (M + M.T) / 2  # Garantir simetria
    
    # Eigendecomposition
    n_compute = min(n_components + 1, n - 1)
    eigenvalues, eigenvectors = eigh(M)
    
    # Ordenar por eigenvalue decrescente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Descartar primeiro componente (trivial, eigenvalue = 1)
    eigenvalues = eigenvalues[1:n_compute]
    eigenvectors = eigenvectors[:, 1:n_compute]
    
    # Converter de volta para espaço original
    embeddings = eigenvectors * d_inv_sqrt[:, np.newaxis]
    
    # Aplicar diffusion time se especificado
    if diffusion_time > 0:
        embeddings = embeddings * np.power(eigenvalues, diffusion_time)
    
    # Normalizar embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=0, keepdims=True)
    
    return embeddings[:, :n_components], eigenvalues[:n_components]


def compute_laplacian_eigenmaps(
    affinity: np.ndarray,
    n_components: int = 10,
    norm_laplacian: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computar Laplacian Eigenmaps embedding.
    
    Laplacian Eigenmaps preservam a estrutura local: pontos próximos
    no espaço original permanecem próximos no embedding.
    
    Parameters
    ----------
    affinity : np.ndarray
        Matriz de afinidade
    n_components : int
        Número de componentes
    norm_laplacian : bool
        Se True, usa Laplaciano normalizado (recomendado)
    random_state : int
        Seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (embeddings, eigenvalues)
    """
    np.random.seed(random_state)
    n = affinity.shape[0]
    
    # Grau de cada nó
    d = np.sum(affinity, axis=1)
    d[d == 0] = 1
    
    if norm_laplacian:
        # Laplaciano normalizado: L = I - D^(-1/2) * W * D^(-1/2)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        L = np.eye(n) - affinity * np.outer(d_inv_sqrt, d_inv_sqrt)
    else:
        # Laplaciano não-normalizado: L = D - W
        L = np.diag(d) - affinity
    
    L = (L + L.T) / 2  # Garantir simetria
    
    # Eigendecomposition (menores eigenvalues)
    n_compute = min(n_components + 1, n - 1)
    eigenvalues, eigenvectors = eigh(L)
    
    # Ordenar por eigenvalue crescente
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Descartar primeiro (trivial)
    eigenvalues = eigenvalues[1:n_compute]
    eigenvectors = eigenvectors[:, 1:n_compute]
    
    # Converter para variância explicada (inverso dos eigenvalues)
    explained = 1 / (eigenvalues + 1e-10)
    explained = explained / np.sum(explained)
    
    return eigenvectors[:, :n_components], eigenvalues[:n_components]


def compute_pca_embedding(
    connectivity: np.ndarray,
    n_components: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computar PCA embedding diretamente na matriz de conectividade.
    
    PCA é rápido e interpretável, mas assume relações lineares.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade
    n_components : int
        Número de componentes
    random_state : int
        Seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (embeddings, eigenvalues, explained_variance_ratio)
    """
    # Preparar matriz
    conn = connectivity.copy()
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    
    # Padronizar
    scaler = StandardScaler()
    conn_scaled = scaler.fit_transform(conn)
    
    # PCA
    pca = PCA(n_components=min(n_components, conn.shape[0] - 1), 
              random_state=random_state)
    embeddings = pca.fit_transform(conn_scaled)
    
    return embeddings, pca.explained_variance_, pca.explained_variance_ratio_


def compute_umap_embedding(
    connectivity: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'correlation',
    random_state: int = 42
) -> np.ndarray:
    """
    Computar UMAP embedding.
    
    UMAP preserva estrutura local e global, mas não fornece
    eigenvalues ou variância explicada diretamente.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade
    n_components : int
        Número de dimensões do embedding
    n_neighbors : int
        Número de vizinhos para construção do grafo
    min_dist : float
        Distância mínima entre pontos no embedding
    metric : str
        Métrica de distância
    random_state : int
        Seed
        
    Returns
    -------
    np.ndarray
        Embeddings
    """
    if not HAS_UMAP:
        raise ImportError("UMAP não instalado. Use: pip install umap-learn")
    
    conn = connectivity.copy()
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    embeddings = reducer.fit_transform(conn)
    
    return embeddings


# =============================================================================
# MAIN GRADIENT FUNCTION
# =============================================================================

def compute_gradients(
    connectivity: np.ndarray,
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    random_state: int = 42,
    retain_affinity: bool = False,
    subject_id: str = None,
    **kwargs
) -> GradientResult:
    """
    Função principal para computar gradientes de conectividade.
    
    Esta é a função recomendada para uso geral. Ela encapsula os diferentes
    métodos de embedding e retorna um resultado estruturado.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade funcional (n_rois, n_rois).
        Pode ser correlação, Fisher-z, ou outra medida de conectividade.
    n_components : int
        Número de componentes (gradientes) a extrair.
        Recomendado: 10 para análise exploratória, 3-5 para visualização.
    approach : str
        Método de embedding:
        - 'dm': Diffusion Map (RECOMENDADO para FC)
          Captura estrutura não-linear, robusto a ruído.
        - 'pca': Principal Component Analysis
          Rápido, linear, interpretável.
        - 'le': Laplacian Eigenmaps
          Preserva estrutura local.
        - 'umap': UMAP (requer umap-learn)
          Preserva estrutura local e global.
    kernel : str
        Kernel para construção de afinidade (usado por 'dm' e 'le'):
        - 'normalized_angle': Ângulo normalizado (RECOMENDADO)
        - 'cosine': Similaridade cosseno
        - 'pearson': Correlação de Pearson
        - 'spearman': Correlação de Spearman
        - 'gaussian': Kernel Gaussiano/RBF
    sparsity : float
        Proporção de conexões mais fracas a zerar (0-1).
        0.9 = manter apenas top 10% mais fortes.
        Ajuda a focar em conexões mais robustas.
    random_state : int
        Seed para reprodutibilidade.
    retain_affinity : bool
        Se True, armazena matriz de afinidade no resultado.
    subject_id : str, optional
        Identificador do sujeito (para organização).
    **kwargs
        Parâmetros adicionais específicos do método:
        - dm: alpha, diffusion_time
        - le: norm_laplacian
        - umap: n_neighbors, min_dist, metric
        
    Returns
    -------
    GradientResult
        Objeto com gradientes e estatísticas.
        
    Examples
    --------
    >>> # Uso básico
    >>> result = compute_gradients(fc_matrix)
    >>> print(f"G1 variance: {result.explained_variance[0]:.3f}")
    >>> g1 = result.G1  # Primeiro gradiente
    
    >>> # Com PCA
    >>> result = compute_gradients(fc_matrix, approach='pca')
    
    >>> # Com parâmetros customizados
    >>> result = compute_gradients(
    ...     fc_matrix,
    ...     approach='dm',
    ...     kernel='cosine',
    ...     sparsity=0.85,
    ...     alpha=0.5,
    ...     diffusion_time=1
    ... )
    """
    # Validar input
    if connectivity.ndim != 2 or connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError("connectivity deve ser uma matriz quadrada")
    
    n_rois = connectivity.shape[0]
    n_components = min(n_components, n_rois - 1)
    
    # Preparar matriz
    conn = connectivity.copy()
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    
    affinity_matrix = None
    
    # =========================================================================
    # Computar embedding baseado no método escolhido
    # =========================================================================
    
    if approach == 'dm':
        # Diffusion Map
        affinity_matrix = compute_affinity(conn, kernel=kernel, sparsity=sparsity)
        
        alpha = kwargs.get('alpha', 0.5)
        diffusion_time = kwargs.get('diffusion_time', 0)
        
        embeddings, eigenvalues = compute_diffusion_map(
            affinity_matrix,
            n_components=n_components,
            alpha=alpha,
            diffusion_time=diffusion_time,
            random_state=random_state
        )
        
        # Variância explicada a partir dos eigenvalues
        explained_variance = eigenvalues / np.sum(eigenvalues)
        
    elif approach == 'le':
        # Laplacian Eigenmaps
        affinity_matrix = compute_affinity(conn, kernel=kernel, sparsity=sparsity)
        
        norm_laplacian = kwargs.get('norm_laplacian', True)
        
        embeddings, eigenvalues = compute_laplacian_eigenmaps(
            affinity_matrix,
            n_components=n_components,
            norm_laplacian=norm_laplacian,
            random_state=random_state
        )
        
        # Para LE, eigenvalues são "inversos" - menores = mais importantes
        explained_variance = (1 / (eigenvalues + 1e-10))
        explained_variance = explained_variance / np.sum(explained_variance)
        
    elif approach == 'pca':
        # PCA
        embeddings, eigenvalues, explained_variance = compute_pca_embedding(
            conn,
            n_components=n_components,
            random_state=random_state
        )
        
        kernel = 'none'  # PCA não usa kernel
        
    elif approach == 'umap':
        # UMAP
        if not HAS_UMAP:
            raise ImportError("UMAP não instalado. Use: pip install umap-learn")
        
        n_neighbors = kwargs.get('n_neighbors', 15)
        min_dist = kwargs.get('min_dist', 0.1)
        metric = kwargs.get('metric', 'correlation')
        
        embeddings = compute_umap_embedding(
            conn,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        # UMAP não fornece eigenvalues naturalmente
        eigenvalues = np.var(embeddings, axis=0)
        explained_variance = eigenvalues / np.sum(eigenvalues)
        kernel = 'umap'
        
    else:
        raise ValueError(f"Método desconhecido: {approach}. "
                        f"Use: dm, pca, le, umap")
    
    # =========================================================================
    # Criar resultado
    # =========================================================================
    
    result = GradientResult(
        gradients=embeddings,
        lambdas=eigenvalues,
        explained_variance=explained_variance,
        n_components=n_components,
        approach=approach,
        kernel=kernel,
        affinity_matrix=affinity_matrix if retain_affinity else None,
        subject_id=subject_id
    )
    
    return result


# =============================================================================
# ALIGNMENT FUNCTIONS
# =============================================================================

def align_gradients_procrustes(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Alinhar gradientes fonte ao alvo usando Procrustes.
    
    Procrustes encontra a rotação/reflexão ótima que minimiza
    a distância entre dois conjuntos de pontos.
    
    Parameters
    ----------
    source : np.ndarray
        Gradientes a serem alinhados (n_rois, n_components)
    target : np.ndarray
        Gradientes de referência (n_rois, n_components)
        
    Returns
    -------
    np.ndarray
        Gradientes alinhados
    """
    from scipy.linalg import orthogonal_procrustes
    
    # Centralizar
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    
    # Encontrar rotação ótima
    R, _ = orthogonal_procrustes(source_centered, target_centered)
    
    # Aplicar rotação
    aligned = source_centered @ R
    
    # Restaurar média do target
    aligned = aligned + target.mean(axis=0)
    
    return aligned


def align_gradients_joint(
    gradients_list: List[np.ndarray],
    reference: str = 'first'
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Alinhar múltiplos conjuntos de gradientes.
    
    Parameters
    ----------
    gradients_list : List[np.ndarray]
        Lista de gradientes a alinhar
    reference : str
        Estratégia de referência:
        - 'first': Usar primeiro como referência
        - 'mean': Iterativamente alinhar à média
        
    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        (gradientes_alinhados, template)
    """
    if len(gradients_list) == 0:
        raise ValueError("Lista de gradientes vazia")
    
    if reference == 'first':
        # Usar primeiro como template
        template = gradients_list[0].copy()
        aligned = [template]
        
        for grads in gradients_list[1:]:
            aligned.append(align_gradients_procrustes(grads, template))
        
    elif reference == 'mean':
        # Iterativamente alinhar à média
        # Iniciar com primeiro como template
        template = gradients_list[0].copy()
        
        for iteration in range(5):  # 5 iterações geralmente suficiente
            aligned = []
            for grads in gradients_list:
                aligned.append(align_gradients_procrustes(grads, template))
            
            # Atualizar template como média
            template = np.mean(aligned, axis=0)
        
    else:
        raise ValueError(f"Referência desconhecida: {reference}")
    
    return aligned, template


def compute_group_gradients(
    connectivity_matrices: Dict[str, np.ndarray],
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    align: bool = True,
    alignment_reference: str = 'mean',
    random_state: int = 42,
    verbose: bool = True
) -> GroupGradientResult:
    """
    Computar e alinhar gradientes para um grupo de sujeitos.
    
    Parameters
    ----------
    connectivity_matrices : Dict[str, np.ndarray]
        Dicionário {subject_id: connectivity_matrix}
    n_components : int
        Número de componentes
    approach : str
        Método de embedding
    kernel : str
        Kernel para afinidade
    sparsity : float
        Threshold de sparsity
    align : bool
        Se True, alinha gradientes entre sujeitos
    alignment_reference : str
        Estratégia de alinhamento ('first' ou 'mean')
    random_state : int
        Seed
    verbose : bool
        Mostrar progresso
        
    Returns
    -------
    GroupGradientResult
        Resultado com gradientes individuais, alinhados e estatísticas de grupo
    """
    if verbose:
        print(f"Computing gradients for {len(connectivity_matrices)} subjects...")
    
    # Computar gradientes individuais
    individual_results = {}
    gradients_list = []
    subject_ids = []
    
    for i, (subject_id, conn) in enumerate(connectivity_matrices.items()):
        if verbose and (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(connectivity_matrices)}")
        
        try:
            result = compute_gradients(
                conn,
                n_components=n_components,
                approach=approach,
                kernel=kernel,
                sparsity=sparsity,
                random_state=random_state + i,
                subject_id=subject_id
            )
            
            individual_results[subject_id] = result
            gradients_list.append(result.gradients)
            subject_ids.append(subject_id)
            
        except Exception as e:
            if verbose:
                warnings.warn(f"Erro em {subject_id}: {e}")
            continue
    
    if len(gradients_list) == 0:
        raise ValueError("Nenhum gradiente computado com sucesso")
    
    # Alinhar se solicitado
    if align:
        if verbose:
            print("Aligning gradients...")
        
        aligned_list, template = align_gradients_joint(
            gradients_list, 
            reference=alignment_reference
        )
        
        aligned_gradients = {
            sid: aligned for sid, aligned in zip(subject_ids, aligned_list)
        }
        
    else:
        aligned_gradients = {
            sid: grads for sid, grads in zip(subject_ids, gradients_list)
        }
        template = gradients_list[0]
    
    # Estatísticas de grupo
    aligned_array = np.stack(list(aligned_gradients.values()), axis=0)
    group_mean = np.mean(aligned_array, axis=0)
    group_std = np.std(aligned_array, axis=0)
    
    if verbose:
        print(f"✓ Computed gradients for {len(individual_results)} subjects")
    
    return GroupGradientResult(
        individual_gradients=individual_results,
        aligned_gradients=aligned_gradients,
        template=template,
        group_mean=group_mean,
        group_std=group_std,
        n_subjects=len(individual_results),
        n_components=n_components
    )


# =============================================================================
# GRADIENT STATISTICS
# =============================================================================

def compute_gradient_stats(gradients: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computar estatísticas descritivas dos gradientes.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes (n_rois, n_components)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Estatísticas por componente
    """
    from scipy import stats as sp_stats
    
    return {
        'mean': np.mean(gradients, axis=0),
        'std': np.std(gradients, axis=0),
        'range': np.ptp(gradients, axis=0),
        'min': np.min(gradients, axis=0),
        'max': np.max(gradients, axis=0),
        'skewness': sp_stats.skew(gradients, axis=0),
        'kurtosis': sp_stats.kurtosis(gradients, axis=0),
        'iqr': sp_stats.iqr(gradients, axis=0),
    }


def compute_gradient_correlation(
    gradients1: np.ndarray,
    gradients2: np.ndarray,
    method: str = 'spearman'
) -> np.ndarray:
    """
    Computar correlação entre dois conjuntos de gradientes.
    
    Parameters
    ----------
    gradients1, gradients2 : np.ndarray
        Gradientes a comparar (n_rois, n_components)
    method : str
        'pearson' ou 'spearman'
        
    Returns
    -------
    np.ndarray
        Correlações por componente
    """
    from scipy import stats as sp_stats
    
    n_components = min(gradients1.shape[1], gradients2.shape[1])
    correlations = np.zeros(n_components)
    
    for c in range(n_components):
        if method == 'pearson':
            correlations[c], _ = sp_stats.pearsonr(
                gradients1[:, c], gradients2[:, c]
            )
        elif method == 'spearman':
            correlations[c], _ = sp_stats.spearmanr(
                gradients1[:, c], gradients2[:, c]
            )
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    return correlations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_gradients(
    connectivity: np.ndarray,
    n_components: int = 3
) -> np.ndarray:
    """
    Função rápida para computar gradientes com defaults sensatos.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade
    n_components : int
        Número de componentes
        
    Returns
    -------
    np.ndarray
        Array de gradientes (n_rois, n_components)
    """
    result = compute_gradients(
        connectivity,
        n_components=n_components,
        approach='dm',
        kernel='normalized_angle',
        sparsity=0.9
    )
    return result.gradients


def compare_methods(
    connectivity: np.ndarray,
    n_components: int = 5,
    methods: List[str] = None
) -> Dict[str, GradientResult]:
    """
    Comparar diferentes métodos de gradiente na mesma matriz.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade
    n_components : int
        Número de componentes
    methods : List[str], optional
        Lista de métodos. Default: ['dm', 'pca', 'le']
        
    Returns
    -------
    Dict[str, GradientResult]
        Resultados por método
    """
    if methods is None:
        methods = ['dm', 'pca', 'le']
        if HAS_UMAP:
            methods.append('umap')
    
    results = {}
    
    for method in methods:
        try:
            results[method] = compute_gradients(
                connectivity,
                n_components=n_components,
                approach=method
            )
        except Exception as e:
            warnings.warn(f"Método {method} falhou: {e}")
    
    return results


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == '__main__':
    print("Gradients Core Module")
    print(f"BrainSpace available: {HAS_BRAINSPACE}")
    print(f"UMAP available: {HAS_UMAP}")
    
    # Quick test
    print("\nTesting with random matrix...")
    np.random.seed(42)
    
    n = 100
    test_matrix = np.random.random((n, n))
    test_matrix = (test_matrix + test_matrix.T) / 2
    
    result = compute_gradients(test_matrix, n_components=5)
    
    print(f"Gradients shape: {result.gradients.shape}")
    print(f"Explained variance (G1-G3): {result.explained_variance[:3]}")
    print(f"Total variance (G1-G3): {np.sum(result.explained_variance[:3]):.3f}")
    
    print("\n✓ Test completed!")
