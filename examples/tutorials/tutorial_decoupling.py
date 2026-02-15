import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# ============================================================
# Imports do sars.config — nomes reais do seu config.py
# ============================================================
from sars.config import (
    ALL_SUBJECT_IDS,
    ATLASES,
    ATLAS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    CONNECTIVITY_DIR,
    TR,
    get_connectivity_path,   # FC: (subject, atlas, kind, strategy) → Path
    get_sc_path,             # SC: (subject, atlas, weight) → Path
    get_timeseries_path,     # Timeseries: (subject, atlas, strategy) → Path
)
from sars.data import (
    symmetrize_matrix,
    normalize_matrix,
    get_upper_triangle,
)

# ============================================================
# Escolha do atlas — trabalhe com um por vez
# ============================================================
atlas_name = "brainnetome"  # 100 ROIs, boa relação custo-benefício
atlas_info = ATLASES[atlas_name]
n_rois = atlas_info['n_rois']   # 100

# ============================================================
# Carregar SC e FC de um sujeito
# ============================================================
subject = "sub-01"

# FC: matriz de correlação de Pearson
fc_path = get_connectivity_path(subject, atlas_name, kind="correlation")
FC_raw = np.load(fc_path)

# SC: matriz de streamline count (pode trocar weight por "fa" se quiser)
sc_path = get_sc_path(subject, atlas_name, weight="sift2")
SC_raw = np.load(sc_path)

print(f"FC path: {fc_path}")
print(f"SC path: {sc_path}")

# ============================================================
# Pré-processamento essencial
# ============================================================

# 1. Simetrizar a SC (tractografia pode produzir assimetrias)
SC = symmetrize_matrix(SC_raw, method="average")

# 2. Remover autoconexões
np.fill_diagonal(SC, 0)
np.fill_diagonal(FC_raw, 0)

# 3. Normalizar SC (importante para comunicabilidade)
SC_norm = normalize_matrix(SC, method="max")

# 4. Log-transform da SC (comprime a distribuição heavy-tailed)
SC_log = np.log1p(SC)

# FC já vem como correlação [-1, 1], não precisa normalizar
FC = FC_raw.copy()

print(f"\nSujeito: {subject}")
print(f"Atlas: {atlas_name} ({n_rois} ROIs)")
print(f"SC shape: {SC.shape}, density: {(SC > 0).sum() / (n_rois*(n_rois-1)):.3f}")
print(f"FC shape: {FC.shape}, range: [{FC.min():.3f}, {FC.max():.3f}]")


def load_all_subjects(atlas_name="brainnetome", sc_weight="sift2"):
    """
    Carrega SC e FC de todos os sujeitos usando os helpers do config.
    
    Parameters
    ----------
    atlas_name : str
        Nome do atlas (chave em ATLASES).
    sc_weight : str
        Tipo de peso da SC: 'streamline_count', 'fa', etc.
    
    Returns
    -------
    dict : {subject_id: {"SC": np.ndarray, "FC": np.ndarray}}
    """
    subjects_data = {}
    
    for sub in ALL_SUBJECT_IDS:
        try:
            fc_path = get_connectivity_path(sub, atlas_name, kind="correlation")
            sc_path = get_sc_path(sub, atlas_name, weight=sc_weight)
            
            fc = np.load(fc_path)
            sc = np.load(sc_path)
            
            sc = symmetrize_matrix(sc, method="average")
            np.fill_diagonal(sc, 0)
            np.fill_diagonal(fc, 0)
            
            subjects_data[sub] = {"SC": sc, "FC": fc}
        except FileNotFoundError as e:
            print(f"  [SKIP] {sub}: {e}")
    
    print(f"Carregados: {len(subjects_data)}/{len(ALL_SUBJECT_IDS)} sujeitos")
    return subjects_data


all_data = load_all_subjects("brainnetome")


def regional_scfc_coupling_baum(SC, FC, min_connections=3):
    """
    SC-FC coupling regional pelo método de Baum et al. (2020, PNAS).
    
    Parameters
    ----------
    SC : np.ndarray (N, N)
        Matriz de conectividade estrutural (streamline count ou FA-weighted).
    FC : np.ndarray (N, N)
        Matriz de conectividade funcional (correlação de Pearson).
    min_connections : int
        Número mínimo de conexões estruturais para computar o coupling.
        Regiões com menos conexões recebem NaN.
    
    Returns
    -------
    coupling : np.ndarray (N,)
        Valor de coupling por região.
    pvalues : np.ndarray (N,)
        P-valor da correlação por região.
    global_coupling : float
        Coupling global (correlação dos triângulos superiores).
    """
    N = SC.shape[0]
    coupling = np.full(N, np.nan)
    pvalues = np.full(N, np.nan)
    
    for i in range(N):
        # Máscara: exclui a própria região + requer conexão estrutural
        mask = (np.arange(N) != i) & (SC[i, :] > 0)
        
        if mask.sum() >= min_connections:
            rho, p = spearmanr(SC[i, mask], FC[i, mask])
            coupling[i] = rho
            pvalues[i] = p
    
    # Coupling global: correlação dos triângulos superiores completos
    sc_triu = get_upper_triangle(SC)
    fc_triu = get_upper_triangle(FC)
    # Para o global, usar apenas edges com SC > 0
    global_mask = sc_triu > 0
    global_coupling, _ = spearmanr(sc_triu[global_mask], fc_triu[global_mask])
    
    return coupling, pvalues, global_coupling


# ============================================================
# Executar para um sujeito
# ============================================================
coupling, pvals, global_c = regional_scfc_coupling_baum(SC, FC)

print(f"\nSC-FC Coupling (Baum) — {subject}")
print(f"  Global coupling: {global_c:.4f}")
print(f"  Regional: mean={np.nanmean(coupling):.4f}, "
      f"std={np.nanstd(coupling):.4f}")
print(f"  Range: [{np.nanmin(coupling):.4f}, {np.nanmax(coupling):.4f}]")
print(f"  Regiões válidas: {np.sum(~np.isnan(coupling))}/{len(coupling)}")


def compute_coupling_all_subjects(all_data):
    """Computa Baum coupling para todos os sujeitos."""
    results = {}
    
    for sub, data in all_data.items():
        coupling_s, pvals_s, global_c = regional_scfc_coupling_baum(
            data["SC"], data["FC"]
        )
        results[sub] = {
            "regional": coupling_s,
            "pvalues": pvals_s,
            "global": global_c
        }
    
    # DataFrame de coupling regional (sujeitos × ROIs)
    regional_df = pd.DataFrame(
        {sub: r["regional"] for sub, r in results.items()}
    ).T
    regional_df.columns = [f"ROI_{i:03d}" for i in range(regional_df.shape[1])]
    
    # DataFrame de coupling global
    global_df = pd.DataFrame(
        {"subject": list(results.keys()),
         "global_coupling": [r["global"] for r in results.values()]}
    )
    
    return results, regional_df, global_df


results_baum, regional_df, global_df = compute_coupling_all_subjects(all_data)

print("\n=== Coupling Global por Sujeito ===")
print(global_df.describe())



from scipy.spatial.distance import pdist, squareform
from scipy.linalg import expm
from sklearn.linear_model import LinearRegression
import networkx as nx

def compute_communication_features(SC, roi_coords=None):
    """
    Computa preditores de comunicação a partir do grafo SC.
    
    Parameters
    ----------
    SC : np.ndarray (N, N)
        Conectividade estrutural.
    roi_coords : np.ndarray (N, 3), optional
        Coordenadas MNI dos ROIs (para distância euclidiana).
    
    Returns
    -------
    features : dict
        'shortest_path': matriz NxN de shortest path lengths
        'communicability': matriz NxN de communicability
        'euclidean': matriz NxN de distâncias euclidianas (se coords fornecidas)
    """
    N = SC.shape[0]
    features = {}
    
    # 1. Shortest path length
    # Converter pesos para "distâncias" (inverso do peso)
    SC_dist = SC.copy()
    SC_dist[SC_dist > 0] = 1.0 / SC_dist[SC_dist > 0]
    G = nx.from_numpy_array(SC_dist)
    
    sp_dict = dict(nx.all_pairs_dijkstra_path_length(G))
    sp_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sp_matrix[i, j] = sp_dict[i].get(j, np.inf)
    # Substituir inf por max finito (regiões desconectadas)
    finite_mask = np.isfinite(sp_matrix)
    if finite_mask.any():
        sp_matrix[~finite_mask] = sp_matrix[finite_mask].max() * 2
    features["shortest_path"] = sp_matrix
    
    # 2. Communicability (matrix exponential)
    # Normalizar pelo spectral radius para estabilidade numérica
    sc_norm = normalize_matrix(SC, method="spectral")
    features["communicability"] = expm(sc_norm)
    
    # 3. Distância euclidiana (se coordenadas disponíveis)
    if roi_coords is not None:
        features["euclidean"] = squareform(pdist(roi_coords, metric="euclidean"))
    
    return features


def regional_scfc_coupling_vazquez(SC, FC, roi_coords=None, min_connections=3):
    """
    SC-FC coupling pelo método de Vázquez-Rodríguez et al. (2019, PNAS).
    
    Parameters
    ----------
    SC : np.ndarray (N, N)
        Conectividade estrutural.
    FC : np.ndarray (N, N)
        Conectividade funcional.
    roi_coords : np.ndarray (N, 3), optional
        Coordenadas MNI (centróides dos ROIs).
    min_connections : int
        Mínimo de preditores válidos para computar.
    
    Returns
    -------
    coupling : np.ndarray (N,)
        R² ajustado por região.
    betas : np.ndarray (N, n_features)
        Coeficientes de regressão por região.
    feature_names : list of str
        Nomes dos preditores usados.
    """
    N = SC.shape[0]
    
    # Computar features de comunicação
    features = compute_communication_features(SC, roi_coords)
    feature_names = list(features.keys())
    n_feat = len(feature_names)
    
    coupling = np.full(N, np.nan)
    betas = np.full((N, n_feat), np.nan)
    
    for i in range(N):
        mask = (np.arange(N) != i)
        
        # Montar matriz X de preditores
        X = np.column_stack([features[f][i, mask] for f in feature_names])
        y = FC[i, mask]
        
        # Remover linhas com inf ou nan
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        if valid.sum() < min_connections + n_feat:
            continue
        
        X_valid = X[valid]
        y_valid = y[valid]
        
        # Regressão linear
        reg = LinearRegression().fit(X_valid, y_valid)
        
        # R² ajustado
        r2 = reg.score(X_valid, y_valid)
        n = len(y_valid)
        p = n_feat
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        coupling[i] = r2_adj
        betas[i, :] = reg.coef_
    
    return coupling, betas, feature_names


# ============================================================
# Executar
# ============================================================

# Carregar coordenadas do atlas (centróide MNI de cada ROI), se disponíveis
# roi_coords = np.load(ATLAS_DIR / f"{atlas_name}_coords_mni.npy")
# Se não tiver coordenadas, passe None (exclui distância euclidiana)
roi_coords = None

coupling_vr, betas_vr, feat_names = regional_scfc_coupling_vazquez(
    SC, FC, roi_coords=roi_coords
)

print(f"\nSC-FC Coupling (Vázquez-Rodríguez) — {subject}")
print(f"  R² ajustado: mean={np.nanmean(coupling_vr):.4f}, "
      f"std={np.nanstd(coupling_vr):.4f}")
print(f"  Features usadas: {feat_names}")
print(f"  Regiões válidas: {np.sum(~np.isnan(coupling_vr))}/{n_rois}")

# Importância relativa dos preditores
for i, name in enumerate(feat_names):
    print(f"  β_{name}: mean={np.nanmean(betas_vr[:, i]):.4f}")
    
    
from scipy.linalg import eigh

def compute_graph_laplacian(SC, normalized=True):
    """
    Computa o Laplaciano do grafo SC.
    
    Parameters
    ----------
    SC : np.ndarray (N, N)
        Conectividade estrutural (pesos positivos).
    normalized : bool
        Se True, retorna o Laplaciano normalizado simétrico.
    
    Returns
    -------
    L : np.ndarray (N, N)
        Laplaciano.
    eigenvalues : np.ndarray (N,)
        Autovalores ordenados.
    eigenvectors : np.ndarray (N, N)
        Autovetores correspondentes (colunas).
    """
    W = SC.copy()
    np.fill_diagonal(W, 0)
    W[W < 0] = 0  # SDI requer pesos positivos
    
    D = np.diag(W.sum(axis=1))
    
    if normalized:
        # D^{-1/2}
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-10)))
        L = np.eye(len(W)) - d_inv_sqrt @ W @ d_inv_sqrt
    else:
        L = D - W
    
    # Decomposição em autovalores (eigh garante autovalores reais e ordenados)
    eigenvalues, eigenvectors = eigh(L)
    
    return L, eigenvalues, eigenvectors


def structural_decoupling_index(SC, FC, split="median"):
    """
    Computa o Structural Decoupling Index (SDI) de Preti & Van De Ville (2019).
    
    Parameters
    ----------
    SC : np.ndarray (N, N)
        Conectividade estrutural.
    FC : np.ndarray (N, N)
        Conectividade funcional (usada como sinais no grafo).
    split : str or int
        Como dividir os autovetores:
        - "median": metade inferior vs superior (N/2)
        - "mean_eigenvalue": split pelo autovalor médio
        - int: índice de corte explícito
    
    Returns
    -------
    sdi : np.ndarray (N,)
        SDI por região.
    coupled_energy : np.ndarray (N,)
        Energia dos componentes de baixa frequência.
    decoupled_energy : np.ndarray (N,)
        Energia dos componentes de alta frequência.
    eigenvalues : np.ndarray (N,)
        Autovalores do Laplaciano.
    """
    N = SC.shape[0]
    
    # 1. Laplaciano e decomposição espectral
    L, eigenvalues, U = compute_graph_laplacian(SC, normalized=True)
    
    # 2. Determinar ponto de corte
    if split == "median":
        k = N // 2
    elif split == "mean_eigenvalue":
        k = np.searchsorted(eigenvalues, eigenvalues.mean())
    elif isinstance(split, int):
        k = split
    else:
        raise ValueError(f"Split method não reconhecido: {split}")
    
    U_low = U[:, :k]    # Autovetores de baixa frequência (coupled)
    U_high = U[:, k:]    # Autovetores de alta frequência (decoupled)
    
    # 3. Projetar cada coluna da FC no espaço espectral do grafo
    # Usamos as colunas da FC como sinais (perfil funcional de cada região)
    sdi = np.zeros(N)
    coupled_energy = np.zeros(N)
    decoupled_energy = np.zeros(N)
    
    for i in range(N):
        signal = FC[:, i]  # Perfil funcional da região i
        
        # GFT: projetar no espaço dos autovetores
        coeff_low = U_low.T @ signal
        coeff_high = U_high.T @ signal
        
        # Reconstruir componentes
        x_low = U_low @ coeff_low   # Componente coupled
        x_high = U_high @ coeff_high  # Componente decoupled
        
        # Energia por nó (valor ao quadrado daquela região)
        e_low = x_low[i] ** 2
        e_high = x_high[i] ** 2
        
        coupled_energy[i] = e_low
        decoupled_energy[i] = e_high
        
        # SDI: log-ratio das energias
        if e_low > 0:
            sdi[i] = np.log2(e_high / e_low)
        else:
            sdi[i] = np.nan  # Região isolada
    
    return sdi, coupled_energy, decoupled_energy, eigenvalues


# ============================================================
# Executar
# ============================================================
sdi, e_coupled, e_decoupled, eigvals = structural_decoupling_index(SC, FC)

print(f"\nSDI (Preti & Van De Ville) — {subject}")
print(f"  SDI: mean={np.nanmean(sdi):.4f}, std={np.nanstd(sdi):.4f}")
print(f"  Decoupled (SDI > 0): {np.sum(sdi > 0)} regiões")
print(f"  Coupled (SDI < 0): {np.sum(sdi < 0)} regiões")
print(f"  Spectral gap (λ₁): {eigvals[1]:.4f}")


def compute_sdi_all_subjects(all_data):
    """Computa SDI para todos os sujeitos."""
    sdi_all = {}
    spectral_gaps = {}
    
    for sub, data in all_data.items():
        sdi_s, _, _, eigvals = structural_decoupling_index(
            data["SC"], data["FC"]
        )
        sdi_all[sub] = sdi_s
        spectral_gaps[sub] = eigvals[1]  # Primeiro autovalor não-trivial
    
    sdi_df = pd.DataFrame(sdi_all).T
    sdi_df.columns = [f"ROI_{i:03d}" for i in range(sdi_df.shape[1])]
    
    return sdi_df, spectral_gaps

sdi_df, spectral_gaps = compute_sdi_all_subjects(all_data)

# O spectral gap indexa a "integridade" da comunidade estrutural
print("\n=== Spectral Gap (λ₁) por Sujeito ===")
for sub, gap in spectral_gaps.items():
    print(f"  {sub}: {gap:.4f}")
    
def multilayer_scfc_analysis(SC, FC, hub_threshold_percentile=80):
    """
    Análise multilayer SC+FC com identificação de hubs cross-modal.
    
    Parameters
    ----------
    SC, FC : np.ndarray (N, N)
    hub_threshold_percentile : float
        Percentil para classificar nós como hubs.
    
    Returns
    -------
    results : dict com métricas multilayer
    """
    N = SC.shape[0]
    
    # 1. Correlação inter-layer por nó
    interlayer_corr = np.zeros(N)
    for i in range(N):
        mask = np.arange(N) != i
        sc_profile = SC[i, mask]
        fc_profile = FC[i, mask]
        if sc_profile.std() > 0 and fc_profile.std() > 0:
            interlayer_corr[i] = np.corrcoef(sc_profile, fc_profile)[0, 1]
    
    # 2. Strength em cada layer
    sc_strength = SC.sum(axis=1)
    fc_strength = np.abs(FC).sum(axis=1)
    
    # 3. Identificar hubs em cada layer
    sc_hub_thresh = np.percentile(sc_strength, hub_threshold_percentile)
    fc_hub_thresh = np.percentile(fc_strength, hub_threshold_percentile)
    
    sc_hubs = sc_strength >= sc_hub_thresh
    fc_hubs = fc_strength >= fc_hub_thresh
    cross_modal_hubs = sc_hubs & fc_hubs
    
    # 4. Hubs exclusivos de cada layer
    sc_only_hubs = sc_hubs & ~fc_hubs
    fc_only_hubs = fc_hubs & ~sc_hubs
    
    # 5. Correlação global (triângulos superiores)
    sc_triu = get_upper_triangle(SC)
    fc_triu = get_upper_triangle(FC)
    global_corr = np.corrcoef(sc_triu, fc_triu)[0, 1]
    
    return {
        "interlayer_correlation": interlayer_corr,
        "global_correlation": global_corr,
        "sc_strength": sc_strength,
        "fc_strength": fc_strength,
        "sc_hubs": sc_hubs,
        "fc_hubs": fc_hubs,
        "cross_modal_hubs": cross_modal_hubs,
        "sc_only_hubs": sc_only_hubs,
        "fc_only_hubs": fc_only_hubs,
        "n_cross_modal": int(cross_modal_hubs.sum()),
    }


ml_results = multilayer_scfc_analysis(SC, FC)

print(f"\nMultilayer Analysis — {subject}")
print(f"  Global SC-FC correlation: {ml_results['global_correlation']:.4f}")
print(f"  Cross-modal hubs: {ml_results['n_cross_modal']} regiões")
print(f"  Interlayer corr: mean={ml_results['interlayer_correlation'].mean():.4f}")

def compare_coupling_methods(SC, FC, roi_coords=None):
    """
    Executa os 4 métodos e compara os resultados.
    """
    # 1. Baum
    coupling_baum, _, global_baum = regional_scfc_coupling_baum(SC, FC)
    
    # 2. Vázquez-Rodríguez
    coupling_vr, _, _ = regional_scfc_coupling_vazquez(SC, FC, roi_coords)
    
    # 3. SDI
    sdi_c, _, _, _ = structural_decoupling_index(SC, FC)
    
    # 4. Multilayer
    ml = multilayer_scfc_analysis(SC, FC)
    coupling_ml = ml["interlayer_correlation"]
    
    # DataFrame comparativo
    comparison = pd.DataFrame({
        "Baum (ρ)": coupling_baum,
        "Vázquez-R (R²)": coupling_vr,
        "SDI": sdi_c,
        "Multilayer (r)": coupling_ml
    })
    
    print("\n=== Correlação entre Métodos (Spearman) ===")
    corr_matrix = comparison.corr(method="spearman")
    print(corr_matrix.round(3))
    
    return comparison

comp = compare_coupling_methods(SC, FC)


def extract_clinical_features(all_data):
    """
    Extrai features de SC-FC coupling para correlação clínica.
    Retorna DataFrame com uma linha por sujeito e colunas de features.
    """
    rows = []
    
    for sub, data in all_data.items():
        SC_s, FC_s = data["SC"], data["FC"]
        n = SC_s.shape[0]
        
        # Baum
        c_baum, _, g_baum = regional_scfc_coupling_baum(SC_s, FC_s)
        
        # SDI
        sdi_s, e_c, e_d, eigvals = structural_decoupling_index(SC_s, FC_s)
        
        # Multilayer
        ml = multilayer_scfc_analysis(SC_s, FC_s)
        
        row = {
            "subject": sub,
            # --- Coupling global ---
            "global_coupling_baum": g_baum,
            "global_correlation_ml": ml["global_correlation"],
            "spectral_gap": eigvals[1],
            
            # --- Coupling regional (estatísticas resumo) ---
            "coupling_baum_mean": np.nanmean(c_baum),
            "coupling_baum_std": np.nanstd(c_baum),
            "coupling_baum_median": np.nanmedian(c_baum),
            
            # --- SDI ---
            "sdi_mean": np.nanmean(sdi_s),
            "sdi_std": np.nanstd(sdi_s),
            "n_decoupled_regions": int(np.nansum(sdi_s > 0)),
            "n_coupled_regions": int(np.nansum(sdi_s < 0)),
            "sdi_skewness": float(pd.Series(sdi_s).skew()),
            
            # --- Multilayer ---
            "interlayer_corr_mean": ml["interlayer_correlation"].mean(),
            "n_cross_modal_hubs": int(ml["n_cross_modal"]),
            "n_sc_only_hubs": int(ml["sc_only_hubs"].sum()),
            "n_fc_only_hubs": int(ml["fc_only_hubs"].sum()),
            
            # --- SC properties (contexto) ---
            "sc_density": float((SC_s > 0).sum() / (n * (n - 1))),
            "sc_mean_strength": float(SC_s.sum(axis=1).mean()),
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


features_df = extract_clinical_features(all_data)

# Salvar
save_dir = OUTPUTS_DIR / "sars" / "scfc_decoupling" / atlas_name
save_dir.mkdir(parents=True, exist_ok=True)
features_df.to_csv(save_dir / "scfc_features_per_subject.csv", index=False)

print("\n=== Features Extraídas ===")
print(features_df.head())
print(f"\nShape: {features_df.shape}")
print(f"\nDescrição:")
print(features_df.describe().round(4))


import matplotlib.pyplot as plt
import seaborn as sns

def plot_sc_fc_scatter(SC, FC, subject_id, atlas_name, save_path=None):
    """
    Scatter plot de SC vs FC (edges), reproduzindo a Fig. 6 de Sporns (2013).
    """
    sc_triu = get_upper_triangle(SC)
    fc_triu = get_upper_triangle(FC)
    has_sc = sc_triu > 0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(sc_triu, fc_triu, s=1, alpha=0.3, c="steelblue", rasterized=True)
    rho_all, _ = spearmanr(sc_triu, fc_triu)
    ax.set_xlabel("Structural Connectivity (streamline count)", fontsize=11)
    ax.set_ylabel("Functional Connectivity (Pearson r)", fontsize=11)
    ax.set_title(f"All edges — ρ = {rho_all:.3f}", fontsize=12)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    
    ax = axes[1]
    ax.scatter(np.log1p(sc_triu[has_sc]), fc_triu[has_sc], 
               s=3, alpha=0.4, c="coral", rasterized=True)
    rho_sc, _ = spearmanr(sc_triu[has_sc], fc_triu[has_sc])
    ax.set_xlabel("log(1 + SC)", fontsize=11)
    ax.set_ylabel("Functional Connectivity (Pearson r)", fontsize=11)
    ax.set_title(f"SC > 0 edges only — ρ = {rho_sc:.3f}", fontsize=12)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    
    fig.suptitle(f"SC-FC Relationship — {subject_id} ({atlas_name})", 
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

fig_dir = FIGURES_DIR / "scfc_decoupling"
fig_dir.mkdir(parents=True, exist_ok=True)
plot_sc_fc_scatter(SC, FC, subject, atlas_name,
                   save_path=fig_dir / f"{subject}_sc_fc_scatter.png")

from sars.config import NETWORK_COLORS

def plot_coupling_by_network(coupling_values, atlas_name="schaefer_100", 
                              method_name="Baum", save_path=None):
    """Bar plot de coupling médio por rede do Yeo 7-networks."""
    network_means, _ = coupling_by_network(coupling_values, atlas_name)
    
    # Cores das 7 networks do Yeo
    yeo_colors = {
        "Vis": "#781286",
        "SomMot": "#4682B4",
        "DorsAttn": "#00760E",
        "SalVentAttn": "#C43AFA",
        "Limbic": "#DCF8A4",
        "Cont": "#E69422",
        "Default": "#CD3E4E"
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    networks = network_means.index.tolist()
    means = network_means["mean"].values
    stds = network_means["std"].values
    colors = [yeo_colors.get(n, "gray") for n in networks]
    
    ax.bar(range(len(networks)), means, yerr=stds, 
           color=colors, edgecolor="black", linewidth=0.5,
           capsize=3, error_kw={"lw": 1})
    
    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels(networks, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(f"SC-FC Coupling ({method_name})", fontsize=11)
    ax.set_title("SC-FC Coupling by Functional Network", fontsize=13)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

plot_coupling_by_network(coupling, "schaefer_100", "Baum")

def plot_coupling_heatmap(regional_df, method_name="Baum", save_path=None):
    """
    Heatmap: sujeitos × ROIs mostrando o coupling regional.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(
        regional_df.values,
        cmap="RdBu_r", center=0,
        xticklabels=False,
        yticklabels=regional_df.index,
        ax=ax,
        cbar_kws={"label": f"SC-FC Coupling ({method_name})"}
    )
    
    ax.set_xlabel(f"ROIs ({regional_df.shape[1]} regions)", fontsize=11)
    ax.set_ylabel("Subject", fontsize=11)
    ax.set_title(f"Regional SC-FC Coupling — {method_name} method", fontsize=13)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

plot_coupling_heatmap(regional_df, "Baum")