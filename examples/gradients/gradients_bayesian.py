import os
import re
import numpy as np
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_group_fc
from nilearn.connectome import ConnectivityMeasure
import pymc as pm
import arviz as az
from scipy.stats import zscore

# =====================================================================
# 1. TEMPLATE NORMATIVO (HCP)
# =====================================================================
fc_hcp = load_group_fc('schaefer', scale=100)

gm_template = GradientMaps(
    n_components=10,
    approach='dm',
    kernel='normalized_angle',
    random_state=42
)
gm_template.fit(fc_hcp)

# =====================================================================
# 2. CARREGAR E PAREAR SUJEITOS (fMRI + DTI)
# =====================================================================
fmri_paths = np.loadtxt(
    "/mnt/nvme1n1p1/sars_cov_2_project/data/output/rsfmri/timeseries_fmri_schaefer100.txt",
    dtype=str
)
dti_paths = np.loadtxt(
    "/mnt/nvme1n1p1/sars_cov_2_project/data/output/diffusion/timeseries_sift2_schaefer100.txt",
    dtype=str
)

# Sujeitos a excluir:
# sub-01: variância rara em FC
# sub-15: NaN em SC
# sub-21: faltou T1w no fMRIPrep (já ausente dos .txt)
# AJUSTAR: adicionar os 2 excluídos por head motion (qual o número deles?)
EXCLUDE = {'sub-01', 'sub-15'}

def get_subject_id(filepath):
    """Extrair sub-XX do path: .../matrices/sub-XX/schaefer100/..."""
    match = re.search(r'(sub-\d+)', filepath)
    return match.group(1) if match else None

fmri_subjects = {get_subject_id(p): p for p in fmri_paths}
dti_subjects = {get_subject_id(p): p for p in dti_paths}

# Interseção, removendo excluídos
common_subjects = sorted(
    (set(fmri_subjects) & set(dti_subjects)) - EXCLUDE
)
print(f"fMRI: {len(fmri_subjects)} | DTI: {len(dti_subjects)}")
print(f"Excluídos: {EXCLUDE}")
print(f"Sujeitos finais: {len(common_subjects)} → {common_subjects}")

# Carregar dados pareados
correlation_measure = ConnectivityMeasure(kind='correlation')

fc_covid = []
structural_connectome = []

for subj in common_subjects:
    # rs-fMRI → FC matrix
    ts = np.load(fmri_subjects[subj])
    fc = correlation_measure.fit_transform([ts])[0]
    fc_covid.append(fc)

    # DTI → structural connectome (log-transform)
    sc = np.load(dti_subjects[subj])
    sc = np.log1p(sc)
    structural_connectome.append(sc)

n_subjects = len(common_subjects)
print(f"\nSujeitos pareados para análise: {n_subjects}")

# =====================================================================
# 3. GRADIENT MAPPING COM ALINHAMENTO AO TEMPLATE HCP
# =====================================================================
gm_covid = GradientMaps(
    n_components=10,
    approach='dm',
    kernel='normalized_angle',
    alignment='procrustes',
    random_state=42
)

# Índice 0 = template HCP, índices 1..n = sujeitos COVID
gm_covid.fit([fc_hcp] + fc_covid)

template_grad = gm_covid.aligned_[0]
covid_grads = [gm_covid.aligned_[i] for i in range(1, n_subjects + 1)]

# =====================================================================
# 4. MÉTRICAS DE DESVIO NORMATIVO
# =====================================================================
deviations = np.array([
    sg[:, 0] - template_grad[:, 0]
    for sg in covid_grads
])  # shape: (n_subjects, 100)

subject_metrics = {
    'spatial_corr': [
        np.corrcoef(template_grad[:, 0], sg[:, 0])[0, 1]
        for sg in covid_grads
    ],
    'mean_abs_deviation': [
        np.mean(np.abs(sg[:, 0] - template_grad[:, 0]))
        for sg in covid_grads
    ],
    'gradient_range': [
        np.ptp(sg[:, 0])
        for sg in covid_grads
    ],
}

print("\n── Métricas de Desvio Normativo (Gradiente 1) ──")
for key, vals in subject_metrics.items():
    print(f"  {key}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

# =====================================================================
# 5. STRUCTURE-FUNCTION COUPLING VIA GRADIENTES
# =====================================================================
gm_structural = GradientMaps(
    n_components=10,
    approach='dm',
    kernel='normalized_angle',
    random_state=42
)

sf_coupling = []
for i in range(n_subjects):
    func_grad = covid_grads[i][:, 0]

    gm_structural.fit(structural_connectome[i])
    struct_grad = gm_structural.gradients_[:, 0]

    r = np.corrcoef(func_grad, struct_grad)[0, 1]
    sf_coupling.append(r)

sf_coupling = np.array(sf_coupling)

print(f"\n── Structure-Function Coupling ──")
print(f"  Mean r: {sf_coupling.mean():.4f}, Std: {sf_coupling.std():.4f}")
print(f"  Range: [{sf_coupling.min():.4f}, {sf_coupling.max():.4f}]")

# =====================================================================
# 6. MODELO BAYESIANO: DESVIO NORMATIVO ~ SF-COUPLING
# =====================================================================
x = zscore(np.array(subject_metrics['mean_abs_deviation']))
y = zscore(sf_coupling)

with pm.Model() as model_multimodal:
    alpha = pm.Normal('alpha', mu=0, sigma=2)
    beta = pm.Normal('beta', mu=0, sigma=2)
    sigma = pm.HalfCauchy('sigma', beta=1)

    mu = alpha + beta * x
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, random_seed=42)

print("\n── Resultado do Modelo Bayesiano ──")
print(az.summary(trace, var_names=['alpha', 'beta', 'sigma'], hdi_prob=0.94))








import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import zscore

# =====================================================================
# 1. MAPEAMENTO PARCELS → REDES YEO 7
# =====================================================================
# No Schaefer-100, os nomes das parcels já contêm a rede Yeo.
# Formato: "7Networks_LH_Vis_1", "7Networks_RH_Default_3", etc.
# Mapeamento: Vis=0, SomMot=1, DorsAttn=2, SalVentAttn=3,
#             Limbic=4, Cont=5, Default=6

from brainspace.datasets import load_parcellation

# Labels do Schaefer-100 (nomes das parcels)
# Se o BrainSpace não der os nomes, usar nilearn:
from nilearn.datasets import fetch_atlas_schaefer_2018
atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
parcel_names = [label.decode() if isinstance(label, bytes) else label 
                for label in atlas.labels]

# Mapear cada parcel à sua rede
network_map = {
    'Vis': 0, 'SomMot': 1, 'DorsAttn': 2, 'SalVentAttn': 3,
    'Limbic': 4, 'Cont': 5, 'Default': 6
}
network_names = list(network_map.keys())

network_idx = []
for name in parcel_names:
    for net_name, net_id in network_map.items():
        if net_name in name:
            network_idx.append(net_id)
            break

network_idx = np.array(network_idx)  # shape: (100,)
n_networks = 7
n_parcels = 100

print("Parcels por rede:")
for name, idx in network_map.items():
    print(f"  {name}: {np.sum(network_idx == idx)} parcels")

# =====================================================================
# 2. PREPARAR DADOS REGIONAIS
# =====================================================================
# Desvio normativo por parcel (já calculado antes)
# deviations: shape (n_subjects, 100)

# SF-coupling POR PARCEL (não global!)
# Precisamos do coupling regional, não de um escalar por sujeito

sf_coupling_regional = np.zeros((n_subjects, n_parcels))

for i in range(n_subjects):
    func_grad = covid_grads[i]  # (100, 10)
    
    gm_structural.fit(structural_connectome[i])
    struct_grad = gm_structural.gradients_  # (100, 10)
    
    # Coupling regional: diferença absoluta entre gradientes
    # nos primeiros 3 componentes (distância no espaço de gradientes)
    for p in range(n_parcels):
        # Correlação entre perfis de gradientes func e struct por parcel
        # Alternativa: distância euclidiana nos primeiros k componentes
        sf_coupling_regional[i, p] = np.abs(
            func_grad[p, 0] - struct_grad[p, 0]
        )

# Padronizar por parcel (across subjects)
X = zscore(deviations, axis=0)        # desvio normativo por parcel
Y = zscore(sf_coupling_regional, axis=0)  # SF-coupling por parcel

# =====================================================================
# 3. MODELO HIERÁRQUICO: EFEITO POR REDE YEO
# =====================================================================
# Formato long para PyMC
# Cada observação = (sujeito, parcel) → Y
subj_idx = np.repeat(np.arange(n_subjects), n_parcels)
parcel_idx = np.tile(np.arange(n_parcels), n_subjects)
net_idx = network_idx[parcel_idx]

X_flat = X.ravel()   # (n_subjects * n_parcels,)
Y_flat = Y.ravel()

# Remover NaNs (parcels com variância zero no zscore)
valid = ~(np.isnan(X_flat) | np.isnan(Y_flat))
X_flat = X_flat[valid]
Y_flat = Y_flat[valid]
subj_idx = subj_idx[valid]
parcel_idx = parcel_idx[valid]
net_idx = net_idx[valid]

print(f"\nObservações válidas: {valid.sum()} / {n_subjects * n_parcels}")

with pm.Model() as model_hierarchical:
    # ── Hyperpriors por rede ──
    # Cada rede tem seu próprio efeito médio e variabilidade
    mu_beta_net = pm.Normal('mu_beta_net', mu=0, sigma=1, shape=n_networks)
    sigma_beta_net = pm.HalfNormal('sigma_beta_net', sigma=0.5, shape=n_networks)
    
    # ── Betas por parcel, hierarquicamente agrupados por rede ──
    beta_parcel = pm.Normal(
        'beta_parcel',
        mu=mu_beta_net[network_idx],
        sigma=sigma_beta_net[network_idx],
        shape=n_parcels
    )
    
    # ── Intercepto aleatório por sujeito ──
    sigma_subj = pm.HalfNormal('sigma_subj', sigma=1)
    alpha_subj = pm.Normal('alpha_subj', mu=0, sigma=sigma_subj, shape=n_subjects)
    
    # ── Residual ──
    sigma = pm.HalfCauchy('sigma', beta=1)
    
    # ── Likelihood ──
    mu = alpha_subj[subj_idx] + beta_parcel[parcel_idx] * X_flat
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=Y_flat)
    
    # Sampling
    trace_hier = pm.sample(
        2000, tune=2000,
        target_accept=0.95,
        random_seed=42,
        cores=4
    )

# =====================================================================
# 4. RESULTADOS
# =====================================================================
# O que mais interessa: mu_beta_net (efeito médio por rede)
print("\n── Efeito por Rede Yeo (mu_beta_net) ──")
summary_net = az.summary(
    trace_hier, var_names=['mu_beta_net'], hdi_prob=0.94
)
summary_net.index = network_names
print(summary_net[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

# Quais redes têm efeito credível? (HDI não cruza zero)
print("\n── Redes com efeito credível (HDI 94% exclui zero) ──")
for net_name, row in summary_net.iterrows():
    credible = "✓" if (row['hdi_3%'] > 0 or row['hdi_97%'] < 0) else "✗"
    direction = "+" if row['mean'] > 0 else "-"
    print(f"  {credible} {net_name}: β={row['mean']:.3f} [{row['hdi_3%']:.3f}, {row['hdi_97%']:.3f}] {direction}")