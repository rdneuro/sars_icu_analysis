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