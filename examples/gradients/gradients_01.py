import numpy as np
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres
from nilearn.connectome import ConnectivityMeasure

ts = np.load("/mnt/nvme1n1p1/sars_cov_2_project/data/output/rsfmri/connectivity/schaefer_100/acompcor/sub-09/timeseries.npy")

correlation_measure = ConnectivityMeasure(kind='correlation')
fc_matrix = correlation_measure.fit_transform([ts])[0]

# 2. Construir o gradiente
gm = GradientMaps(
    n_components=10,          # número de gradientes
    approach='dm',            # diffusion mapping (padrão, recomendado)
    kernel='normalized_angle', # ou 'cosine', 'gaussian'
    alignment=None,           # para sujeito único
    random_state=42
)

# Fit no grupo ou sujeito individual
gm.fit(fc_matrix)

# Gradientes estão em:
gradients = gm.gradients_  # shape: (n_parcels, n_components)
lambdas = gm.lambdas_      # eigenvalues

all_subjects = np.loadtxt("/mnt/nvme1n1p1/sars_cov_2_project/data/output/rsfmri/timeseries_fmri_schaefer100.txt", dtype=str)
fc_matrices = [ConnectivityMeasure(kind='correlation').fit_transform([np.load(subject_fc)])[0] for subject_fc in all_subjects] 

gm_aligned = GradientMaps(
    n_components=10,
    approach='dm',
    kernel='normalized_angle',
    alignment='procrustes',  # alinha todos ao primeiro sujeito
    random_state=42
)
gm_aligned.fit(fc_matrices)

# Gradientes alinhados por sujeito:
for i, grad in enumerate(gm_aligned.aligned_):
    print(f"Sujeito {i}: {grad.shape}")

# Opção 2: Joint embedding (mais robusto para grupos)
gm_joint = GradientMaps(
    n_components=10,
    approach='dm',
    kernel='normalized_angle',
    alignment='joint',  # embedding conjunto
    random_state=42
)
gm_joint.fit(fc_matrices)

# Carregar surfaces (conte69 ou fsaverage)
surf_lh, surf_rh = load_conte69()

# Mapear gradiente parcelado de volta ao surface
from brainspace.datasets import load_parcellation
labeling = load_parcellation('schaefer', scale=100, join=True)

# Plotar o primeiro gradiente
grad1_surface = map_to_labels(
    gm.gradients_[:, 0],  # primeiro gradiente
    labeling,
    mask=labeling != 0,
    fill=np.nan
)

plot_hemispheres(
    surf_lh, surf_rh,
    array_name=grad1_surface,
    size=(1200, 400),
    cmap='viridis_r',
    color_bar=True,
    label_text=['Gradient 1'],
    zoom=1.25
)

