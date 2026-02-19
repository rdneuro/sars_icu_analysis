import warnings
warnings.filterwarnings('ignore')

### ONE SUBJECT
from sars.gradients import compute_gradients, quick_gradients
from sars.utils import get_mtx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fc_matrix = get_mtx('sub-07', 'fmri', 'connectivity_correlation')

# Opção 1: Função rápida
g = quick_gradients(fc_matrix, n_components=3)  # Retorna array

# Opção 2: Função completa (retorna GradientResult)
result = compute_gradients(
    fc_matrix,
    n_components=10,
    approach='dm',
    kernel='normalized_angle',
    sparsity=0.9
)

print(f"G1 variance: {result.explained_variance[0]:.3f}")
g1 = result.G1  # Primeiro gradiente

### LOAD AUTOMATICALLY A SUBJECT
from sars.gradients import compute_subject_gradients
result = compute_subject_gradients('sub-01', atlas_name='schaefer_100')

### GROUP WITH ALIGNMENT
from sars.gradients import compute_cohort_gradients

group = compute_cohort_gradients(
    atlas_name='schaefer_100',
    align=True  # Procrustes alignment
)
print(f"Subjects: {group.n_subjects}")
group_mean = group.group_mean


### METHODS COMPARISONS
from sars.gradients import compare_methods

methods = compare_methods(fc_matrix, methods=['dm', 'pca', 'le'])
for name, result in methods.items():
    print(f"{name}: G1 var = {result.explained_variance[0]:.3f}")
    

### FULL PIPELINE WITH NULL MODELS
from sars.gradients import run_gradient_analysis
from sars.gradients.fast import run_gradient_analysis_fast

results0 = run_gradient_analysis(atlas_name='schaefer_100', compare_to_null=True, n_surrogates=100, generate_figures=True)
results1 = run_gradient_analysis_fast(atlas_name='schaefer_100', n_surrogates=50, approach='pca', n_jobs=16, verbose=True)

for atlas in ['schaefer_100', 'brainnetome', 'aal3', 'synthseg']:
    print(f"\n{'='*50}\n{atlas}\n{'='*50}")
    results = run_gradient_analysis_fast(
        atlas_name=atlas,
        n_surrogates=100,
        approach='dm',
        n_jobs=32,
        verbose=True
    )

### VIZUALIZATIONS
import pandas as pd
from sars.gradients import viz

roi_labels = pd.read_csv("/mnt/nvme1n1p1/sars_cov_2_project/info/atlases/labels_schaefer_100_7networks.csv")['label_roi'].tolist()

# Scatter G1 vs G2
viz.plot_gradient_scatter(result.gradients, components=(1, 2))

# 3D plot
viz.plot_gradient_3d(result.gradients, components=(1, 2, 3))

# Scree plot (variância explicada)
viz.plot_scree(result.explained_variance)

# Ranking de regiões
viz.plot_gradient_ranking(result.gradients, roi_labels, component=1)

# Resumo completo
viz.plot_gradient_summary(result.gradients, result.explained_variance)

# Grupo
viz.plot_group_gradients(group.aligned_gradients, group.group_mean)

### NULL GRADIENTS
## null gradients for vizualization (one subject)
from sars.gradients.fast_null import compute_null_gradients_for_viz
from sars.normative_comparison import config

matrix = config.load_connectivity_matrix('sub-01', 'schaefer_100', 'acompcor')
null_result = compute_null_gradients_for_viz(matrix, n_surrogates=100, n_components=3, approach='dm', n_jobs=32)

G1_observed = null_result['G1_observed']       # (100,) - valor G1 por região
G1_null_mean = null_result['G1_null_mean']     # (100,) - média nula por região  
G1_null_all = null_result['G1_null_all']       # (100, 100) - todos os surrogates

## visualizing
from sars.gradients.brain_viz import plot_gradient_comparison_surface

plot_gradient_comparison_surface(observed_gradient=null_result['G1_observed'], null_mean_gradient=null_result['G1_null_mean'], atlas_name='schaefer_100')

## null gradients for entire group
from sars.gradients.fast_null import compute_group_null_gradients_for_viz

null_results = compute_group_null_gradients_for_viz(atlas_name='schaefer_100', n_surrogates=50, approach='pca', n_jobs=32)
group_G1_observed = null_results['group_G1_observed']   # (100,)
group_G1_null_mean = null_results['group_G1_null_mean'] # (100,)

for sub, sub_result in null_results['individual_results'].items():
    print(f"{sub}: G1 range = [{sub_result['G1_observed'].min():.2f}, {sub_result['G1_observed'].max():.2f}]")

### BRAIN VISUALIZATIONS
from sars.gradients.brain_viz import (
    plot_gradient_on_surface,
    plot_gradient_zscore_surface
)

# Gradiente único
plot_gradient_on_surface(result.G1, atlas_name='schaefer_100', view='lateral')

# Comparação observado vs nulo
plot_gradient_comparison_surface(observed_gradient=result.G1, null_mean_gradient=G1_null_mean, atlas_name='schaefer_100')

# Z-scores por região
plot_gradient_zscore_surface(observed_gradient=result.G1, null_gradients=G1_null_all, threshold=2.0)
