import warnings
warnings.filterwarnings('ignore')

from sars.normative_comparison import run_null_model_analysis, plot_null_model_results

results = run_null_model_analysis(
    atlas_name='schaefer_100',
    null_model='maslov_sneppen',
    n_surrogates=1000  # ≥1000 para publicação
)

figs = plot_null_model_results(results)
