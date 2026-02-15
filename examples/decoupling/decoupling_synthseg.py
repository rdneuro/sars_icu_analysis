# Exemplo de uso com atlas Brainnetome (246 regiões)
import numpy as np
import pandas as pd
from sars.utils import get_mtx, get_lsubs
from sars.graph_analysis.sc_fc_decoupling import sc_fc_decoupling_analysis, create_decoupling_dataframe
from sars.graph_analysis.viz_decoupling_graphs import plot_sc_fc_decoupling_summary

node_labels = pd.read_csv('/mnt/nvme1n1p1/sars_cov_2_project/info/atlases/synthseg_mni_labels.tsv', delimiter='\t')['name'].tolist()
ls_df, ls_10, ls_20 = [], [], []
df_10, df_20 = pd.DataFrame(columns=['sub', 'label', 'decoupling_index', 'ratio_local_efficiency']), pd.DataFrame(columns=['sub', 'label', 'decoupling_index', 'ratio_local_efficiency'])

for sub in get_lsubs():
    sc_matrix = get_mtx(sub, 'dmri', 'connectivity_sift2', atlas='synthseg')
    fc_matrix = get_mtx(sub, 'fmri', 'connectivity_correlation', atlas='synthseg')  

    results = sc_fc_decoupling_analysis(
        sc_matrix=sc_matrix,
        fc_matrix=fc_matrix,
        density=0.15,  # Manter 15% das conexões mais fortes
        method='proportional',
        compute_efficiency_ratio=True,
        compute_hub_analysis=True
    )
    df = create_decoupling_dataframe(results, node_labels)
    ls_df.append(df)
    fig = plot_sc_fc_decoupling_summary(results, node_labels=node_labels, save_path=f"/mnt/nvme1n1p1/sars_cov_2_project/figs/decoupling_graph/synthseg/{sub}_decoupling.png")

    ls_10.append(df[['label']].head(10))
    ls_20.append(df[['label']].head(20))
    
    temp10 = df[['label', 'decoupling_index', 'ratio_local_efficiency']].head(10)
    temp10.insert(0, 'sub', np.repeat(sub, 10).tolist())
    temp20 = df[['label', 'decoupling_index', 'ratio_local_efficiency']].head(20)
    temp20.insert(0, 'sub', np.repeat(sub, 20).tolist())

    df_10 = pd.concat((df_10, temp10), axis=0)
    df_20 = pd.concat((df_20, temp20), axis=0)



a, b = np.unique(ls_10, return_counts=True)
un_10 = pd.DataFrame(dict(roi=a, cnt=b))
a, b = np.unique(ls_20, return_counts=True)
un_20 = pd.DataFrame(dict(roi=a, cnt=b))

un_10 = un_10.sort_values('cnt', ascending=False)
un_20 = un_20.sort_values('cnt', ascending=False)
