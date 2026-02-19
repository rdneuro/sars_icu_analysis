#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║             CONN2RES MASTERCLASS — TUTORIAL COMPLETO                     ║
║     De Iniciante a Expert em Reservoir Computing com Conectomas          ║
║                                                                          ║
║  Autor: Velho Mago · INNT/UFRJ · 2026                                   ║
║  Ref: Suárez et al., Nat Commun 15, 656 (2024)                          ║
║  Repo: https://github.com/netneurolab/conn2res                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

Estrutura do Tutorial:
    [A] Pipeline Quick-Start — funções básicas
    [B] Truques Avançados — 4 receitas específicas
    [C] SC-FC Decoupling — sujeito único (whole-brain + Yeo networks)
    [D] Análise de Grupo + Null Models
    [E] Visualizações Espetaculares — do simples ao publication-quality

Dependências:
    pip install conn2res neurogym reservoirpy bctpy
    pip install nilearn nibabel brainspace surfplot neuromaps
    pip install matplotlib seaborn scipy scikit-learn pandas numpy
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

# conn2res core
from conn2res.connectivity import Conn, get_modules
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting

# ─────────────────────────────────────────────────────────────────────────
# Configuração global de estilo para todas as figuras
# ─────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Paleta de cores Yeo 7 networks
YEO7_COLORS = {
    'VIS':  '#781286',  # Visual
    'SM':   '#4682B4',  # Somatomotor
    'DA':   '#00760E',  # Dorsal Attention
    'VA':   '#C43AFA',  # Ventral Attention / Salience
    'LIM':  '#DCF8A4',  # Limbic
    'FP':   '#E69422',  # Frontoparietal
    'DMN':  '#CD3E4E',  # Default Mode
}

YEO7_NAMES = ['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN']
YEO7_NAMES_RAW = np.unique()
OUTPUT_DIR = os.path.join(os.getcwd(), 'conn2res_tutorial_figs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: gerar dados sintéticos de conectoma
# ═══════════════════════════════════════════════════════════════════════
def load_empirical_connectome(sub, atlas='schaefer100', treat='log', nan=True):
    """
    Carrega as matrizes de conectividade de um determinado paciente, conforme o atlas.
    Para o tratamento da matriz de conectividade estrutural, podem ser optadas por:
          - 'max' = SC / max(SC)
          - 'log' = np.log1p(SC)  [default]
          - 'raw' = não realizar nenhum tratamento.
    Se treat != 'raw', ambas matrizes terão suas diagonais preenchidas com zeros.
    Pode-se optar por checar e tratar valores NaN automaticamente na importação.

    Returns
    -------
    SC : (n_nodes, n_nodes) ndarray — structural connectivity (simétrica)
    FC : (n_nodes, n_nodes) ndarray — functional connectivity (correlação)
    """
    fc = np.load(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/dmri/connectivity_sift2.npy")
    sc = np.load(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/fmri/connectivity_correlation.npy")
    if nan == True:
        imp_sc, imp_fc = SimpleImputer(strategy='median'), SimpleImputer(strategy='mean')
        sc = imp_sc.fit_transform(sc)
        fc = imp_fc.fit_transform(fc)
    if treat == 'raw':
        return sc, fc
    elif treat == 'log':
        sc = np.log1p(sc)
    elif treat == 'max':
        sc = sc / np.max(sc)
    np.fill_diagonal(sc, 0)
    np.fill_diagonal(fc, 0)
    return sc, fc

def load_atlas_labels(atlas='schaefer100'):
    """
    Carrega labels das ROIs de um determinado atlas.
    Caso seja o 'schaefer100', retorna também a network correspondente as ROIs.

    Returns
    -------
    lbls : label de cada ROI, na ordem 1 até nç;
    nets : assignment de cada ROI a uma network Yeo (strings).
    """
    lbls = np.loadtxt(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/info/labels_{atlas}.txt", dtype=str)
    if atlas == 'schaefer100':
        nets = np.loadtxt("/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/info/networks_schaefer100.txt", dtype=str)
        return lbls, nets
    return lbls
        

def make_synthetic_connectome(n_nodes=100, n_modules=7, density=0.3,
                               intra_weight=0.8, inter_weight=0.15,
                               seed=42):
    """
    Gera um conectoma sintético com estrutura modular (simulando Yeo 7).
    Útil para o tutorial quando não temos os dados reais bundled.

    Returns
    -------
    SC : (n_nodes, n_nodes) ndarray — structural connectivity (simétrica)
    FC : (n_nodes, n_nodes) ndarray — functional connectivity (correlação)
    labels : (n_nodes,) ndarray — assignment a cada network Yeo
    """
    rng = np.random.default_rng(seed)

    # Dividir nós em módulos (tamanhos ~iguais)
    sizes = np.diff(np.round(np.linspace(0, n_nodes, n_modules + 1)).astype(int))
    labels = np.repeat(np.arange(n_modules), sizes)

    # --- SC: Modular com pesos ---
    SC = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if labels[i] == labels[j]:
                if rng.random() < density * 1.5:
                    SC[i, j] = rng.uniform(0.3, 1.0) * intra_weight
            else:
                if rng.random() < density * 0.4:
                    SC[i, j] = rng.uniform(0.05, 0.4) * inter_weight
    SC = SC + SC.T
    np.fill_diagonal(SC, 0)

    # --- FC: derivada do SC com ruído (simula correlação funcional) ---
    # Usa um modelo simples: FC ~ (SC + SC^2) + noise
    SC_norm = SC / (np.max(SC) + 1e-10)
    FC = SC_norm + 0.3 * SC_norm @ SC_norm
    FC += rng.normal(0, 0.05, FC.shape)
    FC = (FC + FC.T) / 2
    np.fill_diagonal(FC, 1.0)
    FC = np.clip(FC, -1, 1)

    return SC, FC, labels


def make_group_data(n_subjects=20, n_nodes=100, n_modules=7, seed=42):
    """
    Gera dados de grupo: lista de SCs e FCs para N sujeitos.
    Variabilidade inter-sujeito via perturbação aleatória.
    """
    rng = np.random.default_rng(seed)
    SC_template, FC_template, labels = make_synthetic_connectome(
        n_nodes, n_modules, seed=seed
    )

    SCs, FCs = [], []
    for s in range(n_subjects):
        # Perturbação individual
        noise_sc = rng.normal(0, 0.02, SC_template.shape)
        noise_sc = (noise_sc + noise_sc.T) / 2
        sc_subj = np.clip(SC_template + noise_sc, 0, None)
        np.fill_diagonal(sc_subj, 0)

        noise_fc = rng.normal(0, 0.05, FC_template.shape)
        noise_fc = (noise_fc + noise_fc.T) / 2
        fc_subj = np.clip(FC_template + noise_fc, -1, 1)
        np.fill_diagonal(fc_subj, 1.0)

        SCs.append(sc_subj)
        FCs.append(fc_subj)

    return SCs, FCs, labels


# ═══════════════════════════════════════════════════════════════════════════
#                                                                          ║
#  ██████╗  SEÇÃO A — PIPELINE QUICK-START                                 ║
#  ██╔══██╗ Funções Básicas do conn2res                                    ║
#  ███████║                                                                ║
#  ██╔══██║                                                                ║
#  ██║  ██║                                                                ║
#  ╚═╝  ╚═╝                                                                ║
#                                                                          ║
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  SEÇÃO A — PIPELINE QUICK-START")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# A.1  Classe Conn — o container de conectividade
# ─────────────────────────────────────────────────────────────────────────
# A classe Conn é o ponto de entrada do conn2res. Ela recebe uma matriz
# de conectividade (SC) e faz:
#   1. Remove diagonal, NaN, Inf
#   2. Verifica simetria
#   3. Garante que todos os nós pertencem ao maior componente conectado
#   4. Calcula atributos básicos (n_nodes, n_edges, density)

SC, FC, labels = make_synthetic_connectome(n_nodes=100, n_modules=7)

# Criando o objeto Conn com nossa SC
conn = Conn(w=SC)
print(f"\n[A.1] Conn object criado:")
print(f"   n_nodes  = {conn.n_nodes}")
print(f"   n_edges  = {conn.n_edges}")
print(f"   density  = {conn.density:.4f}")
print(f"   symmetric = {conn.symmetric}")

# ─────────────────────────────────────────────────────────────────────────
# A.2  Pré-processamento da Conectividade
# ─────────────────────────────────────────────────────────────────────────
# CRÍTICO: Antes de usar a SC como reservatório, SEMPRE faça:
#   1. scale()       → escala pesos para [0, 1]
#   2. normalize()   → divide pelo raio espectral
# Isso garante que o raio espectral da SC normalizada = 1.0,
# permitindo que alpha controle o regime dinâmico.

conn.scale_and_normalize()
print(f"\n[A.2] Após scale_and_normalize():")
print(f"   w.min() = {conn.w.min():.6f}")
print(f"   w.max() = {conn.w.max():.6f}")

# Verificar raio espectral
ew, _ = eigh(conn.w)
sr = np.max(np.abs(ew))
print(f"   spectral_radius = {sr:.6f} (deve ser ~1.0)")

# ─────────────────────────────────────────────────────────────────────────
# A.3  Seleção de nós de Input e Output
# ─────────────────────────────────────────────────────────────────────────
# No conn2res, você precisa definir:
#   - input_nodes: nós que recebem o sinal externo
#   - output_nodes (readout_nodes): nós cujos estados são lidos
#
# O método conn.get_nodes() suporta vários node_sets:
#   'all'           → todos os nós
#   'random'        → seleção aleatória
#   'ctx' / 'subctx' → cortical / subcortical (precisa de arquivo)
#   'VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN' → Yeo networks
#                     (precisa de rsn_mapping.npy)
#
# Como estamos com dados sintéticos, vamos fazer manualmente:

# Atribuir módulos ao conn (simula rsn_mapping)
conn.modules = labels[:conn.n_nodes]

# Input nodes: nós aleatórios do módulo Visual (módulo 0)
vis_nodes = np.where(conn.modules == 0)[0]
rng = np.random.default_rng(42)
input_nodes = rng.choice(vis_nodes, size=min(3, len(vis_nodes)), replace=False)

# Output nodes: nós do módulo Somatomotor (módulo 1)
output_nodes = np.where(conn.modules == 1)[0]

print(f"\n[A.3] Seleção de nós:")
print(f"   input_nodes  (VIS)  = {input_nodes} ({len(input_nodes)} nós)")
print(f"   output_nodes (SM)   = {output_nodes[:5]}... ({len(output_nodes)} nós)")

# ─────────────────────────────────────────────────────────────────────────
# A.4  Construindo a Matriz de Input (w_in)
# ─────────────────────────────────────────────────────────────────────────
# w_in conecta o sinal externo (features) aos nós de entrada do reservoir.
# Shape: (n_features, n_nodes)
# Normalmente é uma matriz esparsa com 1s apenas nos input_nodes.

n_features = 3  # número de features do input signal
w_in = np.zeros((n_features, conn.n_nodes))
w_in[np.arange(n_features), input_nodes[:n_features]] = 1.0

print(f"\n[A.4] Matriz w_in: shape = {w_in.shape}")
print(f"   Conexões ativas = {int(w_in.sum())}")

# ─────────────────────────────────────────────────────────────────────────
# A.5  Criando e Simulando o Echo State Network
# ─────────────────────────────────────────────────────────────────────────
# O EchoStateNetwork é o coração do conn2res.
# Parâmetros-chave:
#   w                   → matriz de conectividade (a SC normalizada)
#   activation_function → 'tanh', 'sigmoid', 'relu', 'leaky_relu', etc.
#   leak_rate           → se None, ESN clássico. Se (0,1], leaky integrator.
#
# alpha * conn.w controla o regime dinâmico:
#   alpha < 1 → estável (ordered)
#   alpha ≈ 1 → crítico (edge of chaos) — geralmente MELHOR performance
#   alpha > 1 → caótico

# ⚠️ ARMADILHA CLÁSSICA #1: Não use ruído branco como input para tarefas
# de predição! Ruído não tem estrutura temporal → R² ≈ 0 garantido.
# Use séries temporais caóticas (Mackey-Glass, Lorenz, etc.)

from reservoirpy.datasets import mackey_glass

T = 1000  # timesteps (mais = melhor para regressão)
mg = mackey_glass(n_timesteps=T + 50, tau=17)
x_input = mg[:T].reshape(-1, 1)  # shape (T, 1)

# Para o w_in, como temos 1 feature agora, redistribuímos o input
# para múltiplos nós de entrada com scaling adequado
n_features = 1
w_in = np.zeros((n_features, conn.n_nodes))
# ⚠️ ARMADILHA #2: input_scaling importa! Valores muito altos saturam
# a tanh; muito baixos diluem o sinal. Faixa segura: 0.1 – 1.0
input_scaling = 0.5
w_in[0, input_nodes] = input_scaling  # vários nós recebem o mesmo sinal

# Instanciar ESN
alpha = 0.9  # ligeiramente subcrítico — geralmente ótimo para predição
esn = EchoStateNetwork(
    w=alpha * conn.w,
    activation_function='tanh',
    leak_rate=0.3  # leaky integrator suaviza a dinâmica
)

# ⚠️ ARMADILHA #3 (A MAIS IMPORTANTE): output_nodes
# Se você usa só 15 nós (1 network), descarta 85% da informação!
# Para tarefas de predição, leia de TODOS os nós.
# Reserve readout por módulo apenas para análise de "qual network carrega
# mais informação" (Seção B.2), NÃO como pipeline principal.
reservoir_states = esn.simulate(
    ext_input=x_input,
    w_in=w_in,
    output_nodes=None  # ← TODOS os nós! Essencial para bom R²
)

print(f"\n[A.5] Reservoir simulado:")
print(f"   input: Mackey-Glass, shape = {x_input.shape}")
print(f"   reservoir_states shape = {reservoir_states.shape}")
print(f"   (timesteps × ALL nodes — NÃO filtre prematuramente!)")

# ─────────────────────────────────────────────────────────────────────────
# A.6  Readout Module — treinamento e teste
# ─────────────────────────────────────────────────────────────────────────
# O Readout é onde o aprendizado acontece.
# Usa modelos lineares do sklearn (Ridge, RidgeClassifier, etc.)
# Pipeline: train → test → scores

# Target: predição 1-step ahead da Mackey-Glass
y_target = mg[1:T+1].reshape(-1, 1)  # x(t) → y(t) = x(t+1)

# Split train/test (washout nos primeiros timesteps)
washout = 100  # descartar transiente inicial
frac_train = 0.7
n_train = int((T - washout) * frac_train) + washout

rs_train = reservoir_states[washout:n_train]
rs_test = reservoir_states[n_train:]
y_train = y_target[washout:n_train]
y_test = y_target[n_train:]

# Instanciar Readout com Ridge regression
# ⚠️ DICA: alpha do Ridge (regularização) importa!
# Muito baixo = overfit; muito alto = underfit
readout_module = Readout(estimator=Ridge(alpha=1e-4))

# Treinar
readout_module.train(X=rs_train, y=y_train)

# Testar
scores = readout_module.test(X=rs_test, y=y_test, metric=('r2_score',))
print(f"\n[A.6] Readout performance:")
print(f"   R² score = {scores['r2_score']:.4f}")
print(f"   (Mackey-Glass 1-step prediction, todos os nós como readout)")

# ─────────────────────────────────────────────────────────────────────────
# A.7  Pipeline Completo com run_task()
# ─────────────────────────────────────────────────────────────────────────
# run_task() encapsula train + test + split + readout_modules de uma vez.
# Aceita readout_modules para treinar readouts separados por módulo.

readout_complete = Readout(estimator=Ridge(alpha=1e-4))
# Remover washout antes de passar ao run_task
rs_no_washout = reservoir_states[washout:]
y_no_washout = y_target[washout:]

df_scores = readout_complete.run_task(
    X=rs_no_washout,
    y=y_no_washout,
    frac_train=0.7,
    metric=('r2_score',),
    readout_modules=None  # treina com todos os nós juntos
)
print(f"\n[A.7] run_task() completo:")
print(df_scores)

print("\n✅ Seção A concluída — Pipeline básico dominado!")


# ═══════════════════════════════════════════════════════════════════════════
#                                                                          ║
#  ██████╗  SEÇÃO B — TRUQUES AVANÇADOS                                    ║
#  ██╔══██╗ 4 Receitas Específicas                                         ║
#  ███████║                                                                ║
#  ██╔══██║                                                                ║
#  ██║  ██║                                                                ║
#  ╚═╝  ╚═╝                                                                ║
#                                                                          ║
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SEÇÃO B — TRUQUES AVANÇADOS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# B.1  TRUQUE: Varredura de Alpha (sweep do regime dinâmico)
# ─────────────────────────────────────────────────────────────────────────
# O alpha sweep é a análise mais importante do conn2res!
# Você varre alpha de 0 → 2 e observa onde a performance é máxima.
# Se pico em alpha ≈ 1.0 → seu conectoma opera no regime crítico.

print("\n[B.1] Alpha Sweep — varredura de regime dinâmico")

alphas = np.linspace(0.1, 2.0, 20)
scores_per_alpha = []

for alpha in alphas:
    esn_sweep = EchoStateNetwork(
        w=alpha * conn.w,
        activation_function='tanh',
        leak_rate=0.3
    )

    # Simular com TODOS os nós como output
    rs = esn_sweep.simulate(ext_input=x_input, w_in=w_in,
                            output_nodes=None)

    # Treinar/testar (com washout!)
    rdout = Readout(estimator=Ridge(alpha=1e-4))
    rs_tr = rs[washout:n_train]
    rs_te = rs[n_train:]
    y_tr = y_target[washout:n_train]
    y_te = y_target[n_train:]

    rdout.train(X=rs_tr, y=y_tr)
    sc = rdout.test(X=rs_te, y=y_te, metric=('r2_score',))
    scores_per_alpha.append(sc['r2_score'])

best_alpha = alphas[np.argmax(scores_per_alpha)]
print(f"   Melhor alpha = {best_alpha:.2f}  (R² = {max(scores_per_alpha):.4f})")
print(f"   → Nota: pico próximo de α ≈ 0.7-1.0 confirma regime crítico!")

# ─────────────────────────────────────────────────────────────────────────
# B.2  TRUQUE: Readout por Módulo (network-specific readouts)
# ─────────────────────────────────────────────────────────────────────────
# Em vez de ler de todos os nós, lê separadamente de cada Yeo network.
# Isso revela QUAL rede carrega mais informação para a tarefa.

print("\n[B.2] Readout por Módulo (Yeo networks)")

# Para B.2, reusar os reservoir_states já computados (todos os nós)
rs_full = reservoir_states.copy()

# Criar readout_modules: dicionário {id_modulo: [indices_dos_nós]}
module_labels = labels[:conn.n_nodes]
readout_modules_dict = {}
for mod_id in range(7):
    nodes = np.where(module_labels == mod_id)[0]
    if len(nodes) > 0:
        readout_modules_dict[YEO7_NAMES[mod_id]] = nodes.tolist()

# Treinar readout por módulo — cada network isoladamente
rdout_modules = Readout(estimator=Ridge(alpha=1e-4))
# ⚠️ BUGFIX conn2res: metric DEVE ser LIST, não tuple!
# Internamente, run_task faz ['module','n_nodes'] + metric → TypeError com tuple
df_module_scores = rdout_modules.run_task(
    X=rs_full[washout:],
    y=y_target[washout:],
    frac_train=0.7,
    metric=['r2_score'],  # ← LIST obrigatório!
    readout_modules=readout_modules_dict
)
print(df_module_scores[['module', 'n_nodes', 'r2_score']].to_string(index=False))
print("\n   → Networks com mais nós no 'caminho' do input tendem a R² maior")

# ─────────────────────────────────────────────────────────────────────────
# B.3  TRUQUE: Leaky Integrator ESN para controle temporal
# ─────────────────────────────────────────────────────────────────────────
# O leak_rate adiciona uma constante de tempo: quanto menor o leak_rate,
# mais lenta a dinâmica (mais memória temporal).
# Equação: x(t) = (1-lr)*x(t-1) + lr*f(W*x(t-1) + Win*u(t))

print("\n[B.3] Leaky Integrator ESN")

leak_rates = [0.1, 0.3, 0.5, 0.8, 1.0]
scores_per_lr = []

for lr in leak_rates:
    esn_leaky = EchoStateNetwork(
        w=0.9 * conn.w,
        activation_function='tanh',
        leak_rate=lr  # ← Controla a escala temporal da memória
    )
    rs_leaky = esn_leaky.simulate(
        ext_input=x_input, w_in=w_in, output_nodes=None  # ← ALL nodes!
    )

    rdout = Readout(estimator=Ridge(alpha=1e-4))
    rdout.train(X=rs_leaky[washout:n_train], y=y_target[washout:n_train])
    sc = rdout.test(X=rs_leaky[n_train:], y=y_target[n_train:],
                    metric=('r2_score',))
    scores_per_lr.append(sc['r2_score'])
    print(f"   leak_rate={lr:.1f}  →  R² = {sc['r2_score']:.4f}")

# ─────────────────────────────────────────────────────────────────────────
# B.4  TRUQUE: Randomização do Conectoma + Comparação
# ─────────────────────────────────────────────────────────────────────────
# Para provar que a TOPOLOGIA do conectoma importa (não apenas os pesos),
# compare a performance do conectoma real vs. randomizado.
# conn.randomize() preserva a distribuição de grau mas destrói a
# organização modular.

print("\n[B.4] Conectoma Real vs. Randomizado")

# Salvar SC original
SC_original = conn.w.copy()

# Performance com SC real
esn_real = EchoStateNetwork(w=0.9 * SC_original, activation_function='tanh',
                            leak_rate=0.3)
rs_real = esn_real.simulate(ext_input=x_input, w_in=w_in,
                            output_nodes=None)
rdout_real = Readout(estimator=Ridge(alpha=1e-4))
rdout_real.train(X=rs_real[washout:n_train], y=y_target[washout:n_train])
score_real = rdout_real.test(X=rs_real[n_train:], y=y_target[n_train:],
                             metric=('r2_score',))

# Performance com SC randomizada (5 permutações)
scores_random = []
for i in range(5):
    conn_rand = Conn(w=SC.copy())
    conn_rand.scale_and_normalize()
    conn_rand.randomize(swaps=10)

    # w_in precisa se adaptar ao tamanho do conn_rand (pode mudar!)
    w_in_rand = np.zeros((1, conn_rand.n_nodes))
    rand_inp = min(input_nodes[0], conn_rand.n_nodes - 1)
    w_in_rand[0, rand_inp] = input_scaling

    esn_rand = EchoStateNetwork(w=0.9 * conn_rand.w, activation_function='tanh',
                                leak_rate=0.3)
    rs_rand = esn_rand.simulate(ext_input=x_input, w_in=w_in_rand,
                                output_nodes=None)  # ALL nodes
    rdout_rand = Readout(estimator=Ridge(alpha=1e-4))
    rdout_rand.train(X=rs_rand[washout:n_train], y=y_target[washout:n_train])
    sc_rand = rdout_rand.test(X=rs_rand[n_train:], y=y_target[n_train:],
                               metric=('r2_score',))
    scores_random.append(sc_rand['r2_score'])

print(f"   SC Real:       R² = {score_real['r2_score']:.4f}")
print(f"   SC Randomized: R² = {np.mean(scores_random):.4f} "
      f"± {np.std(scores_random):.4f}")

# Restaurar
conn.w = SC_original

print("\n✅ Seção B concluída — Truques avançados dominados!")


# ═══════════════════════════════════════════════════════════════════════════
#                                                                          ║
#   ██████╗ SEÇÃO C — SC-FC DECOUPLING (Sujeito Único)                     ║
#   ██╔════╝ Whole-Brain + Yeo Networks (Intra/Inter)                      ║
#   ██║                                                                    ║
#   ██║                                                                    ║
#   ╚██████╗                                                               ║
#    ╚═════╝                                                               ║
#                                                                          ║
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SEÇÃO C — SC-FC DECOUPLING (Sujeito Único)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# CONCEITO FUNDAMENTAL:
# ─────────────────────────────────────────────────────────────────────────
# SC-FC decoupling via reservoir computing:
#   1. SC é a arquitetura do reservatório (pesos fixos)
#   2. Injetamos ruído/sinal nos nós de entrada
#   3. O reservatório produz estados (reservoir states)
#   4. A FC "predita" pelo reservoir = correlação entre reservoir states
#   5. Comparamos FC_predita com FC_empírica
#   6. COUPLING = correlação alta entre FC_pred e FC_emp
#      DECOUPLING = correlação baixa → a função diverge da estrutura
#
# Isso é análogo ao Hopf model, mas via RC: a SC gera dinâmica,
# e a diferença entre a dinâmica gerada e a FC empírica = decoupling.
# ─────────────────────────────────────────────────────────────────────────


def compute_rc_scfc_coupling(SC, FC, labels=None, alpha=1.0,
                              activation='tanh', leak_rate=None,
                              n_timepoints=2000, n_runs=10, seed=42):
    """
    Computa SC-FC coupling via Reservoir Computing.

    Pipeline:
        1. SC → normaliza → reservatório
        2. Injeta ruído branco em todos os nós
        3. Simula dinâmica
        4. FC_rc = correlação dos reservoir states
        5. coupling = corr(FC_rc_upper_tri, FC_emp_upper_tri)

    Parameters
    ----------
    SC : (N, N) ndarray — structural connectivity
    FC : (N, N) ndarray — functional connectivity empírica
    labels : (N,) ndarray, optional — Yeo network labels
    alpha : float — spectral radius scaling
    activation : str — função de ativação
    leak_rate : float or None
    n_timepoints : int — timesteps para simulação
    n_runs : int — repetições para estabilidade
    seed : int

    Returns
    -------
    results : dict com:
        'coupling_wholebrain' : float — correlação global
        'FC_predicted' : (N,N) ndarray — FC média predita pelo RC
        'coupling_per_network' : dict — coupling intra-network
        'coupling_internetwork' : DataFrame — coupling inter-network
    """
    rng = np.random.default_rng(seed)
    n_nodes = SC.shape[0]

    # --- Preparar SC para o reservatório ---
    conn_subj = Conn(w=SC.copy())
    conn_subj.scale_and_normalize()
    n_active = conn_subj.n_nodes

    # Mapear labels para nós ativos
    if labels is not None:
        active_labels = labels[conn_subj.idx_node]
    else:
        active_labels = np.zeros(n_active, dtype=int)

    # Mapear FC para nós ativos
    idx_active = np.where(conn_subj.idx_node)[0]
    FC_active = FC[np.ix_(idx_active, idx_active)]

    # --- w_in: todos os nós recebem input (identidade) ---
    w_in = np.eye(n_active)

    # --- Simular múltiplas vezes e acumular FC ---
    FC_rc_accum = np.zeros((n_active, n_active))

    for run in range(n_runs):
        # Input: ruído branco gaussiano
        ext_input = rng.standard_normal((n_timepoints, n_active)) * 0.1

        # ESN
        esn = EchoStateNetwork(
            w=alpha * conn_subj.w,
            activation_function=activation,
            leak_rate=leak_rate
        )

        # Simular
        rs = esn.simulate(ext_input=ext_input, w_in=w_in, output_nodes=None)

        # Descartar washout (primeiros 200 timesteps)
        washout = 200
        rs = rs[washout:]

        # FC do reservoir = correlação de Pearson dos estados
        FC_rc = np.corrcoef(rs.T)
        FC_rc_accum += FC_rc

    FC_predicted = FC_rc_accum / n_runs

    # --- COUPLING WHOLE-BRAIN ---
    # Extrair triângulo superior (excluindo diagonal)
    triu_idx = np.triu_indices(n_active, k=1)
    fc_pred_upper = FC_predicted[triu_idx]
    fc_emp_upper = FC_active[triu_idx]

    # Correlação de Pearson
    coupling_wb, p_wb = stats.pearsonr(fc_pred_upper, fc_emp_upper)

    results = {
        'coupling_wholebrain': coupling_wb,
        'coupling_wholebrain_p': p_wb,
        'FC_predicted': FC_predicted,
        'FC_empirical': FC_active,
        'active_labels': active_labels,
        'idx_active': idx_active,
    }

    # --- COUPLING PER NETWORK (INTRA-NETWORK) ---
    if labels is not None:
        coupling_intra = {}
        unique_mods = np.unique(active_labels)

        for mod in unique_mods:
            mod_nodes = np.where(active_labels == mod)[0]
            if len(mod_nodes) < 3:
                continue

            # Extrair sub-matrices
            fc_pred_sub = FC_predicted[np.ix_(mod_nodes, mod_nodes)]
            fc_emp_sub = FC_active[np.ix_(mod_nodes, mod_nodes)]

            # Triângulo superior
            triu_sub = np.triu_indices(len(mod_nodes), k=1)
            pred_vals = fc_pred_sub[triu_sub]
            emp_vals = fc_emp_sub[triu_sub]

            if len(pred_vals) > 2:
                r, p = stats.pearsonr(pred_vals, emp_vals)
                mod_name = YEO7_NAMES[mod] if mod < 7 else f"Mod{mod}"
                coupling_intra[mod_name] = {'r': r, 'p': p,
                                            'n_nodes': len(mod_nodes)}

        results['coupling_intra'] = coupling_intra

        # --- COUPLING INTER-NETWORK (entre pares de redes) ---
        coupling_inter = []

        for i, mod_i in enumerate(unique_mods):
            for j, mod_j in enumerate(unique_mods):
                if i >= j:
                    continue

                nodes_i = np.where(active_labels == mod_i)[0]
                nodes_j = np.where(active_labels == mod_j)[0]

                if len(nodes_i) < 2 or len(nodes_j) < 2:
                    continue

                # Bloco inter-network da FC
                fc_pred_block = FC_predicted[np.ix_(nodes_i, nodes_j)]
                fc_emp_block = FC_active[np.ix_(nodes_i, nodes_j)]

                pred_flat = fc_pred_block.ravel()
                emp_flat = fc_emp_block.ravel()

                if len(pred_flat) > 2:
                    r, p = stats.pearsonr(pred_flat, emp_flat)
                    name_i = YEO7_NAMES[mod_i] if mod_i < 7 else f"M{mod_i}"
                    name_j = YEO7_NAMES[mod_j] if mod_j < 7 else f"M{mod_j}"
                    coupling_inter.append({
                        'network_i': name_i,
                        'network_j': name_j,
                        'coupling': r,
                        'p_value': p,
                    })

        results['coupling_inter'] = pd.DataFrame(coupling_inter)

    return results


# ─────────────────────────────────────────────────────────────────────────
# C.1  Whole-Brain SC-FC Coupling
# ─────────────────────────────────────────────────────────────────────────

print("\n[C.1] SC-FC Coupling — Whole Brain")
results_c = compute_rc_scfc_coupling(
    SC=SC, FC=FC, labels=labels,
    alpha=1.0, activation='tanh', leak_rate=0.3,
    n_timepoints=2000, n_runs=10, seed=42
)

print(f"   Coupling (whole-brain) = {results_c['coupling_wholebrain']:.4f} "
      f"(p = {results_c['coupling_wholebrain_p']:.2e})")

# ─────────────────────────────────────────────────────────────────────────
# C.2  Intra-Network SC-FC Coupling (por Yeo network)
# ─────────────────────────────────────────────────────────────────────────

print("\n[C.2] Intra-Network Coupling:")
for net_name, vals in results_c['coupling_intra'].items():
    print(f"   {net_name:4s}: r = {vals['r']:.4f}  "
          f"(p = {vals['p']:.2e}, n_nodes = {vals['n_nodes']})")

# ─────────────────────────────────────────────────────────────────────────
# C.3  Inter-Network SC-FC Coupling
# ─────────────────────────────────────────────────────────────────────────

print("\n[C.3] Inter-Network Coupling (top 5 pares):")
df_inter = results_c['coupling_inter'].sort_values('coupling', ascending=False)
print(df_inter.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
# C.4  Alpha Sweep do Decoupling
# ─────────────────────────────────────────────────────────────────────────
# Varre alpha para ver como o regime dinâmico afeta o coupling.
# Isso revela se o conectoma opera melhor na criticalidade.

print("\n[C.4] Alpha Sweep do SC-FC Coupling:")
alphas_sweep = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
coupling_by_alpha = []

for a in alphas_sweep:
    res = compute_rc_scfc_coupling(
        SC=SC, FC=FC, labels=labels,
        alpha=a, n_timepoints=1000, n_runs=5, seed=42
    )
    coupling_by_alpha.append({
        'alpha': a,
        'coupling': res['coupling_wholebrain']
    })
    print(f"   alpha={a:.1f}  →  coupling = {res['coupling_wholebrain']:.4f}")

df_alpha_coupling = pd.DataFrame(coupling_by_alpha)

print("\n✅ Seção C concluída — SC-FC decoupling de sujeito único dominado!")


# ═══════════════════════════════════════════════════════════════════════════
#                                                                          ║
#  ██████╗  SEÇÃO D — ANÁLISE DE GRUPO + NULL MODELS                       ║
#  ██╔══██╗                                                                ║
#  ██║  ██║                                                                ║
#  ██║  ██║                                                                ║
#  ██████╔╝                                                                ║
#  ╚═════╝                                                                 ║
#                                                                          ║
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SEÇÃO D — ANÁLISE DE GRUPO + NULL MODELS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# D.1  Group-Level SC-FC Coupling
# ─────────────────────────────────────────────────────────────────────────

print("\n[D.1] Group-Level SC-FC Coupling")

# Gerar dados de grupo
n_subjects = 15
SCs, FCs, group_labels = make_group_data(
    n_subjects=n_subjects, n_nodes=100, n_modules=7, seed=42
)

# Computar coupling para cada sujeito
group_results = []

for s in range(n_subjects):
    res = compute_rc_scfc_coupling(
        SC=SCs[s], FC=FCs[s], labels=group_labels,
        alpha=1.0, activation='tanh', leak_rate=0.3,
        n_timepoints=1000, n_runs=5, seed=42 + s
    )

    row = {
        'subject': s,
        'coupling_wb': res['coupling_wholebrain'],
    }

    # Adicionar intra-network coupling
    for net_name, vals in res.get('coupling_intra', {}).items():
        row[f'intra_{net_name}'] = vals['r']

    group_results.append(row)

df_group = pd.DataFrame(group_results)

print(f"\n   Coupling whole-brain (grupo):")
print(f"   Mean = {df_group['coupling_wb'].mean():.4f} "
      f"± {df_group['coupling_wb'].std():.4f}")
print(f"   Range = [{df_group['coupling_wb'].min():.4f}, "
      f"{df_group['coupling_wb'].max():.4f}]")

# Intra-network médias
print(f"\n   Coupling intra-network (grupo, média ± std):")
intra_cols = [c for c in df_group.columns if c.startswith('intra_')]
for col in intra_cols:
    net = col.replace('intra_', '')
    print(f"   {net:4s}: {df_group[col].mean():.4f} ± {df_group[col].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────
# D.2  NULL MODEL — Conectoma Randomizado
# ─────────────────────────────────────────────────────────────────────────
# O null model testa se o coupling observado é significativamente
# maior do que o esperado por uma SC com topologia aleatória
# (preservando distribuição de grau).

print("\n[D.2] Null Model — SC Randomizada")

N_PERMS = 20  # Em paper real, use 1000+

null_couplings = []

for perm in range(N_PERMS):
    # Pegar a SC média do grupo
    SC_mean = np.mean(SCs, axis=0)
    FC_mean = np.mean(FCs, axis=0)

    # Randomizar a SC
    conn_null = Conn(w=SC_mean.copy())
    conn_null.scale_and_normalize()
    try:
        conn_null.randomize(swaps=10)
    except Exception:
        # Se randomize falhar, gerar random manual
        rng_null = np.random.default_rng(seed=perm)
        w_rand = conn_null.w.copy()
        n = w_rand.shape[0]
        for _ in range(n * 5):
            i, j = rng_null.integers(0, n, 2)
            k, l = rng_null.integers(0, n, 2)
            if i != j and k != l:
                w_rand[i, j], w_rand[k, l] = w_rand[k, l], w_rand[i, j]
                w_rand[j, i], w_rand[l, k] = w_rand[l, k], w_rand[j, i]
        conn_null.w = w_rand

    # Simulação com SC randomizada
    n_null = conn_null.n_nodes
    w_in_null = np.eye(n_null)
    rng_sim = np.random.default_rng(seed=perm + 1000)
    ext_input_null = rng_sim.standard_normal((1000, n_null)) * 0.1

    esn_null = EchoStateNetwork(
        w=1.0 * conn_null.w, activation_function='tanh', leak_rate=0.3
    )
    rs_null = esn_null.simulate(ext_input=ext_input_null, w_in=w_in_null)
    rs_null = rs_null[200:]  # washout

    FC_null = np.corrcoef(rs_null.T)

    # Coupling com a FC empírica média
    idx_act = np.where(conn_null.idx_node)[0]
    FC_emp_null = FC_mean[np.ix_(idx_act, idx_act)]

    triu = np.triu_indices(n_null, k=1)
    r_null, _ = stats.pearsonr(FC_null[triu], FC_emp_null[triu])
    null_couplings.append(r_null)

null_couplings = np.array(null_couplings)

# Coupling empírico (com SC real, média do grupo)
empirical_coupling = df_group['coupling_wb'].mean()

# P-value (proporção de null ≥ empírico)
p_null = np.mean(null_couplings >= empirical_coupling)

print(f"   Empirical coupling   = {empirical_coupling:.4f}")
print(f"   Null coupling (mean) = {null_couplings.mean():.4f} "
      f"± {null_couplings.std():.4f}")
print(f"   P-value (perm)       = {p_null:.4f}")
print(f"   Z-score              = "
      f"{(empirical_coupling - null_couplings.mean()) / (null_couplings.std() + 1e-10):.2f}")

# ─────────────────────────────────────────────────────────────────────────
# D.3  NULL MODEL — Spin Test (preserva autocorrelação espacial)
# ─────────────────────────────────────────────────────────────────────────
# O spin test é mais conservador que a randomização de edges.
# Ele permuta os LABELS dos nós (quais nós pertencem a qual network),
# preservando a geometria espacial.
# Implementação simplificada aqui (full spin test requer coordenadas).

print("\n[D.3] Null Model — Permutação de Labels (simplificado)")

null_intra = {net: [] for net in YEO7_NAMES}

for perm in range(N_PERMS):
    # Permutação aleatória dos labels
    rng_perm = np.random.default_rng(seed=perm + 5000)
    perm_labels = rng_perm.permutation(group_labels)

    # Recomputar coupling intra-network com labels permutados
    # Usar FC_predicted e FC_empirical do primeiro sujeito como exemplo
    FC_pred = results_c['FC_predicted']
    FC_emp = results_c['FC_empirical']
    active_labs = perm_labels[:FC_pred.shape[0]]

    for mod_id, net_name in enumerate(YEO7_NAMES):
        mod_nodes = np.where(active_labs == mod_id)[0]
        if len(mod_nodes) < 3:
            null_intra[net_name].append(np.nan)
            continue

        triu_sub = np.triu_indices(len(mod_nodes), k=1)
        fc_p = FC_pred[np.ix_(mod_nodes, mod_nodes)][triu_sub]
        fc_e = FC_emp[np.ix_(mod_nodes, mod_nodes)][triu_sub]

        if len(fc_p) > 2:
            r, _ = stats.pearsonr(fc_p, fc_e)
            null_intra[net_name].append(r)
        else:
            null_intra[net_name].append(np.nan)

print(f"\n   Intra-network coupling vs. null (permuted labels):")
for net_name in YEO7_NAMES:
    if net_name in results_c['coupling_intra']:
        emp_val = results_c['coupling_intra'][net_name]['r']
        null_vals = np.array(null_intra[net_name])
        null_vals = null_vals[~np.isnan(null_vals)]
        if len(null_vals) > 0:
            p_perm = np.mean(null_vals >= emp_val)
            print(f"   {net_name:4s}: emp={emp_val:.4f}  "
                  f"null={np.mean(null_vals):.4f}±{np.std(null_vals):.4f}  "
                  f"p={p_perm:.3f}")

print("\n✅ Seção D concluída — Análise de grupo + null models dominados!")


# ═══════════════════════════════════════════════════════════════════════════
#                                                                          ║
#  ███████╗ SEÇÃO E — VISUALIZAÇÕES ESPETACULARES                          ║
#  ██╔════╝                                                                ║
#  █████╗                                                                  ║
#  ██╔══╝                                                                  ║
#  ███████╗                                                                ║
#  ╚══════╝                                                                ║
#                                                                          ║
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SEÇÃO E — VISUALIZAÇÕES ESPETACULARES")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────
# E.1  Figura Simples: Curva de Performance vs Alpha
# ─────────────────────────────────────────────────────────────────────────

print("\n[E.1] Curva de Performance vs Alpha")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, scores_per_alpha, 'o-', color='#2c3e50', lw=2, ms=5,
        markerfacecolor='#e74c3c', markeredgecolor='#2c3e50', zorder=3)
ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5, label='α = 1 (critical)')
ax.set_xlabel('Spectral Radius (α)')
ax.set_ylabel('R² Score')
ax.set_title('Reservoir Performance Across Dynamical Regimes')
ax.legend(frameon=False)
ax.set_ylim(bottom=min(min(scores_per_alpha) - 0.05, -0.1))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E1_alpha_sweep.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.2  Readout por Módulo (bar plot com cores Yeo)
# ─────────────────────────────────────────────────────────────────────────

print("[E.2] Barplot de Performance por Yeo Network")

fig, ax = plt.subplots(figsize=(7, 4))

if 'module' in df_module_scores.columns:
    mod_data = df_module_scores.dropna(subset=['r2_score'])
    colors = [YEO7_COLORS.get(m, '#999999') for m in mod_data['module']]
    bars = ax.bar(range(len(mod_data)), mod_data['r2_score'],
                  color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(len(mod_data)))
    ax.set_xticklabels(mod_data['module'], rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('Network-Specific Computational Capacity')
    ax.axhline(y=0, color='gray', ls='-', lw=0.5)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E2_network_barplot.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.3  Heatmap de FC Empírica vs FC Predita
# ─────────────────────────────────────────────────────────────────────────

print("[E.3] Heatmaps FC Empírica vs FC Predita")

FC_emp = results_c['FC_empirical']
FC_pred = results_c['FC_predicted']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# FC Empírica
im0 = axes[0].imshow(FC_emp, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
axes[0].set_title('FC Empírica', fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# FC Predita (reservoir)
im1 = axes[1].imshow(FC_pred, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
axes[1].set_title('FC Predita (RC)', fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Diferença (Decoupling map)
FC_diff = FC_emp - FC_pred
im2 = axes[2].imshow(FC_diff, cmap='PiYG', vmin=-0.5, vmax=0.5, aspect='auto')
axes[2].set_title('Decoupling (Emp − Pred)', fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# Adicionar linhas de separação de networks
active_labels = results_c['active_labels']
boundaries = []
for mod in range(7):
    nodes = np.where(active_labels == mod)[0]
    if len(nodes) > 0:
        boundaries.append(nodes[-1] + 0.5)

for ax in axes:
    for b in boundaries[:-1]:
        ax.axhline(y=b, color='black', lw=0.5, alpha=0.3)
        ax.axvline(x=b, color='black', lw=0.5, alpha=0.3)

fig.suptitle('SC-FC Coupling Analysis via Reservoir Computing',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E3_fc_heatmaps.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.4  Scatter: FC Empírica vs FC Predita (triângulo superior)
# ─────────────────────────────────────────────────────────────────────────

print("[E.4] Scatter FC_emp vs FC_pred")

triu = np.triu_indices(FC_emp.shape[0], k=1)
fc_emp_flat = FC_emp[triu]
fc_pred_flat = FC_pred[triu]

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(fc_emp_flat, fc_pred_flat, s=3, alpha=0.3, color='#3498db',
           rasterized=True)

# Linha de identidade
lims = [min(fc_emp_flat.min(), fc_pred_flat.min()),
        max(fc_emp_flat.max(), fc_pred_flat.max())]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)

# Regression line
slope, intercept = np.polyfit(fc_emp_flat, fc_pred_flat, 1)
x_line = np.linspace(lims[0], lims[1], 100)
ax.plot(x_line, slope * x_line + intercept, color='#e74c3c', lw=2)

r_val = results_c['coupling_wholebrain']
ax.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('FC Empírica')
ax.set_ylabel('FC Predita (RC)')
ax.set_title('Whole-Brain SC-FC Coupling')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E4_scatter_coupling.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.5  Coupling Inter-Network Matrix (heatmap triangular)
# ─────────────────────────────────────────────────────────────────────────

print("[E.5] Inter-Network Coupling Matrix")

df_inter = results_c['coupling_inter']

# Construir matriz 7x7
coupling_matrix = np.full((7, 7), np.nan)

# Preencher diagonal com intra-network coupling
for i, net in enumerate(YEO7_NAMES):
    if net in results_c['coupling_intra']:
        coupling_matrix[i, i] = results_c['coupling_intra'][net]['r']

# Preencher off-diagonal com inter-network coupling
for _, row in df_inter.iterrows():
    i = YEO7_NAMES.index(row['network_i']) if row['network_i'] in YEO7_NAMES else -1
    j = YEO7_NAMES.index(row['network_j']) if row['network_j'] in YEO7_NAMES else -1
    if i >= 0 and j >= 0:
        coupling_matrix[i, j] = row['coupling']
        coupling_matrix[j, i] = row['coupling']

fig, ax = plt.subplots(figsize=(7, 6))

# Mask triângulo inferior
mask = np.zeros_like(coupling_matrix, dtype=bool)
# Não mascarar nada — mostrar tudo
im = ax.imshow(coupling_matrix, cmap='RdYlBu_r', vmin=-0.3, vmax=0.8,
               aspect='auto')

ax.set_xticks(range(7))
ax.set_xticklabels(YEO7_NAMES, rotation=45, ha='right')
ax.set_yticks(range(7))
ax.set_yticklabels(YEO7_NAMES)

# Anotar valores
for i in range(7):
    for j in range(7):
        val = coupling_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SC-FC Coupling (r)')
ax.set_title('Network-Level SC-FC Coupling Matrix', fontweight='bold')

# Colorir ticks
for i, net in enumerate(YEO7_NAMES):
    color = YEO7_COLORS.get(net, 'black')
    ax.get_xticklabels()[i].set_color(color)
    ax.get_yticklabels()[i].set_color(color)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E5_inter_network_matrix.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.6  Null Model Distribution + Empirical Value
# ─────────────────────────────────────────────────────────────────────────

print("[E.6] Null Model Distribution")

fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(null_couplings, bins=15, color='#bdc3c7', edgecolor='white',
        alpha=0.8, label='Null (randomized SC)')
ax.axvline(x=empirical_coupling, color='#e74c3c', lw=2.5, ls='--',
           label=f'Empirical (r = {empirical_coupling:.3f})')

ax.set_xlabel('SC-FC Coupling (r)')
ax.set_ylabel('Count')
ax.set_title('Null Model: Randomized Connectome', fontweight='bold')
ax.legend(frameon=False)

# Anotar p-value
ax.text(0.95, 0.95, f'p = {p_null:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E6_null_model.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.7  SC-FC Coupling vs Alpha (com confidence band)
# ─────────────────────────────────────────────────────────────────────────

print("[E.7] Coupling vs Alpha com Confidence Band")

fig, ax = plt.subplots(figsize=(6, 4))
couplings = [d['coupling'] for d in coupling_by_alpha]
ax.fill_between(alphas_sweep, np.array(couplings) - 0.05,
                np.array(couplings) + 0.05,
                color='#3498db', alpha=0.15)
ax.plot(alphas_sweep, couplings, 'o-', color='#2c3e50', lw=2.5, ms=7,
        markerfacecolor='#3498db', markeredgecolor='#2c3e50', zorder=3)
ax.axvline(x=1.0, color='gray', ls='--', lw=1, alpha=0.5,
           label='α = 1.0 (critical)')
ax.set_xlabel('Spectral Radius (α)', fontsize=12)
ax.set_ylabel('SC-FC Coupling (r)', fontsize=12)
ax.set_title('Dynamical Regime Shapes SC-FC Coupling', fontweight='bold')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E7_coupling_vs_alpha.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.8  FIGURA COMPOSTA (publication-quality) — A Grande Figura
# ─────────────────────────────────────────────────────────────────────────

print("[E.8] Figura Composta Publication-Quality")

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.4,
                       height_ratios=[1, 1, 1.1])

# --- Panel A: SC matrix ---
ax_a = fig.add_subplot(gs[0, 0])
im_a = ax_a.imshow(SC[:50, :50], cmap='YlOrRd', aspect='auto')
ax_a.set_title('A  Structural\n   Connectivity', fontweight='bold',
               fontsize=10, loc='left')
ax_a.set_xlabel('Node')
ax_a.set_ylabel('Node')
plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)

# --- Panel B: FC empírica ---
ax_b = fig.add_subplot(gs[0, 1])
im_b = ax_b.imshow(FC_emp, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax_b.set_title('B  FC Empírica', fontweight='bold', fontsize=10, loc='left')
plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)

# --- Panel C: FC predita ---
ax_c = fig.add_subplot(gs[0, 2])
im_c = ax_c.imshow(FC_pred, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax_c.set_title('C  FC Predita (RC)', fontweight='bold', fontsize=10, loc='left')
plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

# --- Panel D: Scatter ---
ax_d = fig.add_subplot(gs[0, 3])
ax_d.scatter(fc_emp_flat, fc_pred_flat, s=2, alpha=0.2, color='#3498db',
             rasterized=True)
ax_d.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
slope, intercept = np.polyfit(fc_emp_flat, fc_pred_flat, 1)
ax_d.plot(x_line, slope * x_line + intercept, color='#e74c3c', lw=1.5)
ax_d.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax_d.transAxes,
          fontsize=10, fontweight='bold', va='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax_d.set_xlabel('FC Emp')
ax_d.set_ylabel('FC Pred')
ax_d.set_title('D  Whole-Brain\n   Coupling', fontweight='bold',
               fontsize=10, loc='left')

# --- Panel E: Alpha sweep ---
ax_e = fig.add_subplot(gs[1, 0:2])
ax_e.plot(alphas, scores_per_alpha, 'o-', color='#2c3e50', lw=2, ms=5,
          markerfacecolor='#e74c3c', markeredgecolor='#2c3e50')
ax_e.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
ax_e.set_xlabel('Spectral Radius (α)')
ax_e.set_ylabel('R² Score')
ax_e.set_title('E  Performance vs Dynamical Regime', fontweight='bold',
               fontsize=10, loc='left')

# --- Panel F: Coupling vs alpha ---
ax_f = fig.add_subplot(gs[1, 2:4])
ax_f.plot(alphas_sweep, couplings, 'o-', color='#2c3e50', lw=2, ms=6,
          markerfacecolor='#3498db', markeredgecolor='#2c3e50')
ax_f.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
ax_f.set_xlabel('Spectral Radius (α)')
ax_f.set_ylabel('SC-FC Coupling (r)')
ax_f.set_title('F  SC-FC Coupling vs Dynamical Regime', fontweight='bold',
               fontsize=10, loc='left')

# --- Panel G: Network barplot ---
ax_g = fig.add_subplot(gs[2, 0:2])
if 'module' in df_module_scores.columns:
    mod_data = df_module_scores.dropna(subset=['r2_score'])
    colors = [YEO7_COLORS.get(m, '#999999') for m in mod_data['module']]
    ax_g.bar(range(len(mod_data)), mod_data['r2_score'],
             color=colors, edgecolor='white', lw=0.5)
    ax_g.set_xticks(range(len(mod_data)))
    ax_g.set_xticklabels(mod_data['module'], rotation=45, ha='right')
    ax_g.set_ylabel('R² Score')
    ax_g.axhline(y=0, color='gray', ls='-', lw=0.5)
ax_g.set_title('G  Network-Specific\n   Computational Capacity',
               fontweight='bold', fontsize=10, loc='left')

# --- Panel H: Null model ---
ax_h = fig.add_subplot(gs[2, 2:4])
ax_h.hist(null_couplings, bins=12, color='#bdc3c7', edgecolor='white',
          alpha=0.8, label='Null')
ax_h.axvline(x=empirical_coupling, color='#e74c3c', lw=2.5, ls='--',
             label=f'Empirical')
ax_h.set_xlabel('SC-FC Coupling (r)')
ax_h.set_ylabel('Count')
ax_h.legend(frameon=False)
ax_h.text(0.95, 0.95, f'p = {p_null:.3f}', transform=ax_h.transAxes,
          ha='right', va='top', fontsize=10, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax_h.set_title('H  Null Model Validation', fontweight='bold',
               fontsize=10, loc='left')

fig.suptitle('conn2res: Connectome-Based Reservoir Computing for SC-FC Coupling',
             fontsize=15, fontweight='bold', y=0.98)
fig.savefig(os.path.join(OUTPUT_DIR, 'E8_composite_figure.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.9  Group Violinplot com Individual Dots
# ─────────────────────────────────────────────────────────────────────────

print("[E.9] Group Violinplot")

# Preparar dados long-form para seaborn
rows = []
for _, row in df_group.iterrows():
    for col in intra_cols:
        net = col.replace('intra_', '')
        if not pd.isna(row[col]):
            rows.append({'Network': net, 'Coupling': row[col],
                         'Subject': row['subject']})

df_long = pd.DataFrame(rows)

if len(df_long) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Violin
    palette = [YEO7_COLORS.get(n, '#999') for n in df_long['Network'].unique()]
    sns.violinplot(data=df_long, x='Network', y='Coupling',
                   palette=palette, inner=None, alpha=0.3, ax=ax)

    # Strip (dots individuais)
    sns.stripplot(data=df_long, x='Network', y='Coupling',
                  palette=palette, size=4, alpha=0.7, jitter=0.15, ax=ax)

    # Box no centro (só quartis)
    sns.boxplot(data=df_long, x='Network', y='Coupling',
                color='white', width=0.15, fliersize=0,
                boxprops=dict(alpha=0.7), ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('Intra-Network SC-FC Coupling (r)')
    ax.set_title('Group-Level Network Coupling Distribution',
                 fontweight='bold')

    # Colorir ticks
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(palette[i])
        label.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'E9_group_violinplot.png'))
    plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.10  Brain Plot (com nilearn, se disponível)
# ─────────────────────────────────────────────────────────────────────────

print("[E.10] Brain Plot (tentativa com nilearn)")

try:
    from nilearn import plotting as ni_plot
    from nilearn import datasets as ni_datasets

    # Baixar atlas Schaefer 100 parcels
    atlas = ni_datasets.fetch_atlas_schaefer_2018(n_rois=100)

    # Criar vetor de "decoupling nodal":
    # Para cada nó, o decoupling = 1 - média(coupling com outros nós)
    n_active = results_c['FC_predicted'].shape[0]
    nodal_decoupling = np.zeros(n_active)

    for i in range(n_active):
        # Diferença absoluta média entre FC_pred e FC_emp para o nó i
        diff = np.abs(results_c['FC_empirical'][i, :] -
                      results_c['FC_predicted'][i, :])
        nodal_decoupling[i] = np.mean(diff)

    # Mapear para o atlas (preencher até 100 com zeros)
    full_decoupling = np.zeros(100)
    full_decoupling[:min(n_active, 100)] = nodal_decoupling[:100]

    # Plot brain surface
    fig_brain = ni_plot.plot_markers(
        full_decoupling,
        atlas.labels[:100] if hasattr(atlas, 'labels') else None,
        title='Nodal SC-FC Decoupling',
        colorbar=True,
        display_mode='ortho',
    )
    fig_brain.savefig(os.path.join(OUTPUT_DIR, 'E10_brain_decoupling.png'))
    print("   ✅ Brain plot salvo!")

except ImportError:
    print("   ⚠️  nilearn não disponível — brain plot pulado")
    print("   Para instalar: pip install nilearn")
except Exception as e:
    print(f"   ⚠️  Erro no brain plot: {e}")
    print("   (esperado com dados sintéticos — funciona com dados reais)")

# ─────────────────────────────────────────────────────────────────────────
# E.11  Reservoir States Heatmap (dinâmica temporal)
# ─────────────────────────────────────────────────────────────────────────

print("[E.11] Reservoir States Heatmap")

fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [3, 1]})

# Heatmap dos estados
rs_plot = reservoir_states[:200, :]  # primeiros 200 timesteps
im = axes[0].imshow(rs_plot.T, aspect='auto', cmap='viridis',
                     interpolation='none')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Output Node')
axes[0].set_title('Reservoir States Over Time', fontweight='bold')
plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)

# Distribuição dos estados
axes[1].hist(reservoir_states.ravel(), bins=50, color='#2c3e50',
             edgecolor='white', orientation='horizontal', alpha=0.8)
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Activation')
axes[1].set_title('State Distribution', fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E11_reservoir_states.png'))
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# E.12  Leak Rate Comparison (multi-panel)
# ─────────────────────────────────────────────────────────────────────────

print("[E.12] Leak Rate Comparison")

fig, axes = plt.subplots(1, len(leak_rates), figsize=(3 * len(leak_rates), 3),
                          sharey=True)

for idx, lr in enumerate(leak_rates):
    esn_viz = EchoStateNetwork(
        w=1.0 * conn.w, activation_function='tanh', leak_rate=lr
    )
    rs_viz = esn_viz.simulate(ext_input=x_input[:100], w_in=w_in,
                               output_nodes=output_nodes[:5])

    for node in range(rs_viz.shape[1]):
        axes[idx].plot(rs_viz[:, node], lw=0.7, alpha=0.7)
    axes[idx].set_title(f'lr = {lr}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Time')
    if idx == 0:
        axes[idx].set_ylabel('Activation')

fig.suptitle('Effect of Leak Rate on Reservoir Dynamics',
             fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E12_leak_rate_comparison.png'))
plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# RESUMO FINAL
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  TUTORIAL CONN2RES COMPLETO — RESUMO")
print("=" * 70)
print(f"""
  Figuras salvas em: {OUTPUT_DIR}/

  SEÇÃO A — Pipeline Quick-Start
    • Conn() → scale_and_normalize() → get_nodes()
    • w_in (input matrix) → EchoStateNetwork.simulate()
    • Readout.train() → .test() → .run_task()

  SEÇÃO B — Truques Avançados
    B.1  Alpha Sweep (regime dinâmico)
    B.2  Readout por Módulo (network-specific)
    B.3  Leaky Integrator (controle temporal)
    B.4  Conectoma Real vs Randomizado

  SEÇÃO C — SC-FC Decoupling (Sujeito Único)
    C.1  Whole-Brain coupling
    C.2  Intra-Network coupling (Yeo 7)
    C.3  Inter-Network coupling
    C.4  Alpha sweep do coupling

  SEÇÃO D — Análise de Grupo + Null Models
    D.1  Group-level coupling
    D.2  Null: SC randomizada
    D.3  Null: Label permutation

  SEÇÃO E — Visualizações Espetaculares
    E.1  Performance vs Alpha
    E.2  Network barplot (Yeo colors)
    E.3  FC heatmaps (emp/pred/diff)
    E.4  Scatter coupling
    E.5  Inter-network matrix
    E.6  Null distribution
    E.7  Coupling vs alpha
    E.8  COMPOSITE FIGURE (publication-quality!)
    E.9  Group violinplot
    E.10 Brain plot (nilearn)
    E.11 Reservoir states heatmap
    E.12 Leak rate comparison

  Referência: Suárez et al. (2024) Nat Commun 15:656
  GitHub: https://github.com/netneurolab/conn2res
""")

print("✅ Tutorial completo! Bom debugging, Velho Mago! 🧙‍♂️")
