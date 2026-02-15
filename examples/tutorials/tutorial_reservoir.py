# ============================================================================
# TUTORIAL: Reservoir Computing com a Biblioteca SARS
# ============================================================================
# Ambiente: Spyder (conda: sars)
# Dados: Pacientes pós-UTI COVID-19 (N=23), 4 atlas
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── Imports da biblioteca SARS ──────────────────────────────────────────
from sars.config import (
    ALL_SUBJECT_IDS, ATLASES, ATLAS_DIR,
    get_sc_path, get_connectivity_path, get_timeseries_path,
)

# ── Imports do módulo reservoir ─────────────────────────────────────────
from sars.reservoir.esn import (
    ConnectomeReservoir,
    AdaptiveReservoir,
    memory_capacity,
    kernel_quality,
    echo_state_property_index,
    lyapunov_exponent,
    spectral_analysis,
    characterize_reservoir,
    compare_reservoir_architectures,
    fc_prediction_task,
    spectral_radius_sweep,
)

# ── Configuração de plots ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================================
# PASSO 1: Carregar conectividade estrutural e séries temporais
# ============================================================================

subject = "sub-01"
atlas = "schaefer_100"

# Conectividade estrutural (SC) — do pipeline de dMRI
# Usar streamline count ou SIFT2-weighted
SC = np.load(get_sc_path("sub-01", "schaefer_100"))
print(f"SC shape: {SC.shape}, density: {(SC > 0).sum() / SC.size:.3f}")

# Séries temporais BOLD — do pipeline de rs-fMRI
ts_path = get_timeseries_path(subject, atlas, strategy="acompcor")
timeseries = np.load(ts_path)  # shape: (T, N_rois)

# Conectividade funcional (FC) — para comparação posterior
fc_path = get_connectivity_path(subject, atlas, strategy="acompcor", kind="correlation")
FC = np.load(fc_path)

print(f"Subject:     {subject}")
print(f"Atlas:       {atlas}")
print(f"SC shape:    {SC.shape}")         # (100, 100) para Schaefer
print(f"TS shape:    {timeseries.shape}") # (T, 100) — T volumes temporais
print(f"FC shape:    {FC.shape}")         # (100, 100)
print(f"SC density:  {(SC > 0).sum() / SC.size:.3f}")
print(f"SC range:    [{SC.min():.2f}, {SC.max():.2f}]")

# ============================================================================
# PASSO 2: Inicializar o reservatório com o conectoma do paciente
# ============================================================================

esn = ConnectomeReservoir(
    connectivity=SC,
    spectral_radius=0.9,    # Próximo ao edge of chaos
    leak_rate=0.3,           # Integração adequada para BOLD (TR~2s)
    input_scaling=0.1,       # Regime moderadamente não-linear
    bias_scaling=0.0,        # Sem bias (padrão conn2res)
    input_connectivity=1.0,  # Todos os nós recebem input
    ridge_alpha=1e-5,        # Regularização do readout
    activation="tanh",       # Padrão para ESN
    normalize_connectivity="spectral",  # Normaliza por ρ(W)
    symmetrize=True,         # SC é inerentemente simétrica (dMRI)
    seed=42,                 # Reprodutibilidade
)

# O objeto agora contém:
# - esn.W         → SC normalizada e escalada (100×100)
# - esn.n_neurons → 100 (número de ROIs no Schaefer)
# - esn.states    → None (ainda não rodamos)
# - esn.readout   → None (ainda não treinamos)

print(f"Neurônios do reservatório: {esn.n_neurons}")
print(f"ρ efetivo da matriz W:     {np.max(np.abs(np.linalg.eigvals(esn.W))):.4f}")

# ============================================================================
# PASSO 3: FC Prediction Task — O Teste Fundamental
# ============================================================================

result = fc_prediction_task(
    sc=SC,
    timeseries=timeseries,
    spectral_radius=0.9,
    leak_rate=0.3,
    ridge_alpha=1e-5,
    wash_out=50,            # Descartar os primeiros 50 TRs (warmup)
    train_fraction=0.7,     # 70% treino, 30% teste
    seed=42,
)

print(f"\n{'='*60}")
print(f"FC PREDICTION — {subject} ({atlas})")
print(f"{'='*60}")
print(f"R² global:          {result['r2_global']:.4f}")
print(f"R² por ROI (média): {result['r2_per_roi'].mean():.4f}")
print(f"R² por ROI (std):   {result['r2_per_roi'].std():.4f}")
print(f"R² por ROI (min):   {result['r2_per_roi'].min():.4f}")
print(f"R² por ROI (max):   {result['r2_per_roi'].max():.4f}")

# ── Plot: Predição vs Real para um ROI exemplo ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

roi_idx = np.argmax(result['r2_per_roi'])  # ROI com melhor predição
t = np.arange(result['y_true'].shape[0])

axes[0].plot(t, result['y_true'][:, roi_idx], 'k-', lw=0.8, label='Real (BOLD)')
axes[0].plot(t, result['y_pred'][:, roi_idx], 'r-', lw=0.8, alpha=0.8, label='Predito (ESN)')
axes[0].set_ylabel('Amplitude (z-score)')
axes[0].set_title(
    f'Melhor ROI: R² = {result["r2_per_roi"][roi_idx]:.3f}',
    fontweight='bold'
)
axes[0].legend(frameon=False)

roi_idx_worst = np.argmin(result['r2_per_roi'])
axes[1].plot(t, result['y_true'][:, roi_idx_worst], 'k-', lw=0.8, label='Real (BOLD)')
axes[1].plot(t, result['y_pred'][:, roi_idx_worst], 'r-', lw=0.8, alpha=0.8, label='Predito (ESN)')
axes[1].set_ylabel('Amplitude (z-score)')
axes[1].set_xlabel('Tempo (TRs)')
axes[1].set_title(
    f'Pior ROI: R² = {result["r2_per_roi"][roi_idx_worst]:.3f}',
    fontweight='bold'
)
axes[1].legend(frameon=False)

plt.tight_layout()
plt.savefig('fc_prediction_example.png', dpi=300)
plt.show()

# ============================================================================
# PASSO 4: Memory Capacity — Quantos passos no passado o reservatório lembra?
# ============================================================================

# Memory Capacity (Jaeger, 2001):
# MC = Σ_δ R²(y_δ, ŷ_δ), onde y_δ = u(t - δ) e ŷ_δ é a predição do readout
# MC_total ∈ [0, N] (limitado pelo número de neurônios)

mc_total, mc_profile = memory_capacity(
    esn,                     # O ConnectomeReservoir já inicializado
    max_delay=50,            # Testar delays de 1 a 50
    n_timesteps=3000,        # Timesteps de sinal aleatório
    wash_out=200,            # Warmup
    seed=42,
)

print(f"\nMemory Capacity total: {mc_total:.2f} / {esn.n_neurons}")
print(f"MC relativo (MC/N):   {mc_total / esn.n_neurons:.3f}")
print(f"Delay com maior MC:   {np.argmax(mc_profile) + 1}")

# ── Plot: Perfil de MC por delay ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
delays = np.arange(1, len(mc_profile) + 1)
ax.bar(delays, mc_profile, color='steelblue', alpha=0.7, edgecolor='navy', lw=0.5)
ax.set_xlabel('Delay δ (timesteps)')
ax.set_ylabel('MC(δ)')
ax.set_title(f'Perfil de Memory Capacity — {subject} ({atlas})\nMC total = {mc_total:.2f}',
             fontweight='bold')
ax.axhline(y=0, color='k', lw=0.5)
sns.despine()
plt.tight_layout()
plt.savefig('memory_capacity_profile.png', dpi=300)
plt.show()

# ============================================================================
# PASSO 5: Kernel Quality — Capacidade de separação não-linear
# ============================================================================

# Kernel Quality (Roy & Vetterli, 2007):
# Mede a dimensionalidade efetiva do espaço de estados do reservatório
# KQ alta → o reservatório gera representações diversas (bom para separação)
# KQ baixa → estados altamente correlacionados (pouca diversidade)

kq, kq_rank = kernel_quality(esn, n_patterns=100, seed=42)
print(f"\nKernel Quality:         {kq:.4f}")
print(f"Rank efetivo da matriz: {kq_rank:.4f}")

# ============================================================================
# PASSO 6: Echo State Property Index — O reservatório é estável?
# ============================================================================

# A Echo State Property (ESP) garante que o efeito das condições iniciais
# desaparece ao longo do tempo. É necessária para que o readout seja
# consistente. Se ESP é violada → predições instáveis.

esp_index = echo_state_property_index(
    esn,
    n_trials=20,
    n_timesteps=500,
    seed=42,
)

print(f"\nEcho State Property Index: {esp_index:.4f}")
print(f"ESP satisfeita: {'SIM ✓' if esp_index > 0.95 else 'NÃO ✗ — considere reduzir ρ'}")

# ============================================================================
# PASSO 7: Expoente de Lyapunov — Quão próximo do caos?
# ============================================================================

# O expoente máximo de Lyapunov (λ_max) mede a taxa de divergência 
# exponencial de trajetórias próximas:
# λ_max < 0 → regime ordenado (contrações)
# λ_max ≈ 0 → edge of chaos (criticidade!)
# λ_max > 0 → regime caótico

lambda_max = lyapunov_exponent(
    esn,
    n_timesteps=2000,
    seed=42,
)

print(f"\nλ_max (Lyapunov): {lambda_max:.4f}")
if lambda_max < -0.1:
    print("Regime: ORDENADO (subcrítico)")
elif lambda_max < 0.05:
    print("Regime: EDGE OF CHAOS (próximo à criticidade) ★")
else:
    print("Regime: CAÓTICO (supercrítico)")
    
# ============================================================================
# PASSO 8: Caracterização Completa — Tudo de uma vez
# ============================================================================

# characterize_reservoir() roda todas as métricas acima numa única chamada

full_report = characterize_reservoir(
    esn,
    max_delay=50,
    n_mc_timesteps=3000,
    n_kq_patterns=150,
    n_esp_trials=10,
    compute_lyapunov=True,
    seed=42,
)
from dataclasses import asdict

report_dict = asdict(full_report) if hasattr(full_report, '__dataclass_fields__') else vars(full_report)

print(f"\n{'='*60}")
print(f"RELATÓRIO COMPLETO DO RESERVATÓRIO — {subject} ({atlas})")
print(f"{'='*60}")
for key, value in report_dict.items():
    if isinstance(value, (float, np.floating)):
        print(f"  {key:30s}: {value:.4f}")
    elif isinstance(value, (int, np.integer)):
        print(f"  {key:30s}: {value}")

# ============================================================================
# PASSO 9: Sweep de ρ — O "Sweet Spot" do Conectoma
# ============================================================================

# Para cada valor de spectral radius, mede MC e KQ.
# Redes small-world (como conectomas) tipicamente têm um range ótimo
# mais largo que redes aleatórias (Damicelli et al., 2022).

sweep = spectral_radius_sweep(
    connectivity=SC,
    sr_values=np.arange(0.1, 1.5, 0.05),  # De 0.1 a 1.45
    leak_rate=0.3,
    n_timesteps=3000,
    max_delay=50,
    seed=42,
)

print(f"\nρ ótimo (max MC): {sweep['optimal_sr']:.2f}")
print(f"MC no ρ ótimo:    {sweep['memory_capacities'].max():.2f}")

# ── Plot: Dual-axis MC + KQ vs ρ ───────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

color_mc = '#2166AC'
color_kq = '#B2182B'

ax1.plot(sweep['sr_values'], sweep['memory_capacities'], 'o-',
         color=color_mc, lw=2, markersize=4, label='Memory Capacity')
ax1.set_xlabel('Spectral Radius (ρ)', fontsize=12)
ax1.set_ylabel('Memory Capacity', color=color_mc, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_mc)
ax1.axvline(sweep['optimal_sr'], ls='--', color='gray', alpha=0.5, lw=1)
ax1.annotate(f'ρ* = {sweep["optimal_sr"]:.2f}',
             xy=(sweep['optimal_sr'], sweep['memory_capacities'].max()),
             xytext=(sweep['optimal_sr'] + 0.1, sweep['memory_capacities'].max() * 0.95),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, color='gray')

ax2 = ax1.twinx()
ax2.plot(sweep['sr_values'], sweep['kernel_qualities'], 's-',
         color=color_kq, lw=2, markersize=4, label='Kernel Quality')
ax2.set_ylabel('Kernel Quality', color=color_kq, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_kq)

# Edge of chaos reference
ax1.axvspan(0.85, 1.05, alpha=0.08, color='gold', label='Edge of Chaos')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)

ax1.set_title(f'Spectral Radius Sweep — {subject} ({atlas})', fontweight='bold')
sns.despine(right=False)
plt.tight_layout()
plt.savefig('sr_sweep.png', dpi=300)
plt.show()

# ============================================================================
# PASSO 10: Conectoma vs Redes Nulas
# ============================================================================
# Compara o conectoma real do paciente com redes nulas (aleatória, small-world,
# ring) para demonstrar que a topologia biológica confere vantagem computacional.
#
# Referência: Damicelli et al. (2022), PLOS Computational Biology
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import fields, asdict

from sars.config import get_sc_path, get_timeseries_path
from sars.reservoir.esn import (
    ConnectomeReservoir,
    compare_reservoir_architectures,
)

# ── Configuração de plots ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Carregar dados ─────────────────────────────────────────────────────
subject = "sub-01"
atlas = "schaefer_100"

SC = np.load(get_sc_path(subject, atlas))
timeseries = np.load(get_timeseries_path(subject, atlas, strategy="acompcor"))

# ── Preparar sinais de input/target ────────────────────────────────────
# Dividir ROIs em duas metades: input e output (paradigma conn2res)
N_rois = SC.shape[0]
n_input = N_rois // 2

np.random.seed(42)
roi_order = np.random.permutation(N_rois)
input_rois = roi_order[:n_input]
output_rois = roi_order[n_input:]

u = timeseries[:, input_rois]   # Input: metade dos ROIs
y = timeseries[:, output_rois]  # Target: outra metade

print(f"Subject:      {subject}")
print(f"Atlas:        {atlas} ({N_rois} ROIs)")
print(f"Input ROIs:   {n_input}")
print(f"Output ROIs:  {N_rois - n_input}")
print(f"Timesteps:    {timeseries.shape[0]}")

# ============================================================================
# RODAR A COMPARAÇÃO
# ============================================================================

comparison = compare_reservoir_architectures(
    connectome=SC,
    input_signal=u,
    target_signal=y,
    spectral_radius=0.9,
    leak_rate=0.3,
    ridge_alpha=1e-5,
    wash_out=50,
    max_delay_mc=50,
    seed=42,
)

# ============================================================================
# EXTRAIR RESULTADOS
# ============================================================================

print(f"\n{'='*65}")
print(f"COMPARAÇÃO DE ARQUITETURAS — {subject} ({atlas})")
print(f"{'='*65}")
print(f"Melhor arquitetura: {comparison.best_architecture}\n")

# Converter métricas para formato acessível
# (metrics e task_scores são dicts indexados por nome da arquitetura)
arch_names = comparison.architecture_names

# Tentar extrair MC, KQ, e R² de cada arquitetura
arch_data = {}
for name in arch_names:
    m = comparison.metrics[name]
    t = comparison.task_scores[name]

    # metrics pode ser dict ou dataclass
    if hasattr(m, '__dataclass_fields__'):
        m = asdict(m)
    elif not isinstance(m, dict):
        m = vars(m) if hasattr(m, '__dict__') else {}

    if hasattr(t, '__dataclass_fields__'):
        t = asdict(t)
    elif not isinstance(t, dict):
        t = {'r2': t} if isinstance(t, (float, np.floating)) else {}

    arch_data[name] = {**m, **t}

# Imprimir tabela
# Descobrir quais chaves estão disponíveis
sample_keys = list(arch_data[arch_names[0]].keys())
print(f"Métricas disponíveis: {sample_keys}\n")

# Identificar chaves de MC, KQ, e R² (nomes podem variar)
mc_key = next((k for k in sample_keys if 'memory' in k.lower()), None)
kq_key = next((k for k in sample_keys if 'kernel' in k.lower()), None)
r2_key = next((k for k in sample_keys if 'r2' in k.lower()), None)

header_parts = [f"{'Arquitetura':<20s}"]
if mc_key: header_parts.append(f"{'MC':>10s}")
if kq_key: header_parts.append(f"{'KQ':>10s}")
if r2_key: header_parts.append(f"{'R²':>10s}")
print("".join(header_parts))
print("-" * len("".join(header_parts)))

for name in arch_names:
    d = arch_data[name]
    row = f"{name:<20s}"
    if mc_key: row += f"{d.get(mc_key, 0):>10.2f}"
    if kq_key: row += f"{d.get(kq_key, 0):>10.4f}"
    if r2_key: row += f"{d.get(r2_key, 0):>10.4f}"
    print(row)

# ============================================================================
# PLOT: Barras Agrupadas (Publication-Quality)
# ============================================================================

# Coletar valores para cada métrica
mc_vals = [arch_data[n].get(mc_key, 0) for n in arch_names] if mc_key else None
kq_vals = [arch_data[n].get(kq_key, 0) for n in arch_names] if kq_key else None
r2_vals = [arch_data[n].get(r2_key, 0) for n in arch_names] if r2_key else None

# Quantos painéis precisamos?
panels = [(mc_vals, 'Memory Capacity', mc_key),
          (kq_vals, 'Kernel Quality', kq_key),
          (r2_vals, 'Task R²', r2_key)]
panels = [(v, t, k) for v, t, k in panels if v is not None]
n_panels = len(panels)

# Paleta: conectoma destaca, nulos em tons neutros
colors_map = {
    'connectome': '#2166AC',
    'random':     '#AAAAAA',
    'small_world': '#FDB863',
    'ring':       '#E08214',
    'lattice':    '#D6604D',
}
colors = [colors_map.get(n, '#888888') for n in arch_names]

fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

for ax, (vals, title, key) in zip(axes, panels):
    bars = ax.bar(arch_names, vals, color=colors, edgecolor='black', lw=0.8)

    # Destacar o melhor
    best_idx = arch_names.index(comparison.best_architecture) if comparison.best_architecture in arch_names else 0
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)

    # Valor em cima de cada barra
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel(title)
    ax.set_title(title, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    sns.despine(ax=ax)

fig.suptitle(
    f'Conectoma vs Redes Nulas — {subject} ({atlas})',
    fontweight='bold', fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig('architecture_comparison.png', dpi=300)
plt.show()

# ============================================================================
# INTERPRETAÇÃO
# ============================================================================
print(f"\n{'='*65}")
print("INTERPRETAÇÃO")
print(f"{'='*65}")

if mc_key and comparison.best_architecture == 'connectome':
    conn_mc = arch_data['connectome'].get(mc_key, 0)
    rand_mc = arch_data.get('random', {}).get(mc_key, 0) if 'random' in arch_data else 0
    if rand_mc > 0:
        advantage = ((conn_mc - rand_mc) / rand_mc) * 100
        print(f"O conectoma real tem {advantage:+.1f}% de MC em relação à rede aleatória.")
    print("→ A topologia biológica confere vantagem computacional.")
elif mc_key:
    print(f"→ A melhor arquitetura foi '{comparison.best_architecture}', não o conectoma.")
    print("  Isso pode indicar degradação topológica neste paciente.")

# ============================================================================
# PASSO 11: Reservatório Adaptativo com Plasticidade Hebbiana
# ============================================================================

adaptive_esn = AdaptiveReservoir(
    connectivity=SC,
    spectral_radius=0.9,
    leak_rate=0.3,
    learning_rate=0.001,     # η — plasticidade gradual
    decay_rate=0.01,         # λ — homeostase
    plasticity_mask=None,    # Todas as conexões são plásticas
    renormalize_every=10,    # Renormaliza ρ a cada 10 timesteps
    seed=42,
)

# ── Treinar com os dados reais do paciente ──────────────────────────────
# Dividir ROIs em input/output
N_rois = timeseries.shape[1]
n_input = N_rois // 2

np.random.seed(42)
roi_order = np.random.permutation(N_rois)
input_rois = roi_order[:n_input]
output_rois = roi_order[n_input:]

# Split temporal: 70% treino, 30% teste
T = timeseries.shape[0]
split = int(0.7 * T)

u_train = timeseries[:split, input_rois]
y_train = timeseries[:split, output_rois]
u_test = timeseries[split:, input_rois]
y_test = timeseries[split:, output_rois]

# Treinar (com plasticidade ativa durante o fit)
adaptive_esn.fit(u_train, y_train, wash_out=50)

# ── Analisar evolução dos pesos ─────────────────────────────────────────
W_final = adaptive_esn.W.copy()
W_initial = adaptive_esn.W_initial.copy()
delta_W = W_final - W_initial

print(f"\n{'='*60}")
print(f"EVOLUÇÃO HEBBIANA — {subject} ({atlas})")
print(f"{'='*60}")
print(f"||ΔW||_F:           {np.linalg.norm(delta_W, 'fro'):.4f}")
print(f"Conexões potenciadas: {(delta_W > 0.001).sum()}")
print(f"Conexões deprimidas:  {(delta_W < -0.001).sum()}")
print(f"ρ final:             {np.max(np.abs(np.linalg.eigvals(W_final))):.4f}")

# ── Plot: Matriz de mudança de pesos ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# W inicial
im0 = axes[0].imshow(W_initial, cmap='Blues', aspect='equal')
axes[0].set_title('W inicial (SC normalizado)', fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# W final
im1 = axes[1].imshow(W_final, cmap='Blues', aspect='equal')
axes[1].set_title('W final (pós-Hebbian)', fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# ΔW
vmax = np.percentile(np.abs(delta_W), 99)
im2 = axes[2].imshow(delta_W, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
axes[2].set_title('ΔW (potenciação / depressão)', fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

for ax in axes:
    ax.set_xlabel('ROI')
    ax.set_ylabel('ROI')

plt.tight_layout()
plt.savefig('hebbian_evolution.png', dpi=300)
plt.show()

# ============================================================================
# EXEMPLO AVANÇADO: Plasticidade apenas intra-DMN
# ============================================================================

import pandas as pd

# Carregar labels do Schaefer para identificar redes
labels = pd.read_csv('/mnt/nvme1n1p1/sars_cov_2_project/info/atlases/labels_schaefer_100_7networks.csv')

# Identificar ROIs da Default Mode Network
dmn_mask = labels['label_roi'].str.contains('Default', case=False)
dmn_indices = labels.loc[dmn_mask, 'value_roi'].values - 1  # 0-indexed

# Criar máscara de plasticidade: só conexões intra-DMN são plásticas
plasticity_mask = np.zeros_like(SC)
for i in dmn_indices:
    for j in dmn_indices:
        if i != j:
            plasticity_mask[i, j] = 1.0

print(f"ROIs na DMN: {len(dmn_indices)}")
print(f"Conexões plásticas: {int(plasticity_mask.sum())} / {SC.size}")

# Criar reservatório com plasticidade restrita
esn_dmn_plastic = AdaptiveReservoir(
    connectivity=SC,
    spectral_radius=0.9,
    leak_rate=0.3,
    learning_rate=0.001,
    decay_rate=0.01,
    plasticity_mask=plasticity_mask,  # ← Só intra-DMN!
    seed=42,
)

# Treinar e comparar com plasticidade global...

# ============================================================================
# PASSO 12: Análise Sistemática Através dos 4 Atlas
# ============================================================================

atlas_list = ["schaefer_100", "aal3", "brainnetome"]
# Nota: synthseg_86 tem menos ROIs, incluir se desejado

results_by_atlas = {}

from dataclasses import asdict

for atlas_name in atlas_list:
    print(f"\n>>> Processando {atlas_name}...")

    sc = np.load(get_sc_path(subject, atlas_name))
    ts = np.load(get_timeseries_path(subject, atlas_name, strategy="acompcor"))

    # Verificar compatibilidade SC ↔ TS
    if sc.shape[0] != ts.shape[1]:
        print(f"  ⚠ MISMATCH: SC={sc.shape[0]}, TS={ts.shape[1]} — pulando")
        continue

    fc_result = fc_prediction_task(
        sc=sc, timeseries=ts,
        spectral_radius=0.9, leak_rate=0.3,
        seed=42,
    )

    esn_atlas = ConnectomeReservoir(sc, spectral_radius=0.9, leak_rate=0.3, seed=42)
    metrics = characterize_reservoir(esn_atlas, seed=42)
    metrics_dict = asdict(metrics)  # ← converte dataclass → dict

    results_by_atlas[atlas_name] = {
        'n_rois': sc.shape[0],
        'r2_global': fc_result['r2_global'],
        'r2_per_roi_mean': fc_result['r2_per_roi'].mean(),
        **{k: v for k, v in metrics_dict.items() if isinstance(v, (float, int, np.floating))},
    }
print("Chaves disponíveis:", list(results_by_atlas[atlas_list[0]].keys()))
# ── Tabela comparativa ──────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"COMPARAÇÃO MULTI-ATLAS — {subject}")
print(f"{'='*80}")
print(f"{'Atlas':<20s} {'N ROIs':>8s} {'R² (FC)':>10s} {'MC':>10s} {'KQ':>10s} {'λ_max':>10s}")
print(f"{'-'*68}")
for atlas_name, data in results_by_atlas.items():
    print(f"{atlas_name:<20s} {data['n_rois']:>8d} "
          f"{data['r2_global']:>10.4f} "
          f"{data.get('memory_capacity_total', 0):>10.2f} "
          f"{data.get('kernel_quality', 0):>10.4f} "
          f"{data.get('lyapunov_exponent', 0):>10.4f}")
    
# ============================================================================
# PASSO 13: Análise de Grupo (N=23 pacientes)
# ============================================================================

atlas = "schaefer_100"
all_results = []

for sub in ALL_SUBJECT_IDS:
    print(f"  Processing {sub}...", end=" ")
    try:
        sc = np.load(get_sc_path(sub, atlas))
        ts = np.load(get_timeseries_path(sub, atlas, strategy="acompcor"))

        # FC prediction
        fc_res = fc_prediction_task(sc=sc, timeseries=ts, seed=42)

        # Métricas rápidas
        esn_sub = ConnectomeReservoir(sc, spectral_radius=0.9, leak_rate=0.3, seed=42)
        mc_total, _ = memory_capacity(esn_sub, max_delay=50, seed=42)
        kq, _ = kernel_quality(esn_sub, seed=42)

        all_results.append({
            'subject': sub,
            'r2_global': fc_res['r2_global'],
            'r2_mean': fc_res['r2_per_roi'].mean(),
            'memory_capacity': mc_total,
            'kernel_quality': kq,
        })
        print(f"R²={fc_res['r2_global']:.3f}, MC={mc_total:.1f}")

    except Exception as e:
        print(f"ERRO: {e}")

# ── Converter para DataFrame ────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
print(f"\n{df_results.describe()}")

# ── Plot: Distribuição de MC e R² no grupo ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].hist(df_results['memory_capacity'], bins=10, color='steelblue',
             edgecolor='navy', alpha=0.7)
axes[0].set_xlabel('Memory Capacity')
axes[0].set_ylabel('Contagem')
axes[0].set_title('Distribuição de MC', fontweight='bold')
axes[0].axvline(df_results['memory_capacity'].mean(), ls='--', color='red', lw=1.5)

axes[1].hist(df_results['r2_global'], bins=10, color='#E08214',
             edgecolor='#B35806', alpha=0.7)
axes[1].set_xlabel('R² (FC Prediction)')
axes[1].set_ylabel('Contagem')
axes[1].set_title('Distribuição de R² (FC)', fontweight='bold')

# Scatter: MC vs R²
axes[2].scatter(df_results['memory_capacity'], df_results['r2_global'],
                c='steelblue', edgecolors='navy', s=60, alpha=0.8)
# Regressão
slope, intercept, r, p, _ = stats.linregress(
    df_results['memory_capacity'], df_results['r2_global']
)
x_fit = np.linspace(df_results['memory_capacity'].min(), df_results['memory_capacity'].max(), 50)
axes[2].plot(x_fit, slope * x_fit + intercept, 'r--', lw=1.5)
axes[2].set_xlabel('Memory Capacity')
axes[2].set_ylabel('R² (FC Prediction)')
axes[2].set_title(f'MC vs R² (r={r:.3f}, p={p:.3f})', fontweight='bold')

for ax in axes:
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('group_reservoir_analysis.png', dpi=300)
plt.show()

# ============================================================================
# PASSO 14: Espectro do Reservatório — Modos Oscilatórios e Timescales
# ============================================================================

spec = spectral_analysis(SC, spectral_radius=0.9)
print(f"\nAnálise Espectral do Reservatório:")
print(f"  Gap espectral:        {spec.get('spectral_gap', 'N/A')}")
print(f"  Timescale dominante:  {spec.get('dominant_timescale', 'N/A')}")
print(f"  N° modos oscilatórios: {spec.get('n_oscillatory_modes', 'N/A')}")

# Plot: Eigenspectrum no plano complexo
eigenvalues = spec.get('eigenvalues', np.linalg.eigvals(esn.W))

fig, ax = plt.subplots(figsize=(6, 6))
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.3, label='|z|=1')

ax.scatter(eigenvalues.real, eigenvalues.imag, c=np.abs(eigenvalues),
           cmap='viridis', edgecolors='black', lw=0.3, s=30, zorder=5)
ax.set_xlabel('Re(λ)')
ax.set_ylabel('Im(λ)')
ax.set_title(f'Eigenspectrum do Reservatório — {subject}', fontweight='bold')
ax.set_aspect('equal')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.legend(frameon=False)
plt.colorbar(ax.collections[0], ax=ax, label='|λ|')
sns.despine()
plt.tight_layout()
plt.savefig('eigenspectrum.png', dpi=300)
plt.show()

# ============================================================================
# PASSO 15: Converter para ReservoirPy (se instalado)
# ============================================================================

# O ConnectomeReservoir tem uma implementação NumPy pura por padrão,
# mas pode ser convertido para ReservoirPy para funcionalidades avançadas
# (feedback, deep ESN, etc.)

try:
    rpy_model = esn_atlas.to_reservoirpy()
    print("✓ Modelo ReservoirPy criado com sucesso!")
    print(f"  Pipeline: {rpy_model}")

    # Usar a API nativa do ReservoirPy
    print(rpy_model.nodes)  # ver os nós disponíveis

    # Rodar só o reservatório (sem o readout)
    reservoir_node = rpy_model.nodes[0]
    states = reservoir_node.run(u_train)
    print(f"States shape: {states.shape}")

except ImportError:
    print("ReservoirPy não instalado — usando backend NumPy puro.")
    print("Para instalar: pip install reservoirpy")
    
# ============================================================================
# PASSO 16: SC-FC Decoupling via Reservoir
# ============================================================================

from sars.reservoir.reservoir_dynamics import analyze_sc_fc_decoupling

decoupling = analyze_sc_fc_decoupling(
    sc=SC, fc=FC, timeseries=timeseries,
    spectral_radius=0.9,
    compute_sdi_flag=True,
    atlas_name="schaefer_100",
    seed=42,
)

print(f"Lyapunov:       {decoupling.lyapunov_exponent:.4f}")
print(f"Edge of chaos:  {decoupling.edge_of_chaos_distance:.4f}")
print(f"Entropy:        {decoupling.reservoir_entropy:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (private)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lyapunov_exponent(
    W: np.ndarray,
    states: np.ndarray,
    leak_rate: float = 0.3,
) -> float:
    """Estimate the maximum Lyapunov exponent from reservoir dynamics.

    Uses the Jacobian-based method: at each timestep, the local Jacobian
    of the ESN update rule is:

        J(t) = (1 - α)·I + α · diag(f'(h(t))) · W

    where f'(h) = 1 - tanh²(h) for tanh activation.

    The maximum Lyapunov exponent is estimated as the time-averaged
    log of the largest singular value of J(t).

    Parameters
    ----------
    W : np.ndarray, shape (N, N)
        Reservoir weight matrix (already scaled by spectral radius).
    states : np.ndarray, shape (T, N)
        Reservoir states from a simulation run.
    leak_rate : float
        Leak rate α used during simulation.

    Returns
    -------
    float
        Estimated maximum Lyapunov exponent.
    """
    T, N = states.shape
    alpha = leak_rate

    # Use a subsample for efficiency (every 5th timestep)
    step = max(1, T // 200)
    indices = range(0, T, step)

    log_svs = []
    I = np.eye(N)
    for t in indices:
        x = states[t]
        # Derivative of tanh: f'(h) = 1 - x² (since x = tanh(h))
        f_prime = 1.0 - x ** 2
        # Local Jacobian
        J = (1.0 - alpha) * I + alpha * np.diag(f_prime) @ W
        # Largest singular value
        s_max = np.linalg.norm(J, ord=2)
        if s_max > 0:
            log_svs.append(np.log(s_max))

    if not log_svs:
        return 0.0

    return float(np.mean(log_svs))


def _compute_reservoir_entropy(
    states: np.ndarray,
    n_bins: int = 30,
) -> float:
    """Compute average Shannon entropy of reservoir neuron activations.

    Each neuron's activation distribution is discretized into bins,
    and the Shannon entropy is computed. The result is averaged across
    all neurons and normalized to [0, 1] (dividing by log(n_bins)).

    Higher values indicate richer dynamical repertoire.

    Parameters
    ----------
    states : np.ndarray, shape (T, N)
        Reservoir states.
    n_bins : int
        Number of bins for histogram. Default 30.

    Returns
    -------
    float
        Mean normalized entropy across neurons, in [0, 1].
    """
    T, N = states.shape
    entropies = np.zeros(N)

    for i in range(N):
        counts, _ = np.histogram(states[:, i], bins=n_bins)
        # Normalize to probability
        p = counts / counts.sum()
        # Remove zeros (log(0) is undefined)
        p = p[p > 0]
        # Shannon entropy
        entropies[i] = -np.sum(p * np.log2(p))

    # Normalize by maximum possible entropy
    max_entropy = np.log2(n_bins)
    mean_entropy = float(np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.0

    return mean_entropy