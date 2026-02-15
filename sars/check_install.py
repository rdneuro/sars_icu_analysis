#!/usr/bin/env python3
"""
SARS Library — Verificação de Instalação (UV)
==============================================

Roda após `uv sync` para confirmar que tudo está no lugar.

Uso:
    python check_install.py
    python check_install.py --full    # inclui GPU + diffusion
"""

import sys
import importlib
from pathlib import Path

# ─── Cores ANSI ─────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def check(pkg: str, min_version: str = None, import_name: str = None) -> bool:
    """Tenta importar um pacote e verifica versão mínima."""
    name = import_name or pkg.replace("-", "_")
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {GREEN}✓{RESET} {pkg:.<30s} {ver}")
        return True
    except ImportError:
        print(f"  {RED}✗{RESET} {pkg:.<30s} não encontrado")
        return False
    except Exception as e:
        print(f"  {YELLOW}?{RESET} {pkg:.<30s} erro: {e}")
        return False


def main():
    full = "--full" in sys.argv

    print(f"\n{BOLD}{'='*60}")
    print("  SARS Library — Verificação de Instalação")
    print(f"{'='*60}{RESET}\n")

    # ── Python ──────────────────────────────────────────────────────────
    print(f"{BOLD}Python:{RESET}")
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    symbol = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {symbol} Python {v.major}.{v.minor}.{v.micro} "
          f"({'OK' if ok else 'REQUER >= 3.10'})")
    print(f"  → {sys.executable}\n")

    results = {"core": [], "neuro": [], "reservoir": [], "viz": [],
               "gpu": [], "diffusion": []}

    # ── Core Scientific ─────────────────────────────────────────────────
    print(f"{BOLD}Core Scientific:{RESET}")
    for pkg in ["numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "h5py"]:
        imp = "sklearn" if pkg == "scikit-learn" else None
        results["core"].append(check(pkg, import_name=imp))

    # ── Neuroimaging ────────────────────────────────────────────────────
    print(f"\n{BOLD}Neuroimaging:{RESET}")
    for pkg in ["nibabel", "nilearn"]:
        results["neuro"].append(check(pkg))

    # ── Network Analysis ────────────────────────────────────────────────
    print(f"\n{BOLD}Network Analysis:{RESET}")
    for pkg, imp in [("networkx", None), ("bctpy", "bct")]:
        results["neuro"].append(check(pkg, import_name=imp))

    # ── Reservoir Computing ─────────────────────────────────────────────
    print(f"\n{BOLD}Reservoir Computing:{RESET}")
    results["reservoir"].append(check("reservoirpy"))

    # ── Visualization ───────────────────────────────────────────────────
    print(f"\n{BOLD}Visualization:{RESET}")
    for pkg in ["matplotlib", "seaborn", "plotly"]:
        results["viz"].append(check(pkg))

    # ── Utilities ───────────────────────────────────────────────────────
    print(f"\n{BOLD}Utilities:{RESET}")
    for pkg in ["tqdm", "joblib", "yaml", "jinja2"]:
        imp = "yaml" if pkg == "yaml" else None
        check(pkg, import_name=imp)

    # ── GPU (opcional) ──────────────────────────────────────────────────
    if full:
        print(f"\n{BOLD}GPU / Deep Learning:{RESET}")
        results["gpu"].append(check("torch"))
        results["gpu"].append(check("torch-geometric", import_name="torch_geometric"))

        # CUDA check
        try:
            import torch
            if torch.cuda.is_available():
                dev = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_mem / 1e9
                print(f"  {GREEN}✓{RESET} {'CUDA':.<30s} {dev} ({mem:.1f} GB)")
            else:
                print(f"  {YELLOW}⚠{RESET} {'CUDA':.<30s} não disponível (CPU-only)")
        except:
            pass

    # ── Diffusion (opcional) ────────────────────────────────────────────
    if full:
        print(f"\n{BOLD}Diffusion MRI:{RESET}")
        for pkg in ["dipy", "amico"]:
            results["diffusion"].append(check(pkg))

    # ── SARS library itself ─────────────────────────────────────────────
    print(f"\n{BOLD}SARS Library:{RESET}")
    try:
        import sars
        ver = getattr(sars, "__version__", "dev")
        print(f"  {GREEN}✓{RESET} {'sars':.<30s} {ver}")

        # Testar subpacotes
        subpkgs = [
            ("sars.config", "config"),
            ("sars.reservoir.esn", "reservoir.esn"),
            ("sars.reservoir.reservoir_dynamics", "reservoir.dynamics"),
        ]
        for mod_path, label in subpkgs:
            try:
                importlib.import_module(mod_path)
                print(f"  {GREEN}✓{RESET}   └─ {label}")
            except ImportError as e:
                print(f"  {YELLOW}⚠{RESET}   └─ {label}: {e}")

    except ImportError:
        print(f"  {YELLOW}⚠{RESET} sars não instalado como pacote (use: uv pip install -e .)")
        print(f"      Isso é esperado se você está rodando scripts diretamente.")

    # ── Resumo ──────────────────────────────────────────────────────────
    all_core = results["core"] + results["neuro"] + results["reservoir"] + results["viz"]
    n_ok = sum(all_core)
    n_total = len(all_core)

    print(f"\n{BOLD}{'='*60}")
    if n_ok == n_total:
        print(f"  {GREEN}✓ TODAS as {n_total} dependências core instaladas!{RESET}")
    else:
        print(f"  {YELLOW}⚠ {n_ok}/{n_total} dependências core instaladas{RESET}")
        print(f"  Execute: uv sync")
    print(f"{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
