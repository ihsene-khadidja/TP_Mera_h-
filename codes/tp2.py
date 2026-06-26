"""
TP N°2 — Optimization Problem Initialization (Part 2)
USTHB · Master 2 SII · Module MÉTA · 2025/2026

Fonctionnalités :
  I.   Population Initialization (random + CSV)
  II.  Population Evaluation (best/worst + scatter plot + 3D)
  III. Running Multiple Populations (statistiques)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os, sys, random

# ─────────────────────────────────────────────
#  BENCHMARK FUNCTIONS
# ─────────────────────────────────────────────

class BenchmarkFunctions:

    CONFIG = {
        'F1':  dict(name='Sphere',        type='Unimodal',   range=(-100, 100),    optimal=0,
                    formula='f(x) = Σ xᵢ²'),
        'F2':  dict(name='Schwefel 2.22', type='Unimodal',   range=(-10, 10),      optimal=0,
                    formula='f(x) = Σ|xᵢ| + Π|xᵢ|'),
        'F5':  dict(name='Rosenbrock',    type='Unimodal',   range=(-30, 30),      optimal=0,
                    formula='f(x) = Σ[100(xᵢ₊₁-xᵢ²)² + (1-xᵢ)²]'),
        'F7':  dict(name='Schwefel 1.2',  type='Unimodal',   range=(-128, 128),    optimal=0,
                    formula='f(x) = Σ(Σxⱼ)² + rand'),
        'F8':  dict(name='Schwefel 2.26', type='Multimodal', range=(-500, 500),    optimal=-418.9829,
                    formula='f(x) = Σ[−xᵢ·sin(√|xᵢ|)]'),
        'F9':  dict(name='Rastrigin',     type='Multimodal', range=(-5.12, 5.12),  optimal=0,
                    formula='f(x) = Σ[xᵢ² − 10cos(2πxᵢ) + 10]'),
        'F11': dict(name='Griewank',      type='Multimodal', range=(-600, 600),    optimal=0,
                    formula='f(x) = 1 + (1/4000)Σxᵢ² − Πcos(xᵢ/√i)'),
    }

    @staticmethod
    def evaluate(fname: str, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if fname == 'F1':
            return float(np.sum(x**2))
        elif fname == 'F2':
            return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))
        elif fname == 'F5':
            return float(np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))
        elif fname == 'F7':
            return float(sum((np.sum(x[:i+1]))**2 for i in range(len(x)))) + random.random()
        elif fname == 'F8':
            return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))
        elif fname == 'F9':
            return float(np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10))
        elif fname == 'F11':
            D = len(x)
            return float(1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,D+1)))))
        else:
            raise ValueError(f'Unknown function: {fname}')

    @classmethod
    def rand_solution(cls, fname: str, dim: int) -> np.ndarray:
        lo, hi = cls.CONFIG[fname]['range']
        return np.random.uniform(lo, hi, dim)

    @classmethod
    def optimal_value(cls, fname: str, dim: int) -> float:
        opt = cls.CONFIG[fname]['optimal']
        return opt * dim if fname == 'F8' else opt

    @classmethod
    def contour_grid(cls, fname: str, steps: int = 60):
        lo, hi = cls.CONFIG[fname]['range']
        x1 = np.linspace(lo, hi, steps)
        x2 = np.linspace(lo, hi, steps)
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)
        for i in range(steps):
            for j in range(steps):
                sol = np.zeros(30)
                sol[0] = X1[i,j]; sol[1] = X2[i,j]
                Z[i,j] = cls.evaluate(fname, sol)
        return X1, X2, Z


BF = BenchmarkFunctions()

# ─────────────────────────────────────────────
#  I. POPULATION INITIALIZATION
# ─────────────────────────────────────────────

class Population:

    def __init__(self, fname: str, solutions: np.ndarray):
        self.fname = fname
        self.solutions = solutions          # shape (N, D)
        self.size = solutions.shape[0]
        self.dim  = solutions.shape[1]
        self.fitnesses: np.ndarray | None = None

    # ── random generation ──
    @classmethod
    def generate_random(cls, fname: str, size: int, dim: int = 30) -> 'Population':
        sols = np.array([BF.rand_solution(fname, dim) for _ in range(size)])
        pop = cls(fname, sols)
        print(f'[I] Population generated randomly: {size} solutions × D={dim} for {fname}')
        return pop

    # ── load from CSV ──
    @classmethod
    def load_csv(cls, filepath: str, fname: str) -> 'Population':
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f'CSV not found: {filepath}')
        # auto-detect separator
        with open(path) as f:
            first = f.readline()
        sep = ';' if ';' in first else ','
        df = pd.read_csv(path, sep=sep, header=None)
        sols = df.values.astype(float)
        pop = cls(fname, sols)
        print(f'[I] Population loaded from CSV: {sols.shape[0]} solutions × D={sols.shape[1]} for {fname}')
        return pop

    # ── display (first few rows) ──
    def show(self, max_rows: int = 5, max_cols: int = 8):
        print(f'\n{"─"*60}')
        print(f'  Population — {self.fname} ({BF.CONFIG[self.fname]["name"]})')
        print(f'  Size: {self.size}   Dimension: {self.dim}')
        print(f'{"─"*60}')
        header = '  #  |  ' + '  '.join(f'x{j+1:>3}' for j in range(min(self.dim, max_cols)))
        if self.dim > max_cols: header += '   ...'
        print(header)
        print('  ' + '-'*len(header))
        for i in range(min(self.size, max_rows)):
            row = f'{i+1:>4} |  ' + '  '.join(f'{self.solutions[i,j]:>6.2f}' for j in range(min(self.dim, max_cols)))
            if self.dim > max_cols: row += '   ...'
            print(row)
        if self.size > max_rows:
            print(f'  ... ({self.size - max_rows} more rows)')
        print()


# ─────────────────────────────────────────────
#  II. POPULATION EVALUATION
# ─────────────────────────────────────────────

class PopulationEvaluator:

    def __init__(self, pop: Population):
        self.pop = pop
        self.fitnesses = np.array([BF.evaluate(pop.fname, pop.solutions[i]) for i in range(pop.size)])
        pop.fitnesses = self.fitnesses
        self.best_idx  = int(np.argmin(self.fitnesses))
        self.worst_idx = int(np.argmax(self.fitnesses))

    @property
    def best(self)  -> float: return float(self.fitnesses[self.best_idx])
    @property
    def worst(self) -> float: return float(self.fitnesses[self.worst_idx])

    def report(self):
        fname = self.pop.fname
        cfg   = BF.CONFIG[fname]
        print(f'\n{"═"*60}')
        print(f'  II. Population Evaluation — {fname} ({cfg["name"]})')
        print(f'{"═"*60}')
        print(f'  Population size  : {self.pop.size}')
        print(f'  Dimension        : {self.pop.dim}')
        print(f'  Formula          : {cfg["formula"]}')
        print(f'  Type             : {cfg["type"]}')
        print(f'  Optimal value    : {BF.optimal_value(fname, self.pop.dim):.6f}')
        print(f'  ─────────────────────────────────')
        print(f'  ✓ Best  fitness  : \033[92m{self.best:.6f}\033[0m  (solution #{self.best_idx+1})')
        print(f'  ✗ Worst fitness  : \033[91m{self.worst:.6f}\033[0m  (solution #{self.worst_idx+1})')
        print()

    def plot(self, save_path: str | None = None, show: bool = False):
        fname = self.pop.fname
        cfg   = BF.CONFIG[fname]
        lo, hi = cfg['range']

        fig = plt.figure(figsize=(16, 7), facecolor='#0d1117')
        fig.suptitle(
            f'Population Evaluation — {fname}: {cfg["name"]} ({cfg["type"]})\n{cfg["formula"]}',
            color='white', fontsize=13, y=0.98, fontweight='bold'
        )
        gs = GridSpec(1, 2, figure=fig, wspace=0.35)

        # ── 3D Surface ──
        ax3d = fig.add_subplot(gs[0, 0], projection='3d')
        ax3d.set_facecolor('#0d1117')
        X1, X2, Z = BF.contour_grid(fname, steps=50)
        surf = ax3d.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.85, linewidth=0, antialiased=True)
        ax3d.set_xlabel('x₁', color='#8892aa', fontsize=9)
        ax3d.set_ylabel('x₂', color='#8892aa', fontsize=9)
        ax3d.set_zlabel('f(x)', color='#8892aa', fontsize=9)
        ax3d.set_title('3D Surface', color='#c9d1d9', fontsize=11, pad=8)
        ax3d.tick_params(colors='#555f77', labelsize=7)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        for spine in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
            spine.pane.set_edgecolor('#2a2f3d')

        # ── Scatter + Contour ──
        ax2d = fig.add_subplot(gs[0, 1])
        ax2d.set_facecolor('#0d1117')
        cp = ax2d.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6, linewidths=0.8)
        ax2d.set_xlim(lo, hi); ax2d.set_ylim(lo, hi)

        # all solutions
        px = self.pop.solutions[:, 0]
        py = self.pop.solutions[:, 1]
        ax2d.scatter(px, py, c='#4f8ef7', s=40, alpha=0.8, zorder=5,
                     edgecolors='white', linewidths=0.4, label='Solutions')

        # best solution
        bx = self.pop.solutions[self.best_idx, 0]
        by = self.pop.solutions[self.best_idx, 1]
        ax2d.scatter([bx], [by], c='#4ff7a2', s=160, zorder=10,
                     marker='*', edgecolors='white', linewidths=0.8, label=f'Best ({self.best:.2f})')

        ax2d.set_xlabel('x₁', color='#8892aa', fontsize=10)
        ax2d.set_ylabel('x₂', color='#8892aa', fontsize=10)
        ax2d.set_title(f'Search History ({fname})', color='#c9d1d9', fontsize=11)
        ax2d.tick_params(colors='#555f77')
        for sp in ax2d.spines.values(): sp.set_edgecolor('#2a2f3d')
        legend = ax2d.legend(facecolor='#1a1e2a', edgecolor='#2a2f3d',
                              labelcolor='white', fontsize=9)

        # annotation box
        info = f'Best:  {self.best:.4f}\nWorst: {self.worst:.4f}\nN={self.pop.size}'
        ax2d.text(0.03, 0.97, info, transform=ax2d.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1e2a',
                            edgecolor='#4f8ef7', alpha=0.9),
                  color='white', fontfamily='monospace')

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f'  [Plot saved] {save_path}')
        if show:
            plt.show()
        plt.close()


# ─────────────────────────────────────────────
#  III. MULTIPLE POPULATIONS / RUNS
# ─────────────────────────────────────────────

class MultipleRunsEvaluator:

    def __init__(self, fname: str, pop_size: int, n_runs: int, dim: int = 30):
        self.fname    = fname
        self.pop_size = pop_size
        self.n_runs   = n_runs
        self.dim      = dim
        self.results: list[dict] = []

    def run(self):
        print(f'\n[III] Running {self.n_runs} populations of size {self.pop_size} on {self.fname}...')
        self.results = []
        for r in range(self.n_runs):
            pop = Population.generate_random(self.fname, self.pop_size, self.dim)
            fits = np.array([BF.evaluate(self.fname, pop.solutions[i]) for i in range(self.pop_size)])
            best_idx = int(np.argmin(fits))
            self.results.append({
                'run':       r + 1,
                'best':      float(fits.min()),
                'worst':     float(fits.max()),
                'best_sol':  pop.solutions[best_idx].copy(),
                'all_sols':  pop.solutions.copy(),
                'all_fits':  fits.copy(),
            })
        print(f'  Done.\n')

    # ── Metrics ──
    @property
    def bests(self) -> np.ndarray:
        return np.array([r['best'] for r in self.results])

    @property
    def best_all(self)  -> float: return float(self.bests.min())
    @property
    def worst_all(self) -> float: return float(max(r['worst'] for r in self.results))
    @property
    def mean(self)      -> float: return float(self.bests.mean())
    @property
    def std(self)       -> float: return float(self.bests.std(ddof=1)) if len(self.bests)>1 else 0.0

    def report(self):
        fname = self.fname
        cfg   = BF.CONFIG[fname]
        print(f'\n{"═"*60}')
        print(f'  III. Multiple Runs — {fname} ({cfg["name"]})')
        print(f'{"═"*60}')
        print(f'  Runs        : {self.n_runs}')
        print(f'  Pop. size   : {self.pop_size}')
        print(f'  Dimension   : {self.dim}')
        print()

        # Per-run table
        print(f'  {"Run":>4}  {"Best Fitness":>16}  {"Worst Fitness":>16}')
        print('  ' + '─'*42)
        best_run_idx = int(np.argmin(self.bests))
        for i, r in enumerate(self.results):
            marker = ' ←' if i == best_run_idx else ''
            print(f'  {r["run"]:>4}  {r["best"]:>16.6f}  {r["worst"]:>16.6f}{marker}')

        print()
        print(f'  {"─"*42}')
        print(f'  Best   (min best)  : \033[92m{self.best_all:.6f}\033[0m')
        print(f'  Worst  (max worst) : \033[91m{self.worst_all:.6f}\033[0m')
        print(f'  Mean   (avg error) : \033[93m{self.mean:.6f}\033[0m')
        print(f'  STD                : \033[94m{self.std:.6f}\033[0m')
        print()

        # Interpretation
        print('  Interpretation:')
        if self.std < self.mean * 0.1:
            print('  → Small STD + context suggests stable convergence.')
            if self.mean < 1e4:
                print('  → Good balance: algorithm converges consistently.')
            else:
                print('  → But large Mean: may be stuck in local optima.')
        else:
            print('  → Large STD: high variability across runs (poor stability).')
        print()

    def plot(self, save_path: str | None = None, show: bool = False):
        fname = self.fname
        cfg   = BF.CONFIG[fname]
        lo, hi = cfg['range']

        fig = plt.figure(figsize=(16, 12), facecolor='#0d1117')
        fig.suptitle(
            f'Multiple Runs Evaluation — {fname}: {cfg["name"]}\n'
            f'Runs={self.n_runs}, Pop. Size={self.pop_size}, D={self.dim}',
            color='white', fontsize=13, y=0.98, fontweight='bold'
        )
        gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.45)

        X1, X2, Z = BF.contour_grid(fname, steps=50)

        # ── Scatter: all solutions (all runs) ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#0d1117')
        ax1.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.5, linewidths=0.7)

        colors = cm.plasma(np.linspace(0.2, 0.9, self.n_runs))
        for i, r in enumerate(self.results):
            sols = r['all_sols']
            ax1.scatter(sols[:, 0], sols[:, 1], color=colors[i], s=12, alpha=0.35, zorder=3)

        # Best per run
        best_run_idx = int(np.argmin(self.bests))
        for i, r in enumerate(self.results):
            bsol = r['best_sol']
            ax1.scatter([bsol[0]], [bsol[1]], color='#4ff7a2', s=80, zorder=8,
                        marker='*', edgecolors='white', linewidths=0.5,
                        label='Best/Run' if i == 0 else '')

        ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
        ax1.set_xlabel('x₁', color='#8892aa')
        ax1.set_ylabel('x₂', color='#8892aa')
        ax1.set_title('Search History — All Runs', color='#c9d1d9', fontsize=11)
        ax1.tick_params(colors='#555f77')
        for sp in ax1.spines.values(): sp.set_edgecolor('#2a2f3d')
        ax1.legend(facecolor='#1a1e2a', edgecolor='#2a2f3d', labelcolor='white', fontsize=8)

        # ── Box plot ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#0d1117')
        bp = ax2.boxplot(self.bests, patch_artist=True,
                         boxprops=dict(facecolor=(0.31, 0.56, 0.97, 0.3), color='#4f8ef7'),
                         whiskerprops=dict(color='#4f8ef7'),
                         capprops=dict(color='#4f8ef7'),
                         medianprops=dict(color='#4ff7a2', linewidth=2),
                         flierprops=dict(marker='o', color='#f7a24f', markersize=5))
        ax2.scatter(np.ones(len(self.bests)) + np.random.uniform(-0.2, 0.2, len(self.bests)),
                    self.bests, color='#4f8ef7', s=25, alpha=0.7, zorder=5)
        ax2.set_xticks([1])
        ax2.set_xticklabels([fname], color='#8892aa')
        ax2.set_ylabel('Best Fitness', color='#8892aa')
        ax2.set_title('Best Fitness Distribution', color='#c9d1d9', fontsize=11)
        ax2.tick_params(colors='#555f77')
        for sp in ax2.spines.values(): sp.set_edgecolor('#2a2f3d')

        # ── Bar chart: best per run ──
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#0d1117')
        run_ids = [r['run'] for r in self.results]
        bar_colors = ['#4ff7a2' if i == best_run_idx else '#4f8ef7' for i in range(self.n_runs)]
        ax3.bar(run_ids, self.bests, color=bar_colors, alpha=0.8, edgecolor='#0d1117', width=0.7)
        ax3.axhline(self.mean, color='#f7a24f', linestyle='--', linewidth=1.2,
                    label=f'Mean = {self.mean:.2f}')
        ax3.set_xlabel('Run', color='#8892aa')
        ax3.set_ylabel('Best Fitness', color='#8892aa')
        ax3.set_title('Best Fitness per Run', color='#c9d1d9', fontsize=11)
        ax3.tick_params(colors='#555f77')
        ax3.legend(facecolor='#1a1e2a', edgecolor='#2a2f3d', labelcolor='white', fontsize=8)
        for sp in ax3.spines.values(): sp.set_edgecolor('#2a2f3d')

        # ── Stats summary (text panel) ──
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#13161e')
        ax4.axis('off')

        metrics = [
            ('Best',        f'{self.best_all:.6f}',  '#4ff7a2'),
            ('Worst',       f'{self.worst_all:.6f}',  '#f74f4f'),
            ('Mean (AVG)',  f'{self.mean:.6f}',        '#f7a24f'),
            ('STD',         f'{self.std:.6f}',         '#4f8ef7'),
        ]
        ax4.text(0.5, 0.96, 'Statistics Summary', ha='center', va='top',
                 fontsize=13, color='white', fontweight='bold',
                 transform=ax4.transAxes)
        ax4.text(0.5, 0.86, f'{fname} — {cfg["name"]}  [{cfg["type"]}]',
                 ha='center', va='top', fontsize=10, color='#8892aa',
                 transform=ax4.transAxes)

        for j, (label, value, color) in enumerate(metrics):
            y = 0.72 - j * 0.16
            ax4.text(0.12, y, label, ha='left', va='center',
                     fontsize=11, color='#8892aa', transform=ax4.transAxes)
            ax4.text(0.88, y, value, ha='right', va='center',
                     fontsize=13, color=color, fontweight='bold',
                     fontfamily='monospace', transform=ax4.transAxes)
            ax4.plot([0.08, 0.92], [y-0.055, y-0.055], color="#2a2f3d", linewidth=0.5, transform=ax4.transAxes, clip_on=False)

        # Interpretation
        if self.std < self.mean * 0.1:
            interp = 'Stable convergence'
            icol = '#4ff7a2'
        elif self.std < self.mean * 0.3:
            interp = 'Moderate stability'
            icol = '#f7a24f'
        else:
            interp = 'High variability'
            icol = '#f74f4f'
        ax4.text(0.5, 0.06, interp, ha='center', va='bottom',
                 fontsize=11, color=icol, fontweight='bold',
                 transform=ax4.transAxes)

        ax4.add_patch(plt.Rectangle((0.05, 0.02), 0.9, 0.96,
                                     fill=False, edgecolor='#2a2f3d',
                                     transform=ax4.transAxes, linewidth=1))

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f'  [Plot saved] {save_path}')
        if show:
            plt.show()
        plt.close()


# ─────────────────────────────────────────────
#  MAIN DEMO — runs all 3 parts
# ─────────────────────────────────────────────

def run_demo(output_dir: str = 'tp2_outputs'):
    os.makedirs(output_dir, exist_ok=True)

    print('\n' + '═'*60)
    print('  TP N°2 — Optimization Problem Initialization (Part 2)')
    print('  USTHB · Master 2 SII · Module MÉTA')
    print('═'*60)

    CSV_FILES = {
        'F1':  '/mnt/user-data/uploads/1771748312155_Population_F1-UM.csv',
        'F2':  '/mnt/user-data/uploads/1771748312156_Population_F2-UM.csv',
        'F5':  '/mnt/user-data/uploads/1771748312156_Population_F5-UM.csv',
        'F7':  '/mnt/user-data/uploads/1771748312156_Population_F7-UM.csv',
        'F9':  '/mnt/user-data/uploads/1771748312157_Population_F9-MM.csv',
        'F11': '/mnt/user-data/uploads/1771748312157_Population_F11-MM.csv',
    }

    # ═════════════════════════════════════════
    #  PART I + II — each function
    # ═════════════════════════════════════════
    for fname in ['F1', 'F2', 'F5', 'F7', 'F9', 'F11']:
        csv_path = CSV_FILES.get(fname)

        print(f'\n{"─"*60}')
        print(f'  Function {fname} — {BF.CONFIG[fname]["name"]}')
        print(f'{"─"*60}')

        # Load from CSV if available, else random
        if csv_path and Path(csv_path).exists():
            pop = Population.load_csv(csv_path, fname)
        else:
            print(f'  CSV not found, generating random population...')
            pop = Population.generate_random(fname, size=30)

        pop.show()

        # Evaluate
        ev = PopulationEvaluator(pop)
        ev.report()
        ev.plot(save_path=f'{output_dir}/eval_{fname}.png')

    # ═════════════════════════════════════════
    #  PART III — Multiple Runs
    # ═════════════════════════════════════════
    print('\n' + '═'*60)
    print('  PART III — Running Multiple Populations')
    print('═'*60)

    for fname in ['F1', 'F9', 'F11']:
        mr = MultipleRunsEvaluator(fname, pop_size=30, n_runs=15, dim=30)
        mr.run()
        mr.report()
        mr.plot(save_path=f'{output_dir}/runs_{fname}.png')

    print(f'\n{"═"*60}')
    print(f'  All outputs saved to: {output_dir}/')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    run_demo()