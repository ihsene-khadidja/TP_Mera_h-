"""
PSO — TP N°3  |  F1-UM (Sphere) & F8-MM (Schwefel)
Génère 10 images séparées :
  F1 : 1) Search History 1st Iteration
       2) Search History Final Iteration
       3) Results (info box)
       4) Convergence Curve
       5) Trajectory of 1st solution
       6) Average Fitness
  F8 : idem (7 à 12... mais numérotées 1 à 6 aussi)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# CHEMINS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = r"C:\Users\hp\OneDrive\Desktop\TP Meta-H"

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT POPULATION
# ═══════════════════════════════════════════════════════════════
def load_population(filepath):
    pop = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(';')]
            pop.append(vals)
    return np.array(pop)

pop_f1 = load_population(os.path.join(BASE_DIR, 'Population_F1-UM.csv'))
pop_f8 = load_population(os.path.join(BASE_DIR, 'Population_F8-MM.csv'))

# ═══════════════════════════════════════════════════════════════
# FONCTIONS FITNESS
# ═══════════════════════════════════════════════════════════════
def f1_sphere(x):
    return float(np.sum(x ** 2))

def f8_schwefel(x):
    n = len(x)
    return 418.9829 * n - float(np.sum(x * np.sin(np.sqrt(np.abs(x)))))

# ═══════════════════════════════════════════════════════════════
# PSO
# ═══════════════════════════════════════════════════════════════
def pso(population, fitness_fn, w=0.3, c1=1.4, c2=1.4, T=200):
    N, D = population.shape
    X = population.astype(float).copy()
    V = np.zeros_like(X)

    F = np.array([fitness_fn(X[i]) for i in range(N)])
    g_idx  = int(np.argmin(F))
    g_star = X[g_idx].copy()
    g_fit  = float(F[g_idx])

    P     = X.copy()
    P_fit = F.copy()

    init_best  = g_fit
    init_worst = float(np.max(F))
    hist_best  = [g_fit]
    hist_avg   = [float(np.mean(F))]
    traj_x1    = [X[0, 0]]
    pop_t0     = X.copy()

    no_change = 0
    prev_fit  = g_fit
    last_impr = 0
    stag_iter = 0

    t = 1
    while t <= T:
        for i in range(N):
            if not np.array_equal(g_star, X[i]):
                r1 = np.random.rand(D)
                r2 = np.random.rand(D)
                V[i] = (w  * V[i]
                      + c1 * r1 * (g_star - X[i])
                      + c2 * r2 * (P[i]   - X[i]))
                X[i] = X[i] + V[i]

        F = np.array([fitness_fn(X[i]) for i in range(N)])
        for i in range(N):
            if F[i] < g_fit:
                g_fit  = float(F[i])
                g_star = X[i].copy()
                last_impr = t
            if F[i] < P_fit[i]:
                P_fit[i] = F[i]
                P[i]     = X[i].copy()

        hist_best.append(g_fit)
        hist_avg.append(float(np.mean(F)))
        traj_x1.append(X[0, 0])

        if g_fit >= prev_fit - 1e-12:
            no_change += 1
        else:
            no_change = 0
        prev_fit = g_fit

        if no_change >= 3:
            stag_iter = max(1, t - 2)
            break

        t += 1

    if stag_iter == 0:
        stag_iter = last_impr if last_impr > 0 else t

    return dict(
        init_best  = init_best,
        init_worst = init_worst,
        final_best = g_fit,
        best_pos   = g_star,
        hist_best  = hist_best,
        hist_avg   = hist_avg,
        traj_x1    = traj_x1,
        stagnation = stag_iter,
        final_X    = X,
        pop_t0     = pop_t0,
        iters      = t,
    )
# LANCEMENT PSO
np.random.seed(42)
init_f1 = np.array([f1_sphere(pop_f1[i])   for i in range(len(pop_f1))])
res_f1  = pso(pop_f1.copy(), f1_sphere)

np.random.seed(42)
init_f8 = np.array([f8_schwefel(pop_f8[i]) for i in range(len(pop_f8))])
res_f8  = pso(pop_f8.copy(), f8_schwefel)

for name, res in [("F1-UM", res_f1), ("F8-MM", res_f8)]:
    print(f"=== {name} ===")
    print(f"  Initial : Best={res['init_best']:.2f}, Worst={res['init_worst']:.2f}")
    print(f"  Final   : Best={res['final_best']:.4f}")
    print(f"  Stagnation — Iteration N°{res['stagnation']}\n")

# HELPERS CONTOURS
def contour_f1(ax):
    x = np.linspace(-100, 100, 300)
    y = np.linspace(-100, 100, 300)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ax.contourf(X, Y, Z, levels=25, cmap='YlGn',  alpha=0.35)
    ax.contour( X, Y, Z, levels=25, colors='teal', linewidths=0.5, alpha=0.8)
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

def contour_f8(ax):
    x = np.linspace(-500, 500, 300)
    y = np.linspace(-500, 500, 300)
    X, Y = np.meshgrid(x, y)
    Z = (418.9829*2
         - X*np.sin(np.sqrt(np.abs(X)))
         - Y*np.sin(np.sqrt(np.abs(Y))))
    ax.contourf(X, Y, Z, levels=25, cmap='YlOrRd',       alpha=0.35)
    ax.contour( X, Y, Z, levels=25, colors='saddlebrown', linewidths=0.5, alpha=0.8)
    ax.set_xlim(-500, 500); ax.set_ylim(-500, 500)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sauvegarde : {path}")




def generate_images(res, pop_init, init_fits, contour_fn, fn_name):
    prefix = os.path.join(BASE_DIR, fn_name)
    print(f"\n--- {fn_name} ---")

   
    fig, ax = plt.subplots(figsize=(6, 5))
    contour_fn(ax)
    ax.scatter(pop_init[:, 0], pop_init[:, 1],
               c='black', s=20, zorder=5, label='Particles')
    bi = int(np.argmin(init_fits))
    ax.scatter(pop_init[bi, 0], pop_init[bi, 1],
               c='red', s=90, zorder=7, label='Best')
    ax.set_title(f'Search History ({fn_name}), 1st Iteration', fontsize=11)
    ax.legend(fontsize=8)
    save(fig, f"{prefix}_1_SearchHistory_Init.png")

    # ── Image 2 : Search History Final Iteration ───────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    contour_fn(ax)
    fX = res['final_X']
    ax.scatter(fX[:, 0], fX[:, 1],
               c='black', s=20, zorder=5, label='Particles')
    ax.scatter(fX[:, 0], fX[:, 1],
               c='orange', s=40, zorder=6, alpha=0.5, label='Pbest')
    ax.scatter(res['best_pos'][0], res['best_pos'][1],
               c='red', s=110, zorder=8, label='Gbest')
    ax.set_title(f'Search History ({fn_name}), Final Iteration', fontsize=11)
    ax.legend(fontsize=8)
    save(fig, f"{prefix}_2_SearchHistory_Final.png")

    # ── Image 3 : Boîte résultats ──────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axis('off')
    txt = (
        "Initial population:\n\n"
        f"  Best — {res['init_best']:.2f},  Worst — {res['init_worst']:.2f}\n\n\n"
        "Final population:\n\n"
        f"  Best — {res['final_best']:.4f}\n\n\n"
        f"Stagnation — Iteration N°{res['stagnation']}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=13, va='top',
            bbox=dict(boxstyle='round,pad=0.9',
                      facecolor='white', edgecolor='#aaaaaa', linewidth=1))
    ax.set_title(f'Results — {fn_name}', fontsize=11, fontweight='bold')
    save(fig, f"{prefix}_3_Results.png")

    # ── Image 4 : Convergence Curve ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    iters = np.arange(len(res['hist_best']))
    ax.plot(iters, res['hist_best'], 'r-', linewidth=2)
    ax.set_title(f'Convergence Curve — {fn_name}', fontsize=11)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Fitness')
    ax.set_xlim(0, max(iters))
    save(fig, f"{prefix}_4_ConvergenceCurve.png")

    # ── Image 5 : Trajectory of 1st solution ──────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res['traj_x1'], 'g-', linewidth=2)
    ax.set_title(f'Trajectory of 1st Solution — {fn_name}', fontsize=11)
    ax.set_xlabel('Iteration'); ax.set_ylabel('$x_1^{(1)}$')
    ax.set_xlim(0, max(1, len(res['traj_x1']) - 1))
    save(fig, f"{prefix}_5_Trajectory.png")

    # ── Image 6 : Average Fitness ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res['hist_avg'], 'b-', linewidth=2)
    ax.set_title(f'Average Fitness — {fn_name}', fontsize=11)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Fitness')
    ax.set_xlim(0, max(1, len(res['hist_avg']) - 1))
    save(fig, f"{prefix}_6_AverageFitness.png")

# ── F1-UM ──────────────────────────────────────────────────────
generate_images(res_f1, pop_f1, init_f1, contour_f1, 'F1-UM')

# ── F8-MM ──────────────────────────────────────────────────────
generate_images(res_f8, pop_f8, init_f8, contour_f8, 'F8-MM')

print(f"\nTermine ! 12 images sauvegardees dans : {BASE_DIR}")