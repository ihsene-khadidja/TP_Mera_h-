"""
TP N°5 - Feature Selection with PSO — Part 2
Binary threshold selection: x_i > 0.5 → selected, x_i ≤ 0.5 → not selected
Master 2 SII - Module MÉTA — USTHB
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ─────────────────────────────────────────────
# 1. DATASETS
# ─────────────────────────────────────────────

def load_synthetic_dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=10,
        random_state=42
    )
    return X, y

def load_digits_dataset():
    digits = load_digits()
    return digits.data, digits.target


# ─────────────────────────────────────────────
# 2. KNN CLASSIFIER
# ─────────────────────────────────────────────

def evaluate_knn(X, y, feature_indices, k=5, test_size=0.3, random_state=42):
    if len(feature_indices) == 0:
        return 0.0
    X_selected = X[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state
    )
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return accuracy_score(y_test, knn.predict(X_test))


# ─────────────────────────────────────────────
# 3. OBJECTIVE FUNCTION (Part 2 — binary threshold)
# ─────────────────────────────────────────────

def objective_function(solution, X, y, alpha=0.9, threshold=0.5):
    """
    Part 2: feature i is selected if solution[i] > threshold (0.5)
    SF = number of selected features (dynamic, not fixed)
    f(x) = alpha * f1(x) + (1 - alpha) * f2(x)
    f1(x) = 1 - Accuracy
    f2(x) = SF / D
    """
    D = len(solution)

    # Binary selection: keep features with weight > 0.5
    selected_indices = [i for i, v in enumerate(solution) if v > threshold]

    # Edge case: if no features selected, penalize heavily
    if len(selected_indices) == 0:
        return 1.0, 0.0, []

    SF = len(selected_indices)
    accuracy = evaluate_knn(X, y, selected_indices)
    f1 = 1.0 - accuracy
    f2 = SF / D
    fitness = alpha * f1 + (1 - alpha) * f2
    return fitness, accuracy, selected_indices


# ─────────────────────────────────────────────
# 4. PSO ALGORITHM
# ─────────────────────────────────────────────

def pso_feature_selection(X, y, alpha=0.9, threshold=0.5,
                           n_particles=30, n_iter=50,
                           w=0.7, c1=2.0, c2=1.5,
                           verbose=True):
    D = X.shape[1]

    # Initialisation
    positions  = np.random.uniform(0, 1, (n_particles, D))
    velocities = np.random.uniform(-0.2, 0.2, (n_particles, D))

    pbest_pos     = positions.copy()
    pbest_fitness = np.full(n_particles, np.inf)

    gbest_pos     = None
    gbest_fitness = np.inf
    gbest_acc     = 0.0
    gbest_sf      = 0

    # History for plots
    fitness_history  = []
    accuracy_history = []
    sf_history       = []

    # Scatter data: (iteration, fitness) for every particle every iteration
    scatter_iters   = []
    scatter_fitness = []

    # Evaluate initial population
    for i in range(n_particles):
        fit, acc, sel = objective_function(positions[i], X, y, alpha, threshold)
        pbest_fitness[i] = fit
        if fit < gbest_fitness:
            gbest_fitness = fit
            gbest_acc     = acc
            gbest_sf      = len(sel)
            gbest_pos     = positions[i].copy()

    # Main loop
    for iteration in range(1, n_iter + 1):
        for i in range(n_particles):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)

            velocities[i] = (w  * velocities[i]
                           + c1 * r1 * (pbest_pos[i] - positions[i])
                           + c2 * r2 * (gbest_pos   - positions[i]))

            positions[i] = np.clip(positions[i] + velocities[i], 0, 1)

            fit, acc, sel = objective_function(positions[i], X, y, alpha, threshold)

            scatter_iters.append(iteration)
            scatter_fitness.append(fit)

            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_pos[i]     = positions[i].copy()

            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_acc     = acc
                gbest_sf      = len(sel)
                gbest_pos     = positions[i].copy()

        fitness_history.append(gbest_fitness)
        accuracy_history.append(gbest_acc)
        sf_history.append(gbest_sf)

        if verbose:
            print(f"  [{iteration:3d}/{n_iter}]  "
                  f"fitness={gbest_fitness:.4f}  "
                  f"accuracy={gbest_acc:.4f}  "
                  f"SF={gbest_sf}")

    selected_indices = [i for i, v in enumerate(gbest_pos) if v > threshold]

    return (gbest_pos, gbest_fitness, gbest_acc, selected_indices,
            fitness_history, accuracy_history, sf_history,
            scatter_iters, scatter_fitness)


# ─────────────────────────────────────────────
# 5. MULTI-RUN WRAPPER  ← NEW
#    Produces: Best, Mean, Accuracy, Selected, STD
# ─────────────────────────────────────────────

def run_multiple(X, y, n_runs=15, alpha=0.9, threshold=0.5,
                 n_particles=30, n_iter=50, w=0.7, c1=2.0, c2=1.5):
    """
    Runs PSO n_runs times and collects the evaluation metrics shown in the
    TP interface:
        Best              – lowest fitness across all runs
        Mean (avg error)  – mean best-fitness across runs
        Accuracy          – accuracy of the best run
        Selected          – number of features selected in the best run
        STD               – standard deviation of best-fitness values
    Returns the summary dict and the result of the best run.
    """
    print(f"\nRunning PSO × {n_runs} independent runs …\n")

    best_fits  = []
    all_results = []

    for run in range(1, n_runs + 1):
        print(f"─── Run {run}/{n_runs} ───────────────────────────────")
        result = pso_feature_selection(
            X, y,
            alpha=alpha, threshold=threshold,
            n_particles=n_particles, n_iter=n_iter,
            w=w, c1=c1, c2=c2,
            verbose=True,
        )
        gbest_fitness = result[1]
        best_fits.append(gbest_fitness)
        all_results.append(result)
        print(f"  → Run best fitness: {gbest_fitness:.4f}\n")

    best_fits  = np.array(best_fits)
    best_run_idx = int(np.argmin(best_fits))
    best_result  = all_results[best_run_idx]

    summary = {
        "Best"    : float(np.min(best_fits)),           # best fitness found
        "Mean"    : float(np.mean(best_fits)),           # mean (average error)
        "Accuracy": float(best_result[2]),               # accuracy of best run
        "Selected": len(best_result[3]),                 # # features selected
        "STD"     : float(np.std(best_fits)),            # std of fitness values
    }
    return summary, best_result


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def plot_all(fitness_history, accuracy_history, sf_history,
             scatter_iters, scatter_fitness,
             gbest_pos, selected_indices, D, dataset_name,
             summary, threshold=0.5):

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        f"PSO Feature Selection — Part 2 (Binary threshold > {threshold})\n"
        f"Dataset: {dataset_name}  |  Selected: {len(selected_indices)}/{D} features",
        fontsize=13, fontweight='bold'
    )

    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # ── Plot 1: Fitness convergence ──────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(range(1, len(fitness_history) + 1), fitness_history,
             color='#4f8ef7', linewidth=2, marker='o', markersize=3)
    ax1.fill_between(range(1, len(fitness_history) + 1), fitness_history,
                     alpha=0.1, color='#4f8ef7')
    ax1.set_title('Fitness Convergence', fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Fitness f(x)')
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Scatter — all particles ─────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(scatter_iters, scatter_fitness,
                alpha=0.15, s=8, color='#4f8ef7', label='All particles')
    ax2.plot(range(1, len(fitness_history) + 1), fitness_history,
             color='#f76f6f', linewidth=2, label='Global best', zorder=5)
    ax2.set_title('Scatter Plot — Particle Fitness', fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness f(x)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Evaluation summary panel  ← NEW ─────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    metrics = [
        ("Best",               f"{summary['Best']:.4f}"),
        ("Mean (average error)", f"{summary['Mean']:.4f}"),  # ← renamed to match TP
        ("Accuracy",           f"{summary['Accuracy']:.2f}"),
        ("Selected",           f"{summary['Selected']}"),
        ("STD",                f"{summary['STD']:.4f}"),
    ]

    ax3.set_title('Evaluation', fontweight='bold', pad=12)

    box = dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff',
               edgecolor='#4f8ef7', linewidth=1.5)

    for idx, (label, value) in enumerate(metrics):
        y_pos = 0.82 - idx * 0.17
        ax3.text(0.08, y_pos, f"{label} —", transform=ax3.transAxes,
                 fontsize=11, va='center', color='#333',
                 fontweight='bold')
        ax3.text(0.72, y_pos, value, transform=ax3.transAxes,
                 fontsize=11, va='center', color='#1a4fc4')

    # Outer border
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#4f8ef7')
        spine.set_linewidth(1.5)

    plt.savefig('/mnt/user-data/outputs/pso_results_part2.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved → pso_results_part2.png")


# ─────────────────────────────────────────────
# 7. DISPLAY RESULTS
# ─────────────────────────────────────────────

def display_results(solution, fitness, accuracy, selected_indices,
                    D, alpha, threshold, summary):
    print("\n" + "=" * 60)
    print("RESULTS — Part 2 (Binary threshold)")
    print("=" * 60)
    print(f"Threshold : {threshold}")
    print(f"Fitness   : {fitness:.4f}")
    print(f"Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Selected  : {len(selected_indices)} / {D} features  "
          f"({len(selected_indices) / D * 100:.1f}%)\n")

    print("─── Evaluation (multi-run) ──────────────────────────")
    print(f"  Best               : {summary['Best']:.4f}")
    print(f"  Mean (avg error)   : {summary['Mean']:.4f}")
    print(f"  Accuracy           : {summary['Accuracy']:.2f}")
    print(f"  Selected           : {summary['Selected']}")
    print(f"  STD                : {summary['STD']:.4f}")

    print("\nSolution vector (weights x_i):")
    print(" | ".join(f"{v:.2f}" for v in solution))
    print(f"\nIndices of selected features (x_i > {threshold}):")
    print(" | ".join(str(i) for i in selected_indices))

    f1 = 1 - accuracy
    f2 = len(selected_indices) / D
    f_check = alpha * f1 + (1 - alpha) * f2
    print(f"\nFitness check:")
    print(f"  f1 = 1 - {accuracy:.4f} = {f1:.4f}")
    print(f"  f2 = {len(selected_indices)}/{D} = {f2:.4f}")
    print(f"  f  = {alpha}×{f1:.4f} + {1-alpha}×{f2:.4f} = {f_check:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Choose dataset ───────────────────────────────────────
    DATASET = "synthetic"      # "synthetic" or "digits"

    if DATASET == "synthetic":
        X, y = load_synthetic_dataset()
        print("Dataset: Synthetic  (1000 instances, 50 features)")
    else:
        X, y = load_digits_dataset()
        print("Dataset: Digits     (1797 instances, 64 features)")

    # ── Parameters ───────────────────────────────────────────
    ALPHA       = 0.5
    THRESHOLD   = 0.5
    N_PARTICLES = 30
    N_ITER      = 15
    W           = 0.5
    C1          = 1.5
    C2          = 1.9
    N_RUNS      = 30      

    # ── Run PSO multiple times  ──────────────────────────────
    summary, best_result = run_multiple(
        X, y,
        n_runs=N_RUNS,
        alpha=ALPHA, threshold=THRESHOLD,
        n_particles=N_PARTICLES, n_iter=N_ITER,
        w=W, c1=C1, c2=C2,
    )

    (solution, fitness, accuracy, selected_indices,
     fitness_history, accuracy_history, sf_history,
     scatter_iters, scatter_fitness) = best_result

    # ── Display results ──────────────────────────────────────
    display_results(solution, fitness, accuracy, selected_indices,
                    X.shape[1], ALPHA, THRESHOLD, summary)

    # ── All plots ────────────────────────────────────────────
    plot_all(fitness_history, accuracy_history, sf_history,
             scatter_iters, scatter_fitness,
             solution, selected_indices,
             X.shape[1], DATASET, summary, THRESHOLD)