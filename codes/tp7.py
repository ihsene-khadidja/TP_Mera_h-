"""
TP N°7 - Genetic Algorithm for Feature Selection
Master 2 SII - Module MÉTA - USTHB
================================================
Code basé exactement sur le code source Streamlit du TP.
Paramètres par défaut de l'interface :
  - Data        : Synthetic
  - alpha       : 0.99
  - Selection   : Random
  - Crossover   : 1-Point
  - Replacement : Children
  - Rc          : 0.70
  - Rm          : 0.10
  - Population  : N = 10
  - Max Iter    : T = 20  (exactement 20, pas d'arret anticipe)
  - Run         : 15
================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# 1. CHARGEMENT DU DATASET
# ============================================================

def load_ga_dataset(choice="Synthetic"):
    """
    Synthetic : 1000 samples, 50 features, 5 informatives
    Digits    : sklearn load_digits
    """
    if choice == "Digits":
        data = load_digits()
        return data.data, data.target
    else:
        X, y = make_classification(
            n_samples=1000,
            n_features=50,
            n_informative=5,
            n_redundant=10,
            random_state=42
        )
        return X, y


# ============================================================
# 2. FONCTION FITNESS  (train/test split comme Streamlit)
# ============================================================

def ga_fitness_fn(chromosome, Xtr, Xte, ytr, yte, alpha):
    """
    fitness = alpha * (1 - acc) + (1 - alpha) * (SF / D)
    Evaluation sur split train/test avec KNN k=5.
    """
    selected = np.where(chromosome == 1)[0]
    D  = Xtr.shape[1]
    SF = len(selected)

    if SF == 0:
        return 1.0

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr[:, selected], ytr)
    acc = accuracy_score(yte, knn.predict(Xte[:, selected]))

    return alpha * (1 - acc) + (1 - alpha) * (SF / D)


# ============================================================
# 3. SELECTION  (Random / Tournament / Roulette)
# ============================================================

def selection(pop, fvals, method="Random"):
    N = len(pop)

    if method == "Random":
        idx1 = np.random.randint(N)
        idx2 = np.random.randint(N)
        return pop[idx1].copy(), pop[idx2].copy()

    elif method == "Tournament":
        idx1 = np.random.choice(N, 2, replace=False)
        idx2 = np.random.choice(N, 2, replace=False)
        p1 = pop[idx1[0]] if fvals[idx1[0]] < fvals[idx1[1]] else pop[idx1[1]]
        p2 = pop[idx2[0]] if fvals[idx2[0]] < fvals[idx2[1]] else pop[idx2[1]]
        return p1.copy(), p2.copy()

    else:  # Roulette
        inv   = 1.0 / (fvals + 1e-10)
        probs = inv / inv.sum()
        idx1  = np.random.choice(N, p=probs)
        idx2  = np.random.choice(N, p=probs)
        return pop[idx1].copy(), pop[idx2].copy()


# ============================================================
# 4. CROISEMENT  (1-Point / 2-Point / Uniform)
# ============================================================

def crossover(p1, p2, method="1-Point", rc=0.70):
    D = len(p1)
    r = np.random.rand()

    if r < rc:
        if method == "1-Point":
            k  = np.random.randint(1, D)
            c1 = np.concatenate([p1[:k], p2[k:]])
            c2 = np.concatenate([p2[:k], p1[k:]])

        elif method == "2-Point":
            pts = sorted(np.random.choice(range(1, D), 2, replace=False))
            c1  = np.concatenate([p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]])
            c2  = np.concatenate([p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]])

        else:  # Uniform
            mask = np.random.randint(0, 2, D)
            c1   = np.where(mask, p1, p2)
            c2   = np.where(mask, p2, p1)
    else:
        c1 = p1.copy()
        c2 = p2.copy()

    return c1, c2


# ============================================================
# 5. MUTATION  (Bit-Flip)
# ============================================================

def mutation(chrom, rm=0.10):
    chrom_mut = chrom.copy()
    for j in range(len(chrom)):
        if np.random.rand() < rm:
            chrom_mut[j] = 1 - chrom_mut[j]
    return chrom_mut


# ============================================================
# 6. ALGORITHME GENETIQUE  (un seul run, avec historiques)
#    t=0..T-1 exactement comme le code Streamlit
# ============================================================

def run_ga_full(N, T, rc, rm, alpha,
                Xtr, Xte, ytr, yte,
                selection_method   = "Random",
                crossover_method   = "1-Point",
                replacement_method = "Children",
                verbose            = False):
    """
    GA complet - exactement T iterations (while t < T).
    Identique au code Streamlit + courbe cumulative ajoutee.

    Retourne :
        best_chrom, best_fit,
        convergence_curve, cumulative_curve,
        trajectory_first,  avg_fitness_curve
    """
    D = Xtr.shape[1]

    # Initialisation
    pop   = np.random.randint(0, 2, (N, D))
    fvals = np.array([ga_fitness_fn(c, Xtr, Xte, ytr, yte, alpha) for c in pop])

    best_idx   = np.argmin(fvals)
    best_chrom = pop[best_idx].copy()
    best_fit   = fvals[best_idx]

    # Historiques
    convergence_curve = [best_fit]
    cumulative_curve  = [best_fit]     # monotone decroissante
    trajectory_first  = [fvals[0]]
    avg_fitness_curve = [fvals.mean()]

    cumulative_best = best_fit
    t = 0                              # t <- 0 (code Streamlit)

    if verbose:
        print(f"\n{'Iter':>4} | {'Best':>10} | {'Cum Best':>10} | "
              f"{'Avg':>10} | {'Selected':>8}")
        print("-" * 55)

    # Boucle : exactement T iterations
    while t < T:

        P_new = []

        for _ in range(N // 2):
            p1, p2 = selection(pop, fvals, selection_method)
            c1, c2 = crossover(p1, p2, crossover_method, rc)
            c1     = mutation(c1, rm)
            c2     = mutation(c2, rm)
            P_new.extend([c1, c2])

        P_new     = np.array(P_new[:N])
        new_fvals = np.array([ga_fitness_fn(c, Xtr, Xte, ytr, yte, alpha)
                              for c in P_new])

        # Remplacement
        if replacement_method == "Children":
            pop   = P_new
            fvals = new_fvals
        else:  # Elitism
            combined   = np.vstack([pop, P_new])
            combined_f = np.concatenate([fvals, new_fvals])
            best_ids   = np.argsort(combined_f)[:N]
            pop        = combined[best_ids]
            fvals      = combined_f[best_ids]

        # Mise a jour du meilleur global
        cur_best_idx = np.argmin(fvals)
        if fvals[cur_best_idx] < best_fit:
            best_fit   = fvals[cur_best_idx]
            best_chrom = pop[cur_best_idx].copy()

        # Mise a jour cumulative (monotone)
        if best_fit < cumulative_best:
            cumulative_best = best_fit

        # Enregistrement
        convergence_curve.append(best_fit)
        cumulative_curve.append(cumulative_best)
        trajectory_first.append(fvals[0])
        avg_fitness_curve.append(fvals.mean())

        t += 1

        if verbose:
            print(f"{t:>4} | {best_fit:>10.6f} | {cumulative_best:>10.6f} | "
                  f"{fvals.mean():>10.6f} | {int(best_chrom.sum()):>8}")

    return (best_chrom, best_fit,
            convergence_curve, cumulative_curve,
            trajectory_first,  avg_fitness_curve)


# ============================================================
# 7. MULTIPLE RUNS  (slider Run = 15, comme Streamlit)
# ============================================================

def evaluate_ga(N, T, rc, rm, alpha, ga_runs,
                dataset        = "Synthetic",
                selection_m    = "Random",
                crossover_m    = "1-Point",
                replacement_m  = "Children"):
    """
    Lance le GA ga_runs fois.
    Affiche : Best, Mean error, Accuracy, Selected, STD
    (identique au panel Evaluation de l'interface Streamlit)
    """
    X, y = load_ga_dataset(dataset)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    all_fits     = []
    overall_best = float("inf")
    best_sf_val  = 0
    best_acc_val = 0.0
    best_curves  = None

    print(f"\n{'='*62}")
    print(f"  GA | {ga_runs} Runs | N={N} T={T} Rc={rc} Rm={rm} alpha={alpha}")
    print(f"  Dataset={dataset} | Sel={selection_m} | "
          f"Cross={crossover_m} | Repl={replacement_m}")
    print(f"{'='*62}")

    for run in range(ga_runs):

        (chrom, fit, conv, cum, traj, avg) = run_ga_full(
            N, T, rc, rm, alpha,
            Xtr, Xte, ytr, yte,
            selection_m, crossover_m, replacement_m,
            verbose=False
        )

        selected = np.where(chrom == 1)[0]
        SF       = len(selected)
        acc      = 0.0

        if SF > 0:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(Xtr[:, selected], ytr)
            acc = accuracy_score(yte, knn.predict(Xte[:, selected]))

        all_fits.append(fit)

        if fit < overall_best:
            overall_best = fit
            best_sf_val  = SF
            best_acc_val = acc
            best_curves  = (conv, cum, traj, avg)

        print(f"  Run {run+1:>2}/{ga_runs} | Fitness={fit:.4f} | "
              f"Accuracy={acc*100:.1f}% | Selected={SF:>2}")

    mean_err = float(np.mean(all_fits))
    std_val  = float(np.std(all_fits))

    # Panel Evaluation (comme Streamlit)
    print(f"\n{'='*45}")
    print("           EVALUATION")
    print(f"{'='*45}")
    print(f"  Best               : {overall_best:.4f}")
    print(f"  Mean (average err) : {mean_err:.4f}")
    print(f"  Accuracy           : {best_acc_val:.2f}  ({best_acc_val*100:.1f}%)")
    print(f"  Selected           : {best_sf_val}")
    print(f"  STD                : {std_val:.4f}")
    print(f"{'='*45}")

    return overall_best, mean_err, best_acc_val, best_sf_val, std_val, best_curves


# ============================================================
# 8. VISUALISATION - 4 COURBES
# ============================================================

def plot_results(curves, title="GA Feature Selection"):
    """
    4 courbes du meilleur run :
      1. Convergence Curve          (rouge)
      2. Cumulative Best Fitness    (bordeaux, monotone decroissante)
      3. Trajectory of 1st solution (vert)
      4. Average Fitness of pop     (bleu)
    """
    conv, cum, traj, avg = curves
    iters = range(len(conv))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    plots = [
        (conv, 'red',     'Convergence Curve',            'Fitness'),
        (cum,  'darkred', 'Cumulative Best Fitness',       'Best Fitness (cumul.)'),
        (traj, 'green',   'Trajectory of 1st solution',   'Fitness'),
        (avg,  'blue',    'Average Fitness of population', 'Fitness'),
    ]

    for ax, (data, color, ttl, ylabel) in zip(axes, plots):
        ax.plot(iters, data, color=color, linewidth=1.5,
                marker='o', markersize=2)
        if color == 'darkred':
            ax.fill_between(iters, data, alpha=0.10, color=color)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('Iteration', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    import os
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ga_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Figure sauvegardee : {save_path}")




if __name__ == "__main__":

    # Parametres 
    DATASET     = "Synthetic"   
    ALPHA       = 0.5
    SELECTION   = "Random"     
    CROSSOVER   = "1-Point"     
    REPLACEMENT = "Children"    
    Rc          = 0.70
    Rm          = 0.10
    N           = 10
    T           = 20           
    RUNS        = 15

    np.random.seed(42)

    
    best, mean_err, acc, sf, std, curves = evaluate_ga(
        N=N, T=T, rc=Rc, rm=Rm, alpha=ALPHA, ga_runs=RUNS,
        dataset       = DATASET,
        selection_m   = SELECTION,
        crossover_m   = CROSSOVER,
        replacement_m = REPLACEMENT,
    )

  
    plot_results(
        curves,
        title=(f"TP N°7 - GA Feature Selection | "
               f"Best={best:.4f}  Acc={acc*100:.1f}%  "
               f"Selected={sf}  STD={std:.4f}")
    )

   
    print("\n" + "="*62)
    print("  Single Run verbose - exactement 20 iterations")
    print("="*62)
    X, y = load_ga_dataset(DATASET)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    run_ga_full(N, T, Rc, Rm, ALPHA, Xtr, Xte, ytr, yte,
                SELECTION, CROSSOVER, REPLACEMENT, verbose=True)