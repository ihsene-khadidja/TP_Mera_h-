import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# DATASET
# ============================================================

def load_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=10,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================
# FITNESS
# ============================================================

def fitness(chromosome, Xtr, Xte, ytr, yte, alpha):
    selected = np.where(chromosome == 1)[0]
    D = Xtr.shape[1]
    SF = len(selected)

    if SF == 0:
        return 1.0

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr[:, selected], ytr)
    acc = accuracy_score(yte, knn.predict(Xte[:, selected]))

    return alpha * (1 - acc) + (1 - alpha) * (SF / D)

# ============================================================
# SELECTION
# ============================================================

def selection(pop, fvals, method):

    if method == "Random":
        idx1 = np.random.randint(len(pop))
        idx2 = np.random.randint(len(pop))
        return pop[idx1].copy(), pop[idx2].copy()

    else:  # Roulette (cumulative)
        inv = 1.0 / (fvals + 1e-10)
        probs = inv / inv.sum()
        idx1 = np.random.choice(len(pop), p=probs)
        idx2 = np.random.choice(len(pop), p=probs)
        return pop[idx1].copy(), pop[idx2].copy()

# ============================================================
# CROSSOVER (2-POINT)
# ============================================================

def crossover(p1, p2, rc):
    D = len(p1)
    if np.random.rand() < rc:
        k1, k2 = sorted(np.random.choice(range(1, D), 2, replace=False))
        c1 = np.concatenate([p1[:k1], p2[k1:k2], p1[k2:]])
        c2 = np.concatenate([p2[:k1], p1[k1:k2], p2[k2:]])
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2

# ============================================================
# MUTATION
# ============================================================

def mutation(chrom, rm):
    for j in range(len(chrom)):
        if np.random.rand() < rm:
            chrom[j] = 1 - chrom[j]
    return chrom

# ============================================================
# GA (1 RUN)
# ============================================================

def run_ga(N, T, rc, rm, alpha, Xtr, Xte, ytr, yte,
           selection_method, replacement_method):

    D = Xtr.shape[1]
    pop = np.random.randint(0, 2, (N, D))
    fvals = np.array([fitness(ind, Xtr, Xte, ytr, yte, alpha) for ind in pop])

    best_fit = np.min(fvals)
    best = pop[np.argmin(fvals)].copy()

    t = 0

    while t < T:

        P_new = []

        for _ in range(N // 2):
            p1, p2 = selection(pop, fvals, selection_method)
            c1, c2 = crossover(p1, p2, rc)
            c1 = mutation(c1, rm)
            c2 = mutation(c2, rm)
            P_new.extend([c1, c2])

        P_new = np.array(P_new[:N])
        new_fvals = np.array([fitness(ind, Xtr, Xte, ytr, yte, alpha) for ind in P_new])

        # remplacement
        if replacement_method == "Children":
            pop = P_new
            fvals = new_fvals
        else:  # Best (elitism)
            combined = np.vstack([pop, P_new])
            combined_f = np.concatenate([fvals, new_fvals])
            idx = np.argsort(combined_f)[:N]
            pop = combined[idx]
            fvals = combined_f[idx]

        if np.min(fvals) < best_fit:
            best_fit = np.min(fvals)
            best = pop[np.argmin(fvals)].copy()

        t += 1

    # accuracy finale
    selected = np.where(best == 1)[0]
    acc = 0
    if len(selected) > 0:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr[:, selected], ytr)
        acc = accuracy_score(yte, knn.predict(Xte[:, selected]))

    return best_fit, acc, len(selected), T


# ============================================================
# EVALUATION MULTI-VARIANTES
# ============================================================

def evaluate():

    # paramètres
    ALPHA = 0.5
    Rc = 0.70
    Rm = 0.10
    N = 100
    T = 20
    RUNS = 50

    Xtr, Xte, ytr, yte = load_data()

    selections = ["Random", "Roulette"]
    replacements = ["Children", "Best"]

    for sel in selections:
        for rep in replacements:

            fits, accs, sfs, stops = [], [], [], []

            best_global = float("inf")
            best_acc = 0
            best_sf = 0

            for _ in range(RUNS):
                fit, acc, sf, stop = run_ga(
                    N, T, Rc, Rm, ALPHA,
                    Xtr, Xte, ytr, yte,
                    sel, rep
                )

                fits.append(fit)
                accs.append(acc)
                sfs.append(sf)
                stops.append(stop)

                if fit < best_global:
                    best_global = fit
                    best_acc = acc
                    best_sf = sf

            print("\n========================================")
            print(f"Selection = {sel} | Replacement = {rep}")
            print("========================================")

            print(f"Best                  : {best_global:.4f}")
            print(f"Mean (average error) : {np.mean(fits):.4f}")
            print(f"Accuracy             : {best_acc:.4f} ({best_acc*100:.2f}%)")
            print(f"Selected features    : {best_sf}")
            print(f"STD                  : {np.std(fits):.4f}")
            print(f"Avg stop iteration   : {np.mean(stops):.2f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    evaluate()