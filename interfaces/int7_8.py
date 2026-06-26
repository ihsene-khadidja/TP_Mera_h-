import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Genetic Algorithm UI", layout="wide")

# ==========================================================
# STYLE (comme interface du prof)
# ==========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
h1 {
    text-align: center;
}
.card {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# DATA
# ==========================================================
def load_data(name):
    if name == "Digits":
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

# ==========================================================
# FITNESS
# ==========================================================
def fitness(chrom, Xtr, Xte, ytr, yte, alpha):
    selected = np.where(chrom == 1)[0]
    if len(selected) == 0:
        return 1.0

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr[:, selected], ytr)
    acc = accuracy_score(yte, knn.predict(Xte[:, selected]))

    return alpha * (1 - acc) + (1 - alpha) * (len(selected)/Xtr.shape[1])

# ==========================================================
# OPERATORS
# ==========================================================
def selection(pop, fvals, method):
    if method == "Random":
        i1, i2 = np.random.randint(len(pop)), np.random.randint(len(pop))
        return pop[i1], pop[i2]

    else:  # Roulette
        probs = (1/(fvals+1e-10))
        probs /= probs.sum()
        i1 = np.random.choice(len(pop), p=probs)
        i2 = np.random.choice(len(pop), p=probs)
        return pop[i1], pop[i2]

def crossover(p1, p2, method, rc):
    if np.random.rand() > rc:
        return p1.copy(), p2.copy()

    D = len(p1)

    if method == "1-Point":
        k = np.random.randint(1, D)
        return (np.concatenate([p1[:k], p2[k:]]),
                np.concatenate([p2[:k], p1[k:]]))

    elif method == "2-Point":
        k1, k2 = sorted(np.random.choice(range(1, D), 2, replace=False))
        return (np.concatenate([p1[:k1], p2[k1:k2], p1[k2:]]),
                np.concatenate([p2[:k1], p1[k1:k2], p2[k2:]]))

    else:  # Uniform
        mask = np.random.randint(0, 2, D)
        return np.where(mask, p1, p2), np.where(mask, p2, p1)

def mutation(chrom, rm):
    for i in range(len(chrom)):
        if np.random.rand() < rm:
            chrom[i] = 1 - chrom[i]
    return chrom

# ==========================================================
# GA CORE
# ==========================================================
def run_ga(N, T, rc, rm, alpha, Xtr, Xte, ytr, yte,
           sel, cross, repl):

    D = Xtr.shape[1]
    pop = np.random.randint(0, 2, (N, D))

    best_curve = []
    avg_curve = []

    for t in range(T):

        fvals = np.array([fitness(c, Xtr, Xte, ytr, yte, alpha) for c in pop])
        best_curve.append(np.min(fvals))
        avg_curve.append(np.mean(fvals))

        new_pop = []

        for _ in range(N//2):
            p1, p2 = selection(pop, fvals, sel)
            c1, c2 = crossover(p1, p2, cross, rc)
            new_pop.append(mutation(c1, rm))
            new_pop.append(mutation(c2, rm))

        new_pop = np.array(new_pop[:N])
        new_fvals = np.array([fitness(c, Xtr, Xte, ytr, yte, alpha) for c in new_pop])

        if repl == "Children":
            pop = new_pop
        else:
            combined = np.vstack([pop, new_pop])
            combined_f = np.concatenate([fvals, new_fvals])
            idx = np.argsort(combined_f)[:N]
            pop = combined[idx]

    best_idx = np.argmin(fvals)
    best = pop[best_idx]

    selected = np.where(best == 1)[0]
    acc = 0
    if len(selected) > 0:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr[:, selected], ytr)
        acc = accuracy_score(yte, knn.predict(Xte[:, selected]))

    return best_curve, avg_curve, acc, len(selected)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title(" Parameters")

dataset = st.sidebar.selectbox("Dataset", ["Synthetic", "Digits"])
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.99)

N = st.sidebar.slider("Population (N)", 5, 50, 10)
T = st.sidebar.slider("Iterations (T)", 5, 50, 20)
runs = st.sidebar.slider("Runs", 1, 30, 15)

selection_method = st.sidebar.selectbox("Selection", ["Random", "Roulette"])
crossover_method = st.sidebar.selectbox("Crossover", ["1-Point", "2-Point", "Uniform"])
replacement_method = st.sidebar.selectbox("Replacement", ["Children", "Elitism"])

Rc = st.sidebar.slider("Rc", 0.0, 1.0, 0.7)
Rm = st.sidebar.slider("Rm", 0.0, 1.0, 0.1)

# ==========================================================
# TABS
# ==========================================================
tab1, tab2 = st.tabs([" TP7 (Part 1)", " TP8 (Part 2)"])

X, y = load_data(dataset)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3)

# ==========================================================
# RUN BUTTON
# ==========================================================
if st.button(" Run GA"):

    best_scores = []

    for _ in range(runs):
        best_curve, avg_curve, acc, sf = run_ga(
            N, T, Rc, Rm, alpha,
            Xtr, Xte, ytr, yte,
            selection_method,
            crossover_method,
            replacement_method
        )
        best_scores.append(best_curve[-1])

    # ======================================================
    # RESULTATS
    # ======================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Best", f"{np.min(best_scores):.4f}")
    col2.metric("Mean", f"{np.mean(best_scores):.4f}")
    col3.metric("Accuracy", f"{acc:.2f}")
    col4.metric("Selected", sf)

    # ======================================================
    # GRAPHS
    # ======================================================
    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].plot(best_curve)
    ax[0].set_title("Convergence")

    ax[1].plot(avg_curve)
    ax[1].set_title("Average Fitness")

    st.pyplot(fig)

    # ======================================================
    # TP8 EXTRA     streamlit run int7_8.py  
    # ======================================================
    with tab2:
        st.success("Variants du GA activés (TP8)")
        st.write("Tu peux comparer les variantes ici ")
        st.line_chart(best_curve)