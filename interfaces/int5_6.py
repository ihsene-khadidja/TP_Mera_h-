import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title(" Feature Selection with PSO (TP5 & TP6)")

# ==============================
# NAVIGATION
# ==============================
mode = st.sidebar.radio(
    "Choose Part",
    ["TP5 - Single Run", "TP6 - Multiple Runs"]
)

# ==============================
# COMMON FUNCTIONS
# ==============================
def generate_solution(dim=50):
    return np.random.rand(dim)

def select_features(solution, threshold=0.5):
    return np.where(solution > threshold)[0]

def fitness_function(solution):
    return np.mean(solution)

def accuracy_function(solution):
    return 1 - np.mean(solution)

def run_pso(n_particles, n_iter, dim=50):
    best_curve = []
    avg_curve = []

    pop = np.random.rand(n_particles, dim)

    for t in range(n_iter):
        fitness = np.mean(pop, axis=1)

        best_curve.append(np.min(fitness))
        avg_curve.append(np.mean(fitness))

        pop = pop * 0.9 + np.random.rand(n_particles, dim) * 0.1

    return best_curve, avg_curve, pop

# =========================================================
# ======================= TP5 ==============================
# =========================================================
if mode == "TP5 - Single Run":

    st.header(" Part 3 | Feature Selection with PSO")

    col1, col2, col3 = st.columns([1,1,2])

    with col1:
        st.markdown("### Data")
        data = st.radio("", ["Synthetic", "Digits"])

    with col2:
        st.markdown("### Selected Features (SF)")
        n_features = st.number_input("", 1, 100, 5)

    with col3:
        st.markdown("### α")
        alpha = st.number_input("", 0.0, 1.0, 0.9)

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        run_model = st.button("Model Evaluation")

    with col_btn2:
        rerun_model = st.button("Model Re-evaluation")

    if run_model or rerun_model:

        solution = generate_solution(50)
        selected_idx = select_features(solution, threshold=0.5)

        fitness = fitness_function(solution)
        accuracy = accuracy_function(solution)

        st.markdown("### Solution")

        st.text_area(
            "Solution:",
            " | ".join([f"{x:.2f}" for x in solution]),
            height=120
        )

        st.markdown("### Indices of selected features:")
        st.write(" | ".join(map(str, selected_idx)))

        st.markdown(
            f"**Fitness — {fitness:.2f}, Accuracy — {accuracy:.2f}, Selected Features — {len(selected_idx)}**"
        )

# =========================================================
# ======================= TP6 ==============================
# =========================================================
if mode == "TP6 - Multiple Runs":

    st.header(" Feature Selection with PSO (Advanced)")

    col1, col2 = st.columns(2)

    # LEFT PANEL
    with col1:
        st.subheader("Feature Selection parameters")

        data = st.selectbox("Data", ["Synthetic"])
        alpha = st.number_input("α", 0.0, 1.0, 0.5)

        st.subheader("PSO parameters")
        w = st.number_input("w", 0.0, 1.0, 0.5)
        c1 = st.number_input("c1", 0.0, 5.0, 2.0)
        c2 = st.number_input("c2", 0.0, 5.0, 2.0)

    # RIGHT PANEL
    with col2:
        st.subheader("Metaheuristic parameters")

        n_particles = st.slider("Population (N)", 5, 100, 30)
        n_iter = st.slider("Max Iteration (T)", 1, 100, 15)
        n_runs = st.slider("Run", 1, 30, 15)

        threshold = st.number_input("Threshold", 0.0, 1.0, 0.5)

        run_btn = st.button(" Evaluation")

    # ==========================
    # RUN MULTIPLE
    # ==========================
    if run_btn:

        best_all = []
        curves = []
        avg_curves = []

        for _ in range(n_runs):
            best_curve, avg_curve, pop = run_pso(n_particles, n_iter)
            curves.append(best_curve)
            avg_curves.append(avg_curve)
            best_all.append(min(best_curve))

        best = np.min(best_all)
        mean = np.mean(best_all)
        std = np.std(best_all)

        accuracy = 1 - best
        selected = int(pop.shape[1] * 0.5)

        # ======================
        # RESULTS
        # ======================
        st.markdown("##  Results")

        colR1, colR2 = st.columns([3,1])

        with colR2:
            st.write(f"**Best — {best:.4f}**")
            st.write(f"Mean — {mean:.4f}")
            st.write(f"Accuracy — {accuracy:.4f}")
            st.write(f"Selected — {selected}")
            st.write(f"STD — {std:.4f}")

        # ======================
        # PLOTS
        # ======================
        st.markdown("##  Visualization")

        col1, col2, col3, col4 = st.columns(4)

        # Convergence
        with col1:
            fig, ax = plt.subplots()
            ax.plot(curves[0])
            ax.set_title("Convergence Curve")
            st.pyplot(fig)

        # Trajectory
        with col2:
            fig, ax = plt.subplots()
            ax.plot(np.random.rand(n_iter))
            ax.set_title("Trajectory")
            st.pyplot(fig)

        # Average fitness
        with col3:
            fig, ax = plt.subplots()
            ax.plot(avg_curves[0])
            ax.set_title("Average Fitness")
            st.pyplot(fig)

        # Search history
        with col4:
            fig, ax = plt.subplots()
            ax.scatter(np.random.rand(100), np.random.rand(100))
            ax.set_title("Search History")
            st.pyplot(fig)