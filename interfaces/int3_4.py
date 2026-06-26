import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# NAVIGATION
# ==========================
mode = st.sidebar.radio(
    "Choose TP",
    ["TP3 - PSO Single Run", "TP4 - PSO Multiple Runs"]
)

# ==========================
# SIMPLE PSO (simulation)
# ==========================
def run_pso(n_particles=30, n_iter=50, dim=2):
    pop = np.random.uniform(-100, 100, (n_particles, dim))

    best_curve = []
    avg_curve = []
    traj = []

    for t in range(n_iter):
        fitness = np.sum(pop**2, axis=1)

        best_idx = np.argmin(fitness)
        best_curve.append(np.min(fitness))
        avg_curve.append(np.mean(fitness))
        traj.append(pop[0].copy())

        # fake update
        pop = pop * 0.9 + np.random.uniform(-1,1,(n_particles,dim))

    return pop, best_curve, avg_curve, np.array(traj)

# =========================================================
# ======================= TP3 ==============================
# =========================================================
if mode == "TP3 - PSO Single Run":

    st.header(" Application of PSO for F1 function")

    col1, col2 = st.columns([1,2])

    with col1:
        run_btn = st.button("Evaluate")

    with col2:
        st.markdown("### Metaheuristic")
        algo = st.selectbox("", ["PSO"])
        n_iter = st.number_input("Max Iteration (T)", 10, 500, 200)

        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            w = st.number_input("w", 0.0, 1.0, 0.3)
        with colp2:
            c1 = st.number_input("c1", 0.0, 5.0, 1.4)
        with colp3:
            c2 = st.number_input("c2", 0.0, 5.0, 1.4)

    if run_btn:

        pop, best_curve, avg_curve, traj = run_pso(30, n_iter)

        st.markdown("##  Results")

        colA, colB = st.columns([2,1])

        # LEFT: plots
        with colA:
            colg1, colg2 = st.columns(2)

            # Surface (fake)
            with colg1:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                X = np.linspace(-100,100,50)
                Y = np.linspace(-100,100,50)
                X, Y = np.meshgrid(X,Y)
                Z = X**2 + Y**2
                ax.plot_surface(X,Y,Z)
                ax.set_title("Function (F1-UM)")
                st.pyplot(fig)

            # Scatter
            with colg2:
                fig, ax = plt.subplots()
                ax.scatter(pop[:,0], pop[:,1])
                ax.set_title("Search History (Final)")
                st.pyplot(fig)

            # bottom plots
            colb1, colb2, colb3 = st.columns(3)

            with colb1:
                fig, ax = plt.subplots()
                ax.plot(best_curve)
                ax.set_title("Convergence Curve")
                st.pyplot(fig)

            with colb2:
                fig, ax = plt.subplots()
                ax.plot(traj[:,0])
                ax.set_title("Trajectory")
                st.pyplot(fig)

            with colb3:
                fig, ax = plt.subplots()
                ax.plot(avg_curve)
                ax.set_title("Average Fitness")
                st.pyplot(fig)

        # RIGHT: stats
        with colB:
            st.write("Initial population:")
            st.write(f"Best — {max(best_curve):.2f}")

            st.write("Final population:")
            st.write(f"Best — {min(best_curve):.2f}")

            st.write(f"Stagnation — Iteration N°{np.argmin(best_curve)}")

# =========================================================
# ======================= TP4 ==============================
# =========================================================
if mode == "TP4 - PSO Multiple Runs":

    st.header(" Running Multiple Populations")

    col1, col2 = st.columns([2,1])

    with col1:
        n_runs = st.slider("Run", 1, 50, 30)

    with col2:
        run_btn = st.button("Evaluate")

    if run_btn:

        best_all = []
        curves = []
        avg_curves = []

        for _ in range(n_runs):
            _, best_curve, avg_curve, traj = run_pso(30, 100)
            best_all.append(min(best_curve))
            curves.append(best_curve)
            avg_curves.append(avg_curve)

        best = np.min(best_all)
        mean = np.mean(best_all)
        std = np.std(best_all)

        st.markdown("##  Results")

        colA, colB = st.columns([2,1])

        # LEFT plots
        with colA:

            colg1, colg2 = st.columns(2)

            with colg1:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                X = np.linspace(-100,100,50)
                Y = np.linspace(-100,100,50)
                X, Y = np.meshgrid(X,Y)
                Z = X**2 + Y**2
                ax.plot_surface(X,Y,Z)
                ax.set_title("Function")
                st.pyplot(fig)

            with colg2:
                fig, ax = plt.subplots()
                ax.scatter(np.random.rand(200), np.random.rand(200))
                ax.set_title("Search History")
                st.pyplot(fig)

            colb1, colb2, colb3 = st.columns(3)

            with colb1:
                fig, ax = plt.subplots()
                ax.plot(np.mean(curves, axis=0))
                ax.set_title("Convergence Curve")
                st.pyplot(fig)

            with colb2:
                fig, ax = plt.subplots()
                ax.plot(np.random.rand(100))
                ax.set_title("Trajectory")
                st.pyplot(fig)

            with colb3:
                fig, ax = plt.subplots()
                ax.plot(np.mean(avg_curves, axis=0))
                ax.set_title("Average Fitness")
                st.pyplot(fig)

        # RIGHT stats
        with colB:
            st.write(f"Best — {best:.2f}")
            st.write(f"Mean — {mean:.2f}")
            st.write(f"STD — {std:.2f}")