import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT TON CODE
from metah_tp1 import functions
from tp2 import Population, PopulationEvaluator

st.set_page_config(layout="wide")

st.title(" TP - Metaheuristics")
st.markdown("## Part 1 | Optimization Problem Initialization")

# ==========================
# NAVIGATION
# ==========================
mode = st.sidebar.radio(
    "Choose Part",
    ["TP1 - Solution", "TP2 - Population"]
)

# =========================================================
# ======================= TP1 ==============================
# =========================================================
if mode == "TP1 - Solution":

    st.markdown("### Standard Continuous Optimization Benchmark Problems")

    col1, col2, col3, col4 = st.columns([1,1,1,1])

    with col1:
        dim = st.number_input("Dimension (D)", 2, 100, 30)

    with col2:
        min_val = st.number_input("Min", -100.0)

    with col3:
        max_val = st.number_input("Max", 100.0)

    with col4:
        generate = st.button("Generate solution")

    st.markdown("### Solution")

    if generate:
        solution = np.random.uniform(min_val, max_val, dim)
        st.session_state["solution"] = solution

    if "solution" in st.session_state:
        st.text_area(
            "Candidate solution example",
            " | ".join([f"{x:.2f}" for x in st.session_state["solution"]]),
            height=100
        )

    colf1, colf2 = st.columns([1,2])

    with colf1:
        func_name = st.selectbox("Function", list(functions.keys()))

    with colf2:
        st.latex("f(x) = \\sum x_i^2")

    evaluate = st.button("Evaluate solution")

    if evaluate and "solution" in st.session_state:
        f = functions[func_name]
        fitness = f(st.session_state["solution"])
        st.success(f"Fitness = {fitness:.4f}")

# =========================================================
# ======================= TP2 ==============================
# =========================================================
if mode == "TP2 - Population":

    st.markdown("## Population Initialization")

    col1, col2 = st.columns([3,1])

    with col1:
        size = st.slider("Population Size", 5, 100, 30)

    with col2:
        generate_pop = st.button("Generate population")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if generate_pop:
        pop = Population.generate_random("F1", size)
        st.session_state["pop"] = pop

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        pop = Population("F1", df.values)
        st.session_state["pop"] = pop

    # ======================
    # DISPLAY
    # ======================
    if "pop" in st.session_state:

        pop = st.session_state["pop"]

        st.markdown("### Population")
        st.dataframe(pop.solutions)

        evaluate = st.button("Evaluate population")

        if evaluate:
            evaluator = PopulationEvaluator(pop)

            st.success(f"Best — {evaluator.best:.2f}")
            st.error(f"Worst — {evaluator.worst:.2f}")

            # ======================
            # PLOTS
            # ======================
            col1, col2 = st.columns(2)

            # 3D
            with col1:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

                X = np.linspace(-100,100,50)
                Y = np.linspace(-100,100,50)
                X, Y = np.meshgrid(X,Y)
                Z = X**2 + Y**2

                ax.plot_surface(X,Y,Z)
                st.pyplot(fig)

            # Contour + scatter
            with col2:
                fig, ax = plt.subplots()

                X = np.linspace(-100,100,100)
                Y = np.linspace(-100,100,100)
                X, Y = np.meshgrid(X,Y)
                Z = X**2 + Y**2

                ax.contour(X,Y,Z)

                sols = pop.solutions
                ax.scatter(sols[:,0], sols[:,1], c="black")

                best_sol = sols[np.argmin([np.sum(s**2) for s in sols])]
                ax.scatter(best_sol[0], best_sol[1], c="red", s=100)

                st.pyplot(fig)