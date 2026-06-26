import numpy as np
import pandas as pd

def F1(x):
    return np.sum(x**2)

def F8(x):
    d = len(x)
    return 418.9829*d - np.sum(x*np.sin(np.sqrt(abs(x))))

def load_population(file):

    df = pd.read_csv(file)

    df = df.select_dtypes(include=[np.number])

    return df.to_numpy()


def PSO(func, X):

    n_particles, dim = X.shape

    V = np.zeros((n_particles,dim))

    pbest = X.copy()
    pbest_val = np.array([func(x) for x in X])

    gbest_index = np.argmin(pbest_val)
    gbest = pbest[gbest_index]
    gbest_val = pbest_val[gbest_index]

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    for _ in range(20):

        r1 = np.random.rand(n_particles,dim)
        r2 = np.random.rand(n_particles,dim)

        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)

        X = X + V

        fitness = np.array([func(x) for x in X])

        for i in range(n_particles):

            if fitness[i] < pbest_val[i]:
                pbest[i] = X[i]
                pbest_val[i] = fitness[i]

        best_index = np.argmin(pbest_val)

        if pbest_val[best_index] < gbest_val:
            gbest = pbest[best_index]
            gbest_val = pbest_val[best_index]

    return gbest_val


popF1 = load_population("Population_F1-UM.csv")
popF8 = load_population("Population_F8-MM.csv")

bestF1 = PSO(F1, popF1)
bestF8 = PSO(F8, popF8)

print("Best F1 =", bestF1)
print("Best F8 =", bestF8)