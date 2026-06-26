import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. Dataset
# =========================
digits = load_digits()
X = digits.data
y = digits.target

D = X.shape[1]

# 🔥 FIX IMPORTANT (split une seule fois)
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 2. Paramètres
# =========================
n_particles = 10
max_iter = 20
alpha = 0.99

w = 0.5
c1 = 2
c2 = 2

n_runs = 15

# =========================
# 3. Fitness
# =========================
def fitness(x):
    indices = np.where(x > 0.5)[0]
    
    if len(indices) == 0:
        return 1
    
    X_train = X_train_full[:, indices]
    X_test = X_test_full[:, indices]
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    f1 = 1 - acc
    f2 = len(indices) / D
    
    return alpha * f1 + (1 - alpha) * f2

# =========================
# 4. Stockage
# =========================
accuracies = []
n_selected_features = []

best_global_acc = 0
best_global_features = None

# =========================
# 5. Multi-run PSO
# =========================
for run in range(n_runs):
    
    particles = np.random.rand(n_particles, D)
    velocities = np.random.rand(n_particles, D)

    pbest = particles.copy()
    pbest_scores = np.array([fitness(p) for p in particles])

    gbest = pbest[np.argmin(pbest_scores)]
    gbest_score = np.min(pbest_scores)

    for iteration in range(max_iter):
        for i in range(n_particles):
            
            r1, r2 = np.random.rand(), np.random.rand()
            
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)
            
            score = fitness(particles[i])
            
            if score < pbest_scores[i]:
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score

        best_index = np.argmin(pbest_scores)
        
        if pbest_scores[best_index] < gbest_score:
            gbest = pbest[best_index].copy()
            gbest_score = pbest_scores[best_index]

    # =========================
    # Evaluation finale
    # =========================
    indices = np.where(gbest > 0.5)[0]
    
    if len(indices) > 0:
        X_train = X_train_full[:, indices]
        X_test = X_test_full[:, indices]
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        accuracies.append(acc)
        n_selected_features.append(len(indices))
        
        print(f"Run {run+1}: Accuracy={acc:.4f}, Features={len(indices)}")
        
        if acc > best_global_acc:
            best_global_acc = acc
            best_global_features = indices

# =========================
# 6. Résultats finaux
# =========================
print("\n==============================")
print(" RESULTATS FINAUX")
print("==============================")

print(f" Best Accuracy = {best_global_acc:.4f}")
print(f" Best Features Count = {len(best_global_features)}")
print(f" Best Features = {best_global_features}")

print("\n--- Moyenne & stabilité ---")
print(f" Mean Accuracy = {np.mean(accuracies):.4f}")
print(f" Std Accuracy  = {np.std(accuracies):.4f}")
print(f" Mean Features = {np.mean(n_selected_features):.2f}")
print(f" Std Features  = {np.std(n_selected_features):.2f}")