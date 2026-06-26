import numpy as np

# Définition des fonctions
def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F5(x):
    return np.sum(100 * (x[:-1]**2 - x[1:])**2 + (1 - x[:-1])**2)

def F7(x):
    D = len(x)
    return np.sum([i * (x[i-1]**4) for i in range(1, D+1)]) + np.random.rand()

def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F11(x):
    D = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod([np.cos(x[i] / np.sqrt(i+1)) for i in range(D)])
    return 1 + sum_term - prod_term


# Solutions candidates
solutions = {
    'F1': np.array([-27.81, -71.96, -47.13, 54.63, -86.58, -96.77, 63.39, 75.60, -39.94, -45.13, 
                    90.77, 70.68, 36.61, 18.50, -46.11, -91.04, -34.74, 94.34, 61.98, 77.94, 
                    -78.75, -3.11, -2.81, 80.69, -76.95, 43.46, 3.65, -26.73, 49.26, 0.72]),
    
    'F2': np.array([9.79, -1.60, 6.62, 6.07, 8.16, -1.29, 3.97, 7.41, 1.13, -7.32, 
                    7.15, 3.59, 5.92, -9.07, 2.34, -2.47, -7.22, 4.24, -7.75, -9.37, 
                    7.08, 5.02, -2.03, -4.33, 1.50, -2.29, 7.32, 9.72, -9.67, -2.52]),
    
    'F5': np.array([-23.85, -19.42, 6.20, -9.92, -1.11, 6.19, -3.25, 10.30, 22.39, -10.27, 
                    8.92, 28.76, -10.92, 28.92, 12.16, -23.71, 13.74, -12.67, -23.86, 19.71, 
                    -25.63, 28.86, -28.97, 11.99, 6.37, -8.29, -6.86, 1.50, -25.89, 19.02]),
    
    'F7': np.array([15.30, -124.59, 15.30, 76.61, -24.73, 60.58, -37.78, 89.80, -77.46, 71.79, 
                    1.68, 54.56, -9.37, -48.23, 10.54, -46.11, -94.84, 9.99, -10.71, 19.30, 
                    -46.78, 20.22, -109.43, -66.11, 103.04, 124.89, -33.52, 43.97, -52.42, -71.94]),
    
    'F8': np.array([294.94, 93.76, 77.03, -44.49, 338.08, 252.79, 318.38, 54.94, 428.00, 466.38, 
                    306.97, 344.46, 469.53, 251.11, -198.22, 182.60, 362.80, 322.47, 377.28, 237.63, 
                    131.96, -58.09, -240.77, -484.17, -464.34, -152.18, -38.75, 369.36, -135.30, -249.50]),
    
    'F9': np.array([-1.20, -1.63, 2.42, -4.95, -2.28, 1.16, 0.02, 0.71, 0.80, 4.00, 
                    4.92, -1.61, -0.77, 2.36, 4.36, -5.09, -1.42, 0.21, -1.96, 3.47, 
                    3.08, -3.75, 3.71, 3.97, 3.80, -3.91, -0.09, 4.51, -1.06, 0.68]),
    
    'F11': np.array([377.01, -369.41, 81.54, -546.71, -60.41, -338.54, -561.90, -268.59, -21.93, 118.09, 
                     150.80, -507.26, 25.70, -540.82, 146.39, 243.18, -107.89, 242.99, -281.18, 195.16, 
                     301.57, 78.46, -447.01, -492.81, -488.54, -24.41, -22.75, -324.18, 437.47, -595.42])
}

# Fonctions correspondantes
functions = {
    'F1': F1,
    'F2': F2,
    'F5': F5,
    'F7': F7,
    'F8': F8,
    'F9': F9,
    'F11': F11
}

# Optimum théoriques (valeurs idéales)
optimums = {
    'F1': 0,
    'F2': 0,
    'F5': 0,
    'F7': 0,
    'F8': -418.9829 * 30,  # -418.9829 * dimension
    'F9': 0,
    'F11': 0
}

print("="*80)
print("ÉVALUATION DES SOLUTIONS CANDIDATES")
print("="*80)

for func_name in ['F1', 'F2', 'F5', 'F7', 'F8', 'F9', 'F11']:
    print(f"\n{'='*80}")
    print(f"FONCTION {func_name}")
    print(f"{'='*80}")
    
    x = solutions[func_name]
    func = functions[func_name]
    optimum = optimums[func_name]
    
    # Pour F7, calculer plusieurs fois car elle contient un terme aléatoire
    if func_name == 'F7':
        results = [func(x) for _ in range(10)]
        result = np.mean(results)
        std = np.std(results)
        print(f"Dimension: {len(x)}")
        print(f"Valeur moyenne (10 exécutions): {result:.6f}")
        print(f"Écart-type: {std:.6f}")
        print(f"Min-Max: [{min(results):.6f}, {max(results):.6f}]")
    else:
        result = func(x)
        print(f"Dimension: {len(x)}")
        print(f"Valeur de la fonction: {result:.6f}")
    
    print(f"Optimum théorique: {optimum:.6f}")
    print(f"Distance à l'optimum: {abs(result - optimum):.6f}")
    
    # Statistiques sur la solution
    print(f"\nStatistiques de la solution:")
    print(f"  Min: {np.min(x):.2f}")
    print(f"  Max: {np.max(x):.2f}")
    print(f"  Moyenne: {np.mean(x):.2f}")
    print(f"  Écart-type: {np.std(x):.2f}")
    
    # Top 3 contributions pour certaines fonctions
    if func_name in ['F1', 'F9']:
        print(f"\nTop 3 valeurs (en valeur absolue):")
        abs_x = np.abs(x)
        top_indices = np.argsort(abs_x)[::-1][:3]
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. x[{idx}] = {x[idx]:.2f} (|x| = {abs_x[idx]:.2f})")

print(f"\n{'='*80}")
print("RÉSUMÉ COMPARATIF")
print(f"{'='*80}")
print(f"{'Fonction':<10} {'Valeur obtenue':<20} {'Optimum':<15} {'Distance':<15}")
print("-"*80)

for func_name in ['F1', 'F2', 'F5', 'F7', 'F8', 'F9', 'F11']:
    x = solutions[func_name]
    func = functions[func_name]
    optimum = optimums[func_name]
    
    if func_name == 'F7':
        result = np.mean([func(x) for _ in range(10)])
    else:
        result = func(x)
    
    distance = abs(result - optimum)
    print(f"{func_name:<10} {result:<20.6f} {optimum:<15.6f} {distance:<15.6f}")

print("="*80)