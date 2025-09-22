# dlps.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import time

from perceptron import Perceptron
from util import plot_decision_regions

# Criar diretório para imagens
os.makedirs('images', exist_ok=True)

print("=" * 50)
print("EXERCÍCIO 5: DATASET LINEARMENTE SEPARÁVEL PERSONALIZADO (DLPS)")
print("=" * 50)

# Criar dataset personalizado conforme especificação
print("\n=== Criando dataset personalizado ===")

np.random.seed(42)

# Classe 0: centro em (-2, -2)
class_0 = np.random.randn(100, 2) + [-2, -2]

# Classe 1: centro em (2, 2)
class_1 = np.random.randn(100, 2) + [2, 2]

# Combinar
X = np.vstack([class_0, class_1])
y = np.hstack([np.zeros(100), np.ones(100)])

print(f"Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Centro Classe 0: ({np.mean(class_0[:, 0]):.1f}, {np.mean(class_0[:, 1]):.1f})")
print(f"Centro Classe 1: ({np.mean(class_1[:, 0]):.1f}, {np.mean(class_1[:, 1]):.1f})")

# Visualizar dataset original
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='o', label='Classe 0', alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='s', label='Classe 1', alpha=0.7)
plt.title('Dataset Linearmente Separável Personalizado (DLPS)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('images/dlps_original.png', dpi=300, bbox_inches='tight')
plt.show()

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ===========================================
# ANÁLISE PRINCIPAL
# ===========================================

print("\n=== ANÁLISE PRINCIPAL: Geometria da Solução ===")

ppn = Perceptron(learning_rate=0.01, n_epochs=50)
start_time = time.time()
ppn.fit(X_train_std, y_train)
training_time = time.time() - start_time

# Avaliar
y_pred_train = ppn.predict(X_train_std)
y_pred_test = ppn.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Verificar convergência
converged = 0 in ppn.errors_history
epochs_to_converge = ppn.errors_history.index(0) + 1 if converged else len(ppn.errors_history)

print(f"Resultados:")
print(f"- Acurácia treino: {train_accuracy:.2%}")
print(f"- Acurácia teste: {test_accuracy:.2%}")
print(f"- Convergiu: {'Sim' if converged else 'Não'}")
print(f"- Épocas até convergência: {epochs_to_converge}")
print(f"- Tempo: {training_time:.4f}s")

# Equação da reta de decisão
print(f"\nEquação da fronteira de decisão:")
print(f"- w1: {ppn.weights[0]:.3f}, w2: {ppn.weights[1]:.3f}, bias: {ppn.bias:.3f}")

if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"- x2 = {slope:.3f} * x1 + {intercept:.3f}")

# Verificar pontos classificados corretamente
correct_predictions = sum(y_pred_train == y_train)
total_points = len(y_train)
print(f"- Pontos corretos: {correct_predictions}/{total_points} ({correct_predictions/total_points:.1%})")

# ===========================================
# EXPERIMENTO DE PROXIMIDADE
# ===========================================

print(f"\n=== EXPERIMENTO DE PROXIMIDADE ===")

distances = [5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6]
proximity_results = []

print("Testando diferentes distâncias entre centros:")

for distance in distances:
    # Criar dataset com centros à distância específica
    center_offset = distance / 2
    
    np.random.seed(42)
    class_0_prox = np.random.randn(50, 2) + [-center_offset, -center_offset]
    class_1_prox = np.random.randn(50, 2) + [center_offset, center_offset]
    
    X_prox = np.vstack([class_0_prox, class_1_prox])
    y_prox = np.hstack([np.zeros(50), np.ones(50)])
    
    # Dividir e normalizar
    X_train_prox, X_test_prox, y_train_prox, y_test_prox = train_test_split(
        X_prox, y_prox, test_size=0.3, random_state=42, stratify=y_prox
    )
    
    scaler_prox = StandardScaler()
    X_train_prox_std = scaler_prox.fit_transform(X_train_prox)
    X_test_prox_std = scaler_prox.transform(X_test_prox)
    
    # Treinar
    ppn_prox = Perceptron(learning_rate=0.01, n_epochs=50)
    ppn_prox.fit(X_train_prox_std, y_train_prox)
    
    # Avaliar
    test_acc_prox = accuracy_score(y_test_prox, ppn_prox.predict(X_test_prox_std))
    converged_prox = 0 in ppn_prox.errors_history
    epochs_prox = ppn_prox.errors_history.index(0) + 1 if converged_prox else len(ppn_prox.errors_history)
    
    proximity_results.append({
        'distance': distance,
        'accuracy': test_acc_prox,
        'converged': converged_prox,
        'epochs': epochs_prox
    })
    
    print(f"Distância {distance}: {test_acc_prox:.1%} ({'Convergiu' if converged_prox else 'Não convergiu'})")

# Identificar threshold de falha
failure_threshold = None
for result in proximity_results:
    if result['accuracy'] < 0.80:
        failure_threshold = result['distance']
        break

if failure_threshold:
    print(f"\nThreshold de falha: distância ≤ {failure_threshold}")
else:
    print(f"\nPerceptron robusto em todas as distâncias testadas")

# ===========================================
# VISUALIZAÇÕES
# ===========================================

print(f"\n=== VISUALIZAÇÕES ===")

# Visualização combinada
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Regiões de decisão
plt.subplot(2, 2, 1)
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Fronteira de Decisão')
plt.xlabel('Feature 1 (normalizada)')
plt.ylabel('Feature 2 (normalizada)')

# 2. Convergência
plt.subplot(2, 2, 2)
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, 'b-o')
plt.xlabel('Épocas')
plt.ylabel('Erros')
plt.title('Convergência')
plt.grid(True, alpha=0.3)

# 3. Função de decisão 3D
ax = plt.subplot(2, 2, 3, projection='3d')
x1_range = np.linspace(X_train_std[:, 0].min()-1, X_train_std[:, 0].max()+1, 20)
x2_range = np.linspace(X_train_std[:, 1].min()-1, X_train_std[:, 1].max()+1, 20)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
Z = ppn.weights[0] * X1_grid + ppn.weights[1] * X2_grid + ppn.bias

ax.plot_surface(X1_grid, X2_grid, Z, alpha=0.3)
colors = ['red' if label == 0 else 'blue' for label in y_train]
ax.scatter(X_train_std[:, 0], X_train_std[:, 1], [0]*len(X_train_std), c=colors)
ax.set_title('Superfície de Decisão 3D')

# 4. Experimento de proximidade
plt.subplot(2, 2, 4)
distances_list = [r['distance'] for r in proximity_results]
accuracies = [r['accuracy'] for r in proximity_results]

plt.plot(distances_list, accuracies, 'g-o', linewidth=2, markersize=6)
plt.xlabel('Distância entre Centros')
plt.ylabel('Acurácia')
plt.title('Experimento de Proximidade')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.8, color='orange', linestyle='--', label='Threshold (80%)')
if failure_threshold:
    plt.axvline(x=failure_threshold, color='red', linestyle='--', label=f'Falha ≤{failure_threshold}')
plt.legend()

plt.tight_layout()
plt.savefig('images/dlps_geometric_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico detalhado do experimento de proximidade
plt.figure(figsize=(10, 6))
plt.plot(distances_list, accuracies, 'g-o', linewidth=3, markersize=8)
plt.xlabel('Distância entre Centros')
plt.ylabel('Acurácia de Teste')
plt.title('Experimento de Proximidade: Threshold de Falha')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Threshold (80%)')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Acaso (50%)')

for i, result in enumerate(proximity_results):
    if result['accuracy'] < 0.8:
        plt.scatter(result['distance'], result['accuracy'], color='red', s=100, marker='x')

plt.legend()
plt.ylim(0.4, 1.05)
plt.savefig('images/dlps_proximity_experiment.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================================
# ANÁLISE DOS RESULTADOS
# ===========================================

print(f"\n=== ANÁLISE DOS RESULTADOS ===")

print(f"\nObjetivos atendidos:")
print(f"1. Dataset customizado: centros em (-2,-2) e (2,2)")
if ppn.weights[1] != 0:
    print(f"2. Equação da reta: x2 = {slope:.3f} * x1 + {intercept:.3f}")
print(f"3. Verificação pontos: {correct_predictions}/{total_points} corretos")
if failure_threshold:
    print(f"4. Threshold de falha: distância ≤ {failure_threshold}")
else:
    print(f"4. Robusto até distância {min(distances_list)}")

print(f"\nGráficos salvos na pasta 'images/':")
print("- dlps_original.png")
print("- dlps_geometric_analysis.png") 
print("- dlps_proximity_experiment.png")