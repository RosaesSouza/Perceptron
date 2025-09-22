# moons.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import time

from perceptron import Perceptron
from util import plot_decision_regions

# Criar diretório para imagens
os.makedirs('images', exist_ok=True)

# PASSO 1: Gerar o Dataset
print("=" * 50)
print("EXEMPLO: MOONS DATASET")
print("=" * 50)

# make_moons gera duas classes em formato de lua
X, y = datasets.make_moons(
    n_samples=200, # Total de pontos
    noise=0.15, # Ruído gaussiano adicionado
    random_state=42 # Seed para reprodutibilidade
)

print(f"Dataset gerado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {np.unique(y)}")
print(f"- Formato: Duas luas entrelaçadas (não linearmente separável)")

# PASSO 2: Dividir em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # 30% para teste
    random_state=42,
    stratify=y # Mantém proporção das classes
)

print(f"\nDivisão treino/teste:")
print(f"- Treino: {len(X_train)} amostras")
print(f"- Teste: {len(X_test)} amostras")

# PASSO 3: Normalização (Importante!)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# PASSO 4: Treinar o Perceptron
start_time = time.time()

ppn = Perceptron(learning_rate=0.01, n_epochs=100)  # Aumentando épocas para ver comportamento
ppn.fit(X_train_std, y_train)

training_time = time.time() - start_time
print(f"\nTempo de treinamento: {training_time:.4f} segundos")

# PASSO 5: Avaliar o Modelo
# Acurácia no treino
y_pred_train = ppn.predict(X_train_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
# Acurácia no teste
y_pred_test = ppn.predict(X_test_std)
test_accuracy = accuracy_score(y_test, y_pred_test)
# Acurácia total
y_pred_total = np.concatenate([y_pred_train, y_pred_test])
y_total = np.concatenate([y_train, y_test])
total_accuracy = accuracy_score(y_total, y_pred_total)
# Total de amostras classificadas erradas (treino + teste)
train_errors = np.sum(y_pred_train != y_train)
test_errors = np.sum(y_pred_test != y_test)
total_errors = int(train_errors + test_errors)

print(f"\nResultados:")
print(f"- Acurácia no conjunto de treinamento: {train_accuracy:.2%}")
print(f"- Acurácia no conjunto de teste: {test_accuracy:.2%}")
print(f"- Acurácia em todo o conjunto: {total_accuracy:.2%}")
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")
print(f"- Total de amostras classificadas erradas: {total_errors}")

# Verificar convergência
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente (esperado para dados não linearmente separáveis)")

# PASSO 6: Visualizar Dataset Original
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='o', label='Classe 0', alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='s', label='Classe 1', alpha=0.7)
plt.title('Dataset Moons Original')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('images/moons_original.png', dpi=300, bbox_inches='tight')
plt.show()

# PASSO 7: Visualizar Resultados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Regiões de Decisão
plt.subplot(1,2,1)
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Regiões de Decisão - Moons')
plt.xlabel('Feature 1 (normalizada)')
plt.ylabel('Feature 2 (normalizada)')
plt.legend(loc='upper left')

# Subplot 2: Curva de Convergência
plt.subplot(1,2,2)
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/moons_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráficos individuais
# Gráfico 1: Regiões de Decisão
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Regiões de Decisão - Moons Dataset')
plt.xlabel('Feature 1 (normalizada)')
plt.ylabel('Feature 2 (normalizada)')
plt.legend(loc='upper left')
plt.savefig('images/moons_decision_regions.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 2: Convergência
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o', color='orange', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento - Moons (Não Converge)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Convergência Ideal')
plt.legend()
plt.savefig('images/moons_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise dos Pesos Aprendidos
print(f"\nPesos aprendidos:")
print(f"- w1: {ppn.weights[0]:.4f}")
print(f"- w2: {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

# A equação da fronteira de decisão é:
# w1*x1 + w2*x2 + bias = 0
# ou seja: x2 = -(w1/w2)*x1 - (bias/w2)
if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")
    print(f"Nota: Esta fronteira linear não consegue separar bem as classes em forma de lua")

print("\nGráficos salvos na pasta 'images/':")
print("- moons_original.png (dataset original)")
print("- moons_results.png (gráfico combinado)")
print("- moons_decision_regions.png")
print("- moons_convergence.png")