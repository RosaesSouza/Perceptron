# iris.py

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

# PASSO 1: Carregar o Dataset
print("=" * 50)
print("EXEMPLO: IRIS DATASET (SETOSA VS VERSICOLOR)")
print("=" * 50)

# Carregar o dataset Iris
iris = datasets.load_iris()

# IMPORTANTE: Usar apenas as classes 0 e 1 (Setosa e Versicolor)
# Classe 2 (Virginica) não é linearmente separável das outras
mask = iris.target != 2
X = iris.data[mask]
y = iris.target[mask]

# Usar apenas 2 features para visualização
# Índices [0, 2] = comprimento da sépala e comprimento da pétala
X = X[:, [0, 2]]

print(f"Dataset carregado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]} (comprimento da sépala, comprimento da pétala)")
print(f"- Classes: {np.unique(y)}")
print(f"- Nomes das classes: {iris.target_names[:2]}")

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

ppn = Perceptron(learning_rate=0.01, n_epochs=50)
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

print(f"\nResultados:")
print(f"- Acurácia no conjunto de treinamento: {train_accuracy:.2%}")
print(f"- Acurácia no conjunto de teste: {test_accuracy:.2%}")
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")

# Verificar convergência
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente")

# PASSO 6: Visualizar Resultados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Regiões de Decisão
plt.subplot(1,2,1)
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Regiões de Decisão - Iris (Setosa vs Versicolor)')
plt.xlabel('Comprimento da Sépala (normalizada)')
plt.ylabel('Comprimento da Pétala (normalizada)')
plt.legend(loc='upper left')

# Subplot 2: Curva de Convergência
plt.subplot(1,2,2)
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/iris_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráficos individuais
# Gráfico 1: Regiões de Decisão
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Regiões de Decisão - Iris (Setosa vs Versicolor)')
plt.xlabel('Comprimento da Sépala (normalizada)')
plt.ylabel('Comprimento da Pétala (normalizada)')
plt.legend(loc='upper left')
plt.savefig('images/iris_decision_regions.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico 2: Convergência
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o', color='green', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Convergência do Treinamento - Iris')
plt.grid(True, alpha=0.3)
plt.savefig('images/iris_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise dos Pesos Aprendidos
print(f"\nPesos aprendidos:")
print(f"- w1: {ppn.weights[0]:.4f}")
print(f"- w2: {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

# A equação da fronteira de decisão
if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")

print("\nGráficos salvos na pasta 'images/':")
print("- iris_results.png (gráfico combinado)")
print("- iris_decision_regions.png")
print("- iris_convergence.png")


