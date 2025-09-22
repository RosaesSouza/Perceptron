# breast.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import time

from perceptron import Perceptron
from util import plot_decision_regions

# Criar diretório para imagens
os.makedirs('images', exist_ok=True)

# PASSO 1: Carregar o Dataset
print("=" * 50)
print("EXEMPLO: BREAST CANCER WISCONSIN")
print("=" * 50)

# Carregar o dataset de câncer de mama
breast_cancer = datasets.load_breast_cancer()

print(f"Dataset carregado:")
print(f"- Amostras: {breast_cancer.data.shape[0]}")
print(f"- Features: {breast_cancer.data.shape[1]}")
print(f"- Classes: {np.unique(breast_cancer.target)}")
print(f"- Nomes das classes: {breast_cancer.target_names}")
print(f"- Nomes das features: {breast_cancer.feature_names[:5]}... (total: {len(breast_cancer.feature_names)})")

# VERSÃO A: Usando apenas 2 features para visualização
print("\n=== VERSÃO A: Usando 2 features ===")
# Escolhendo as duas primeiras features (mean radius e mean texture)
X_2features = breast_cancer.data[:, :2] 
y = breast_cancer.target

# PASSO 2: Dividir em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X_2features, y,
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

ppn_2f = Perceptron(learning_rate=0.01, n_epochs=100)
ppn_2f.fit(X_train_std, y_train)

training_time_2f = time.time() - start_time
print(f"\nTempo de treinamento (2 features): {training_time_2f:.4f} segundos")

# PASSO 5: Avaliar o Modelo (2 features)
# Acurácia no treino
y_pred_train_2f = ppn_2f.predict(X_train_std)
train_accuracy_2f = accuracy_score(y_train, y_pred_train_2f)
# Acurácia no teste
y_pred_test_2f = ppn_2f.predict(X_test_std)
test_accuracy_2f = accuracy_score(y_test, y_pred_test_2f)

print(f"\nResultados (2 features):")
print(f"- Acurácia no conjunto de treinamento: {train_accuracy_2f:.2%}")
print(f"- Acurácia no conjunto de teste: {test_accuracy_2f:.2%}")
print(f"- Erros finais no treino: {ppn_2f.errors_history[-1]}")

# Verificar convergência
if 0 in ppn_2f.errors_history:
    conv_epoch = ppn_2f.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente")

# Matriz de confusão e relatório de classificação (2 features)
cm_2f = confusion_matrix(y_test, y_pred_test_2f)
print("\nMatriz de Confusão (2 features):")
print(cm_2f)
print("\nRelatório de Classificação (2 features):")
print(classification_report(y_test, y_pred_test_2f, target_names=breast_cancer.target_names))

# VERSÃO B: Usando todas as 30 features
print("\n\n=== VERSÃO B: Usando todas as 30 features ===")

# Dividir em Treino e Teste (todas as features)
X_all = breast_cancer.data
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y,
    test_size=0.3, # 30% para teste
    random_state=42,
    stratify=y # Mantém proporção das classes
)

# Normalização (todas as features)
scaler_all = StandardScaler()
X_train_all_std = scaler_all.fit_transform(X_train_all)
X_test_all_std = scaler_all.transform(X_test_all)

# Treinar o Perceptron (todas as features)
start_time = time.time()

ppn_all = Perceptron(learning_rate=0.01, n_epochs=100)
ppn_all.fit(X_train_all_std, y_train_all)

training_time_all = time.time() - start_time
print(f"\nTempo de treinamento (todas as features): {training_time_all:.4f} segundos")

# Avaliar o Modelo (todas as features)
# Acurácia no treino
y_pred_train_all = ppn_all.predict(X_train_all_std)
train_accuracy_all = accuracy_score(y_train_all, y_pred_train_all)
# Acurácia no teste
y_pred_test_all = ppn_all.predict(X_test_all_std)
test_accuracy_all = accuracy_score(y_test_all, y_pred_test_all)

print(f"\nResultados (todas as features):")
print(f"- Acurácia no conjunto de treinamento: {train_accuracy_all:.2%}")
print(f"- Acurácia no conjunto de teste: {test_accuracy_all:.2%}")
print(f"- Erros finais no treino: {ppn_all.errors_history[-1]}")

# Verificar convergência
if 0 in ppn_all.errors_history:
    conv_epoch = ppn_all.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente")

# Matriz de confusão e relatório de classificação (todas as features)
cm_all = confusion_matrix(y_test_all, y_pred_test_all)
print("\nMatriz de Confusão (todas as features):")
print(cm_all)
print("\nRelatório de Classificação (todas as features):")
print(classification_report(y_test_all, y_pred_test_all, target_names=breast_cancer.target_names))

# PASSO 6: Visualizar Resultados
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Regiões de Decisão (apenas para versão com 2 features)
plt.subplot(1,2,1)
plot_decision_regions(X_train_std, y_train, classifier=ppn_2f)
plt.title('Regiões de Decisão - Breast Cancer (2 features)')
plt.xlabel('Mean Radius (normalizado)')
plt.ylabel('Mean Texture (normalizado)')
plt.legend(loc='upper left')

# Subplot 2: Curvas de Convergência (comparação)
plt.subplot(1,2,2)
plt.plot(range(1, len(ppn_2f.errors_history) + 1), ppn_2f.errors_history, marker='o', label='2 features')
plt.plot(range(1, len(ppn_all.errors_history) + 1), ppn_all.errors_history, marker='s', label='30 features')
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Comparação da Convergência')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/breast_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico individual: Regiões de Decisão
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_std, y_train, classifier=ppn_2f)
plt.title('Regiões de Decisão - Breast Cancer (2 features)')
plt.xlabel('Mean Radius (normalizado)')
plt.ylabel('Mean Texture (normalizado)')
plt.legend(loc='upper left')
plt.savefig('images/breast_decision_regions.png', dpi=300, bbox_inches='tight')
plt.show()

# Gráfico individual: Comparação de Convergência
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(ppn_2f.errors_history) + 1), ppn_2f.errors_history, marker='o', label='2 features', linewidth=2)
plt.plot(range(1, len(ppn_all.errors_history) + 1), ppn_all.errors_history, marker='s', label='30 features', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Número de erros')
plt.title('Comparação da Convergência - Breast Cancer')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('images/breast_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise adicional: Visualização das matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Matriz de confusão (2 features)
axes[0].matshow(cm_2f, cmap=plt.cm.Blues)
axes[0].set_title('Matriz de Confusão (2 features)')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Maligno', 'Benigno'])
axes[0].set_yticklabels(['Maligno', 'Benigno'])

# Adicionar valores na matriz
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{cm_2f[i, j]}", 
                    ha="center", va="center", color="white" if cm_2f[i, j] > cm_2f.max()/2 else "black")

# Matriz de confusão (todas as features)
axes[1].matshow(cm_all, cmap=plt.cm.Blues)
axes[1].set_title('Matriz de Confusão (30 features)')
axes[1].set_xlabel('Predito')
axes[1].set_ylabel('Real')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Maligno', 'Benigno'])
axes[1].set_yticklabels(['Maligno', 'Benigno'])

# Adicionar valores na matriz
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{cm_all[i, j]}", 
                    ha="center", va="center", color="white" if cm_all[i, j] > cm_all.max()/2 else "black")

plt.tight_layout()
plt.savefig('images/breast_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nComparação entre as versões:")
print(f"- Acurácia de teste (2 features): {test_accuracy_2f:.2%}")
print(f"- Acurácia de teste (30 features): {test_accuracy_all:.2%}")
print(f"- Melhoria com todas as features: {test_accuracy_all - test_accuracy_2f:.2%}")
print(f"- Tempo de treinamento (2 features): {training_time_2f:.4f} segundos")
print(f"- Tempo de treinamento (30 features): {training_time_all:.4f} segundos")

# Análise dos falsos positivos e falsos negativos
fp_2f = cm_2f[0, 1]  # Falsos positivos (2 features): Maligno predito como Benigno
fn_2f = cm_2f[1, 0]  # Falsos negativos (2 features): Benigno predito como Maligno

fp_all = cm_all[0, 1]  # Falsos positivos (todas features)
fn_all = cm_all[1, 0]  # Falsos negativos (todas features)

print("\nAnálise de erros críticos:")
print(f"- Falsos positivos (2 features): {fp_2f}")
print(f"- Falsos negativos (2 features): {fn_2f}")
print(f"- Falsos positivos (30 features): {fp_all}")
print(f"- Falsos negativos (30 features): {fn_all}")

print("\nGráficos salvos na pasta 'images/':")
print("- breast_results.png (gráfico combinado)")
print("- breast_decision_regions.png")
print("- breast_convergence.png")
print("- breast_confusion_matrix.png")