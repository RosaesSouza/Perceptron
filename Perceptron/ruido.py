# ruido.py

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
print("EXEMPLO: CLASSIFICAÇÃO COM RUÍDO")
print("=" * 50)

# Experimento 1: Variando a separação entre classes
print("\n=== EXPERIMENTO 1: Variando separação (class_sep) ===")

separations = [0.5, 1.0, 2.0, 3.0]
sep_results = []

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, sep in enumerate(separations):
    print(f"\nTestando separação: {sep}")
    
    # Gerar dataset com separação específica
    X, y = datasets.make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=sep,
        flip_y=0.05,  # 5% de ruído
        random_state=42
    )
    
    # Dividir e normalizar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Treinar perceptron
    ppn = Perceptron(learning_rate=0.01, n_epochs=50)
    ppn.fit(X_train_std, y_train)
    
    # Avaliar
    y_pred = ppn.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    
    sep_results.append({
        'separation': sep,
        'accuracy': accuracy,
        'epochs_to_converge': len(ppn.errors_history) if 0 in ppn.errors_history else 'Não convergiu',
        'final_errors': ppn.errors_history[-1]
    })
    
    print(f"- Acurácia: {accuracy:.2%}")
    print(f"- Erros finais: {ppn.errors_history[-1]}")
    
    # Plotar regiões de decisão
    plt.subplot(2, 2, i+1)
    plot_decision_regions(X_train_std, y_train, classifier=ppn)
    plt.title(f'Separação = {sep} (Acc: {accuracy:.1%})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.savefig('images/ruido_separation_experiment.png', dpi=300, bbox_inches='tight')
plt.show()

# Experimento 2: Variando o ruído nos rótulos
print("\n=== EXPERIMENTO 2: Variando ruído (flip_y) ===")

noise_levels = [0.0, 0.05, 0.10, 0.20]
noise_results = []

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

base_X, base_y = datasets.make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=2.0, flip_y=0.0,  # SEM ruído inicial
    random_state=42
)

for i, noise in enumerate(noise_levels):
    print(f"\nTestando ruído: {noise*100:.0f}%")
    
    # Aplicar ruído manualmente para controle total
    y_noisy = base_y.copy()
    if noise > 0:
        n_flip = int(len(y_noisy) * noise)
        flip_indices = np.random.choice(len(y_noisy), n_flip, replace=False)
        y_noisy[flip_indices] = 1 - y_noisy[flip_indices]  # Inverter rótulos
    
    # Dividir e normalizar
    X_train, X_test, y_train, y_test = train_test_split(
        base_X, y_noisy, test_size=0.3, random_state=42, stratify=y_noisy
    )
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Treinar perceptron
    ppn = Perceptron(learning_rate=0.01, n_epochs=50)
    ppn.fit(X_train_std, y_train)
    
    # Avaliar
    y_pred = ppn.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    
    noise_results.append({
        'noise': noise,
        'accuracy': accuracy,
        'epochs_to_converge': len(ppn.errors_history) if 0 in ppn.errors_history else 'Não convergiu',
        'final_errors': ppn.errors_history[-1]
    })
    
    print(f"- Acurácia: {accuracy:.2%}")
    print(f"- Erros finais: {ppn.errors_history[-1]}")
    
    # Plotar regiões de decisão
    plt.subplot(2, 2, i+1)
    plot_decision_regions(X_train_std, y_train, classifier=ppn)
    plt.title(f'Ruído = {noise*100:.0f}% (Acc: {accuracy:.1%})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.savefig('images/ruido_noise_experiment.png', dpi=300, bbox_inches='tight')
plt.show()

# Experimento 3: Early Stopping
print("\n=== EXPERIMENTO 3: Early Stopping ===")

# Implementar perceptron com early stopping
class PerceptronEarlyStopping(Perceptron):
    def fit(self, X, y, X_val=None, y_val=None, patience=5):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_history = []
        self.val_accuracy_history = []
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            errors = 0
            
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                error = y[idx] - y_predicted
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0)
            
            self.errors_history.append(errors)
            
            # Validação
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                self.val_accuracy_history.append(val_accuracy)
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping na época {epoch + 1}")
                    break

# Testar com early stopping
X, y = datasets.make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=1.5, flip_y=0.1, random_state=42
)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

# Perceptron normal
ppn_normal = Perceptron(learning_rate=0.01, n_epochs=100)
ppn_normal.fit(X_train_std, y_train)

# Perceptron com early stopping
ppn_early = PerceptronEarlyStopping(learning_rate=0.01, n_epochs=100)
ppn_early.fit(X_train_std, y_train, X_val_std, y_val, patience=10)

# Comparar resultados
acc_normal = accuracy_score(y_test, ppn_normal.predict(X_test_std))
acc_early = accuracy_score(y_test, ppn_early.predict(X_test_std))

print(f"\nComparação:")
print(f"- Perceptron normal: {acc_normal:.2%} (épocas: {len(ppn_normal.errors_history)})")
print(f"- Perceptron early stopping: {acc_early:.2%} (épocas: {len(ppn_early.errors_history)})")

# Gráfico de comparação
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Comparação de erros
plt.subplot(1, 2, 1)
plt.plot(range(1, len(ppn_normal.errors_history) + 1), ppn_normal.errors_history, 
         label='Normal', marker='o')
plt.plot(range(1, len(ppn_early.errors_history) + 1), ppn_early.errors_history, 
         label='Early Stopping', marker='s')
plt.xlabel('Épocas')
plt.ylabel('Erros de Treinamento')
plt.title('Comparação: Erros de Treinamento')
plt.legend()
plt.grid(True, alpha=0.3)

# Acurácia de validação (early stopping)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(ppn_early.val_accuracy_history) + 1), ppn_early.val_accuracy_history, 
         marker='o', color='green')
plt.xlabel('Épocas')
plt.ylabel('Acurácia de Validação')
plt.title('Early Stopping: Acurácia de Validação')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/ruido_early_stopping.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise final dos resultados
print("\n=== ANÁLISE DOS RESULTADOS ===")

print("\n1. Impacto da separação entre classes:")
for result in sep_results:
    print(f"   Separação {result['separation']}: {result['accuracy']:.1%} de acurácia")

print("\n2. Impacto do ruído nos rótulos:")
for result in noise_results:
    print(f"   Ruído {result['noise']*100:.0f}%: {result['accuracy']:.1%} de acurácia")

# Gráfico de resumo
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: Separação vs Acurácia
separations_list = [r['separation'] for r in sep_results]
accuracies_sep = [r['accuracy'] for r in sep_results]

plt.subplot(1, 2, 1)
plt.plot(separations_list, accuracies_sep, marker='o', linewidth=2, markersize=8)
plt.xlabel('Separação entre Classes')
plt.ylabel('Acurácia')
plt.title('Impacto da Separação na Performance')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

# Gráfico 2: Ruído vs Acurácia
noise_list = [r['noise']*100 for r in noise_results]
accuracies_noise = [r['accuracy'] for r in noise_results]

plt.subplot(1, 2, 2)
plt.plot(noise_list, accuracies_noise, marker='s', linewidth=2, markersize=8, color='red')
plt.xlabel('Ruído nos Rótulos (%)')
plt.ylabel('Acurácia')
plt.title('Impacto do Ruído na Performance')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('images/ruido_analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGráficos salvos na pasta 'images/':")
print("- ruido_separation_experiment.png")
print("- ruido_noise_experiment.png") 
print("- ruido_early_stopping.png")
print("- ruido_analysis_summary.png")