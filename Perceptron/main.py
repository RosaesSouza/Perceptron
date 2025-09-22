# main.py
import sys

def print_menu():
    print("\n===== PERCEPTRON - MENU DE EXEMPLOS =====")
    print("0. Blobs (demonstração básica)")
    print("1. Iris Dataset (Setosa vs Versicolor)")
    print("2. Moons Dataset (não linearmente separável)")
    print("3. Breast Cancer Wisconsin")
    print("4. Classificação com Ruído")
    print("5. Dataset Linearmente Separável Personalizado (DLPS)")
    print("6. Sair")
    print("=======================================")
    return input("Escolha um exemplo para executar (0-6): ")

def main():
    while True:
        choice = print_menu()
        
        if choice == '6':
            print("Programa encerrado.")
            sys.exit(0)
            
        elif choice == '0':
            print("\nExecutando exemplo de Blobs (demonstração básica)...\n")
            import blobs
            
        elif choice == '1':
            print("\nExecutando exemplo de Iris Dataset...\n")
            import iris
            
        elif choice == '2':
            print("\nExecutando exemplo de Moons Dataset...\n")
            import moons
            
        elif choice == '3':
            print("\nExecutando exemplo de Breast Cancer Wisconsin...\n")
            import breast
            
        elif choice == '4':
            print("\nExecutando exemplo de Classificação com Ruído...\n")
            import ruido
            
        elif choice == '5':
            print("\nExecutando exemplo de Dataset Linearmente Separável Personalizado...\n")
            import dlps
            
        else:
            print("Opção inválida. Por favor, escolha um número entre 0 e 6.")
        
        input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    print("Atividade Prática Perceptron")
    print("Implementação do algoritmo de Perceptron com diferentes datasets")
    main()