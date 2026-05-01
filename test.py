import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Assure-toi que le chemin correspond bien à l'emplacement de ton dossier ASCAD sur le Bureau
DATASET_PATH = "C:/Users/<TonUser>/Desktop/ASCAD/ASCAD_databases/ASCAD.h5" 
# Remplace <TonUser> par ton nom d'utilisateur Windows ou mets le chemin complet absolu

def explore_ascad_dataset(filepath):
    print(f"--- Ouverture du dataset : {filepath} ---\n")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Lister les clés principales (Groupes)
            print("Clés principales trouvées (Groupes HDF5) :")
            for key in f.keys():
                print(f" - {key}")
            
            # 2. inspecter le groupe 'Profiling' (Entraînement)
            if 'Profiling' in f:
                profiling = f['Profiling']
                print("\n--- Détails du groupe 'Profiling' ---")
                
                # Accéder aux traces et aux labels
                traces = profiling['traces']
                labels = profiling['labels']
                
                print(f"Shape des Traces (Profiling): {traces.shape}")
                print(f"Type de données: {traces.dtype}")
                print(f"Shape des Labels (Profiling): {labels.shape}")
                print(f"Type de labels: {labels.dtype}")
                
                # Vérifier les métadonnées (souvent présentes dans ASCAD)
                if 'metadata' in profiling:
                    print("Métadonnées présentes.")
                    
                # 3. inspecter le groupe 'Attack' (Test/validation)
            if 'Attack' in f:
                attack = f['Attack']
                print("\n--- Détails du groupe 'Attack' ---")
                print(f"Shape des Traces (Attack): {attack['traces'].shape}")
                print(f"Shape des Labels (Attack): {attack['labels'].shape}")

            # 4. Visualisation d'une trace pour comprendre le signal
            print("\nGénération d'un graphique de la première trace de profiling...")
            plt.figure(figsize=(12, 4))
            # On prend juste la première trace
            plt.plot(traces[0, :]) 
            plt.title("Exemple d'une trace de consommation (Profiling trace #0)")
            plt.xlabel("Temps (échantillons)")
            plt.ylabel("Tension / Puissance")
            plt.grid(True)
            plt.show()

            # Calcul basique pour voir l'amplitude
            print(f"\nStatistiques rapides sur la trace #0 :")
            print(f"Min: {np.min(traces[0])}, Max: {np.max(traces[0])}, Mean: {np.mean(traces[0])}")

    except FileNotFoundError:
        print("ERREUR : Le fichier n'a pas été trouvé. Vérifie le chemin dans la variable DATASET_PATH.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    # N'oublie pas de modifier le chemin ci-dessus si besoin !
    explore_ascad_dataset(DATASET_PATH)