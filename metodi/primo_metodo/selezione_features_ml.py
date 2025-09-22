"""
SCRIPT: SELEZIONE FEATURES PER MACHINE LEARNING

Questo script:
1. Carica il dataset filtrato dalla fase precedente
2. Seleziona solo le 15 features più significative per il ML
3. Mantiene il target 'readmitted_NO'
4. Salva il dataset ML-ready nella cartella third_clean
5. Genera report di selezione
"""

import pandas as pd
import numpy as np
import os

def main():
    print("SELEZIONE FEATURES PER MACHINE LEARNING")
    print("=" * 50)

    # 1. Crea cartella di output
    os.makedirs('outputs/datasets_clean/third_clean', exist_ok=True)

    # 2. Carica dataset filtrato
    input_path = 'outputs/datasets_clean/second_clean/diabetes_clean_filtered.csv'
    print(f"\nCaricando dataset da: {input_path}")

    df = pd.read_csv(input_path, sep=';')
    print(f"Dataset originale: {df.shape[0]} righe, {df.shape[1]} colonne")

    # 3. CARICA FEATURES SIGNIFICATIVE DA SIGNIFICATIVITA.PY
    print("Caricando features significative da significativita.py...")

    # Legge le features significative dal file generato da significativita.py
    features_file = 'outputs/dataset_pvalue/selected_features.txt'
    selected_features = []

    try:
        with open(features_file, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        # Estrae i nomi delle features dal file (ignora commenti)
        for line in lines:
            line = line.strip()
            if line.startswith("'") and ',' in line:
                # Estrae il nome della feature tra apici
                feature_name = line.split("'")[1]
                selected_features.append(feature_name)

        print(f"Features caricate da file: {len(selected_features)}")
        print("Prime 10 features:", selected_features[:10])

    except FileNotFoundError:
        print(f"ERRORE: File {features_file} non trovato!")
        print("Esegui prima 'python significativita.py' per generare le features significative.")
        return
    except Exception as e:
        print(f"ERRORE nel leggere il file: {e}")
        return

    # Target variable
    target_column = 'readmitted_NO'

    # 4. Verifica presenza features
    print(f"\nVerificando presenza features...")
    missing_features = []
    available_features = []

    for feature in selected_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)

    # Controlla target
    if target_column not in df.columns:
        print(f"ERRORE: Target '{target_column}' non trovato nel dataset!")
        return

    print(f"Features disponibili: {len(available_features)}/{len(selected_features)}")
    if missing_features:
        print(f"Features mancanti: {missing_features}")

    # 5. Crea dataset ML-ready
    print(f"\nCreando dataset ML-ready...")

    # Seleziona features + target
    ml_columns = available_features + [target_column]
    df_ml = df[ml_columns].copy()

    print(f"Dataset ML: {df_ml.shape[0]} righe, {df_ml.shape[1]} colonne")

    # 6. Verifica qualità dati
    print(f"\nVerifica qualità dati:")
    print(f"- Valori mancanti: {df_ml.isnull().sum().sum()}")
    print(f"- Righe duplicate: {df_ml.duplicated().sum()}")

    # Distribuzione target
    target_dist = df_ml[target_column].value_counts()
    print(f"- Distribuzione target:")
    for val, count in target_dist.items():
        percentage = (count / len(df_ml)) * 100
        print(f"  {target_column}={val}: {count} ({percentage:.1f}%)")

    # 7. Statistiche features per tipo
    binary_features = []
    continuous_features = []

    for feature in available_features:
        unique_vals = df_ml[feature].nunique()
        if unique_vals == 2:
            binary_features.append(feature)
        else:
            continuous_features.append(feature)

    print(f"\nTipologie features:")
    print(f"- Features binarie: {len(binary_features)}")
    print(f"- Features continue: {len(continuous_features)}")

    # 8. Salva dataset ML-ready
    output_path = 'outputs/datasets_clean/third_clean/diabetes_ml_ready.csv'
    df_ml.to_csv(output_path, index=False, sep=';')
    print(f"\nDataset ML salvato in: {output_path}")

    # 9. Genera report dettagliato
    report_path = 'outputs/datasets_clean/third_clean/feature_selection_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("REPORT SELEZIONE FEATURES PER MACHINE LEARNING\n")
        f.write("=" * 60 + "\n\n")

        f.write("TRASFORMAZIONE DATASET:\n")
        f.write(f"- Dataset input: {df.shape}\n")
        f.write(f"- Dataset output: {df_ml.shape}\n")
        f.write(f"- Riduzione colonne: {df.shape[1]} → {df_ml.shape[1]} (-{df.shape[1] - df_ml.shape[1]})\n")
        f.write(f"- Percentuale riduzione: {((df.shape[1] - df_ml.shape[1]) / df.shape[1]) * 100:.1f}%\n\n")

        f.write(f"FEATURES SELEZIONATE ({len(available_features)} + target):\n")
        f.write("-" * 40 + "\n")

        f.write("\nFEATURES SIGNIFICATIVE (ordinate per p-value):\n")
        for i, feature in enumerate(available_features, 1):
            feat_type = 'binaria' if feature in binary_features else 'continua'
            f.write(f"  {i:2d}. {feature:<25} ({feat_type})\n")

        f.write(f"\nTARGET VARIABLE:\n")
        f.write(f"- {target_column}\n")

        if missing_features:
            f.write(f"\nFEATURES NON TROVATE:\n")
            for feature in missing_features:
                f.write(f"- {feature}\n")

        f.write(f"\nQUALITA' DATI:\n")
        f.write(f"- Valori mancanti: {df_ml.isnull().sum().sum()}\n")
        f.write(f"- Righe duplicate: {df_ml.duplicated().sum()}\n")

        f.write(f"\nDISTRIBUZIONE TARGET:\n")
        for val, count in target_dist.items():
            percentage = (count / len(df_ml)) * 100
            f.write(f"- {target_column}={val}: {count} righe ({percentage:.1f}%)\n")

        f.write(f"\nTIPOLOGIE FEATURES:\n")
        f.write(f"- Features binarie: {len(binary_features)}\n")
        for feature in binary_features:
            f.write(f"  * {feature}\n")
        f.write(f"- Features continue: {len(continuous_features)}\n")
        for feature in continuous_features:
            f.write(f"  * {feature}\n")

        f.write(f"\nFILE GENERATI:\n")
        f.write(f"- Dataset ML-ready: diabetes_ml_ready.csv\n")
        f.write(f"- Questo report: feature_selection_report.txt\n")

    print(f"Report salvato in: {report_path}")

    # 10. Summary finale
    print(f"\n" + "="*50)
    print(f"SELEZIONE FEATURES COMPLETATA!")
    print(f"="*50)
    print(f"Dataset ML-ready creato con successo:")
    print(f"- Righe: {df_ml.shape[0]:,}")
    print(f"- Features: {len(available_features)}")
    print(f"- Target: {target_column}")
    print(f"- Riduzione features: {((df.shape[1] - df_ml.shape[1]) / df.shape[1]) * 100:.1f}%")
    print(f"\nIl dataset è ora pronto per il machine learning!")
    print(f"File: {output_path}")

    return df_ml

if __name__ == "__main__":
    dataset_ml = main()