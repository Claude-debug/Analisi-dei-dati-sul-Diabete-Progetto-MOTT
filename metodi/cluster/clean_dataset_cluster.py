#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Carica e pulisce il dataset diabetico"""
    print("Caricamento dataset...")

    # Carica il dataset con separator corretto
    df = pd.read_csv('database/diabetic_data.csv', sep=';')
    print(f"Dataset originale: {df.shape}")
    print(f"Pazienti unici: {df['patient_nbr'].nunique()}")

    return df

def remove_duplicate_patients(df):
    """Rimuove i record duplicati per paziente, mantenendo solo il primo encounter"""
    print("\nRimozione duplicati per paziente...")

    # Ordina per patient_nbr e encounter_id per mantenere il primo encounter
    df_sorted = df.sort_values(['patient_nbr', 'encounter_id'])

    # Mantieni solo il primo record per ogni paziente
    df_unique = df_sorted.drop_duplicates(subset=['patient_nbr'], keep='first')

    print(f"Record rimossi: {len(df) - len(df_unique)}")
    print(f"Dataset dopo rimozione duplicati: {df_unique.shape}")

    return df_unique

def replace_question_marks_with_nan(df):
    """Sostituisce tutti i valori '?' con NaN"""
    print("\nSostituzione '?' con NaN...")

    # Conta i '?' per colonna prima della sostituzione
    question_counts = {}
    for col in df.columns:
        question_counts[col] = (df[col] == '?').sum()

    # Sostituisci '?' con NaN
    df_clean = df.replace('?', np.nan)

    # Mostra statistiche sui valori mancanti
    print("Colonne con valori mancanti:")
    for col, count in question_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} ({percentage:.1f}%)")

    return df_clean

def handle_missing_values(df):
    """Gestisce i valori mancanti del dataset"""
    print("\nGestione valori mancanti...")

    # Identifica colonne con troppi valori mancanti (>80%)
    missing_threshold = 0.8
    high_missing_cols = []

    for col in df.columns:
        missing_percentage = df[col].isnull().sum() / len(df)
        if missing_percentage > missing_threshold:
            high_missing_cols.append(col)

    print(f"Colonne con >{missing_threshold*100}% valori mancanti: {len(high_missing_cols)}")
    for col in high_missing_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        print(f"  {col}: {missing_pct:.1f}%")

    # Rimuovi colonne con troppi valori mancanti
    df_cleaned = df.drop(columns=high_missing_cols)
    print(f"Dataset dopo rimozione colonne: {df_cleaned.shape}")

    # Per le colonne rimanenti, gestisci i valori mancanti
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                # Per colonne numeriche, usa la mediana
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            else:
                # Per colonne categoriche, usa la moda
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    print(f"Valori mancanti rimanenti: {df_cleaned.isnull().sum().sum()}")
    return df_cleaned

def create_age_clusters(df):
    """Crea cluster basati sull'età dei pazienti"""
    print("\nCreazione cluster per età...")

    # Mapping fasce d'età a valori numerici
    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }

    # Converti età in valori numerici
    df['age_numeric'] = df['age'].map(age_mapping)

    # Prepara features per clustering
    cluster_features = ['age_numeric']

    # Aggiungi altre features demografiche se disponibili
    if 'gender' in df.columns:
        # Codifica gender
        gender_encoder = LabelEncoder()
        df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])
        cluster_features.append('gender_encoded')

    # Standardizza features per clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[cluster_features])

    # Applica K-means clustering
    n_clusters = 3  # Giovani, mezza età, anziani
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['age_cluster'] = kmeans.fit_predict(features_scaled)

    # Analizza i cluster risultanti
    print("Analisi cluster età:")
    for cluster_id in range(n_clusters):
        cluster_data = df[df['age_cluster'] == cluster_id]
        age_range = f"{cluster_data['age_numeric'].min():.0f}-{cluster_data['age_numeric'].max():.0f}"
        print(f"  Cluster {cluster_id}: {len(cluster_data)} pazienti, età {age_range}")

    return df

def encode_categorical_variables(df):
    """Codifica le variabili categoriche"""
    print("\nCodifica variabili categoriche...")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Rimuovi colonne che non vogliamo codificare
    exclude_cols = ['patient_nbr', 'encounter_id']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    print(f"Colonne categoriche da codificare: {len(categorical_cols)}")

    # Applica Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df, label_encoders

def prepare_target_variable(df):
    """Prepara la variabile target per la predizione"""
    print("\nPreparazione variabile target...")

    if 'readmitted' in df.columns:
        # Mappa readmitted in variabile binaria
        # 'NO' -> 0, '<30' e '>30' -> 1
        readmit_mapping = {'NO': 0, '<30': 1, '>30': 1}
        df['readmitted_binary'] = df['readmitted'].map(readmit_mapping)

        # Statistiche target
        target_dist = df['readmitted_binary'].value_counts()
        print("Distribuzione target:")
        print(f"  No riammissione (0): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
        print(f"  Riammissione (1): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")

    return df

def save_cleaned_dataset(df, output_path):
    """Salva il dataset pulito"""
    print(f"\nSalvataggio dataset pulito in: {output_path}")

    # Crea directory se non esiste
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva dataset
    df.to_csv(output_path, index=False)

    print(f"Dataset salvato: {df.shape}")
    print(f"Colonne finali: {len(df.columns)}")

    return output_path

def analyze_cluster_performance(df):
    """Analizza le performance dei cluster per la predizione"""
    print("\nAnalisi performance cluster...")

    if 'readmitted_binary' in df.columns and 'age_cluster' in df.columns:
        # Analizza distribuzione target per cluster
        for cluster_id in df['age_cluster'].unique():
            cluster_data = df[df['age_cluster'] == cluster_id]
            readmit_rate = cluster_data['readmitted_binary'].mean()
            print(f"  Cluster {cluster_id}: {len(cluster_data)} pazienti, riammissione {readmit_rate:.1%}")

def main():
    """Pipeline principale di pulizia dataset"""
    print("PIPELINE PULIZIA DATASET DIABETICI")
    print("="*50)

    # Step 1: Carica dataset originale
    df = load_and_clean_data()

    # Step 2: Rimuovi duplicati per paziente
    df = remove_duplicate_patients(df)

    # Step 3: Sostituisci '?' con NaN
    df = replace_question_marks_with_nan(df)

    # Step 4: Gestisci valori mancanti
    df = handle_missing_values(df)

    # Step 5: Crea cluster per età
    df = create_age_clusters(df)

    # Step 6: Codifica variabili categoriche
    df, encoders = encode_categorical_variables(df)

    # Step 7: Prepara variabile target
    df = prepare_target_variable(df)

    # Step 8: Analizza performance cluster
    analyze_cluster_performance(df)

    # Step 9: Salva dataset pulito
    output_path = 'outputs/datasets_clean/cluster/db_clean_cluster.csv'
    save_cleaned_dataset(df, output_path)

    print("\n" + "="*50)
    print("PULIZIA DATASET COMPLETATA!")
    print(f"Output: {output_path}")
    print(f"Pazienti finali: {len(df)}")
    print(f"Features finali: {len(df.columns)}")

    return df, output_path

if __name__ == "__main__":
    df_cleaned, output_file = main()