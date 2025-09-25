"""
Pulizia e Preprocessing del Dataset Diabetici per Analisi Riammissione

Questo modulo si occupa della pulizia iniziale del dataset diabetici, includendo:
- Rimozione colonne irrilevanti
- Gestione valori mancanti
- Deduplicazione pazienti
- Conversione features categoriche
- Normalizzazione dati

Il dataset risultante è pronto per analisi statistiche e machine learning.

Input: database/diabetic_data.csv (dataset grezzo)
Output: outputs/datasets_clean/first_clean/diabetes_clean.csv (dataset pulito)

Autore: Progetto MOTT - Predizione Riammissione Diabetici
Data: 2024
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import json
from typing import Dict, List, Tuple

def load_raw_dataset(filepath: str) -> pd.DataFrame:
    """
    Carica il dataset grezzo dei pazienti diabetici.

    Parameters:
    -----------
    filepath : str
        Percorso al file CSV del dataset grezzo

    Returns:
    --------
    pd.DataFrame
        Dataset caricato con informazioni di debugging
    """
    print("Caricamento dataset grezzo...")
    df = pd.read_csv(filepath, sep=';')

    print(f"Dataset caricato: {df.shape[0]:,} righe x {df.shape[1]:,} colonne")
    print(f"Colonne disponibili: {len(df.columns)} colonne")

    return df

def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove le colonne irrilevanti per l'analisi di riammissione.

    Colonne rimosse:
    - encounter_id: Identificatore incontro (non predittivo)
    - weight: Troppi valori mancanti
    - admission_type_id, admission_source_id, discharge_disposition_id: ID numerici non interpretativi
    - payer_code: Informazioni di pagamento (non mediche)
    - medical_specialty: Specialità medica (alta cardinalità)
    - diag_1, diag_2, diag_3: Diagnosi specifiche (troppo granulari)
    - max_glu_serum, A1Cresult: Risultati laboratorio (molti missing)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset originale

    Returns:
    --------
    pd.DataFrame
        Dataset con colonne irrilevanti rimosse
    """
    # Lista colonne da rimuovere con motivazione
    cols_to_drop = [
        "encounter_id",          # ID incontro - non predittivo
        "weight",                # Troppi valori mancanti
        "admission_type_id",     # ID numerico non interpretabile
        "admission_source_id",   # ID numerico non interpretabile
        "discharge_disposition_id", # ID numerico non interpretabile
        "payer_code",            # Info pagamento - non medica
        "medical_specialty",     # Alta cardinalità, difficile generalizzare
        "diag_1",               # Diagnosi troppo granulare
        "diag_2",               # Diagnosi troppo granulare
        "diag_3",               # Diagnosi troppo granulare
        "max_glu_serum",        # Molti missing values
        "A1Cresult"             # Molti missing values
    ]

    # Filtro solo colonne esistenti per evitare errori
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    missing_cols = [col for col in cols_to_drop if col not in df.columns]

    print(f"\nRimozione colonne irrilevanti:")
    print(f"- Colonne da rimuovere presenti: {len(existing_cols_to_drop)}")
    print(f"- Colonne da rimuovere già assenti: {len(missing_cols)}")

    if missing_cols:
        print(f"- Colonne non trovate: {missing_cols}")

    # Rimozione colonne
    df_clean = df.drop(columns=existing_cols_to_drop)

    print(f"- Shape dopo rimozione: {df_clean.shape[0]:,} righe x {df_clean.shape[1]:,} colonne")

    return df_clean

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gestisce i valori mancanti nel dataset.

    Nel dataset diabetici, i valori mancanti sono rappresentati da "?".
    Questa funzione li converte in NaN e rimuove le righe incomplete.

    Strategia: Rimozione completa delle righe con missing values perché:
    - Il dataset è sufficientemente grande
    - I missing values sono distribuiti su molte colonne
    - L'imputation potrebbe introdurre bias

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con potenziali valori mancanti

    Returns:
    --------
    pd.DataFrame
        Dataset senza valori mancanti
    """
    print(f"\nGestione valori mancanti:")

    # Conteggio valori "?" prima della conversione
    question_marks = (df == "?").sum().sum()
    print(f"- Valori '?' trovati: {question_marks:,}")

    # Conversione "?" in NaN pandas
    df_clean = df.replace("?", pd.NA)

    # Conteggio righe con valori mancanti
    rows_with_na = df_clean.isna().any(axis=1).sum()
    print(f"- Righe con valori mancanti: {rows_with_na:,}")

    # Rimozione righe incomplete
    df_clean = df_clean.dropna()

    print(f"- Shape dopo pulizia: {df_clean.shape[0]:,} righe x {df_clean.shape[1]:,} colonne")
    print(f"- Righe rimosse: {df.shape[0] - df_clean.shape[0]:,}")

    return df_clean

def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove i pazienti duplicati mantenendo solo la prima occorrenza.

    Nel dataset diabetici, alcuni pazienti possono avere multiple visite.
    Per l'analisi di riammissione, consideriamo solo il primo incontro
    per evitare bias e dipendenze temporali.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con potenziali pazienti duplicati

    Returns:
    --------
    pd.DataFrame
        Dataset con pazienti unici (senza colonna patient_nbr)
    """
    print(f"\nDeduplicazione pazienti:")

    # Conteggio duplicati prima della rimozione
    initial_patients = df['patient_nbr'].nunique()
    total_records = len(df)
    duplicates = total_records - initial_patients

    print(f"- Pazienti unici: {initial_patients:,}")
    print(f"- Record totali: {total_records:,}")
    print(f"- Record duplicati: {duplicates:,}")

    # Rimozione duplicati (mantiene prima occorrenza)
    df_clean = df.drop_duplicates(subset="patient_nbr", keep="first")

    # Rimozione colonna identificativa (non più necessaria)
    df_clean = df_clean.drop(columns=["patient_nbr"])

    print(f"- Pazienti finali: {len(df_clean):,}")
    print(f"- Record rimossi: {total_records - len(df_clean):,}")

    return df_clean

def convert_age_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte i range di età in valori numerici usando la media del range.

    Nel dataset originale, l'età è rappresentata come range (es. "[70-80)").
    Questa funzione converte ogni range nel suo valore medio per permettere
    analisi numeriche.

    Esempi di conversione:
    - "[0-10)" -> 5.0
    - "[70-80)" -> 75.0
    - "[90-100)" -> 95.0

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con colonna 'age' contenente range

    Returns:
    --------
    pd.DataFrame
        Dataset con colonna 'age' numerica
    """
    def age_range_to_mean(age_range: str) -> float:
        """
        Converte un singolo range di età nel suo valore medio.

        Parameters:
        -----------
        age_range : str
            Range di età nel formato "[start-end)"

        Returns:
        --------
        float
            Valore medio del range
        """
        # Rimozione caratteri di formattazione
        clean_range = age_range.strip("[]()")

        # Estrazione valori inizio e fine
        start, end = clean_range.split("-")

        # Calcolo media
        return (int(start) + int(end)) / 2

    print(f"\nConversione range età:")

    # Esempi di range presenti
    unique_ranges = df['age'].unique()
    print(f"- Range di età presenti: {len(unique_ranges)}")
    print(f"- Esempi: {list(unique_ranges)[:5]}")

    # Applicazione conversione
    df_converted = df.copy()
    df_converted["age"] = df["age"].apply(age_range_to_mean)

    # Statistiche della conversione
    print(f"- Età minima: {df_converted['age'].min():.1f} anni")
    print(f"- Età massima: {df_converted['age'].max():.1f} anni")
    print(f"- Età media: {df_converted['age'].mean():.1f} anni")

    return df_converted

def normalize_readmission_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza la variabile target 'readmitted' in formato binario.

    Nel dataset originale, 'readmitted' ha 3 valori:
    - 'NO': Non riammesso
    - '<30': Riammesso entro 30 giorni
    - '>30': Riammesso dopo 30 giorni

    Questa funzione crea una classificazione binaria:
    - 'No': Non riammesso (era 'NO')
    - 'Si': Riammesso (era '<30' o '>30')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con variabile 'readmitted' originale

    Returns:
    --------
    pd.DataFrame
        Dataset con variabile 'readmitted' binaria
    """
    print(f"\nNormalizzazione target riammissione:")

    # Analisi distribuzione originale
    original_counts = df['readmitted'].value_counts()
    print(f"- Distribuzione originale:")
    for value, count in original_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {value}: {count:,} ({percentage:.1f}%)")

    # Conversione binaria
    df_normalized = df.copy()
    readmission_mapping = {
        'NO': 'No',    # Non riammesso
        '<30': 'Si',   # Riammesso entro 30 giorni
        '>30': 'Si'    # Riammesso dopo 30 giorni
    }

    df_normalized['readmitted'] = df['readmitted'].replace(readmission_mapping)

    # Analisi distribuzione finale
    final_counts = df_normalized['readmitted'].value_counts()
    print(f"- Distribuzione finale:")
    for value, count in final_counts.items():
        percentage = (count / len(df_normalized)) * 100
        print(f"  {value}: {count:,} ({percentage:.1f}%)")

    return df_normalized

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Codifica le variabili categoriche usando Label Encoding.

    Identifica automaticamente le colonne categoriche (tipo object) e le converte
    in valori numerici preservando le relazioni ordinali quando possibile.

    Label Encoding è appropriato per questo dataset perché:
    - Molte features sono naturalmente ordinali (es. livelli di medicazione)
    - L'algoritmo finale può gestire encoding numerici
    - Mantiene la dimensionalità bassa rispetto a One-Hot Encoding

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con features categoriche

    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Dataset con features codificate e dizionario dei mapping
    """
    print(f"\nCodifica features categoriche:")

    # Identificazione colonne categoriche
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"- Features categoriche trovate: {len(categorical_cols)}")
    print(f"- Colonne: {list(categorical_cols)}")

    # Inizializzazione strutture per encoders e mappings
    label_encoders = {}
    encoding_mappings = {}
    df_encoded = df.copy()

    # Processo di encoding per ogni colonna categorica
    for col in categorical_cols:
        print(f"\n  Encoding '{col}':")

        # Valori unici prima dell'encoding
        unique_values = df[col].unique()
        print(f"    - Valori unici: {len(unique_values)}")
        print(f"    - Esempi: {list(unique_values)[:3]}")

        # Applicazione Label Encoding
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])

        # Salvataggio encoder per possibili inversioni future
        label_encoders[col] = le

        # Creazione mapping valore_originale -> valore_codificato
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        encoding_mappings[col] = mapping

        print(f"    - Range codificato: {df_encoded[col].min()} - {df_encoded[col].max()}")

    print(f"\nEncoding completato per {len(categorical_cols)} colonne.")

    return df_encoded, encoding_mappings


def save_cleaned_dataset(df: pd.DataFrame, encoding_mappings: Dict,
                        output_dir: str = 'outputs/datasets_clean/first_clean') -> str:
    """
    Salva il dataset pulito e i mapping di codifica.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset pulito e codificato
    encoding_mappings : Dict
        Dizionario con i mapping di codifica
    output_dir : str
        Directory di output

    Returns:
    --------
    str
        Percorso del file dataset salvato
    """
    print(f"\nSalvataggio dataset pulito:")

    # Creazione directory se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Salvataggio dataset principale
    dataset_path = os.path.join(output_dir, 'diabetes_clean.csv')
    df.to_csv(dataset_path, index=False)

    # Salvataggio mapping encodings per riferimento futuro
    mappings_path = os.path.join(output_dir, 'encoding_mappings.json')
    with open(mappings_path, 'w') as f:
        json.dump(encoding_mappings, f, indent=2)

    # Report finale
    print(f"- Dataset salvato: {dataset_path}")
    print(f"- Shape finale: {df.shape[0]:,} righe x {df.shape[1]:,} colonne")
    print(f"- Mapping salvati: {mappings_path}")

    return dataset_path


def print_final_summary(original_shape: Tuple[int, int], final_shape: Tuple[int, int],
                       encoding_mappings: Dict) -> None:
    """
    Stampa un riepilogo finale della pulizia del dataset.

    Parameters:
    -----------
    original_shape : Tuple[int, int]
        Dimensioni dataset originale (righe, colonne)
    final_shape : Tuple[int, int]
        Dimensioni dataset finale (righe, colonne)
    encoding_mappings : Dict
        Mapping delle codifiche applicate
    """
    print("\n" + "="*60)
    print("           RIEPILOGO PULIZIA DATASET")
    print("="*60)

    # Statistiche di riduzione
    rows_removed = original_shape[0] - final_shape[0]
    cols_removed = original_shape[1] - final_shape[1]

    print(f"\nDataset originale:  {original_shape[0]:,} righe x {original_shape[1]:,} colonne")
    print(f"Dataset finale:     {final_shape[0]:,} righe x {final_shape[1]:,} colonne")
    print(f"Righe rimosse:      {rows_removed:,} ({rows_removed/original_shape[0]*100:.1f}%)")
    print(f"Colonne rimosse:    {cols_removed:,} ({cols_removed/original_shape[1]*100:.1f}%)")

    print(f"\nFeatures categoriche codificate: {len(encoding_mappings)}")
    for col, mapping in encoding_mappings.items():
        print(f"- {col}: {len(mapping)} categorie")

    print(f"\nOutput generato:")
    print(f"- diabetes_clean.csv (dataset pulito)")
    print(f"- encoding_mappings.json (mapping codifiche)")
    print("\n" + "="*60)


def main() -> None:
    """
    Funzione principale che esegue l'intera pipeline di pulizia del dataset.

    Workflow:
    1. Caricamento dataset grezzo
    2. Rimozione colonne irrilevanti
    3. Gestione valori mancanti
    4. Deduplicazione pazienti
    5. Conversione range età
    6. Normalizzazione target riammissione
    7. Encoding features categoriche
    8. Salvataggio risultati
    9. Report finale
    """
    print("Avvio pipeline di pulizia dataset diabetici...")

    # 1. Caricamento dataset grezzo
    df = load_raw_dataset("database/diabetic_data.csv")
    original_shape = df.shape

    # 2. Pulizia strutturale
    df = remove_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = deduplicate_patients(df)

    # 3. Trasformazioni features
    df = convert_age_ranges(df)
    df = normalize_readmission_target(df)

    # 4. Encoding features categoriche
    df, encoding_mappings = encode_categorical_features(df)

    # 5. Salvataggio risultati
    output_path = save_cleaned_dataset(df, encoding_mappings)

    # 6. Report finale
    print_final_summary(original_shape, df.shape, encoding_mappings)

    print(f"\nPulizia completata con successo!")
    print(f"Dataset disponibile in: {output_path}")


if __name__ == "__main__":
    main()
