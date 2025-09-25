"""
Modulo per l'applicazione di filtri addizionali al dataset diabetico pulito.

Questo modulo implementa il secondo stadio di pulizia del dataset diabetico,
applicando filtri specifici per rimuovere categorie con bassa rappresentativitàa
o dati di qualità insufficiente. Il processo include:

1. Rimozione di pazienti con race_Other = 1 (categoria troppo generica)
2. Rimozione di pazienti con gender_Unknown/Invalid = 1 (dati di genere non affidabili)
3. Eliminazione delle colonne corrispondenti e della colonna readmitted_>30

Il dataset processato viene salvato nella cartella second_clean per l'utilizzo
nelle fasi successive del pipeline di machine learning.

Author: [Nome del progetto]
Date: 2024
"""

import pandas as pd
import os
from typing import Tuple, Optional
from pathlib import Path


def create_output_directories(base_path: str = 'outputs/datasets_clean') -> None:
    """
    Crea le directory di output necessarie per il salvataggio dei dataset puliti.

    Args:
        base_path (str): Percorso base per le directory di output

    Returns:
        None

    Note:
        Crea le cartelle first_clean e second_clean se non esistono già
    """
    first_clean_path = os.path.join(base_path, 'first_clean')
    second_clean_path = os.path.join(base_path, 'second_clean')

    # Crea le cartelle di output se non esistono
    os.makedirs(first_clean_path, exist_ok=True)
    os.makedirs(second_clean_path, exist_ok=True)

    print(f"Directory create/verificate:")
    print(f"  - {first_clean_path}")
    print(f"  - {second_clean_path}")


def load_first_clean_dataset(input_path: str = 'outputs/datasets_clean/first_clean/diabetes_clean.csv') -> pd.DataFrame:
    """
    Carica il dataset dal primo stadio di pulizia.

    Args:
        input_path (str): Percorso del file CSV del dataset pre-pulito

    Returns:
        pd.DataFrame: Dataset caricato dal primo stadio di pulizia

    Raises:
        FileNotFoundError: Se il file di input non esiste
        pd.errors.EmptyDataError: Se il file è vuoto

    Note:
        Il dataset deve essere stato precedentemente processato dalla fase di pulizia iniziale
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset non trovato: {input_path}")

    try:
        df = pd.read_csv(input_path)
        print(f"Dataset caricato da: {input_path}")
        print(f"Dimensioni iniziali: {df.shape}")
        print(f"Righe: {len(df):,}, Colonne: {len(df.columns)}")
        return df
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Il file {input_path} è vuoto")
    except Exception as e:
        raise Exception(f"Errore durante il caricamento del dataset: {str(e)}")


def apply_race_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Rimuove i pazienti con categoria race_Other = 1.

    La categoria 'race_Other' rappresenta una classificazione troppo generica
    che non fornisce informazioni utili per l'analisi medica. La rimozione
    di questi record migliora la qualità del dataset.

    Args:
        df (pd.DataFrame): Dataset di input

    Returns:
        Tuple[pd.DataFrame, int]: Dataset filtrato e numero di righe rimosse

    Note:
        Mantiene solo i record dove race_Other != 1
    """
    initial_rows = len(df)

    # Verifica che la colonna esista
    if 'race_Other' not in df.columns:
        print("ATTENZIONE: Colonna 'race_Other' non presente nel dataset")
        return df, 0

    # Applica il filtro: mantieni righe dove race_Other != 1
    filtered_df = df[df['race_Other'] != 1].copy()

    rows_removed = initial_rows - len(filtered_df)
    print(f"Filtro race_Other applicato:")
    print(f"  - Righe rimosse: {rows_removed:,}")
    print(f"  - Righe rimanenti: {len(filtered_df):,}")

    return filtered_df, rows_removed


def apply_gender_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Rimuove i pazienti con categoria gender_Unknown/Invalid = 1.

    I dati di genere con categoria 'Unknown/Invalid' indicano informazioni
    demografiche inaffidabili o mancanti, che possono introdurre bias
    nell'analisi. La loro rimozione migliora la qualità del dataset.

    Args:
        df (pd.DataFrame): Dataset di input

    Returns:
        Tuple[pd.DataFrame, int]: Dataset filtrato e numero di righe rimosse

    Note:
        Mantiene solo i record dove gender_Unknown/Invalid != 1
    """
    initial_rows = len(df)

    # Verifica che la colonna esista
    if 'gender_Unknown/Invalid' not in df.columns:
        print("ATTENZIONE: Colonna 'gender_Unknown/Invalid' non presente nel dataset")
        return df, 0

    # Applica il filtro: mantieni righe dove gender_Unknown/Invalid != 1
    filtered_df = df[df['gender_Unknown/Invalid'] != 1].copy()

    rows_removed = initial_rows - len(filtered_df)
    print(f"Filtro gender_Unknown/Invalid applicato:")
    print(f"  - Righe rimosse: {rows_removed:,}")
    print(f"  - Righe rimanenti: {len(filtered_df):,}")

    return filtered_df, rows_removed


def remove_unnecessary_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Rimuove le colonne non più necessarie dopo l'applicazione dei filtri.

    Elimina le colonne utilizzate per il filtraggio e altre colonne non utili
    per l'analisi successiva:
    - race_Other: utilizzata per filtraggio, ora non più necessaria
    - gender_Unknown/Invalid: utilizzata per filtraggio, ora non più necessaria
    - readmitted_>30: specifica categoria di riammissione che verrà accorpata

    Args:
        df (pd.DataFrame): Dataset di input

    Returns:
        Tuple[pd.DataFrame, list]: Dataset senza le colonne specificate e lista colonne rimosse

    Note:
        Le colonne vengono rimosse solo se presenti nel dataset
    """
    columns_to_remove = ['race_Other', 'gender_Unknown/Invalid', 'readmitted_>30']

    # Identifica quali colonne sono effettivamente presenti
    columns_to_remove_present = [col for col in columns_to_remove if col in df.columns]
    columns_missing = [col for col in columns_to_remove if col not in df.columns]

    if columns_missing:
        print(f"ATTENZIONE: Colonne non trovate nel dataset: {columns_missing}")

    if not columns_to_remove_present:
        print("Nessuna colonna da rimuovere trovata nel dataset")
        return df, []

    # Rimuovi le colonne presenti
    cleaned_df = df.drop(columns=columns_to_remove_present)

    print(f"Colonne rimosse: {columns_to_remove_present}")
    print(f"Colonne rimanenti: {len(cleaned_df.columns)}")
    print(f"  (era: {len(df.columns)}, ora: {len(cleaned_df.columns)})")

    return cleaned_df, columns_to_remove_present


def save_filtered_dataset(df: pd.DataFrame, output_path: str = 'outputs/datasets_clean/second_clean/diabetes_clean_filtered.csv') -> str:
    """
    Salva il dataset filtrato nel percorso di output specificato.

    Args:
        df (pd.DataFrame): Dataset filtrato da salvare
        output_path (str): Percorso di output per il file CSV

    Returns:
        str: Percorso del file salvato

    Raises:
        Exception: Se si verifica un errore durante il salvataggio

    Note:
        Crea la directory di output se non esiste
    """
    try:
        # Crea la directory se non esiste
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Salva il dataset
        df.to_csv(output_path, index=False)

        print(f"Dataset filtrato salvato in: {output_path}")
        print(f"Dimensioni finali: {df.shape}")

        return output_path

    except Exception as e:
        raise Exception(f"Errore durante il salvataggio del dataset: {str(e)}")


def print_filtering_summary(initial_rows: int, final_rows: int, race_removed: int, gender_removed: int, columns_removed: list) -> None:
    """
    Stampa un riepilogo dettagliato dell'operazione di filtraggio.

    Args:
        initial_rows (int): Numero di righe iniziali
        final_rows (int): Numero di righe finali
        race_removed (int): Numero di righe rimosse dal filtro race
        gender_removed (int): Numero di righe rimosse dal filtro gender
        columns_removed (list): Lista delle colonne rimosse

    Returns:
        None
    """
    total_removed = initial_rows - final_rows
    retention_rate = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0

    print("\n" + "="*60)
    print("RIEPILOGO OPERAZIONI DI FILTRAGGIO")
    print("="*60)
    print(f"Righe iniziali:                {initial_rows:,}")
    print(f"Righe rimosse filtro race:     {race_removed:,}")
    print(f"Righe rimosse filtro gender:   {gender_removed:,}")
    print(f"Totale righe rimosse:          {total_removed:,}")
    print(f"Righe finali:                  {final_rows:,}")
    print(f"Tasso di ritenzione:           {retention_rate:.1f}%")
    print(f"Colonne rimosse:               {len(columns_removed)}")
    for col in columns_removed:
        print(f"  - {col}")
    print("="*60)


def apply_additional_filters(input_path: Optional[str] = None, output_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Applica tutti i filtri addizionali al dataset in sequenza.

    Questa è la funzione principale che orchestra tutte le operazioni di filtraggio:
    1. Carica il dataset pre-pulito
    2. Applica il filtro per race_Other
    3. Applica il filtro per gender_Unknown/Invalid
    4. Rimuove le colonne non necessarie
    5. Salva il dataset finale

    Args:
        input_path (Optional[str]): Percorso del dataset di input. Se None, usa il default
        output_path (Optional[str]): Percorso del dataset di output. Se None, usa il default

    Returns:
        Tuple[pd.DataFrame, str]: Dataset processato e percorso del file salvato

    Raises:
        Exception: Se si verifica un errore durante il processo di filtraggio
    """
    try:
        # Definisci percorsi default se non specificati
        if input_path is None:
            input_path = 'outputs/datasets_clean/first_clean/diabetes_clean.csv'
        if output_path is None:
            output_path = 'outputs/datasets_clean/second_clean/diabetes_clean_filtered.csv'

        print("AVVIO PROCESSO DI FILTRAGGIO ADDIZIONALE")
        print("="*50)

        # Step 1: Carica il dataset
        df = load_first_clean_dataset(input_path)
        initial_rows = len(df)

        # Step 2: Applica filtro race
        df, race_removed = apply_race_filter(df)

        # Step 3: Applica filtro gender
        df, gender_removed = apply_gender_filter(df)

        # Step 4: Rimuovi colonne non necessarie
        df, columns_removed = remove_unnecessary_columns(df)

        # Step 5: Salva dataset filtrato
        saved_path = save_filtered_dataset(df, output_path)

        # Step 6: Stampa riepilogo
        print_filtering_summary(initial_rows, len(df), race_removed, gender_removed, columns_removed)

        return df, saved_path

    except Exception as e:
        print(f"ERRORE durante il processo di filtraggio: {str(e)}")
        raise


def main() -> None:
    """
    Funzione principale che esegue il processo completo di filtraggio addizionale.

    Coordina l'esecuzione di tutti i passaggi necessari:
    1. Creazione delle directory di output
    2. Applicazione di tutti i filtri addizionali
    3. Salvataggio del dataset finale

    Returns:
        None

    Note:
        Questa funzione può essere chiamata direttamente o tramite esecuzione dello script
    """
    print("MODULO: Rimozione Ulteriori Filtri - Dataset Diabetico")
    print("="*60)
    print("Applicazione di filtri addizionali per migliorare la qualità dei dati")
    print("="*60)

    try:
        # Step 1: Crea le directory necessarie
        create_output_directories()

        # Step 2: Applica tutti i filtri
        processed_df, output_file = apply_additional_filters()

        # Messaggio finale di successo
        print("\nPROCESSO COMPLETATO CON SUCCESSO!")
        print(f"Dataset filtrato disponibile in: {output_file}")
        print(f"Record finali: {len(processed_df):,}")
        print(f"Features finali: {len(processed_df.columns)}")

    except Exception as e:
        print(f"\nERRORE CRITICO: {str(e)}")
        print("Il processo non è stato completato correttamente.")
        raise


if __name__ == "__main__":
    main()