import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def optimize_for_regression():
    """
    Preprocessing ottimizzato per REGRESSIONE basato sull'approccio dell'utente
    ma con miglioramenti specifici per modelli predittivi.
    """
    
    print("=== PREPROCESSING OTTIMIZZATO PER REGRESSIONE ===\n")
    
    # 1. Carica dataset
    df = pd.read_csv("database/diabetic_data.csv")
    print(f"Dataset iniziale: {df.shape}")

    # 2. STEP 1: Definisci TARGET VARIABLE (cruciale per regressione!)
    print(f"\n=== SELEZIONE TARGET VARIABLE ===")
    
    # Opzioni target per regressione
    target_options = {
        'time_in_hospital': 'Predire durata ricovero (giorni)',
        'num_medications': 'Predire numero farmaci necessari',
        'readmitted_binary': 'Predire probabilità riammissione (0/1)'
    }
    
    print("Opzioni target disponibili:")
    for target, desc in target_options.items():
        print(f"  - {target}: {desc}")
    
    # Scegli target (modifica qui per cambiare obiettivo)
    TARGET = 'readmitted_binary'  # <-- CAMBIA QUI per target diverso
    print(f"\nTARGET SCELTO: {TARGET}")
    print(f"Descrizione: {target_options[TARGET]}")

    # 3. Elimina colonne NON predittive (mantieni quelle cliniche utili)
    print(f"\n=== RIMOZIONE COLONNE NON PREDITTIVE ===")
    
    cols_to_drop = [
        "encounter_id",           # ID non predittivo
        "weight",                 # Rimossa come richiesto
        "admission_type_id",      # Admin info
        "discharge_disposition_id", # Admin info
        "number_diagnoses",       # Rimossa come richiesto
        "payer_code",             # Info assicurativa
        "medical_specialty",      # Rimossa come richiesto
        "diag_1",                 # Rimossa come richiesto
        "diag_2",                 # Rimossa come richiesto
        "diag_3",                 # Rimossa come richiesto
        "max_glu_serum",          # Rimossa come richiesto
        "A1Cresult"               # Rimossa come richiesto
    ]
    
    # NON eliminare subito target se è tra le colonne da droppare
    if TARGET in cols_to_drop:
        cols_to_drop.remove(TARGET)
        print(f"ATTENZIONE: {TARGET} era in cols_to_drop, rimosso dalla lista")
    
    # Elimina solo le colonne che esistono realmente
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    missing_cols = [col for col in cols_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"Colonne non trovate nel dataset: {missing_cols}")
    
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True)
        print(f"Colonne rimosse: {existing_cols_to_drop}")
    
    print(f"Dataset dopo rimozione: {df.shape}")

    # 4. Gestione MISSING VALUES
    print(f"\n=== GESTIONE MISSING VALUES ===")
    
    df.replace("?", pd.NA, inplace=True)
    
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    print(f"Missing values rimanenti dopo rimozione colonne:")
    
    # NO IMPUTAZIONE - Elimina righe con missing per mantenere dati autentici
    print(f"Missing rimanenti per colonna:")
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)
    print(remaining_missing)
    
    if remaining_missing.sum() > 0:
        print(f"\nRighe con missing da eliminare: {df.isnull().any(axis=1).sum()}")
        print("NESSUNA IMPUTAZIONE - Manteniamo solo dati autentici")
        df = df.dropna()
        print(f"Dataset dopo eliminazione missing: {df.shape}")
    else:
        print("\nNessun missing value rimanente")

    # 5. Gestione duplicati pazienti (CRUCIALE per regressione!)
    print(f"\n=== GESTIONE DUPLICATI PAZIENTI ===")
    if 'patient_nbr' in df.columns:
        print("Analisi duplicati prima della rimozione:")
        total_records = df.shape[0]
        unique_patients = df['patient_nbr'].nunique()
        duplicate_records = total_records - unique_patients
        
        print(f"  - Ricoveri totali: {total_records}")
        print(f"  - Pazienti unici: {unique_patients}")
        print(f"  - Ricoveri duplicati da rimuovere: {duplicate_records}")
        
        if duplicate_records > 0:
            duplicate_patients = df[df['patient_nbr'].duplicated(keep=False)]['patient_nbr'].nunique()
            print(f"  - Pazienti con ricoveri multipli: {duplicate_patients}")
            
            # Mostra esempio duplicati
            sample_duplicates = df[df['patient_nbr'].duplicated(keep=False)].groupby('patient_nbr').first().head(3)
            print(f"  - Esempi pazienti con duplicati: {list(sample_duplicates.index)}")
        
        # Rimuovi duplicati (mantieni primo ricovero per paziente)
        df = df.drop_duplicates(subset="patient_nbr", keep="first")
        
        print(f"\nDopo rimozione duplicati:")
        print(f"  - Dataset finale: {df.shape}")
        print(f"  - Verifica: pazienti unici = {df['patient_nbr'].nunique()}")
        
        # Controllo finale
        if len(df) == df['patient_nbr'].nunique():
            print("  SUCCESSO: Ogni paziente appare una sola volta")
        else:
            print("  ERRORE: Duplicati rimanenti!")
    else:
        print("ATTENZIONE: patient_nbr non trovato - impossibile rimuovere duplicati!")
    
    # 6. Conversione età
    print(f"\n=== CONVERSIONE ETÀ ===")
    def age_to_mean(age_range):
        if pd.isna(age_range):
            return np.nan
        age_range = str(age_range).strip("[]()")
        if "-" not in age_range:
            return float(age_range)
        start, end = age_range.split("-")
        return (int(start) + int(end)) / 2

    if 'age' in df.columns:
        df["age"] = df["age"].apply(age_to_mean)
        print(f"Età convertita: range {df['age'].min()}-{df['age'].max()}")
    
    # 6.1 Crea target readmitted_binary se necessario
    if TARGET == 'readmitted_binary' and 'readmitted' in df.columns:
        df['readmitted_binary'] = df['readmitted'].map({'NO': 0, '<30': 1, '>30': 1})
        print(f"Target readmitted_binary creato: NO=0, riammissione=1")
        print(f"Distribuzione: {df['readmitted_binary'].value_counts().to_dict()}")

    # 6.5 SALVA DATASET PULITO (prima dell'encoding) per encoding manuale
    print(f"\n=== SALVATAGGIO DATASET PULITO (pre-encoding) ===")
    clean_output = "outputs/cleaned_data/diabetes_clean_no_encoding.csv"
    df.to_csv(clean_output, index=False)
    print(f"Dataset pulito salvato: {clean_output}")
    print(f"  - {df.shape[0]} pazienti")
    print(f"  - {df.shape[1]} variabili")
    print(f"  - NO duplicati, NO missing, NO imputazione")
    print(f"  - Categorie testuali per encoding manuale")
    
    # 7. ENCODING OTTIMIZZATO PER REGRESSIONE
    print(f"\n=== ENCODING PER REGRESSIONE ===")
    
    # 7.1 Variabili ordinali cliniche (mantieni ordine)
    ordinal_mappings = {
        'change': {'No': 0, 'Ch': 1},
        'diabetesMed': {'No': 0, 'Yes': 1},
        'readmitted': {'NO': 0, '<30': 1, '>30': 2}  # Ordine temporale
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns and col != TARGET:  # Non encodare il target se è categorico
            df[col] = df[col].map(mapping)
            print(f"  {col}: encoded ordinalmente -> {mapping}")
    
    # 7.2 Gestione categoriche rimanenti (se presenti)
    categorical_remaining = df.select_dtypes(include=['object']).columns
    categorical_for_encoding = [col for col in categorical_remaining if col != TARGET]
    
    if len(categorical_for_encoding) > 0:
        print(f"  Variabili categoriche rimanenti: {list(categorical_for_encoding)}")
        
        # Per le poche variabili rimanenti, usa encoding semplice
        for col in categorical_for_encoding:
            if df[col].nunique() > 10:  # Alta cardinalità
                target_means = df.groupby(col)[TARGET].mean()
                df[col + '_encoded'] = df[col].map(target_means)
                df.drop(columns=[col], inplace=True)
                print(f"    {col} -> {col}_encoded (target encoding)")
            # Per bassa cardinalità, mantieni per one-hot successivo
    else:
        print("  Nessuna variabile categorica rimanente da encodare")
    
    # 7.3 Resto categoriche: One-hot encoding
    categorical_final = df.select_dtypes(include=['object']).columns
    categorical_for_dummies = [col for col in categorical_final if col != TARGET]
    
    if len(categorical_for_dummies) > 0:
        print(f"  Colonne per one-hot encoding: {categorical_for_dummies}")
        df = pd.get_dummies(df, columns=categorical_for_dummies, drop_first=True, dtype=int)
        print(f"  One-hot encoding applicato, nuove dimensioni: {df.shape}")

    # 8. GESTIONE TARGET VARIABLE
    print(f"\n=== PREPARAZIONE TARGET VARIABLE ===")
    
    if TARGET not in df.columns:
        print(f"ERRORE: Target '{TARGET}' non trovato nel dataset!")
        print(f"Colonne disponibili: {list(df.columns)}")
        return None
    
    # Se target è categorico, convertilo
    if df[TARGET].dtype == 'object':
        le = LabelEncoder()
        df[TARGET] = le.fit_transform(df[TARGET].astype(str))
        print(f"Target categorico convertito a numerico: {le.classes_}")
    
    print(f"Target statistics:")
    print(f"  Tipo: {df[TARGET].dtype}")
    print(f"  Range: {df[TARGET].min()} - {df[TARGET].max()}")
    print(f"  Media: {df[TARGET].mean():.2f}")
    print(f"  Missing: {df[TARGET].isnull().sum()}")

    # 9. OUTLIERS (importante per regressione)
    print(f"\n=== GESTIONE OUTLIERS ===")
    
    # Rimuovi outlier estremi solo dalle feature numeriche (non target)
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != TARGET]
    
    outliers_removed = 0
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR  # 3 IQR per essere conservativi
        upper = Q3 + 3 * IQR
        
        before_size = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        outliers_col = before_size - len(df)
        outliers_removed += outliers_col
        
        if outliers_col > 0:
            print(f"  {col}: rimossi {outliers_col} outlier")
    
    print(f"Outlier totali rimossi: {outliers_removed}")
    print(f"Dataset finale: {df.shape}")

    # 10. FEATURE SELECTION per regressione
    print(f"\n=== FEATURE SELECTION ===")
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # Rimuovi patient_nbr se presente (non predittivo)
    if 'patient_nbr' in X.columns:
        X = X.drop(columns=['patient_nbr'])
        print("patient_nbr rimosso dalle features")
    
    print(f"Features prima della selezione: {X.shape[1]}")
    
    # Seleziona top K features più correlate al target
    k_features = min(20, X.shape[1])  # Max 20 features
    selector = SelectKBest(score_func=f_regression, k=k_features)
    X_selected = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    print(f"Features selezionate: {k_features}")
    print("Top 10 features per score:")
    for i, (feat, score) in enumerate(zip(selected_features, feature_scores)):
        if i < 10:
            print(f"  {feat}: {score:.2f}")

    # 11. TRAIN/TEST SPLIT
    print(f"\n=== TRAIN/TEST SPLIT ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 12. SCALING (IMPORTANTE per regressione!)
    print(f"\n=== FEATURE SCALING ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("StandardScaler applicato a train e test set")

    # 13. SALVATAGGIO FINALE
    print(f"\n=== SALVATAGGIO DATASET PRONTO PER REGRESSIONE ===")
    
    # Salva dataset processato (per analisi)
    final_df = pd.DataFrame(X_selected, columns=selected_features)
    final_df[TARGET] = y.values
    
    output_file = "outputs/cleaned_data/diabetes_regression_ready.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Dataset finale salvato: {output_file}")
    print(f"  - {final_df.shape[0]} pazienti")
    print(f"  - {final_df.shape[1]-1} features + 1 target")
    print(f"  - Target: {TARGET}")
    
    # Salva anche arrays per ML (opzionale)
    np.savez(
        'outputs/cleaned_data/diabetes_ml_arrays.npz',
        X_train=X_train_scaled,
        X_test=X_test_scaled, 
        y_train=y_train.values,
        y_test=y_test.values,
        feature_names=selected_features,
        target_name=TARGET
    )
    print("Arrays ML salvati in: outputs/cleaned_data/diabetes_ml_arrays.npz")
    
    print(f"\n=== SUMMARY FINALE ===")
    print(f"Target variable: {TARGET}")
    print(f"Tipo regressione: {'Lineare' if y.nunique() > 10 else 'Logistica'}")
    print(f"Features finali: {len(selected_features)}")
    print(f"Campioni training: {X_train.shape[0]}")
    print(f"Campioni test: {X_test.shape[0]}")
    print(f"\nPRONTO PER:")
    print("- Linear Regression")
    print("- Ridge/Lasso Regression") 
    print("- Random Forest")
    print("- Gradient Boosting")
    
    return final_df, X_train_scaled, X_test_scaled, y_train, y_test, selected_features

if __name__ == "__main__":
    # Esegui preprocessing ottimizzato
    try:
        result = optimize_for_regression()
        if result is not None:
            df_final, X_train, X_test, y_train, y_test, features = result
            print(f"\n*** PREPROCESSING COMPLETATO CON SUCCESSO ***")
            print(f"*** Usa 'diabetes_regression_ready.csv' per analisi ***")
            print(f"*** Usa 'diabetes_ml_arrays.npz' per modelli ML ***")
        else:
            print("ERRORE nel preprocessing!")
    except Exception as e:
        print(f"ERRORE: {e}")
        import traceback
        traceback.print_exc()