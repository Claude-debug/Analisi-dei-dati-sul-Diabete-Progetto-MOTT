#!/usr/bin/env python3
"""
Sistema Avanzato di Clustering per Dataset Diabetico con Predizione Riammissione.

Questo modulo implementa un sistema sofisticato di clustering multi-metodologico
per la classificazione e predizione del rischio di riammissione ospedaliera
di pazienti diabetici. Il sistema confronta 4 approcci diversi:

1. **K-means Gerarchico**: Clustering basato su macro-gruppi di età + features significative
2. **Decision Tree Funzionale**: Clustering tramite regole del decision tree per ogni fascia d'età
3. **Ibrido Età + Risk**: Combinazione di segmentazione per età e scoring del rischio
4. **Age-Based Fisso**: 4 fasce d'età fisse (metodo di baseline sempre utilizzato)

Architettura del Sistema:
- Analisi delle features significative per ogni fascia d'età
- Comparazione competitiva tra i primi 3 metodi
- Valutazione multi-metrica (silhouette, omogeneità, separazione, utilità predittiva)
- Selezione automatica del metodo migliore
- Output multipli per utilizzo nel sistema integrato finale

Processo Pipeline:
1. Caricamento e pulizia dataset diabetico grezzo
2. Rimozione duplicati per paziente (mantenendo primo encounter)
3. Gestione valori mancanti con strategie specifiche per tipo dato
4. Analisi correlazioni features per fascia d'età
5. Applicazione e confronto dei 4 metodi di clustering
6. Valutazione performance con metriche multiple
7. Selezione metodo vincente e salvataggio dataset processati

Output:
- Dataset clustered per ogni metodo in outputs/datasets_clean/cluster/terzo_metodo/
- Confronto performance dettagliato
- Cluster analisi per utilizzo downstream

Author: [Nome del progetto]
Date: 2024
Version: 3.0 - Sistema Ibrido Integrato
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data() -> pd.DataFrame:
    """
    Carica il dataset diabetico originale dal database e verifica la struttura.

    Legge il dataset diabetico da database/diabetic_data.csv utilizzando il
    separator corretto (';') e fornisce statistiche di base sui dati caricati.

    Returns:
        pd.DataFrame: Dataset diabetico grezzo con tutte le features originali

    Note:
        - Il dataset contiene encounter multipli per paziente
        - Sono presenti valori mancanti codificati come '?'
        - Il separator del CSV è ';' non la virgola standard
    """
    print("Caricamento dataset...")

    # Carica il dataset con separator corretto
    df = pd.read_csv('database/diabetic_data.csv', sep=';')
    print(f"Dataset originale: {df.shape}")
    print(f"Pazienti unici: {df['patient_nbr'].nunique()}")

    return df

def remove_duplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove i record duplicati per paziente, mantenendo solo il primo encounter.

    Poiché il dataset contiene encounter multipli per lo stesso paziente,
    questa funzione mantiene solo il primo encounter ordinato per patient_nbr
    e encounter_id per garantire consistenza e evitare data leakage.

    Args:
        df (pd.DataFrame): Dataset con potenziali duplicati per paziente

    Returns:
        pd.DataFrame: Dataset con un record unico per paziente

    Note:
        - Ordina per patient_nbr e encounter_id prima della deduplica
        - Mantiene il primo encounter cronologico per paziente
        - Rimuove bias da pazienti con encounter frequenti
    """
    print("\nRimozione duplicati per paziente...")

    # Ordina per patient_nbr e encounter_id per mantenere il primo encounter
    df_sorted = df.sort_values(['patient_nbr', 'encounter_id'])

    # Mantieni solo il primo record per ogni paziente
    df_unique = df_sorted.drop_duplicates(subset=['patient_nbr'], keep='first')

    print(f"Record rimossi: {len(df) - len(df_unique)}")
    print(f"Dataset dopo rimozione duplicati: {df_unique.shape}")

    return df_unique

def replace_question_marks_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sostituisce tutti i valori '?' con NaN per standardizzare i valori mancanti.

    Il dataset originale usa '?' per indicare valori mancanti, che deve essere
    convertito in NaN per consentire il corretto handling da parte di pandas
    e scikit-learn.

    Args:
        df (pd.DataFrame): Dataset con valori '?' da convertire

    Returns:
        pd.DataFrame: Dataset con '?' sostituiti da NaN

    Note:
        - Fornisce statistiche dettagliate sui valori mancanti per colonna
        - Calcola percentuali di missing values per prioritizzare la pulizia
    """
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

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gestisce i valori mancanti con strategie specifiche per tipo di dato.

    Implementa una strategia a due livelli:
    1. Rimuove colonne con >80% valori mancanti (troppo sparse)
    2. Imputa valori mancanti rimasti con mediana (numeriche) o moda (categoriche)

    Args:
        df (pd.DataFrame): Dataset con valori mancanti da gestire

    Returns:
        pd.DataFrame: Dataset pulito senza valori mancanti

    Note:
        - Soglia di rimozione colonne: 80% missing values
        - Imputazione numerica: mediana (robusta agli outliers)
        - Imputazione categorica: moda (valore più frequente)
    """
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

def age_mapping(df: pd.DataFrame) -> dict:
    """
    Analizza le features più significative per ogni fascia d'età tramite correlazione.

    Per ogni fascia d'età nel dataset, calcola le correlazioni tra features numeriche
    e la variabile target readmitted_binary, identificando le 5 features più
    correlate (|correlation| > 0.05) per personalizzare i modelli di clustering.

    Args:
        df (pd.DataFrame): Dataset con fasce d'età e features numeriche

    Returns:
        dict: Dizionario con analisi per fascia d'età contenente:
            - count: numero pazienti nella fascia
            - readmit_rate: tasso riammissione
            - significant_features: top 5 features correlate
            - feature_correlations: correlazioni complete

    Note:
        - Minimum 50 pazienti per fascia per analisi affidabile
        - Soglia correlazione minima: 0.05
        - Features analizzate: cliniche e administrative
    """
    print("\nAnalisi features significative per fascia d'età...")

    # Mantieni le fasce d'età originali
    age_groups = df['age'].unique()
    print(f"Fasce d'età trovate: {len(age_groups)}")

    # Prepara features numeriche per l'analisi
    numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                       'number_diagnoses', 'num_procedures', 'number_outpatient',
                       'number_emergency', 'number_inpatient']

    # Filtra features esistenti
    available_features = [f for f in numeric_features if f in df.columns]
    print(f"Features numeriche disponibili: {len(available_features)}")

    age_group_analysis = {}

    for age_group in age_groups:
        if pd.isna(age_group):
            continue

        age_data = df[df['age'] == age_group].copy()
        if len(age_data) < 50:  # Skip gruppi troppo piccoli
            continue

        print(f"\nAnalisi fascia {age_group}: {len(age_data)} pazienti")

        # Calcola correlazioni con riammissione
        correlations = {}
        significant_features = []

        if 'readmitted_binary' in age_data.columns:
            for feature in available_features:
                if age_data[feature].notna().sum() > 10:  # Minimo dati per correlazione
                    corr = age_data[feature].corr(age_data['readmitted_binary'])
                    correlations[feature] = abs(corr) if not pd.isna(corr) else 0

            # Seleziona top 5 features più correlate
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:5] if f[1] > 0.05]  # Soglia minima
            significant_features = top_features

        age_group_analysis[age_group] = {
            'count': len(age_data),
            'readmit_rate': age_data['readmitted_binary'].mean() if 'readmitted_binary' in age_data.columns else 0,
            'significant_features': significant_features,
            'feature_correlations': correlations
        }

        print(f"  Tasso riammissione: {age_group_analysis[age_group]['readmit_rate']:.1%}")
        print(f"  Features significative: {significant_features}")

    return age_group_analysis

def create_hierarchical_age_clusters(df: pd.DataFrame, age_analysis: dict) -> tuple[pd.DataFrame, list]:
    """
    Crea cluster gerarchici: prima per età, poi per features significative tramite K-means.

    Implementa un approccio gerarchico a due livelli:
    1. Raggruppa fasce d'età in 4 macro-gruppi (giovani, adulti, anziani, molto_anziani)
    2. All'interno di ogni macro-gruppo, applica K-means sulle features significative

    Il numero di sub-cluster per macro-gruppo è adattivo (2-3) basato sulla dimensione
    del gruppo e disponibilità di features significative.

    Args:
        df (pd.DataFrame): Dataset con informazioni di età e features
        age_analysis (dict): Analisi features significative per fascia d'età

    Returns:
        tuple[pd.DataFrame, list]:
            - Dataset con colonna 'final_cluster' assegnata
            - Lista info cluster con dettagli metodo e features usate

    Note:
        - Minimum 100 pazienti per macro-gruppo per sub-clustering
        - Minimum 2 features significative per K-means
        - StandardScaler applicato prima del clustering
    """
    print("\nCreazione cluster gerarchici eta -> features...")

    # Step 1: Raggruppa fasce d'età simili in macro-gruppi
    age_macro_groups = {
        'giovani': ['[0-10)', '[10-20)', '[20-30)', '[30-40)'],
        'adulti': ['[40-50)', '[50-60)'],
        'anziani': ['[60-70)', '[70-80)'],
        'molto_anziani': ['[80-90)', '[90-100)']
    }

    # Mappa ogni paziente al suo macro-gruppo
    def get_age_macro_group(age):
        for group_name, age_ranges in age_macro_groups.items():
            if age in age_ranges:
                return group_name
        return 'unknown'

    df['age_macro_group'] = df['age'].apply(get_age_macro_group)

    # Step 2: Per ogni macro-gruppo, crea sub-cluster basati sulle features significative
    final_clusters = []
    cluster_id = 0

    # Inizializza la colonna final_cluster
    df['final_cluster'] = np.nan

    for group_name, age_ranges in age_macro_groups.items():
        group_data = df[df['age_macro_group'] == group_name].copy()

        if len(group_data) < 100:  # Skip gruppi troppo piccoli
            df.loc[group_data.index, 'final_cluster'] = cluster_id
            final_clusters.append({
                'cluster_id': cluster_id,
                'age_group': group_name,
                'sub_method': 'single',
                'count': len(group_data),
                'features_used': []
            })
            cluster_id += 1
            continue

        print(f"\nProcessing {group_name}: {len(group_data)} pazienti")

        # Raccogli features significative per questo macro-gruppo
        all_significant_features = set()
        for age_range in age_ranges:
            if age_range in age_analysis:
                all_significant_features.update(age_analysis[age_range]['significant_features'])

        significant_features_list = list(all_significant_features)
        available_significant = [f for f in significant_features_list if f in group_data.columns]

        print(f"  Features significative per {group_name}: {available_significant}")

        if len(available_significant) >= 2 and len(group_data) >= 200:
            # Crea 2-3 sub-cluster basati sulle features significative
            scaler = StandardScaler()
            features_data = group_data[available_significant].fillna(group_data[available_significant].median())
            features_scaled = scaler.fit_transform(features_data)

            n_subclusters = min(3, max(2, len(group_data) // 100))  # 2-3 cluster
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subclusters = kmeans.fit_predict(features_scaled)

            for subcluster_id in range(n_subclusters):
                mask = subclusters == subcluster_id
                df.loc[group_data.index[mask], 'final_cluster'] = cluster_id

                final_clusters.append({
                    'cluster_id': cluster_id,
                    'age_group': group_name,
                    'sub_method': 'features_kmeans',
                    'count': mask.sum(),
                    'features_used': available_significant
                })

                cluster_id += 1

        else:
            # Un singolo cluster per tutto il macro-gruppo
            df.loc[group_data.index, 'final_cluster'] = cluster_id
            final_clusters.append({
                'cluster_id': cluster_id,
                'age_group': group_name,
                'sub_method': 'single',
                'count': len(group_data),
                'features_used': available_significant
            })
            cluster_id += 1

    # Step 3: Assegna cluster di default per dati mancanti
    df['final_cluster'].fillna(cluster_id, inplace=True)

    print(f"\nCreati {cluster_id + 1} cluster finali:")
    for cluster_info in final_clusters:
        print(f"  Cluster {cluster_info['cluster_id']}: {cluster_info['age_group']} - "
              f"{cluster_info['count']} pazienti - {cluster_info['sub_method']} - "
              f"features: {cluster_info['features_used'][:3]}...")

    return df, final_clusters

def create_decision_tree_clusters(df: pd.DataFrame, age_analysis: dict) -> tuple[pd.DataFrame, list]:
    """
    Crea cluster funzionali usando Decision Tree per ogni macro-gruppo di età.

    Utilizza DecisionTreeClassifier per identificare split ottimali sulle features
    significative, creando cluster basati sulle foglie dell'albero. Ogni foglia
    diventa un cluster con regole interpretabili per la predizione del rischio.

    Args:
        df (pd.DataFrame): Dataset con informazioni di età e features
        age_analysis (dict): Analisi features significative per fascia d'età

    Returns:
        tuple[pd.DataFrame, list]:
            - Dataset con colonna 'dt_cluster' assegnata
            - Lista info cluster con regole del decision tree estratte

    Note:
        - Decision Tree parametri: max_depth=3, min_samples_split/leaf adattivi
        - Class_weight='balanced' per gestire classi sbilanciate
        - Estrazione automatica regole da radice a foglia
    """
    print("\nCreazione cluster funzionali con Decision Tree...")

    # Step 1: Crea macro-gruppi di età
    age_macro_groups = {
        'giovani': ['[0-10)', '[10-20)', '[20-30)', '[30-40)'],
        'adulti': ['[40-50)', '[50-60)'],
        'anziani': ['[60-70)', '[70-80)'],
        'molto_anziani': ['[80-90)', '[90-100)']
    }

    def get_age_macro_group(age):
        for group_name, age_ranges in age_macro_groups.items():
            if age in age_ranges:
                return group_name
        return 'unknown'

    df['age_macro_group_dt'] = df['age'].apply(get_age_macro_group)

    # Step 2: Inizializza colonne per decision tree clusters
    df['dt_cluster'] = np.nan
    dt_cluster_info = []
    cluster_id = 0

    for group_name, age_ranges in age_macro_groups.items():
        group_data = df[df['age_macro_group_dt'] == group_name].copy()

        if len(group_data) < 100:
            # Gruppo troppo piccolo, assegna cluster singolo
            df.loc[group_data.index, 'dt_cluster'] = cluster_id
            dt_cluster_info.append({
                'cluster_id': cluster_id,
                'age_group': group_name,
                'method': 'single',
                'count': len(group_data),
                'features_used': [],
                'tree_rules': 'Too small group'
            })
            cluster_id += 1
            continue

        print(f"\nProcessing {group_name}: {len(group_data)} pazienti")

        # Step 3: Raccogli features significative per questo macro-gruppo
        all_significant_features = set()
        for age_range in age_ranges:
            if age_range in age_analysis:
                all_significant_features.update(age_analysis[age_range]['significant_features'])

        significant_features_list = list(all_significant_features)
        available_significant = [f for f in significant_features_list if f in group_data.columns]

        print(f"  Features significative per {group_name}: {available_significant}")

        if len(available_significant) >= 2 and len(group_data) >= 200:
            # Step 4: Usa Decision Tree per trovare split ottimali
            features_data = group_data[available_significant].fillna(group_data[available_significant].median())

            # Decision Tree con parametri ottimizzati per clustering funzionale
            dt = DecisionTreeClassifier(
                max_depth=3,  # Non troppo profondo
                min_samples_split=max(50, len(group_data) // 20),  # Min 50 pazienti per split
                min_samples_leaf=max(30, len(group_data) // 30),   # Min 30 pazienti per foglia
                random_state=42,
                class_weight='balanced'  # Bilancia le classi
            )

            # Fit del decision tree
            dt.fit(features_data, group_data['readmitted_binary'])

            # Step 5: Usa le foglie del tree come cluster
            leaf_predictions = dt.apply(features_data)
            unique_leaves = np.unique(leaf_predictions)

            print(f"  Decision Tree creato con {len(unique_leaves)} foglie/cluster")

            # Step 6: Assegna cluster basati sulle foglie del tree
            for leaf_id in unique_leaves:
                mask = leaf_predictions == leaf_id
                leaf_data = group_data.iloc[mask]

                # Calcola statistiche per questa foglia
                readmit_rate = leaf_data['readmitted_binary'].mean()

                # Ottieni la regola che porta a questa foglia
                tree_rules = get_decision_path_rules(dt, features_data.columns, leaf_id)

                df.loc[leaf_data.index, 'dt_cluster'] = cluster_id

                dt_cluster_info.append({
                    'cluster_id': cluster_id,
                    'age_group': group_name,
                    'method': 'decision_tree',
                    'count': len(leaf_data),
                    'readmit_rate': readmit_rate,
                    'features_used': available_significant,
                    'tree_rules': tree_rules,
                    'leaf_id': leaf_id
                })

                print(f"    Cluster {cluster_id}: {len(leaf_data)} pazienti, "
                      f"riammissione {readmit_rate:.1%}, regole: {tree_rules[:100]}...")

                cluster_id += 1
        else:
            # Non abbastanza features o pazienti, cluster singolo
            df.loc[group_data.index, 'dt_cluster'] = cluster_id
            dt_cluster_info.append({
                'cluster_id': cluster_id,
                'age_group': group_name,
                'method': 'single',
                'count': len(group_data),
                'features_used': available_significant,
                'tree_rules': 'Single cluster - insufficient features/data'
            })
            cluster_id += 1

    # Step 7: Riempi valori mancanti
    df['dt_cluster'].fillna(cluster_id, inplace=True)

    print(f"\nCreati {cluster_id + 1} cluster funzionali con Decision Tree")
    return df, dt_cluster_info

def get_decision_path_rules(tree, feature_names, leaf_id):
    """Estrae le regole del decision tree che portano a una specifica foglia"""
    try:
        # Trova il percorso dalla radice alla foglia
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        # Trova il percorso verso la foglia
        def find_path_to_leaf(node_id, target_leaf, path=[]):
            if children_left[node_id] == children_right[node_id]:  # È una foglia
                if node_id == target_leaf:
                    return path
                else:
                    return None

            # Prova percorso sinistro
            left_path = find_path_to_leaf(children_left[node_id], target_leaf,
                                        path + [(node_id, 'left')])
            if left_path is not None:
                return left_path

            # Prova percorso destro
            right_path = find_path_to_leaf(children_right[node_id], target_leaf,
                                         path + [(node_id, 'right')])
            return right_path

        path = find_path_to_leaf(0, leaf_id)
        if path is None:
            return "Cannot determine path"

        # Costruisci le regole
        rules = []
        for node_id, direction in path:
            feature_name = feature_names[feature[node_id]]
            threshold_val = threshold[node_id]

            if direction == 'left':
                rules.append(f"{feature_name} <= {threshold_val:.2f}")
            else:
                rules.append(f"{feature_name} > {threshold_val:.2f}")

        return " AND ".join(rules)

    except Exception as e:
        return f"Error extracting rules: {str(e)}"

def create_hybrid_clusters(df: pd.DataFrame, age_analysis: dict) -> tuple[pd.DataFrame, list]:
    """
    Approccio ibrido: Combina segmentazione per età + scoring del rischio di riammissione.

    Implementa un metodo a due fasi:
    1. Segmentazione in macro-gruppi di età (giovani, adulti, anziani, molto_anziani)
    2. All'interno di ogni gruppo, calcola risk score con LogisticRegression
    3. Segmenta ogni gruppo in 3 livelli di rischio (low, medium, high) tramite percentili

    Risultato: cluster che combinano caratteristiche demografiche e rischio clinico.

    Args:
        df (pd.DataFrame): Dataset con informazioni di età e features cliniche
        age_analysis (dict): Analisi features significative per fascia d'età

    Returns:
        tuple[pd.DataFrame, list]:
            - Dataset con colonna 'hybrid_cluster' assegnata
            - Lista info cluster con risk score e tasso riammissione effettivo

    Note:
        - Risk score calcolato con LogisticRegression class_weight='balanced'
        - Segmentazione rischio: percentili 33° e 67°
        - Confronto tra predicted risk score e actual readmit rate
    """
    print("\nCreazione cluster ibridi età + pattern riammissione...")

    # Step 1: Crea macro-gruppi di età
    age_macro_groups = {
        'giovani': ['[0-10)', '[10-20)', '[20-30)', '[30-40)'],
        'adulti': ['[40-50)', '[50-60)'],
        'anziani': ['[60-70)', '[70-80)'],
        'molto_anziani': ['[80-90)', '[90-100)']
    }

    def get_age_macro_group(age):
        for group_name, age_ranges in age_macro_groups.items():
            if age in age_ranges:
                return group_name
        return 'unknown'

    df['age_macro_group_hybrid'] = df['age'].apply(get_age_macro_group)

    # Step 2: Per ogni macro-gruppo, calcola risk score e segmenta
    df['hybrid_cluster'] = np.nan
    hybrid_cluster_info = []
    cluster_id = 0

    for group_name, age_ranges in age_macro_groups.items():
        group_data = df[df['age_macro_group_hybrid'] == group_name].copy()

        if len(group_data) < 100:
            df.loc[group_data.index, 'hybrid_cluster'] = cluster_id
            hybrid_cluster_info.append({
                'cluster_id': cluster_id,
                'age_group': group_name,
                'risk_level': 'all',
                'method': 'single',
                'count': len(group_data),
                'features_used': []
            })
            cluster_id += 1
            continue

        print(f"\nProcessing {group_name}: {len(group_data)} pazienti")

        # Step 3: Raccogli features significative
        all_significant_features = set()
        for age_range in age_ranges:
            if age_range in age_analysis:
                all_significant_features.update(age_analysis[age_range]['significant_features'])

        significant_features_list = list(all_significant_features)
        available_significant = [f for f in significant_features_list if f in group_data.columns]

        if len(available_significant) >= 2:
            # Step 4: Calcola risk score per ogni paziente
            features_data = group_data[available_significant].fillna(group_data[available_significant].median())

            # Usa un modello semplice per calcolare risk score
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(random_state=42, class_weight='balanced')
            lr.fit(features_data, group_data['readmitted_binary'])
            risk_scores = lr.predict_proba(features_data)[:, 1]  # Probabilità di riammissione

            # Step 5: Segmenta in base al risk score
            # Usa percentili per creare 3 gruppi bilanciati
            try:
                risk_low_thresh = np.percentile(risk_scores, 33)
                risk_high_thresh = np.percentile(risk_scores, 67)

                # Assegna risk levels
                risk_levels = []
                for score in risk_scores:
                    if score <= risk_low_thresh:
                        risk_levels.append('low_risk')
                    elif score <= risk_high_thresh:
                        risk_levels.append('medium_risk')
                    else:
                        risk_levels.append('high_risk')

                risk_levels = np.array(risk_levels)

                # Crea cluster per ogni combinazione età + risk level
                for risk_level in ['low_risk', 'medium_risk', 'high_risk']:
                    mask = risk_levels == risk_level
                    if mask.sum() == 0:
                        continue

                    risk_group_data = group_data.iloc[mask]
                    actual_readmit_rate = risk_group_data['readmitted_binary'].mean()

                    df.loc[risk_group_data.index, 'hybrid_cluster'] = cluster_id

                    hybrid_cluster_info.append({
                        'cluster_id': cluster_id,
                        'age_group': group_name,
                        'risk_level': risk_level,
                        'method': 'hybrid_age_risk',
                        'count': len(risk_group_data),
                        'predicted_risk_avg': risk_scores[mask].mean(),
                        'actual_readmit_rate': actual_readmit_rate,
                        'features_used': available_significant
                    })

                    print(f"    Cluster {cluster_id}: {group_name}_{risk_level} - "
                          f"{len(risk_group_data)} pazienti, "
                          f"risk score: {risk_scores[mask].mean():.2f}, "
                          f"actual readmit: {actual_readmit_rate:.1%}")

                    cluster_id += 1

            except Exception as e:
                print(f"  Errore nel calcolo risk score per {group_name}: {e}")
                # Fallback a cluster singolo
                df.loc[group_data.index, 'hybrid_cluster'] = cluster_id
                hybrid_cluster_info.append({
                    'cluster_id': cluster_id,
                    'age_group': group_name,
                    'risk_level': 'all',
                    'method': 'single_fallback',
                    'count': len(group_data),
                    'features_used': available_significant
                })
                cluster_id += 1
        else:
            # Non abbastanza features, cluster singolo
            df.loc[group_data.index, 'hybrid_cluster'] = cluster_id
            hybrid_cluster_info.append({
                'cluster_id': cluster_id,
                'age_group': group_name,
                'risk_level': 'all',
                'method': 'single',
                'count': len(group_data),
                'features_used': available_significant
            })
            cluster_id += 1

    # Step 6: Riempi valori mancanti
    df['hybrid_cluster'].fillna(cluster_id, inplace=True)

    print(f"\nCreati {cluster_id + 1} cluster ibridi età + risk")
    return df, hybrid_cluster_info

def create_age_based_clusters(df: pd.DataFrame, age_analysis: dict) -> tuple[pd.DataFrame, list]:
    """
    Approccio Age-Based: 4 fasce d'età fisse come cluster di baseline.

    Implementa il metodo di clustering più semplice e interpretabile, utilizzando
    4 fasce d'età fisse basate su età_encoded:
    - young_0_40: età 0-40 anni (encoded 1-3)
    - middle_40_60: età 40-60 anni (encoded 4-5)
    - elderly_60_80: età 60-80 anni (encoded 6-7)
    - very_elderly_80_100: età 80+ anni (encoded 8-10)

    Questo metodo serve come baseline e viene SEMPRE utilizzato nel sistema
    integrato finale indipendentemente dai risultati della competizione.

    Args:
        df (pd.DataFrame): Dataset con colonna 'age' o 'age_encoded'
        age_analysis (dict): Analisi features (utilizzata per statistiche)

    Returns:
        tuple[pd.DataFrame, list]:
            - Dataset con colonne 'age_based_cluster' e 'age_based_cluster_id'
            - Lista info cluster con mapping età e statistiche

    Note:
        - Metodo fisso NON in competizione con gli altri 3
        - Sempre presente nel sistema integrato finale
        - Crea age_encoded se non presente (mapping range età -> numero)
    """
    print("\nCreazione cluster Age-Based (4 fasce d'età fisse)...")

    # Step 1: Definisce le 4 fasce d'età fisse
    age_clusters = {
        'young_0_40': {'min_encoded': 0, 'max_encoded': 3, 'age_ranges': ['[0-10)', '[10-20)', '[20-30)', '[30-40)']},
        'middle_40_60': {'min_encoded': 4, 'max_encoded': 5, 'age_ranges': ['[40-50)', '[50-60)']},
        'elderly_60_80': {'min_encoded': 6, 'max_encoded': 7, 'age_ranges': ['[60-70)', '[70-80)']},
        'very_elderly_80_100': {'min_encoded': 8, 'max_encoded': 10, 'age_ranges': ['[80-90)', '[90-100)']}
    }

    # Step 2: Crea age_encoded se non esiste
    if 'age_encoded' not in df.columns:
        # Mappa age ranges a numeri
        age_mapping = {
            '[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4,
            '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8,
            '[80-90)': 9, '[90-100)': 10
        }
        df['age_encoded'] = df['age'].map(age_mapping).fillna(5)  # Default middle age

    # Step 3: Assegna cluster basati su age_encoded
    def get_age_cluster(age_encoded):
        if pd.isna(age_encoded):
            return 'middle_40_60'  # Default

        age_encoded = int(age_encoded)
        if age_encoded <= 3:
            return 'young_0_40'
        elif age_encoded <= 5:
            return 'middle_40_60'
        elif age_encoded <= 7:
            return 'elderly_60_80'
        else:
            return 'very_elderly_80_100'

    df['age_based_cluster'] = df['age_encoded'].apply(get_age_cluster)

    # Step 4: Crea cluster numerici per compatibilità
    cluster_mapping = {
        'young_0_40': 0,
        'middle_40_60': 1,
        'elderly_60_80': 2,
        'very_elderly_80_100': 3
    }
    df['age_based_cluster_id'] = df['age_based_cluster'].map(cluster_mapping)

    # Step 5: Analizza ogni cluster
    age_based_cluster_info = []

    for cluster_name, cluster_id in cluster_mapping.items():
        cluster_data = df[df['age_based_cluster'] == cluster_name]

        if len(cluster_data) == 0:
            continue

        readmit_rate = cluster_data['readmitted_binary'].mean()

        # Raccogli features significative per questa fascia
        significant_features = []
        if cluster_name.replace('_', ' ').replace(' ', '_') in age_analysis:
            analysis_key = None
            # Trova la chiave corrispondente in age_analysis
            for key in age_analysis.keys():
                if any(age_range in age_clusters[cluster_name]['age_ranges'] for age_range in [key]):
                    analysis_key = key
                    break

            if analysis_key and 'significant_features' in age_analysis[analysis_key]:
                significant_features = age_analysis[analysis_key]['significant_features'][:5]

        age_based_cluster_info.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'age_ranges': age_clusters[cluster_name]['age_ranges'],
            'method': 'age_based_fixed',
            'count': len(cluster_data),
            'readmit_rate': readmit_rate,
            'features_used': significant_features,
            'age_encoded_range': f"{age_clusters[cluster_name]['min_encoded']}-{age_clusters[cluster_name]['max_encoded']}"
        })

        print(f"  Cluster {cluster_id} ({cluster_name}): {len(cluster_data)} pazienti, "
              f"riammissione {readmit_rate:.1%}, "
              f"età encoded {age_clusters[cluster_name]['min_encoded']}-{age_clusters[cluster_name]['max_encoded']}")

    print(f"\nCreati {len(age_based_cluster_info)} cluster Age-Based fissi")

    return df, age_based_cluster_info

def encode_categorical_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Codifica le variabili categoriche utilizzando Label Encoding.

    Applica LabelEncoder a tutte le colonne categoriche (tipo object) eccetto
    gli identificativi (patient_nbr, encounter_id). Crea nuove colonne con
    suffisso '_encoded' mantenendo le originali per riferimento.

    Args:
        df (pd.DataFrame): Dataset con variabili categoriche da codificare

    Returns:
        tuple[pd.DataFrame, dict]:
            - Dataset con colonne categoriche codificate aggiunte
            - Dizionario degli encoders per eventuale reverse mapping

    Note:
        - Mantiene colonne originali per interpretabilità
        - Esclude colonne identificative dal processo
        - Encoders salvati per coerenza con nuovi dati
    """
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

def prepare_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara la variabile target binaria per la predizione di riammissione.

    Converte la variabile 'readmitted' multi-classe in una variabile binaria
    'readmitted_binary' mappando:
    - 'NO' -> 0 (nessuna riammissione)
    - '<30' e '>30' -> 1 (riammissione entro o oltre 30 giorni)

    Args:
        df (pd.DataFrame): Dataset con variabile 'readmitted' originale

    Returns:
        pd.DataFrame: Dataset con variabile 'readmitted_binary' aggiunta

    Note:
        - Semplifica il problema da multi-classe a binario
        - Preserva la variabile originale per analisi dettagliate
        - Fornisce statistiche distribuzione classi
    """
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

def save_cleaned_dataset(df: pd.DataFrame, output_path: str) -> str:
    """
    Salva il dataset pulito nel percorso specificato.

    Crea le directory necessarie se non esistenti e salva il DataFrame
    in formato CSV senza indice per compatibilità downstream.

    Args:
        df (pd.DataFrame): Dataset pulito da salvare
        output_path (str): Percorso completo file di output

    Returns:
        str: Percorso del file salvato (confermato)

    Note:
        - Crea directory automaticamente se necessario
        - Formato CSV senza indice per pulizia
        - Logging dimensioni finali per verifica
    """
    print(f"\nSalvataggio dataset pulito in: {output_path}")

    # Crea directory se non esiste
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva dataset
    df.to_csv(output_path, index=False)

    print(f"Dataset salvato: {df.shape}")
    print(f"Colonne finali: {len(df.columns)}")

    return output_path

def compare_clustering_methods(df: pd.DataFrame, age_analysis: dict) -> tuple[dict, str]:
    """
    Confronta 3 metodi di clustering competitivi + Age-Based fisso con valutazione multi-metrica.

    Esegue una competizione tra 3 metodi avanzati (K-means Gerarchico, Decision Tree,
    Ibrido Età+Risk) utilizzando metriche multiple per determinare il metodo migliore.
    Il metodo Age-Based viene sempre generato ma NON partecipa alla competizione.

    Pipeline di Valutazione:
    1. Esecuzione parallela dei 4 metodi di clustering
    2. Valutazione con 5 metriche ponderate:
       - Silhouette Score (15%): qualità geometrica cluster
       - Intra-cluster Homogeneity (20%): omogeneità interna
       - Inter-cluster Separation (25%): separazione tra cluster
       - Prediction Utility (25%): utilità per predizione ML
       - Risk Discrimination (15%): capacità discriminare rischio
    3. Calcolo Overall Score pesato per selezione vincente

    Args:
        df (pd.DataFrame): Dataset pulito per clustering
        age_analysis (dict): Analisi features significative per età

    Returns:
        tuple[dict, str]:
            - Dizionario completo risultati tutti i metodi (incluso age_based)
            - Nome del metodo vincente della competizione (esclude age_based)

    Note:
        - Solo 3 metodi in competizione (kmeans, decision_tree, hybrid)
        - Age-based sempre disponibile ma fuori competizione
        - Sistema integrato finale usa SEMPRE age_based + metodo vincente
    """
    print("\n" + "="*80)
    print("COMPARAZIONE 3 METODI DI CLUSTERING + AGE-BASED FISSO")
    print("="*80)

    competitive_results = {}

    # METODI IN COMPETIZIONE (3)
    print("\n" + "="*50)
    print("METODI IN COMPETIZIONE")
    print("="*50)

    # Metodo 1: K-means gerarchico
    print("\n1. METODO K-MEANS GERARCHICO")
    df_kmeans, kmeans_info = create_hierarchical_age_clusters(df.copy(), age_analysis)
    kmeans_metrics = evaluate_clustering_performance(df_kmeans, 'final_cluster', 'K-means Gerarchico')
    competitive_results['kmeans'] = {
        'df': df_kmeans,
        'info': kmeans_info,
        'metrics': kmeans_metrics
    }

    # Metodo 2: Decision Tree
    print("\n2. METODO DECISION TREE FUNZIONALE")
    df_dt, dt_info = create_decision_tree_clusters(df.copy(), age_analysis)
    dt_metrics = evaluate_clustering_performance(df_dt, 'dt_cluster', 'Decision Tree')
    competitive_results['decision_tree'] = {
        'df': df_dt,
        'info': dt_info,
        'metrics': dt_metrics
    }

    # Metodo 3: Ibrido età + risk
    print("\n3. METODO IBRIDO ETÀ + RISK")
    df_hybrid, hybrid_info = create_hybrid_clusters(df.copy(), age_analysis)
    hybrid_metrics = evaluate_clustering_performance(df_hybrid, 'hybrid_cluster', 'Ibrido Età+Risk')
    competitive_results['hybrid'] = {
        'df': df_hybrid,
        'info': hybrid_info,
        'metrics': hybrid_metrics
    }

    # METODO FISSO (non in competizione)
    print("\n" + "="*50)
    print("METODO FISSO (NON IN COMPETIZIONE)")
    print("="*50)
    print("\n4. METODO AGE-BASED FISSO (SEMPRE USATO NEL SISTEMA FINALE)")
    df_age_based, age_based_info = create_age_based_clusters(df.copy(), age_analysis)
    age_based_metrics = evaluate_clustering_performance(df_age_based, 'age_based_cluster_id', 'Age-Based Fisso')

    # Combina tutti i risultati
    results = competitive_results.copy()
    results['age_based'] = {
        'df': df_age_based,
        'info': age_based_info,
        'metrics': age_based_metrics
    }

    # Confronto finale SOLO per i metodi in competizione
    print("\n" + "="*80)
    print("RISULTATI COMPARAZIONE COMPETITIVA (3 METODI)")
    print("="*80)

    competitive_comparison_table = []
    for method_name, method_data in competitive_results.items():
        metrics = method_data['metrics']
        competitive_comparison_table.append({
            'Metodo': method_name.upper(),
            'N_Clusters': len(set(method_data['df'][metrics['cluster_column']].dropna())),
            'Silhouette_Score': f"{metrics['silhouette_score']:.3f}",
            'Intra_Cluster_Homogeneity': f"{metrics['intra_cluster_homogeneity']:.3f}",
            'Inter_Cluster_Separation': f"{metrics['inter_cluster_separation']:.3f}",
            'Prediction_Utility': f"{metrics['prediction_utility']:.3f}",
            'Risk_Discrimination': f"{metrics['risk_discrimination']:.3f}",
            'Overall_Score': f"{metrics['overall_score']:.3f}"
        })

    # Stampa tabella comparativa SOLO per metodi competitivi
    competitive_comparison_df = pd.DataFrame(competitive_comparison_table)
    print(competitive_comparison_df.to_string(index=False))

    # Determina il migliore TRA I COMPETITIVI
    best_competitive_method = max(competitive_results.keys(), key=lambda x: competitive_results[x]['metrics']['overall_score'])
    best_competitive_score = competitive_results[best_competitive_method]['metrics']['overall_score']

    print(f"\nMETODO VINCENTE DELLA COMPETIZIONE: {best_competitive_method.upper()}")
    print(f"Overall Score: {best_competitive_score:.3f}")

    # Informazioni sul metodo fisso
    age_based_score = results['age_based']['metrics']['overall_score']
    print(f"\nMETODO FISSO AGE-BASED:")
    print(f"Overall Score: {age_based_score:.3f} (NON IN COMPETIZIONE)")
    print("Questo metodo sarà SEMPRE usato nel sistema integrato finale")

    # Analisi dettagliata del metodo vincente della competizione
    print(f"\nANALISI DETTAGLIATA - VINCENTE COMPETIZIONE: {best_competitive_method.upper()}")
    best_competitive_df = competitive_results[best_competitive_method]['df']
    best_competitive_info = competitive_results[best_competitive_method]['info']
    best_competitive_cluster_col = competitive_results[best_competitive_method]['metrics']['cluster_column']

    analyze_detailed_cluster_performance(best_competitive_df, best_competitive_info, best_competitive_cluster_col)

    return results, best_competitive_method

def evaluate_clustering_performance(df, cluster_column, method_name):
    """Valuta la performance di un metodo di clustering"""
    print(f"\nValutazione performance {method_name}...")

    metrics = {'cluster_column': cluster_column}

    # 1. Silhouette Score (per valutare la qualità dei cluster)
    try:
        # Prepara features per silhouette
        numeric_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                           'number_diagnoses', 'num_procedures', 'number_outpatient',
                           'number_emergency', 'number_inpatient']
        available_features = [f for f in numeric_features if f in df.columns]

        if len(available_features) >= 2:
            features_data = df[available_features].fillna(df[available_features].median())
            clusters = df[cluster_column].fillna(-1)

            # Solo se abbiamo più di 1 cluster
            n_clusters = len(set(clusters))
            if n_clusters > 1:
                silhouette_avg = silhouette_score(features_data, clusters)
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0

        metrics['silhouette_score'] = max(silhouette_avg, 0.0)

    except Exception as e:
        print(f"  Errore calcolo silhouette: {e}")
        metrics['silhouette_score'] = 0.0

    # 2. Intra-cluster Homogeneity (omogeneità interna)
    try:
        intra_homogeneity_scores = []
        for cluster_id in df[cluster_column].unique():
            if pd.isna(cluster_id):
                continue
            cluster_data = df[df[cluster_column] == cluster_id]
            if len(cluster_data) > 1:
                readmit_var = cluster_data['readmitted_binary'].var()
                homogeneity = 1 - readmit_var  # Più bassa varianza = più omogeneo
                intra_homogeneity_scores.append(max(homogeneity, 0))

        metrics['intra_cluster_homogeneity'] = np.mean(intra_homogeneity_scores) if intra_homogeneity_scores else 0.0
    except Exception as e:
        print(f"  Errore calcolo omogeneità: {e}")
        metrics['intra_cluster_homogeneity'] = 0.0

    # 3. Inter-cluster Separation (separazione tra cluster)
    try:
        cluster_means = []
        for cluster_id in df[cluster_column].unique():
            if pd.isna(cluster_id):
                continue
            cluster_data = df[df[cluster_column] == cluster_id]
            cluster_means.append(cluster_data['readmitted_binary'].mean())

        if len(cluster_means) > 1:
            separation = np.var(cluster_means)  # Più varianza tra cluster = migliore separazione
        else:
            separation = 0.0

        metrics['inter_cluster_separation'] = separation
    except Exception as e:
        print(f"  Errore calcolo separazione: {e}")
        metrics['inter_cluster_separation'] = 0.0

    # 4. Prediction Utility (utilità predittiva)
    try:
        # Usa un semplice RandomForest per testare utilità predittiva dei cluster
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Feature set: include cluster come feature
        cluster_encoded = LabelEncoder().fit_transform(df[cluster_column].fillna(-1))
        prediction_features = np.column_stack([
            cluster_encoded,
            df['time_in_hospital'].fillna(df['time_in_hospital'].median()),
            df['num_medications'].fillna(df['num_medications'].median()),
            df['number_diagnoses'].fillna(df['number_diagnoses'].median())
        ])

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        cv_scores = cross_val_score(rf, prediction_features, df['readmitted_binary'], cv=3, scoring='roc_auc')
        metrics['prediction_utility'] = cv_scores.mean()

    except Exception as e:
        print(f"  Errore calcolo utilità predittiva: {e}")
        metrics['prediction_utility'] = 0.5

    # 5. Risk Discrimination (capacità di discriminare rischio)
    try:
        cluster_readmit_rates = []
        for cluster_id in df[cluster_column].unique():
            if pd.isna(cluster_id):
                continue
            cluster_data = df[df[cluster_column] == cluster_id]
            if len(cluster_data) >= 10:  # Minimo 10 pazienti
                readmit_rate = cluster_data['readmitted_binary'].mean()
                cluster_readmit_rates.append(readmit_rate)

        if len(cluster_readmit_rates) >= 2:
            # Calcola differenza tra cluster ad alto e basso rischio
            max_rate = max(cluster_readmit_rates)
            min_rate = min(cluster_readmit_rates)
            risk_discrimination = max_rate - min_rate
        else:
            risk_discrimination = 0.0

        metrics['risk_discrimination'] = risk_discrimination
    except Exception as e:
        print(f"  Errore calcolo discriminazione rischio: {e}")
        metrics['risk_discrimination'] = 0.0

    # 6. Overall Score (punteggio complessivo pesato)
    weights = {
        'silhouette_score': 0.15,
        'intra_cluster_homogeneity': 0.20,
        'inter_cluster_separation': 0.25,
        'prediction_utility': 0.25,
        'risk_discrimination': 0.15
    }

    overall_score = sum(metrics[key] * weight for key, weight in weights.items())
    metrics['overall_score'] = overall_score

    # Stampa risultati
    print(f"  Risultati {method_name}:")
    print(f"    - N. Clusters: {len(set(df[cluster_column].dropna()))}")
    print(f"    - Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"    - Intra-cluster Homogeneity: {metrics['intra_cluster_homogeneity']:.3f}")
    print(f"    - Inter-cluster Separation: {metrics['inter_cluster_separation']:.3f}")
    print(f"    - Prediction Utility: {metrics['prediction_utility']:.3f}")
    print(f"    - Risk Discrimination: {metrics['risk_discrimination']:.3f}")
    print(f"    - Overall Score: {metrics['overall_score']:.3f}")

    return metrics

def analyze_detailed_cluster_performance(df, cluster_info, cluster_column):
    """Analisi dettagliata delle performance dei cluster del metodo migliore"""
    print("\nAnalisi dettagliata cluster:")

    for cluster_id in sorted(df[cluster_column].unique()):
        if pd.isna(cluster_id):
            continue

        cluster_data = df[df[cluster_column] == cluster_id]
        readmit_rate = cluster_data['readmitted_binary'].mean()

        # Trova info del cluster
        if isinstance(cluster_info, list) and len(cluster_info) > 0:
            cluster_detail = next((c for c in cluster_info if c.get('cluster_id') == cluster_id), {})

            print(f"\n  Cluster {int(cluster_id)}: {len(cluster_data)} pazienti")
            print(f"    - Tasso riammissione: {readmit_rate:.1%}")
            print(f"    - Gruppo età: {cluster_detail.get('age_group', 'unknown')}")
            print(f"    - Metodo: {cluster_detail.get('method', 'unknown')}")

            if 'tree_rules' in cluster_detail:
                print(f"    - Regole: {cluster_detail['tree_rules'][:120]}...")
            elif 'risk_level' in cluster_detail:
                print(f"    - Risk level: {cluster_detail['risk_level']}")
                if 'predicted_risk_avg' in cluster_detail:
                    print(f"    - Risk score medio: {cluster_detail['predicted_risk_avg']:.3f}")

def analyze_final_cluster_performance(df, cluster_info):
    """Mantiene la funzione originale per compatibilità"""
    print("\nAnalisi performance cluster finali...")

    if 'readmitted_binary' in df.columns and 'final_cluster' in df.columns:
        print("\nPerformance per cluster:")
        # Analizza distribuzione target per cluster
        for cluster_id in sorted(df['final_cluster'].unique()):
            if pd.isna(cluster_id):
                continue

            cluster_data = df[df['final_cluster'] == cluster_id]
            readmit_rate = cluster_data['readmitted_binary'].mean()

            # Trova info del cluster
            cluster_detail = next((c for c in cluster_info if c['cluster_id'] == cluster_id), None)
            age_group = cluster_detail['age_group'] if cluster_detail else 'unknown'
            method = cluster_detail['sub_method'] if cluster_detail else 'unknown'

            print(f"  Cluster {int(cluster_id)}: {len(cluster_data)} pazienti, "
                  f"riammissione {readmit_rate:.1%}, {age_group} ({method})")

        # Analisi per macro-gruppo di età
        print("\nPerformance per macro-gruppo età:")
        for age_group in df['age_macro_group'].unique():
            if pd.isna(age_group):
                continue
            group_data = df[df['age_macro_group'] == age_group]
            readmit_rate = group_data['readmitted_binary'].mean()
            print(f"  {age_group}: {len(group_data)} pazienti, riammissione {readmit_rate:.1%}")

def main() -> tuple[pd.DataFrame, str, dict, str]:
    """
    Pipeline principale di pulizia dataset e comparazione metodi clustering.

    Esegue la pipeline completa end-to-end:
    1. Caricamento e pulizia dataset diabetico originale
    2. Preprocessing (duplicati, valori mancanti, encoding)
    3. Analisi features significative per fascia d'età
    4. Comparazione 4 metodi di clustering (3 competitivi + 1 fisso)
    5. Selezione metodo vincente e salvataggio dataset multipli

    La funzione produce dataset clustered per tutti i metodi, consentendo
    al sistema integrato di utilizzare sia il metodo vincente che l'age-based.

    Returns:
        tuple[pd.DataFrame, str, dict, str]:
            - Dataset del metodo vincente (per compatibilità)
            - Percorso file output principale
            - Dizionario completo risultati tutti i metodi
            - Nome metodo vincente della competizione

    Note:
        - Salva dataset per TUTTI i metodi in outputs/datasets_clean/cluster/terzo_metodo/
        - Dataset principal: metodo vincente competizione
        - Age-based sempre disponibile per sistema integrato
        - Performance logging completo per analisi
    """
    print("PIPELINE PULIZIA DATASET DIABETICI")
    print("="*50)

    # Step 1: Carica dataset originale
    df = load_and_clean_data()

    # OPZIONALE: Usa un subset per test veloce (commenta per dataset completo)
    print(f"USANDO DATASET COMPLETO per comparazione finale...")
    # df = df.sample(n=5000, random_state=42)

    # Step 2: Rimuovi duplicati per paziente
    df = remove_duplicate_patients(df)

    # Step 3: Sostituisci '?' con NaN
    df = replace_question_marks_with_nan(df)

    # Step 4: Gestisci valori mancanti
    df = handle_missing_values(df)

    # Step 5: Prepara variabile target (necessaria per l'analisi delle correlazioni)
    df = prepare_target_variable(df)

    # Step 6: Analizza features per fascia d'età
    age_analysis = age_mapping(df)

    # Step 7: COMPARAZIONE METODI DI CLUSTERING 
    clustering_results, best_method = compare_clustering_methods(df, age_analysis)

    # Step 8: Usa il dataset del metodo migliore per le fasi successive
    best_df = clustering_results[best_method]['df']
    best_info = clustering_results[best_method]['info']

    # Step 9: Codifica variabili categoriche sul dataset migliore
    best_df, encoders = encode_categorical_variables(best_df)

    # Step 10: Salva dataset pulito con il metodo migliore
    output_path = f'outputs/datasets_clean/cluster/terzo_metodo/db_clean_cluster_{best_method}.csv'
    save_cleaned_dataset(best_df, output_path)

    # Step 11: Salva anche tutti i metodi per confronti futuri
    for method_name, method_data in clustering_results.items():
        method_df, _ = encode_categorical_variables(method_data['df'].copy())
        method_path = f'outputs/datasets_clean/cluster/terzo_metodo/db_clean_cluster_{method_name}.csv'
        save_cleaned_dataset(method_df, method_path)
        print(f"Salvato dataset {method_name}: {method_path}")

    print("\n" + "="*50)
    print("PULIZIA DATASET COMPLETATA!")
    print(f"METODO VINCENTE: {best_method.upper()}")
    print(f"Output principale: {output_path}")
    print(f"Pazienti finali: {len(best_df)}")
    print(f"Features finali: {len(best_df.columns)}")
    print(f"Overall Score: {clustering_results[best_method]['metrics']['overall_score']:.3f}")

    return best_df, output_path, clustering_results, best_method

if __name__ == "__main__":
    df_cleaned, output_file, results, best_method = main()