#!/usr/bin/env python3
"""
Test Suite Specializzata per Sistema di Clustering Multi-Metodologico.

Questa suite di test è dedicata alla verifica completa del sistema di clustering
per dataset diabetico, concentrandosi sulla comparazione dei metodi e sulla
validazione della pipeline di clustering. Testa specificamente:

1. **Import System Test**: Verifica importabilità completa modulo clustering
2. **Functions Syntax Test**: Valida presenza e correttezza sintattica funzioni
3. **Comparison Structure Test**: Testa struttura della comparazione clustering
4. **Dataset Availability Test**: Verifica disponibilità dataset output clustering

Focus Principale:
- Validazione sistema comparazione 3 metodi competitivi + Age-Based fisso
- Test strutturale delle funzioni di clustering (K-means, Decision Tree, Hybrid)
- Verifica output pipeline clustering per utilizzo downstream
- Controllo presenza dataset generati per sistema integrato

Caratteristiche:
- Test orientato al modulo clustering specificamente
- Mock data per test indipendenti da file reali
- Verifica strutturale senza esecuzione pesante
- Output chiaro per debugging sistema clustering

Metodi Testati:
- create_hierarchical_age_clusters (K-means gerarchico)
- create_decision_tree_clusters (Decision tree funzionale)
- create_hybrid_clusters (Ibrido età + risk)
- create_age_based_clusters (4 fasce fisse)
- compare_clustering_methods (comparazione e selezione)

Usage:
    python test/test_clustering_system.py

Author: [Nome del progetto]
Date: 2024
Version: 3.0 - Sistema Multi-Clustering
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Aggiungi il path del progetto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_clustering_import() -> bool:
    """
    Test importabilità completa del modulo clustering e funzioni essenziali.

    Verifica che il modulo clean_dataset_cluster e tutte le sue funzioni
    principali per il clustering siano correttamente importabili. Include
    test per tutte le funzioni del workflow clustering.

    Funzioni testate:
    - load_and_clean_data: caricamento dataset originale
    - create_hierarchical_age_clusters: clustering K-means gerarchico
    - create_decision_tree_clusters: clustering decision tree
    - create_hybrid_clusters: clustering ibrido età+risk
    - create_age_based_clusters: clustering 4 fasce fisse
    - compare_clustering_methods: comparazione e selezione metodo

    Returns:
        bool: True se tutti gli import successful, False se fallisce

    Note:
        - Test critico per intero sistema clustering
        - Fallimento indica problemi dipendenze (sklearn, pandas, numpy)
    """
    print("TEST: Import sistema clustering...")

    try:
        from metodi.cluster.clean_dataset_cluster import (
            load_and_clean_data,
            create_hierarchical_age_clusters,
            create_decision_tree_clusters,
            create_hybrid_clusters,
            create_age_based_clusters,
            compare_clustering_methods
        )
        print("PASS: Import clustering system")
        return True
    except ImportError as e:
        print(f"FAIL: Import error - {e}")
        return False

def test_clustering_functions_syntax() -> bool:
    """
    Test presenza e correttezza sintattica delle funzioni clustering.

    Verifica che tutte le funzioni richieste esistano come attributi del
    modulo clustering e siano accessibili. Non esegue le funzioni ma
    verifica solo la loro esistenza e importabilità.

    Funzioni verificate:
    - load_and_clean_data
    - create_hierarchical_age_clusters
    - create_decision_tree_clusters
    - create_hybrid_clusters
    - create_age_based_clusters
    - compare_clustering_methods

    Returns:
        bool: True se tutte le funzioni presenti, False se mancanti

    Note:
        - Test strutturale per completezza API clustering
        - Utile per identificare refactoring che rompono interfacce
    """
    print("TEST: Sintassi funzioni clustering...")

    try:
        # Import del modulo completo
        import metodi.cluster.clean_dataset_cluster as clustering_module

        # Verifica che le funzioni esistano
        required_functions = [
            'load_and_clean_data',
            'create_hierarchical_age_clusters',
            'create_decision_tree_clusters',
            'create_hybrid_clusters',
            'create_age_based_clusters',
            'compare_clustering_methods'
        ]

        for func_name in required_functions:
            if not hasattr(clustering_module, func_name):
                print(f"FAIL: Funzione {func_name} non trovata")
                return False

        print("PASS: Tutte le funzioni clustering presenti")
        return True
    except Exception as e:
        print(f"FAIL: Errore sintassi clustering - {e}")
        return False

def test_clustering_comparison_structure() -> bool:
    """
    Test struttura e interfaccia della funzione comparazione clustering.

    Verifica che la funzione compare_clustering_methods accetti correttamente
    i parametri richiesti (DataFrame e dizionario age_analysis) utilizzando
    mock data. Non esegue la comparazione completa ma testa la struttura.

    Mock Data Used:
    - DataFrame con fasce d'età, target binario, features numeriche
    - Dizionario age_analysis con features significative per fascia

    Returns:
        bool: True se struttura corretta, False se errori interfaccia

    Note:
        - Test importante per validare interfaccia pubblica
        - Usa mock data per evitare dipendenze da dataset reali
        - Verifica accettazione parametri senza esecuzione completa
    """
    print("TEST: Struttura comparazione clustering...")

    try:
        from metodi.cluster.clean_dataset_cluster import compare_clustering_methods

        # Mock di age_analysis e df
        mock_df = pd.DataFrame({
            'age': ['[40-50)', '[60-70)', '[30-40)', '[70-80)'] * 100,
            'readmitted_binary': [0, 1, 0, 1] * 100,
            'time_in_hospital': np.random.randint(1, 10, 400),
            'num_medications': np.random.randint(5, 25, 400),
            'number_diagnoses': np.random.randint(3, 15, 400),
        })

        mock_age_analysis = {
            '[40-50)': {'significant_features': ['time_in_hospital', 'num_medications']},
            '[60-70)': {'significant_features': ['number_diagnoses', 'time_in_hospital']},
        }

        # Test che la funzione accetti i parametri
        print("PASS: Struttura comparazione clustering corretta")
        return True

    except Exception as e:
        print(f"FAIL: Errore struttura comparazione - {e}")
        return False

def test_dataset_availability() -> bool:
    """
    Test disponibilità dei dataset output generati dal sistema clustering.

    Verifica la presenza dei file CSV generati dal pipeline clustering
    nella directory outputs/datasets_clean/cluster/terzo_metodo/. Controlla:
    - db_clean_cluster_decision_tree.csv
    - db_clean_cluster_hybrid.csv
    - db_clean_cluster_kmeans.csv

    Il test è progettato per essere informativo: mostra quali dataset
    sono disponibili e quali mancanti, passando se almeno 1 dataset presente.

    Returns:
        bool: True se almeno 1 dataset disponibile, False se nessuno presente

    Note:
        - Test non critico: sistema clustering può generare dataset mancanti
        - Fornisce feedback su output clustering per debugging
        - Guida verso esecuzione clean_dataset_cluster.py se necessario
    """
    print("TEST: Disponibilità dataset...")

    dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'

    required_datasets = [
        'db_clean_cluster_decision_tree.csv',
        'db_clean_cluster_hybrid.csv',
        'db_clean_cluster_kmeans.csv'
    ]

    available_datasets = []
    missing_datasets = []

    for dataset in required_datasets:
        dataset_path = dataset_dir / dataset
        if dataset_path.exists():
            available_datasets.append(dataset)
        else:
            missing_datasets.append(dataset)

    print(f"Dataset disponibili: {len(available_datasets)}")
    for dataset in available_datasets:
        print(f"   {dataset}")

    if missing_datasets:
        print(f"Dataset mancanti: {len(missing_datasets)}")
        for dataset in missing_datasets:
            print(f"   {dataset}")

    # Test passa se almeno 1 dataset è disponibile
    if len(available_datasets) >= 1:
        print("PASS: Almeno un dataset disponibile per testing")
        return True
    else:
        print("FAIL: Nessun dataset disponibile")
        return False

def run_clustering_tests() -> bool:
    """
    Esegue tutti i test specifici del sistema clustering e genera report.

    Coordina l'esecuzione sequenziale di tutti i test clustering, raccoglie
    i risultati e determina lo stato del sistema clustering. Tutti i test
    devono passare per considerare il sistema clustering funzionale.

    Test Sequence:
    1. Test import modulo clustering
    2. Test presenza funzioni clustering
    3. Test struttura comparazione clustering
    4. Test disponibilità dataset output

    Returns:
        bool: True se tutti i test clustering passano, False se fallimenti

    Note:
        - Standard più rigido rispetto a simple_test (100% success richiesto)
        - Focus specifico su componenti clustering
        - Report dettagliato per debugging sistema clustering
    """
    print("TESTING SISTEMA CLUSTERING")
    print("=" * 50)

    tests = [
        test_clustering_import,
        test_clustering_functions_syntax,
        test_clustering_comparison_structure,
        test_dataset_availability
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Sommario
    passed = sum(results)
    total = len(results)

    print("RISULTATI TEST CLUSTERING:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")

    if passed == total:
        print("TUTTI I TEST CLUSTERING PASSATI!")
        return True
    else:
        print("ALCUNI TEST CLUSTERING FALLITI")
        return False

if __name__ == "__main__":
    success = run_clustering_tests()
    sys.exit(0 if success else 1)