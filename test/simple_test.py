#!/usr/bin/env python3
"""
Test Suite Semplificata per Sistema Predizione Riammissione Diabetici.

Questa suite di test fornisce una verifica rapida e completa della funzionalità
di base del sistema integrato di predizione riammissione per pazienti diabetici.
I test sono ottimizzati per compatibilità cross-platform (incluso Windows) ed
eseguono controlli essenziali senza dipendenze pesanti.

Test Coverage:
1. **Import System Test**: Verifica importabilità moduli clustering e integrato
2. **Class Instantiation Test**: Test istanziazione IntegratedHybridPredictor
3. **Age Mapping Test**: Verifica correttezza mapping fasce d'età
4. **Dataset Availability Test**: Controlla presenza dataset generati dal clustering
5. **Real System Functionality Test**: Test funzionalità con dati reali (se disponibili)

Caratteristiche:
- Esecuzione rapida (<30 secondi)
- Output clear senza emoji (compatibilità Windows)
- Graceful degradation se dataset non disponibili
- Success rate threshold: 80% per sistema funzionale
- Logging dettagliato per debugging

Usage:
    python test/simple_test.py

Exit Codes:
    0: Sistema funzionale (success rate >= 80%)
    1: Sistema con problemi critici

Author: [Nome del progetto]
Date: 2024
Version: 3.0 - Sistema Integrato
"""

import sys
import os
import time
from pathlib import Path

# Aggiungi il path del progetto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_clustering_import() -> bool:
    """
    Test importabilità del sistema di clustering e delle funzioni core.

    Verifica che il modulo clean_dataset_cluster e le sue funzioni principali
    (compare_clustering_methods, create_age_based_clusters) siano correttamente
    importabili. Questo è un test fondamentale per il funzionamento del sistema.

    Returns:
        bool: True se import successful, False se fallisce

    Note:
        - Test critico: fallimento indica problemi di installazione/dipendenze
        - Verifica solo import, non esecuzione funzioni
    """
    print("TEST: Import sistema clustering...")

    try:
        from metodi.cluster.clean_dataset_cluster import (
            compare_clustering_methods,
            create_age_based_clusters
        )
        print("PASS: Import clustering system")
        return True
    except ImportError as e:
        print(f"FAIL: Import error - {e}")
        return False

def test_integrated_system_import() -> bool:
    """
    Test importabilità del sistema integrato e classe principale.

    Verifica che il modulo hybrid_ml_clinical_rules_integrated e la classe
    IntegratedHybridPredictor siano correttamente importabili. Questo test
    è essenziale per il funzionamento del sistema di predizione finale.

    Returns:
        bool: True se import successful, False se fallisce

    Note:
        - Test critico per sistema di predizione integrato
        - Fallimento indica problemi con dipendenze ML (scikit-learn, pandas)
    """
    print("TEST: Import sistema integrato...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import (
            IntegratedHybridPredictor
        )
        print("PASS: Import sistema integrato")
        return True
    except ImportError as e:
        print(f"FAIL: Import error - {e}")
        return False

def test_integrated_system_class() -> bool:
    """
    Test istanziazione della classe IntegratedHybridPredictor con parametri dummy.

    Verifica che la classe principale possa essere istanziata correttamente
    con parametri dummy (senza caricare file reali) e che tutti gli attributi
    essenziali siano presenti e inizializzati.

    Attributi verificati:
    - age_based_path: percorso dataset age-based
    - decision_tree_path: percorso dataset decision tree

    Returns:
        bool: True se istanziazione successful, False se fallisce

    Note:
        - Usa path dummy per evitare dipendenze da file
        - Test costruttore e inizializzazione attributi base
    """
    print("TEST: Istanziazione IntegratedHybridPredictor...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Test con path dummy
        system = IntegratedHybridPredictor('dummy1.csv', 'dummy2.csv')

        # Verifica attributi essenziali
        if not hasattr(system, 'age_based_path'):
            print("FAIL: Attributo age_based_path mancante")
            return False

        if not hasattr(system, 'decision_tree_path'):
            print("FAIL: Attributo decision_tree_path mancante")
            return False

        print("PASS: Classe istanziata correttamente")
        return True

    except Exception as e:
        print(f"FAIL: Errore istanziazione - {e}")
        return False

def test_age_mapping() -> bool:
    """
    Test correttezza del mapping delle fasce d'età nel sistema integrato.

    Verifica che la funzione get_patient_age_cluster del sistema integrato
    mappi correttamente i valori age_encoded alle fasce d'età predefinite:
    - 1-3 -> young_0_40
    - 4-5 -> middle_40_60
    - 6-7 -> elderly_60_80
    - 8-10 -> very_elderly_80_100

    Returns:
        bool: True se mapping corretto, False se errori

    Note:
        - Test critico per correttezza predizioni
        - Mapping deve essere coerente con clustering age-based
    """
    print("TEST: Age-based mapping...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        system = IntegratedHybridPredictor('dummy1.csv', 'dummy2.csv')

        # Test mapping età
        test_cases = [
            {'age_encoded': 2, 'expected': 'young_0_40'},
            {'age_encoded': 5, 'expected': 'middle_40_60'},
            {'age_encoded': 7, 'expected': 'elderly_60_80'},
            {'age_encoded': 9, 'expected': 'very_elderly_80_100'},
        ]

        for case in test_cases:
            result = system.get_patient_age_cluster(case)
            if result != case['expected']:
                print(f"FAIL: Age mapping error - {case['age_encoded']} -> {result}")
                return False

        print("PASS: Age mapping funziona")
        return True

    except Exception as e:
        print(f"FAIL: Errore age mapping - {e}")
        return False

def test_dataset_availability() -> bool:
    """
    Test disponibilità dei dataset generati dal sistema di clustering.

    Controlla la presenza dei file dataset nelle directory di output del
    sistema di clustering. I dataset richiesti sono:
    - db_clean_cluster_decision_tree.csv
    - db_clean_cluster_hybrid.csv
    - db_clean_cluster_kmeans.csv

    Il test passa se almeno 1 dataset è disponibile, consentendo test
    con funzionalità ridotta.

    Returns:
        bool: True se almeno 1 dataset disponibile, False se nessuno

    Note:
        - Non è critico: sistema può funzionare con dataset parziali
        - Guida verso esecuzione clean_dataset_cluster.py se necessario
    """
    print("TEST: Dataset disponibili...")

    dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'

    required_datasets = [
        'db_clean_cluster_decision_tree.csv',
        'db_clean_cluster_hybrid.csv',
        'db_clean_cluster_kmeans.csv'
    ]

    available_count = 0
    for dataset in required_datasets:
        dataset_path = dataset_dir / dataset
        if dataset_path.exists():
            available_count += 1
            print(f"   FOUND: {dataset}")
        else:
            print(f"   MISSING: {dataset}")

    if available_count >= 1:
        print(f"PASS: {available_count}/{len(required_datasets)} dataset disponibili")
        return True
    else:
        print("FAIL: Nessun dataset disponibile")
        return False

def test_real_system_functionality() -> bool:
    """
    Test funzionalità del sistema integrato con dataset reali (se disponibili).

    Se i dataset sono disponibili, testa l'inizializzazione completa del
    sistema IntegratedHybridPredictor con dati reali. Verifica che:
    - Il sistema possa caricare i dataset
    - L'inizializzazione completa senza errori
    - I componenti interni siano correttamente configurati

    Returns:
        bool: True se sistema funziona o dataset non disponibili, False se errori

    Note:
        - Test non critico: skip se dataset mancanti
        - Verifica funzionalità end-to-end quando possibile
        - Importante per validare pipeline completa
    """
    print("TEST: Funzionalità sistema con dati reali...")

    try:
        dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'
        dt_path = dataset_dir / 'db_clean_cluster_decision_tree.csv'

        if not dt_path.exists():
            print("SKIP: Dataset non disponibile")
            return True

        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Test inizializzazione con dati reali
        system = IntegratedHybridPredictor(str(dt_path), str(dt_path))
        result = system.initialize_systems()

        if not result:
            print("FAIL: Inizializzazione fallita")
            return False

        print("PASS: Sistema funziona con dati reali")
        return True

    except Exception as e:
        print(f"FAIL: Errore sistema reale - {e}")
        return False

def main() -> bool:
    """
    Esegue la suite completa di test e genera report finale.

    Coordina l'esecuzione sequenziale di tutti i test, raccoglie i risultati
    e genera statistiche complete. Determina lo stato generale del sistema
    basato su success rate threshold (80%).

    Test Flow:
    1. Esecuzione tutti i test in sequenza
    2. Raccolta risultati e timing
    3. Calcolo statistiche (passed/failed/success rate)
    4. Determinazione stato finale sistema

    Returns:
        bool: True se sistema funzionale (success rate >= 80%), False altrimenti

    Note:
        - Success rate threshold: 80% per sistema funzionale
        - Logging completo per debugging
        - Tempo esecuzione tracked per performance monitoring
    """
    print("SIMPLE TEST SUITE - DIABETES READMISSION PREDICTION")
    print("=" * 60)

    tests = [
        ("Import Clustering", test_clustering_import),
        ("Import Integrato", test_integrated_system_import),
        ("Istanziazione Classe", test_integrated_system_class),
        ("Age Mapping", test_age_mapping),
        ("Dataset Disponibilità", test_dataset_availability),
        ("Funzionalità Reale", test_real_system_functionality)
    ]

    start_time = time.time()
    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))

    end_time = time.time()
    total_time = end_time - start_time

    # Sommario
    print("\n" + "=" * 60)
    print("RISULTATI FINALI")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:4} {test_name}")
        if success:
            passed += 1

    failed = len(results) - passed
    success_rate = passed / len(results) * 100

    print(f"\nSTATISTICHE:")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Total Time: {total_time:.1f}s")

    if success_rate >= 80:
        print(f"\nRISULTATO: SISTEMA FUNZIONALE")
        return True
    else:
        print(f"\nRISULTATO: SISTEMA CON PROBLEMI")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)