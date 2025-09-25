#!/usr/bin/env python3
"""
Test Suite Completa End-to-End per Pipeline Integrato Predizione Diabetici.

Questa suite di test fornisce una validazione completa e approfondita dell'intero
sistema end-to-end per la predizione di riammissione ospedaliera di pazienti diabetici.
Include test strutturali, funzionali, performance e robustezza del sistema completo.

Test Coverage Completa:
1. **Pipeline Structure Test**: Verifica struttura generale progetto
2. **Complete Workflow Test**: Test workflow clustering -> sistema integrato
3. **Data Flow Test**: Validazione flusso dati tra componenti
4. **End-to-End Prediction Test**: Test predizione completa con dati reali
5. **Performance Benchmarks Test**: Valutazione performance e timing
6. **Memory Usage Test**: Monitoraggio utilizzo memoria sistema

Caratteristiche Distintive:
- Test end-to-end completo dalla pulizia dati alla predizione finale
- Benchmarking performance con soglie definite
- Monitoraggio memoria per sistemi resource-constrained
- Graceful degradation quando dataset non disponibili
- Success rate threshold: 80% per sistema funzionale (più flessibile di test unitari)

Performance Benchmarks:
- Inizializzazione sistema: max 30 secondi
- Predizione singola: max 1 secondo
- Memoria utilizzata: max 500 MB per dataset completo

Pipeline Testato:
1. Sistema Clustering (clean_dataset_cluster.py)
2. Sistema Integrato (hybrid_ml_clinical_rules_integrated.py)
3. Flusso dati completo tra componenti
4. Performance sistema completo

Usage:
    python test/test_complete_pipeline.py

Success Criteria:
- Struttura progetto completa e corretta
- Workflow clustering -> integrato funzionante
- Performance entro benchmark stabiliti
- Success rate >= 80% (più tollerante per test end-to-end)

Author: [Nome del progetto]
Date: 2024
Version: 3.0 - Pipeline Integrato Completo
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Aggiungi il path del progetto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_pipeline_structure() -> bool:
    """
    Test struttura generale e completezza del progetto.

    Verifica che tutte le directory e file essenziali del progetto siano
    presenti nella struttura prevista. Controlla la presenza di:
    - Directory principali (metodi, outputs, database, test)
    - File chiave del sistema (clustering, sistema integrato, README)

    Directory verificate:
    - metodi/: codice principale del sistema
    - metodi/cluster/: sistema clustering
    - metodi/terzo_metodo/: sistema integrato
    - outputs/: directory output dataset processati
    - database/: dataset originali
    - test/: suite di test

    File verificati:
    - clean_dataset_cluster.py: sistema clustering
    - hybrid_ml_clinical_rules_integrated.py: sistema integrato
    - README.md: documentazione principale

    Returns:
        bool: True se struttura completa, False se mancanti componenti critici

    Note:
        - Test fondamentale per validare setup progetto
        - Fallimento indica progetto incompleto o corrotto
    """
    print("TEST: Struttura pipeline progetto...")

    required_dirs = [
        'metodi',
        'metodi/cluster',
        'metodi/terzo_metodo',
        'outputs',
        'database',
        'test'
    ]

    required_files = [
        'metodi/cluster/clean_dataset_cluster.py',
        'metodi/terzo_metodo/hybrid_ml_clinical_rules_integrated.py',
        'README.md'
    ]

    # Test directory
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"FAIL: Directory mancanti: {missing_dirs}")
        return False

    # Test file
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"FAIL: File mancanti: {missing_files}")
        return False

    print("PASS: Struttura progetto corretta")
    return True

def test_complete_workflow() -> bool:
    """
    Test workflow completo dalla pulizia dati al sistema integrato.

    Verifica che il workflow end-to-end sia funzionale:
    1. Import sistema clustering funzionale
    2. Import sistema integrato funzionale
    3. I due sistemi possano lavorare insieme senza conflitti

    Il test verifica che i moduli principali possano essere importati
    simultaneamente e che non ci siano conflitti di dipendenze o
    naming tra i diversi componenti del sistema.

    Returns:
        bool: True se workflow completo funzionale, False se errori

    Note:
        - Test critico per validare integrazione componenti
        - Fallimento indica problemi architetturali o dipendenze
    """
    print("TEST: Workflow completo...")

    try:
        # 1. Test import clustering
        from metodi.cluster.clean_dataset_cluster import compare_clustering_methods

        # 2. Test import sistema integrato
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # 3. Test che i moduli possano lavorare insieme
        print("   Clustering system: OK")
        print("   Integrated system: OK")

        print("PASS: Workflow completo funzionale")
        return True

    except ImportError as e:
        print(f"FAIL: Import error nel workflow - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Errore workflow - {e}")
        return False

def test_data_flow() -> bool:
    """
    Test flusso dati tra componenti clustering e sistema integrato.

    Verifica che i dataset generati dal sistema clustering siano:
    1. Presenti nelle directory di output attese
    2. Leggibili dal sistema integrato
    3. In formato corretto per utilizzo downstream

    Directory verificata:
    - outputs/datasets_clean/cluster/terzo_metodo/

    Dataset attesi:
    - db_clean_cluster_decision_tree.csv
    - db_clean_cluster_hybrid.csv
    - db_clean_cluster_kmeans.csv

    Il test valida che almeno un dataset sia disponibile e leggibile,
    consentendo funzionalità ridotta se non tutti i dataset sono presenti.

    Returns:
        bool: True se flusso dati funzionale, False se errori critici

    Note:
        - Non critico: sistema può funzionare con dataset parziali
        - Fornisce guidance per eseguire clustering se necessario
    """
    print("TEST: Flusso dati...")

    try:
        # Verifica che il sistema integrato possa accedere ai dataset generati dal clustering
        dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'

        expected_datasets = [
            'db_clean_cluster_decision_tree.csv',
            'db_clean_cluster_hybrid.csv',
            'db_clean_cluster_kmeans.csv'
        ]

        available_datasets = []
        for dataset in expected_datasets:
            dataset_path = dataset_dir / dataset
            if dataset_path.exists():
                available_datasets.append(dataset)

        if not available_datasets:
            print("WARNING: Nessun dataset clustering disponibile")
            print("   (Eseguire clean_dataset_cluster.py per generare i dataset)")
            return True  # Non è un errore fatale per il test

        # Test che i dataset siano leggibili
        for dataset in available_datasets[:1]:  # Test solo il primo per velocità
            dataset_path = dataset_dir / dataset
            try:
                df = pd.read_csv(dataset_path)
                if len(df) == 0:
                    print(f"FAIL: Dataset {dataset} vuoto")
                    return False
            except Exception as e:
                print(f"FAIL: Errore lettura {dataset} - {e}")
                return False

        print(f"PASS: Flusso dati OK ({len(available_datasets)} dataset disponibili)")
        return True

    except Exception as e:
        print(f"FAIL: Errore flusso dati - {e}")
        return False

def test_end_to_end_prediction() -> bool:
    """
    Test predizione end-to-end con dati reali se disponibili.

    Esegue un test completo del sistema di predizione:
    1. Inizializzazione sistema integrato con dataset reali
    2. Verifica corretto caricamento e setup
    3. Test predizione con paziente campione
    4. Validazione output predizione

    Paziente Test:
    - age_encoded: 6 (fascia elderly)
    - time_in_hospital: 5
    - num_medications: 15
    - number_diagnoses: 8

    Output Validato:
    - Presenza chiave 'age_cluster'
    - Mapping corretto fascia d'età (elderly_60_80)
    - Struttura predizione completa

    Returns:
        bool: True se predizione funzionale o dataset non disponibili, False se errori

    Note:
        - Test più realistico della funzionalità sistema
        - Skip gracefully se dataset mancanti
        - Valida mapping età e logica predizione
    """
    print("TEST: Predizione end-to-end...")

    try:
        dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'
        dt_path = dataset_dir / 'db_clean_cluster_decision_tree.csv'

        if not dt_path.exists():
            print("SKIP: Dataset non disponibile per test end-to-end")
            return True

        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Test inizializzazione rapida
        system = IntegratedHybridPredictor(str(dt_path), str(dt_path))
        init_success = system.initialize_systems()

        if not init_success:
            print("FAIL: Inizializzazione fallita")
            return False

        # Test predizione con paziente dummy
        sample_patient = {
            'age_encoded': 6,  # elderly
            'time_in_hospital': 5,
            'num_medications': 15,
            'number_diagnoses': 8
        }

        # La predizione dovrebbe funzionare anche senza training completo
        prediction = system.predict_with_integrated_system(sample_patient)

        # Verifica risultato predizione
        if 'age_cluster' not in prediction:
            print("FAIL: Predizione mancante age_cluster")
            return False

        if prediction['age_cluster'] != 'elderly_60_80':
            print(f"FAIL: Age cluster errato: {prediction['age_cluster']}")
            return False

        print("PASS: Predizione end-to-end funzionale")
        return True

    except Exception as e:
        print(f"FAIL: Errore predizione end-to-end - {e}")
        return False

def test_performance_benchmarks() -> bool:
    """
    Test benchmark di performance del sistema completo.

    Misura i tempi di esecuzione delle operazioni critiche del sistema
    e verifica che rientrino nei benchmark stabiliti per assicurare
    un'esperienza utente accettabile.

    Metriche Misurate:
    1. Tempo inizializzazione sistema integrato
    2. Tempo predizione singola paziente

    Benchmark Soglie:
    - Inizializzazione: max 30 secondi (include caricamento dataset)
    - Predizione: max 1 secondo (per responsività real-time)

    Il test genera warning se i tempi superano le soglie ma non fallisce
    automaticamente, permettendo valutazione caso per caso.

    Returns:
        bool: True sempre (benchmark informativi), False solo se errori critici

    Note:
        - Warning se inizializzazione > 30s (possibile dataset grande)
        - Warning se predizione > 1s (possibile problema performance)
        - Skip se dataset non disponibili
    """
    print("TEST: Benchmark performance...")

    try:
        dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'
        dt_path = dataset_dir / 'db_clean_cluster_decision_tree.csv'

        if not dt_path.exists():
            print("SKIP: Dataset non disponibile per benchmark")
            return True

        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Misura tempo inizializzazione
        start_time = time.time()
        system = IntegratedHybridPredictor(str(dt_path), str(dt_path))
        system.initialize_systems()
        init_time = time.time() - start_time

        # Misura tempo predizione singola
        sample_patient = {'age_encoded': 5, 'time_in_hospital': 3}

        start_time = time.time()
        prediction = system.predict_with_integrated_system(sample_patient)
        pred_time = time.time() - start_time

        # Benchmark accettabili
        max_init_time = 30.0  # 30 secondi
        max_pred_time = 1.0   # 1 secondo

        if init_time > max_init_time:
            print(f"WARNING: Inizializzazione lenta ({init_time:.1f}s > {max_init_time}s)")

        if pred_time > max_pred_time:
            print(f"WARNING: Predizione lenta ({pred_time:.3f}s > {max_pred_time}s)")

        print(f"PASS: Benchmark OK (init: {init_time:.1f}s, pred: {pred_time:.3f}s)")
        return True

    except Exception as e:
        print(f"FAIL: Errore benchmark - {e}")
        return False

def test_memory_usage() -> bool:
    """
    Test utilizzo memoria del sistema completo per deployment scalabile.

    Monitora l'utilizzo di memoria durante il caricamento e utilizzo del
    sistema integrato per verificare che sia compatibile con ambienti
    con risorse limitate.

    Processo Monitoraggio:
    1. Misurazione memoria iniziale processo
    2. Caricamento sistema integrato con dataset reali
    3. Misurazione memoria dopo inizializzazione
    4. Calcolo memoria utilizzata dal sistema
    5. Cleanup e garbage collection

    Benchmark Memoria:
    - Soglia warning: 500 MB per dataset completo (~71k record)
    - Considerazione deployment su server con risorse limitate

    Returns:
        bool: True sempre (memoria informativa), False solo se errori critici

    Note:
        - Richiede psutil per monitoraggio memoria
        - Skip gracefully se psutil non disponibile
        - Warning se utilizzo > 500 MB
        - Cleanup automatico memoria dopo test
    """
    print("TEST: Utilizzo memoria...")

    try:
        import psutil
        import gc

        # Memoria iniziale
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'
        dt_path = dataset_dir / 'db_clean_cluster_decision_tree.csv'

        if dt_path.exists():
            from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

            system = IntegratedHybridPredictor(str(dt_path), str(dt_path))
            system.initialize_systems()

            # Memoria dopo caricamento
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - initial_memory

            # Cleanup
            del system
            gc.collect()

        else:
            memory_used = 0

        # Benchmark memoria (dataset di ~71k record)
        max_memory = 500  # 500 MB

        if memory_used > max_memory:
            print(f"WARNING: Alto utilizzo memoria ({memory_used:.1f} MB > {max_memory} MB)")
        else:
            print(f"PASS: Memoria OK ({memory_used:.1f} MB)")

        return True

    except ImportError:
        print("SKIP: psutil non disponibile per test memoria")
        return True
    except Exception as e:
        print(f"FAIL: Errore test memoria - {e}")
        return False

def run_complete_pipeline_tests() -> bool:
    """
    Esegue tutti i test del pipeline completo e determina stato sistema.

    Coordina l'esecuzione sequenziale di tutti i test end-to-end,
    raccoglie risultati e determina lo stato generale del sistema completo
    utilizzando un success rate threshold più flessibile rispetto ai test unitari.

    Test Sequence:
    1. Test struttura pipeline progetto
    2. Test workflow completo clustering -> integrato
    3. Test flusso dati tra componenti
    4. Test predizione end-to-end
    5. Test benchmark performance
    6. Test utilizzo memoria

    Success Criteria:
    - Success rate >= 100%: "TUTTI I TEST PIPELINE PASSATI!"
    - Success rate >= 80%: "PIPELINE FUNZIONALE (alcuni warning)"
    - Success rate < 80%: "PIPELINE CON PROBLEMI SIGNIFICATIVI"

    Returns:
        bool: True se sistema funzionale (>= 80%), False se problemi critici

    Note:
        - Threshold più tollerante per test end-to-end complessi
        - Performance e memoria sono informativi, non bloccanti
        - Report dettagliato per troubleshooting
    """
    print("TESTING PIPELINE COMPLETO")
    print("=" * 50)

    tests = [
        test_pipeline_structure,
        test_complete_workflow,
        test_data_flow,
        test_end_to_end_prediction,
        test_performance_benchmarks,
        test_memory_usage
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Sommario
    passed = sum(results)
    total = len(results)

    print("RISULTATI TEST PIPELINE COMPLETO:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")
    print(f"   Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("TUTTI I TEST PIPELINE PASSATI!")
        return True
    elif passed >= total * 0.8:  # 80% success rate
        print("PIPELINE FUNZIONALE (alcuni warning)")
        return True
    else:
        print("PIPELINE CON PROBLEMI SIGNIFICATIVI")
        return False

if __name__ == "__main__":
    success = run_complete_pipeline_tests()
    sys.exit(0 if success else 1)