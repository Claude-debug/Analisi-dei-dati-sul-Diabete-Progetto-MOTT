#!/usr/bin/env python3
"""
Test Suite Specializzata per Sistema Integrato di Predizione Riammissione.

Questa suite di test è dedicata alla validazione completa del sistema integrato
IntegratedHybridPredictor, che combina clustering Age-Based (primario) con
Decision Tree (secondario) per predizioni di riammissione ospedaliera accurate.

Test Coverage Specializzato:
1. **Import System Test**: Verifica importabilità sistema integrato
2. **Class Instantiation Test**: Test istanziazione IntegratedHybridPredictor
3. **Age-Based Mapping Test**: Validazione mapping fasce d'età (componente primario)
4. **Prediction with Dummy Data Test**: Test predizione con dati mock
5. **System with Real Data Test**: Test con dataset reali (se disponibili)
6. **Main Function Test**: Verifica funzione main del sistema

Architettura Sistema Testato:
- **Age-Based Clustering (Primario)**: 4 fasce d'età fisse sempre utilizzate
- **Decision Tree (Secondario)**: Clustering avanzato dal vincitore competizione
- **Sistema Integrato**: Combina entrambi per predizioni robuste

Focus Specifico:
- Validazione corretta integrazione Age-Based + vincitore competizione
- Test mapping età per consistenza predizioni
- Verifica caricamento e inizializzazione con dataset reali
- Validazione struttura output predizioni

Caratteristiche:
- Test orientato specificamente al sistema integrato
- Mock data per test indipendenti da dataset esterni
- Graceful handling quando dataset non disponibili
- Verifica funzionalità critiche per deployment

Sistema Integrato Testato:
- IntegratedHybridPredictor class
- get_patient_age_cluster function
- initialize_systems method
- predict_with_integrated_system method

Usage:
    python test/test_integrated_system.py

Success Criteria:
- Tutti i test devono passare per sistema integrato funzionale
- Standard rigoroso per componente finale del sistema

Author: [Nome del progetto]
Date: 2024
Version: 3.0 - Sistema Integrato Age-Based + Competitive Winner
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Aggiungi il path del progetto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_integrated_system_import() -> bool:
    """
    Test importabilità completa del sistema integrato e componenti principali.

    Verifica che il modulo hybrid_ml_clinical_rules_integrated e le sue
    componenti principali (classe IntegratedHybridPredictor e funzione main)
    siano correttamente importabili senza errori di dipendenze.

    Componenti testati:
    - IntegratedHybridPredictor: classe principale del sistema integrato
    - main: funzione entry point per esecuzione standalone

    Returns:
        bool: True se import successful, False se errori di importazione

    Note:
        - Test critico per funzionamento sistema integrato
        - Fallimento indica problemi con dipendenze ML complesse
        - Essenziale per deployment del sistema finale
    """
    print("TEST: Import sistema integrato...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import (
            IntegratedHybridPredictor,
            main
        )
        print("PASS: Import sistema integrato")
        return True
    except ImportError as e:
        print(f"FAIL: Import error - {e}")
        return False

def test_integrated_system_class() -> bool:
    """
    Test istanziazione e inizializzazione della classe IntegratedHybridPredictor.

    Verifica che la classe principale possa essere istanziata con path dummy
    e che tutti gli attributi essenziali siano presenti e correttamente
    inizializzati per il funzionamento del sistema integrato.

    Attributi verificati:
    - age_based_path: percorso dataset clustering age-based
    - decision_tree_path: percorso dataset clustering vincitore competizione
    - age_based_df: DataFrame per clustering age-based
    - decision_tree_df: DataFrame per clustering secondario
    - age_models: modelli ML per ciascuna fascia d'età
    - age_features: features selezionate per fascia d'età

    Returns:
        bool: True se istanziazione corretta, False se attributi mancanti

    Note:
        - Test fondamentale per architettura sistema integrato
        - Usa path dummy per evitare dipendenze da file
        - Verifica solo costruttore, non caricamento dati
    """
    print("TEST: Istanziazione classe IntegratedHybridPredictor...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Test con path dummy (non deve caricare i file)
        system = IntegratedHybridPredictor('dummy1.csv', 'dummy2.csv')

        # Verifica attributi essenziali
        required_attributes = [
            'age_based_path', 'decision_tree_path',
            'age_based_df', 'decision_tree_df',
            'age_models', 'age_features'
        ]

        for attr in required_attributes:
            if not hasattr(system, attr):
                print(f"FAIL: Attributo {attr} mancante")
                return False

        print("PASS: Classe IntegratedHybridPredictor istanziata correttamente")
        return True

    except Exception as e:
        print(f"FAIL: Errore istanziazione classe - {e}")
        return False

def test_age_based_mapping() -> bool:
    """
    Test correttezza del mapping delle fasce d'età nel sistema integrato.

    Verifica che la funzione get_patient_age_cluster mappi correttamente
    i valori age_encoded alle 4 fasce d'età fisse del sistema Age-Based:

    Mapping testato:
    - age_encoded 1-3 -> young_0_40 (0-40 anni)
    - age_encoded 4-5 -> middle_40_60 (40-60 anni)
    - age_encoded 6-7 -> elderly_60_80 (60-80 anni)
    - age_encoded 8-10 -> very_elderly_80_100 (80+ anni)

    Returns:
        bool: True se mapping corretto per tutti i casi test, False se errori

    Note:
        - Test critico per correttezza predizioni Age-Based
        - Mapping deve essere consistente con clustering age-based
        - Errori causerebbero predizioni errate per fascia d'età
    """
    print("TEST: Age-based mapping...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        system = IntegratedHybridPredictor('dummy1.csv', 'dummy2.csv')

        # Test get_patient_age_cluster
        test_cases = [
            {'age_encoded': 2, 'expected': 'young_0_40'},
            {'age_encoded': 5, 'expected': 'middle_40_60'},
            {'age_encoded': 7, 'expected': 'elderly_60_80'},
            {'age_encoded': 9, 'expected': 'very_elderly_80_100'},
        ]

        for case in test_cases:
            result = system.get_patient_age_cluster(case)
            if result != case['expected']:
                print(f"FAIL: Age mapping error - age_encoded {case['age_encoded']} -> {result}, expected {case['expected']}")
                return False

        print("PASS: Age-based mapping funziona correttamente")
        return True

    except Exception as e:
        print(f"FAIL: Errore age-based mapping - {e}")
        return False

def test_prediction_with_dummy_data() -> bool:
    """
    Test funzionalità predizione del sistema integrato con dati simulati.

    Crea mock data che simula un dataset reale con distribuzione realistica
    e testa la funzione predict_with_integrated_system per verificare che:
    1. Il sistema accetti i dati di input correttamente
    2. Generi predizioni senza errori critici
    3. Restituisca output con struttura corretta

    Mock Data:
    - 200 pazienti simulati
    - 4 fasce d'età (encoded 2,5,7,9)
    - Features cliniche realistiche (time_in_hospital, num_medications, etc.)
    - Target binario per riammissione

    Output Validato:
    - age_cluster: fascia d'età assegnata
    - prediction: predizione riammissione
    - probability: probabilità predizione
    - confidence: livello confidenza

    Returns:
        bool: True se predizione funzionale, False se errori struttura output

    Note:
        - Test indipendente da dataset esterni
        - Simula inizializzazione con mock per test isolato
        - Verifica robustezza sistema anche senza training completo
    """
    print("TEST: Predizione con dati dummy...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Mock data che simula un dataset reale
        mock_data = {
            'age_encoded': [2, 5, 7, 9] * 50,
            'age': ['[20-30)', '[50-60)', '[70-80)', '[90-100)'] * 50,
            'readmitted_binary': [0, 1, 0, 1] * 50,
            'time_in_hospital': np.random.randint(1, 10, 200),
            'num_medications': np.random.randint(5, 25, 200),
            'number_diagnoses': np.random.randint(3, 15, 200),
            'number_inpatient': np.random.randint(0, 3, 200),
            'number_emergency': np.random.randint(0, 2, 200),
            'number_outpatient': np.random.randint(0, 5, 200)
        }

        mock_df = pd.DataFrame(mock_data)

        system = IntegratedHybridPredictor('dummy1.csv', 'dummy2.csv')

        # Simula inizializzazione con mock data
        system.age_based_df = mock_df
        system.decision_tree_df = mock_df
        system.df = mock_df

        # Test che la predizione non crashi (anche senza modelli trainati)
        sample_patient = {
            'age_encoded': 5,
            'time_in_hospital': 3,
            'num_medications': 10
        }

        prediction = system.predict_with_integrated_system(sample_patient)

        # Verifica struttura risposta
        required_keys = ['age_cluster', 'prediction', 'probability', 'confidence']
        for key in required_keys:
            if key not in prediction:
                print(f"FAIL: Chiave {key} mancante nella predizione")
                return False

        print("PASS: Predizione con dati dummy funziona")
        return True

    except Exception as e:
        print(f"FAIL: Errore predizione dummy - {e}")
        return False

def test_system_with_real_data() -> bool:
    """
    Test sistema integrato con dataset reali generati dal clustering.

    Se i dataset sono disponibili nella directory outputs, esegue test
    con dati reali per validare:
    1. Caricamento corretto dei dataset clustered
    2. Inizializzazione completa del sistema senza errori
    3. Popolamento corretto dei DataFrame interni

    Dataset utilizzati (se disponibili):
    - db_clean_cluster_decision_tree.csv (per entrambi i path in test)

    Validazioni:
    - initialize_systems() completa con successo
    - age_based_df caricato e popolato (len > 0)
    - decision_tree_df caricato e popolato (len > 0)

    Returns:
        bool: True se sistema funziona con dati reali o dataset non disponibili,
              False se errori di inizializzazione

    Note:
        - Test più realistico per deployment
        - Skip gracefully se dataset mancanti
        - Importante per validare pipeline clustering -> sistema integrato
    """
    print("TEST: Sistema con dati reali...")

    try:
        dataset_dir = project_root / 'outputs' / 'datasets_clean' / 'cluster' / 'terzo_metodo'
        dt_path = dataset_dir / 'db_clean_cluster_decision_tree.csv'

        if not dt_path.exists():
            print("SKIP: Dataset Decision Tree non disponibile")
            return True

        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import IntegratedHybridPredictor

        # Test con dataset reali
        system = IntegratedHybridPredictor(str(dt_path), str(dt_path))

        # Test solo inizializzazione (non training completo per velocità)
        result = system.initialize_systems()

        if not result:
            print("FAIL: Inizializzazione sistema fallita")
            return False

        # Verifica che i dataframe siano caricati
        if system.age_based_df is None or len(system.age_based_df) == 0:
            print("FAIL: Dataset age-based non caricato")
            return False

        if system.decision_tree_df is None or len(system.decision_tree_df) == 0:
            print("FAIL: Dataset decision tree non caricato")
            return False

        print("PASS: Sistema con dati reali inizializzato")
        return True

    except Exception as e:
        print(f"FAIL: Errore sistema dati reali - {e}")
        return False

def test_main_function() -> bool:
    """
    Test disponibilità e accessibilità della funzione main del sistema integrato.

    Verifica che la funzione main sia:
    1. Correttamente importabile dal modulo
    2. Callable (eseguibile come funzione)
    3. Disponibile per esecuzione standalone del sistema

    La funzione main è l'entry point per l'utilizzo del sistema integrato
    come script standalone, quindi la sua presenza e accessibilità sono
    critiche per deployment e utilizzo del sistema.

    Returns:
        bool: True se funzione main disponibile e callable, False se errori

    Note:
        - Test essenziale per utilizzo standalone del sistema
        - Non esegue main() per evitare side effects nei test
        - Verifica solo disponibilità e callability
    """
    print("TEST: Funzione main sistema integrato...")

    try:
        from metodi.terzo_metodo.hybrid_ml_clinical_rules_integrated import main

        # Test che la funzione main esista e sia callable
        if not callable(main):
            print("FAIL: Funzione main non callable")
            return False

        print("PASS: Funzione main disponibile")
        return True

    except ImportError:
        print("FAIL: Funzione main non importabile")
        return False
    except Exception as e:
        print(f"FAIL: Errore funzione main - {e}")
        return False

def run_integrated_system_tests() -> bool:
    """
    Esegue tutti i test specializzati del sistema integrato con standard rigoroso.

    Coordina l'esecuzione sequenziale di tutti i test specifici per il sistema
    integrato e determina lo stato del componente finale del sistema. Utilizza
    standard rigoroso (100% success) per il componente critico finale.

    Test Sequence:
    1. Test import sistema integrato
    2. Test istanziazione classe IntegratedHybridPredictor
    3. Test mapping fasce d'età Age-Based
    4. Test predizione con dati dummy
    5. Test sistema con dati reali
    6. Test funzione main

    Success Criteria:
    - Tutti i test devono passare (100% success rate)
    - Standard più rigoroso rispetto a test end-to-end
    - Sistema integrato è componente critico finale

    Returns:
        bool: True se tutti i test passano, False se qualsiasi fallimento

    Note:
        - Standard rigoroso per componente finale sistema
        - Fallimento indica problemi critici nel sistema integrato
        - Report dettagliato per troubleshooting problemi
    """
    print("TESTING SISTEMA INTEGRATO")
    print("=" * 50)

    tests = [
        test_integrated_system_import,
        test_integrated_system_class,
        test_age_based_mapping,
        test_prediction_with_dummy_data,
        test_system_with_real_data,
        test_main_function
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Sommario
    passed = sum(results)
    total = len(results)

    print("RISULTATI TEST SISTEMA INTEGRATO:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")

    if passed == total:
        print("TUTTI I TEST SISTEMA INTEGRATO PASSATI!")
        return True
    else:
        print("ALCUNI TEST SISTEMA INTEGRATO FALLITI")
        return False

if __name__ == "__main__":
    success = run_integrated_system_tests()
    sys.exit(0 if success else 1)