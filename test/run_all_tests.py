#!/usr/bin/env python3
"""
Test Runner - Esegue tutti i test del progetto
"""

import sys
import os
import time
from pathlib import Path

# Aggiungi il path del progetto
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_test_suite(test_name, test_function):
    """Esegue una suite di test con timing"""
    print(f"\nESECUZIONE: {test_name}")
    print("=" * 60)

    start_time = time.time()
    try:
        success = test_function()
        end_time = time.time()
        duration = end_time - start_time

        if success:
            print(f"PASS: {test_name} COMPLETATO ({duration:.1f}s)")
            return True
        else:
            print(f"FAIL: {test_name} FALLITO ({duration:.1f}s)")
            return False

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"ERROR: {test_name} ERRORE: {e} ({duration:.1f}s)")
        return False

def main():
    """Main test runner"""
    print("TEST RUNNER - PROGETTO DIABETES READMISSION PREDICTION")
    print("=" * 70)
    print("Eseguendo suite completa di test...")

    # Import delle suite di test
    try:
        from test_clustering_system import run_clustering_tests
        from test_integrated_system import run_integrated_system_tests
        from test_complete_pipeline import run_complete_pipeline_tests
    except ImportError as e:
        print(f"ERRORE IMPORT TEST: {e}")
        sys.exit(1)

    # Definizione test suite
    test_suites = [
        ("CLUSTERING SYSTEM", run_clustering_tests),
        ("INTEGRATED SYSTEM", run_integrated_system_tests),
        ("COMPLETE PIPELINE", run_complete_pipeline_tests)
    ]

    # Esecuzione test
    overall_start = time.time()
    results = []

    for test_name, test_func in test_suites:
        result = run_test_suite(test_name, test_func)
        results.append((test_name, result))

    overall_end = time.time()
    total_duration = overall_end - overall_start

    # Sommario finale
    print("\n" + "=" * 70)
    print("SOMMARIO FINALE TEST")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nSTATISTICHE:")
    print(f"   Passed: {passed}/{len(results)}")
    print(f"   Failed: {failed}/{len(results)}")
    print(f"   Success Rate: {passed/len(results)*100:.1f}%")
    print(f"   Total Time: {total_duration:.1f}s")

    # Valutazione finale
    if passed == len(results):
        print("\nTUTTI I TEST PASSATI! PROGETTO FUNZIONALE AL 100%")
        exit_code = 0
    elif passed >= len(results) * 0.8:
        print("\nPROGETTO SOSTANZIALMENTE FUNZIONALE (>80% test passati)")
        exit_code = 0
    else:
        print("\nPROGETTO CON PROBLEMI SIGNIFICATIVI (<80% test passati)")
        exit_code = 1

    # Raccomandazioni
    if failed > 0:
        print("\nRACCOMANDAZIONI:")
        if any("CLUSTERING" in name for name, success in results if not success):
            print("   - Verificare sistema di clustering: clean_dataset_cluster.py")
        if any("INTEGRATED" in name for name, success in results if not success):
            print("   - Verificare sistema integrato: hybrid_ml_clinical_rules_integrated.py")
        if any("PIPELINE" in name for name, success in results if not success):
            print("   - Verificare disponibilità dataset e struttura progetto")
    else:
        print("\nPROGETTO PRONTO PER:")
        print("   - Esecuzione clustering comparison")
        print("   - Training sistema integrato")
        print("   - Deployment produzione")

    print("\n" + "=" * 70)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()