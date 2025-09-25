# Test Suite - Diabetes Readmission Prediction Pipeline

Questa cartella contiene una suite completa di test per verificare il funzionamento dell'intero pipeline di predizione riammissione ospedaliera per pazienti diabetici.

## Struttura Test

### Suite di Test Disponibili

1. **`test_clustering_system.py`** - Test Sistema Clustering
   - Verifica import e sintassi del sistema di clustering
   - Test comparazione 3 metodi + Age-Based fisso
   - Controllo disponibilità dataset

2. **`test_integrated_system.py`** - Test Sistema Integrato
   - Test import e istanziazione IntegratedHybridPredictor
   - Verifica funzionalità Age-Based mapping
   - Test predizione con dati dummy e reali

3. **`test_complete_pipeline.py`** - Test Pipeline Completo
   - Test struttura progetto end-to-end
   - Verifica workflow clustering → sistema integrato
   - Benchmark performance e memoria

4. **`run_all_tests.py`** - Test Runner Principale
   - Esegue tutte le suite in sequenza
   - Fornisce sommario completo con statistiche
   - Raccomandazioni basate sui risultati

## Come Eseguire i Test

### Esecuzione Completa (Raccomandato)
```bash
# Esegui tutti i test
cd test
python run_all_tests.py
```

### Esecuzione Singola Suite
```bash
# Solo test clustering
python test_clustering_system.py

# Solo test sistema integrato
python test_integrated_system.py

# Solo test pipeline completo
python test_complete_pipeline.py
```

## Output Test

### Formato Risultati
Ogni test produce output strutturato:
- **TEST**: Descrizione del test
- **PASS**: Test superato
- **FAIL**: Test fallito
- **WARNING**: Test con avvisi
- **SKIP**: Test saltato (es. dataset mancanti)

### Esempio Output
```
TEST: Import sistema clustering...
PASS: Import clustering system

TEST: Sintassi funzioni clustering...
PASS: Tutte le funzioni clustering presenti

RISULTATI TEST CLUSTERING:
   Passed: 4/4
   Failed: 0/4
```

## Test Coverage

### Sistema Clustering
- Import moduli e funzioni
- Sintassi e struttura codice
- Disponibilità dataset
- Logica comparazione metodi

### Sistema Integrato
- Istanziazione classe
- Age-based mapping
- Predizione con dati dummy
- Integrazione con dataset reali
- Funzione main

### Pipeline Completo
- Struttura progetto
- Workflow end-to-end
- Flusso dati tra componenti
- Predizione end-to-end
- Benchmark performance
- Utilizzo memoria

## Prerequisiti

### Dipendenze Python
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
psutil>=5.8.0  # Per test memoria (opzionale)
```

### Struttura Dataset
I test funzionano anche senza dataset, ma per test completi servono:
```
outputs/datasets_clean/cluster/terzo_metodo/
├── db_clean_cluster_decision_tree.csv
├── db_clean_cluster_hybrid.csv
└── db_clean_cluster_kmeans.csv
```

## Interpretazione Risultati

### Success Rate
- **100%**: Sistema completamente funzionale
- **80-99%**: Sistema sostanzialmente funzionale con warning
- **<80%**: Sistema con problemi significativi

### Raccomandazioni Automatiche
Il test runner fornisce raccomandazioni specifiche basate sui fallimenti:
- Problemi clustering → Verificare `clean_dataset_cluster.py`
- Problemi integrato → Verificare `hybrid_ml_clinical_rules_integrated.py`
- Problemi pipeline → Verificare dataset e struttura

## Troubleshooting

### Errori Comuni

**ImportError**: Moduli non trovati
```bash
# Soluzione: Eseguire dalla directory test
cd test
python run_all_tests.py
```

**FileNotFoundError**: Dataset mancanti
```bash
# Soluzione: Generare dataset prima
python metodi/cluster/clean_dataset_cluster.py
```

**MemoryError**: Insufficiente memoria
```bash
# Soluzione: Chiudere altre applicazioni o usare subset dati
```

### Test in Modalità Debug
```python
# Aggiungi debug in run_all_tests.py
import traceback

try:
    # test code
except Exception as e:
    traceback.print_exc()
```

## Estensione Test

### Aggiungere Nuovi Test
1. Creare nuovo file `test_nome_feature.py`
2. Implementare funzione `run_nome_feature_tests()`
3. Aggiungere import in `run_all_tests.py`
4. Aggiungere alla lista `test_suites`

### Template Test
```python
def test_new_feature():
    """Test nuova funzionalità"""
    print("TEST: Descrizione test...")

    try:
        # Logica test
        assert condition, "Messaggio errore"
        print("PASS: Test superato")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False
```

---

**Versione**: 1.0.0 | **Ultima Modifica**: Settembre 2025 | **Coverage**: Sistema Completo