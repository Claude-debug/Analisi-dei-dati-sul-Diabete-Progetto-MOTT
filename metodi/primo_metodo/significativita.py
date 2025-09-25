"""
Analisi Significatività Statistica delle Features per Predizione Riammissione Diabetici

Questo modulo esegue un'analisi statistica completa delle features per identificare
le variabili più significative nella predizione della riammissione ospedaliera di
pazienti diabetici. Utilizza diversi test statistici appropriati per variabili
binarie e continue.

Metodologie Implementate:
- Test Chi-quadrato per variabili binarie
- Test Shapiro-Wilk per normalità
- T-test per variabili continue normali
- Test Mann-Whitney U per variabili continue non normali
- Mutual Information per valutazione generale
- Effect Size (Cramér's V) per variabili binarie

Output:
- Analisi completa in CSV
- Lista features significative per uso in ML
- Report statistico dettagliato

Autore: Progetto MOTT - Predizione Riammissione Diabetici
Data: 2024
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def load_dataset(filepath: str) -> tuple[pd.DataFrame, list, list]:
    """
    Carica il dataset e separa le variabili in binarie e continue.

    Parameters:
    -----------
    filepath : str
        Percorso al file CSV del dataset

    Returns:
    --------
    tuple[pd.DataFrame, list, list]
        Dataset caricato, lista features binarie, lista features continue
    """
    # Caricamento del dataset pulito e filtrato
    df = pd.read_csv(filepath, sep=';')

    # Identificazione automatica delle variabili binarie (2 valori unici)
    binary_features = [col for col in df.columns if df[col].nunique() == 2 and col != "readmitted_NO"]

    # Tutte le altre variabili sono considerate continue (escluso il target)
    continuous_features = [col for col in df.columns if col not in binary_features and col != "readmitted_NO"]

    return df, binary_features, continuous_features

def analyze_binary_features(df: pd.DataFrame, binary_features: list, target: str = "readmitted_NO") -> dict:
    """
    Analizza la significatività delle variabili binarie usando il test Chi-quadrato.

    Il test Chi-quadrato verifica l'indipendenza tra due variabili categoriali.
    H0: Le variabili sono indipendenti (non c'è associazione)
    H1: Le variabili sono associate

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset contenente le features e il target
    binary_features : list
        Lista delle features binarie da analizzare
    target : str, default="readmitted_NO"
        Nome della variabile target

    Returns:
    --------
    dict
        Dizionario con p-value per ogni feature binaria, ordinato per significatività
    """
    binary_significance = {}

    for feature in binary_features:
        # Creazione tabella di contingenza (crosstab)
        contingency_table = pd.crosstab(df[feature], df[target])

        # Test Chi-quadrato: restituisce chi2, p-value, dof, expected_frequencies
        chi2, p, _, _ = chi2_contingency(contingency_table)
        binary_significance[feature] = p

    # Ordinamento per p-value crescente (più significativo = p-value più basso)
    return dict(sorted(binary_significance.items(), key=lambda x: x[1]))

def test_normality(df: pd.DataFrame, continuous_features: list) -> dict:
    """
    Testa la normalità delle distribuzioni usando il test Shapiro-Wilk.

    Il test Shapiro-Wilk è appropriato per campioni < 5000.
    H0: I dati seguono una distribuzione normale
    H1: I dati non seguono una distribuzione normale

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset contenente le features continue
    continuous_features : list
        Lista delle features continue da testare

    Returns:
    --------
    dict
        Dizionario con p-value del test di normalità per ogni feature
    """
    normality_results = {}

    for feature in continuous_features:
        # Test Shapiro-Wilk: restituisce statistica e p-value
        stat, p = shapiro(df[feature])
        normality_results[feature] = p

    return normality_results


def analyze_continuous_features(df: pd.DataFrame, continuous_features: list,
                               normality_results: dict, target: str = "readmitted_NO") -> dict:
    """
    Analizza la significatività delle variabili continue usando test appropriati.

    Sceglie automaticamente il test statistico basandosi sulla normalità:
    - T-test per dati normali (p > 0.05 nel test Shapiro-Wilk)
    - Mann-Whitney U per dati non normali (p ≤ 0.05 nel test Shapiro-Wilk)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset contenente le features e il target
    continuous_features : list
        Lista delle features continue da analizzare
    normality_results : dict
        Risultati del test di normalità
    target : str, default="readmitted_NO"
        Nome della variabile target

    Returns:
    --------
    dict
        Dizionario con p-value per ogni feature continua, ordinato per significatività
    """
    continuous_significance = {}

    for feature in continuous_features:
        # Separazione dei gruppi: riammessi (1) vs non riammessi (0)
        group_readmitted = df[df[target] == 1][feature]
        group_not_readmitted = df[df[target] == 0][feature]

        if normality_results[feature] > 0.05:
            # Dati normali: uso T-test per campioni indipendenti
            # H0: Le medie dei due gruppi sono uguali
            stat, p = ttest_ind(group_readmitted, group_not_readmitted)
        else:
            # Dati non normali: uso Mann-Whitney U (test non parametrico)
            # H0: Le distribuzioni dei due gruppi sono identiche
            stat, p = mannwhitneyu(group_readmitted, group_not_readmitted)

        continuous_significance[feature] = p

    # Ordinamento per p-value crescente (più significativo = p-value più basso)
    return dict(sorted(continuous_significance.items(), key=lambda x: x[1]))

def combine_significance_results(binary_results: dict, continuous_results: dict) -> dict:
    """
    Combina i risultati di significatività di variabili binarie e continue.

    Parameters:
    -----------
    binary_results : dict
        Risultati significatività per features binarie
    continuous_results : dict
        Risultati significatività per features continue

    Returns:
    --------
    dict
        Dizionario unificato ordinato per significatività (p-value crescente)
    """
    # Unione dei due dizionari
    all_significance = {**binary_results, **continuous_results}

    # Ordinamento globale per p-value crescente
    return dict(sorted(all_significance.items(), key=lambda x: x[1]))

def calculate_mutual_information(df: pd.DataFrame, target: str = "readmitted_NO") -> dict:
    """
    Calcola la Mutual Information tra le features e il target.

    La Mutual Information misura la dipendenza statistica tra due variabili.
    Valori più alti indicano maggiore dipendenza (più informazione condivisa).
    È utile per rilevare relazioni non lineari che i test tradizionali potrebbero perdere.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset completo
    target : str, default="readmitted_NO"
        Nome della variabile target

    Returns:
    --------
    dict
        Dizionario con score di Mutual Information per ogni feature,
        ordinato per score decrescente
    """
    # Separazione features (X) e target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Calcolo Mutual Information con seed fisso per riproducibilità
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Creazione dizionario feature -> score
    mi_results = dict(zip(X.columns, mi_scores))

    # Ordinamento per score decrescente (più informativo = score più alto)
    return dict(sorted(mi_results.items(), key=lambda x: x[1], reverse=True))


def calculate_effect_sizes(df: pd.DataFrame, binary_features: list, target: str = "readmitted_NO") -> dict:
    """
    Calcola l'Effect Size (Cramér's V) per le features binarie.

    Cramér's V misura l'intensità dell'associazione tra due variabili categoriali.
    Range: [0, 1] dove:
    - 0 = nessuna associazione
    - 0.1 = associazione debole
    - 0.3 = associazione moderata
    - 0.5 = associazione forte
    - 1 = associazione perfetta

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset contenente le features e il target
    binary_features : list
        Lista delle features binarie
    target : str, default="readmitted_NO"
        Nome della variabile target

    Returns:
    --------
    dict
        Dizionario con Cramér's V per ogni feature binaria
    """
    effect_sizes = {}

    for feature in binary_features:
        # Tabella di contingenza
        contingency_table = pd.crosstab(df[feature], df[target])

        # Calcolo Chi-quadrato
        chi2, _, _, _ = chi2_contingency(contingency_table)

        # Calcolo Cramér's V
        n = contingency_table.sum().sum()  # Numero totale osservazioni
        min_dimension = min(contingency_table.shape) - 1  # Gradi di libertà
        cramers_v = np.sqrt(chi2 / (n * min_dimension))

        effect_sizes[feature] = cramers_v

    return effect_sizes

def create_comprehensive_results(features: list, binary_features: list, continuous_features: list,
                                mi_results: dict, binary_significance: dict,
                                continuous_significance: dict, normality_results: dict,
                                effect_sizes: dict) -> pd.DataFrame:
    """
    Crea un DataFrame completo con tutti i risultati dell'analisi.

    Combina tutti i risultati statistici in un'unica tabella per confronto e ranking.

    Parameters:
    -----------
    features : list
        Lista di tutte le features
    binary_features : list
        Lista delle features binarie
    continuous_features : list
        Lista delle features continue
    mi_results : dict
        Risultati Mutual Information
    binary_significance : dict
        Risultati significatività features binarie
    continuous_significance : dict
        Risultati significatività features continue
    normality_results : dict
        Risultati test normalità
    effect_sizes : dict
        Effect sizes per features binarie

    Returns:
    --------
    pd.DataFrame
        DataFrame completo con tutti i risultati per ogni feature
    """
    combined_results = []

    for feature in features:
        # Informazioni base per ogni feature
        result = {
            'feature': feature,
            'type': 'binary' if feature in binary_features else 'continuous',
            'mutual_info': mi_results.get(feature, 0)
        }

        # Metriche specifiche per features binarie
        if feature in binary_features:
            result['chi2_pvalue'] = binary_significance.get(feature, 1)
            result['cramers_v'] = effect_sizes.get(feature, 0)
            result['test_used'] = 'Chi-quadrato'
        # Metriche specifiche per features continue
        else:
            result['statistical_pvalue'] = continuous_significance.get(feature, 1)
            result['normality_p'] = normality_results.get(feature, 1)
            # Determinazione del test usato basandosi sulla normalità
            result['test_used'] = 'T-test' if normality_results.get(feature, 0) > 0.05 else 'Mann-Whitney'

        combined_results.append(result)

    # Conversione in DataFrame per analisi facilitata
    return pd.DataFrame(combined_results)

def save_results(results_df: pd.DataFrame, output_dir: str = 'outputs/dataset_pvalue') -> None:
    """
    Salva i risultati dell'analisi in file CSV.

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame con tutti i risultati dell'analisi
    output_dir : str
        Directory dove salvare i risultati
    """
    # Creazione directory se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Salvataggio analisi completa
    filepath = os.path.join(output_dir, 'analisi_significativita_completa.csv')
    results_df.to_csv(filepath, index=False)
    print(f"Analisi completa salvata in: {filepath}")

def select_significant_features(binary_significance: dict, continuous_significance: dict,
                               binary_features: list, alpha: float = 0.05) -> tuple[list, list]:
    """
    Seleziona automaticamente le features statisticamente significative.

    Applica il threshold di significatività (di default p < 0.05) per identificare
    le features che mostrano associazione statisticamente significativa con il target.

    Parameters:
    -----------
    binary_significance : dict
        P-value per features binarie
    continuous_significance : dict
        P-value per features continue
    binary_features : list
        Lista features binarie per classificazione
    alpha : float, default=0.05
        Livello di significatività (soglia p-value)

    Returns:
    --------
    tuple[list, list]
        Lista features significative e lista con (feature, p-value) ordinata
    """
    significant_features = []

    # Collezione features binarie significative
    for feature, p_value in binary_significance.items():
        if p_value < alpha:
            significant_features.append(feature)

    # Collezione features continue significative
    for feature, p_value in continuous_significance.items():
        if p_value < alpha:
            significant_features.append(feature)

    # Creazione lista con p-value per ordinamento
    significant_with_pvalues = []
    for feature in significant_features:
        if feature in binary_significance:
            significant_with_pvalues.append((feature, binary_significance[feature]))
        else:
            significant_with_pvalues.append((feature, continuous_significance[feature]))

    # Ordinamento per p-value crescente (più significativo primo)
    significant_with_pvalues.sort(key=lambda x: x[1])

    return significant_features, significant_with_pvalues


def save_selected_features(significant_with_pvalues: list, binary_features: list,
                          output_dir: str = 'outputs/dataset_pvalue') -> None:
    """
    Salva la lista delle features significative per uso in machine learning.

    Crea un file di testo formattato per essere facilmente importato
    nel modulo di selezione features per ML.

    Parameters:
    -----------
    significant_with_pvalues : list
        Lista di tuple (feature, p-value) ordinata per significatività
    binary_features : list
        Lista features binarie per classificazione tipo
    output_dir : str
        Directory dove salvare il file
    """
    filepath = os.path.join(output_dir, 'selected_features.txt')

    with open(filepath, 'w') as f:
        # Header del file
        f.write("# FEATURES SIGNIFICATIVE (p < 0.05) per selezione_features_ml.py\n")
        f.write("# Ordinate per significatività (p-value crescente)\n")
        f.write("# Formato: 'nome_feature',  # rank. p=valore (tipo)\n\n")

        # Lista features con metadati
        for i, (feature, p_value) in enumerate(significant_with_pvalues, 1):
            feat_type = 'binaria' if feature in binary_features else 'continua'
            f.write(f"'{feature}',  # {i}. p={p_value:.2e} ({feat_type})\n")

    print(f"Features significative salvate in: {filepath}")

def print_summary_report(df_shape: tuple, binary_features: list, continuous_features: list,
                        significant_features: list, all_significance: dict) -> None:
    """
    Stampa un report di riepilogo dell'analisi di significatività.

    Parameters:
    -----------
    df_shape : tuple
        Dimensioni del dataset (righe, colonne)
    binary_features : list
        Lista features binarie
    continuous_features : list
        Lista features continue
    significant_features : list
        Lista features significative
    all_significance : dict
        Tutti i risultati di significatività
    """
    print("\n" + "="*60)
    print("           REPORT ANALISI SIGNIFICATIVITÀ")
    print("="*60)

    # Statistiche dataset
    print(f"\nDataset caricato: {df_shape[0]:,} righe × {df_shape[1]:,} colonne")
    print(f"Features binarie: {len(binary_features):,}")
    print(f"Features continue: {len(continuous_features):,}")
    print(f"Target: readmitted_NO")

    # Risultati significatività
    print(f"\nFeatures significative trovate: {len(significant_features):,}")
    print(f"Percentuale significative: {len(significant_features)/len(all_significance)*100:.1f}%")

    # Top 5 features più significative
    print(f"\nTOP 5 FEATURES PIÙ SIGNIFICATIVE:")
    print("-" * 45)
    for i, (feature, p_value) in enumerate(list(all_significance.items())[:5], 1):
        feat_type = 'binaria' if feature in binary_features else 'continua'
        print(f"{i:2d}. {feature:<25} p={p_value:.2e} ({feat_type})")

    print("\nFile di output generati:")
    print("- analisi_significativita_completa.csv (analisi completa)")
    print("- selected_features.txt (features per ML)")
    print("\n" + "="*60)


def main() -> None:
    """
    Funzione principale che esegue l'intera pipeline di analisi statisticsa.

    Workflow:
    1. Caricamento dataset
    2. Identificazione tipo features (binarie/continue)
    3. Test statistici appropriati per ogni tipo
    4. Calcolo Mutual Information e Effect Size
    5. Combinazione risultati
    6. Selezione features significative
    7. Salvataggio risultati
    8. Report finale
    """
    # 1. Caricamento e preparazione dati
    df, binary_features, continuous_features = load_dataset(
        "outputs/datasets_clean/second_clean/diabetes_clean_filtered.csv"
    )

    # 2. Analisi statistiche
    print("Esecuzione test statistici...")

    # Test per variabili binarie (Chi-quadrato)
    binary_significance = analyze_binary_features(df, binary_features)

    # Test per variabili continue (normalità + test appropriato)
    normality_results = test_normality(df, continuous_features)
    continuous_significance = analyze_continuous_features(
        df, continuous_features, normality_results
    )

    # Combinazione risultati
    all_significance = combine_significance_results(binary_significance, continuous_significance)

    # 3. Metodologie aggiuntive
    print("Calcolo Mutual Information e Effect Size...")

    # Mutual Information per tutte le features
    mi_results = calculate_mutual_information(df)

    # Effect Size per features binarie
    effect_sizes = calculate_effect_sizes(df, binary_features)

    # 4. Creazione risultati completi
    all_features = binary_features + continuous_features
    results_df = create_comprehensive_results(
        all_features, binary_features, continuous_features,
        mi_results, binary_significance, continuous_significance,
        normality_results, effect_sizes
    )

    # 5. Selezione features significative
    significant_features, significant_with_pvalues = select_significant_features(
        binary_significance, continuous_significance, binary_features
    )

    # 6. Salvataggio risultati
    print("Salvataggio risultati...")
    save_results(results_df)
    save_selected_features(significant_with_pvalues, binary_features)

    # 7. Report finale
    print_summary_report(
        df.shape, binary_features, continuous_features,
        significant_features, all_significance
    )


if __name__ == "__main__":
    main()
