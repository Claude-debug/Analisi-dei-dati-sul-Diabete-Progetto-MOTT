import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# 1. Caricare il dataset
df = pd.read_csv("outputs/datasets_clean/second_clean/diabetes_clean_filtered.csv", sep=';')

# 2. Separare le variabili binarie e continue
binary_features = [col for col in df.columns if df[col].nunique() == 2 and col != "readmitted_NO"]
continuous_features = [col for col in df.columns if col not in binary_features and col != "readmitted_NO"]

# 3. Test per le variabili binarie (Chi-quadrato)
binary_significance = {}
for feature in binary_features:
    contingency_table = pd.crosstab(df[feature], df["readmitted_NO"])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    binary_significance[feature] = p

# Ordinare le feature binarie per significatività
binary_significance = dict(sorted(binary_significance.items(), key=lambda x: x[1]))

# 4. Test per le variabili continue
# Verifica della normalità (Shapiro-Wilk)
normality_results = {}
for feature in continuous_features:
    stat, p = shapiro(df[feature])
    normality_results[feature] = p

# Test statistico appropriato per le variabili continue
continuous_significance = {}
for feature in continuous_features:
    if normality_results[feature] > 0.05:  # Distribuzione normale
        stat, p = ttest_ind(df[df["readmitted_NO"] == 1][feature],
                            df[df["readmitted_NO"] == 0][feature])
    else:  # Distribuzione non normale
        stat, p = mannwhitneyu(df[df["readmitted_NO"] == 1][feature],
                               df[df["readmitted_NO"] == 0][feature])
    continuous_significance[feature] = p

# Ordinare le feature continue per significatività
continuous_significance = dict(sorted(continuous_significance.items(), key=lambda x: x[1]))

# 5. Combinare i risultati
all_significance = {**binary_significance, **continuous_significance}
all_significance = dict(sorted(all_significance.items(), key=lambda x: x[1]))

# 6. METODOLOGIE AGGIUNTIVE

print(f"\nDataset caricato: {df.shape}")
print(f"Features binarie: {len(binary_features)}")
print(f"Features continue: {len(continuous_features)}")
print(f"Target: readmitted_NO")

# 6.1 Mutual Information per tutte le features
print("\n=== MUTUAL INFORMATION ===")
X = df.drop(columns=['readmitted_NO'])
y = df['readmitted_NO']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_results = dict(zip(X.columns, mi_scores))
mi_results = dict(sorted(mi_results.items(), key=lambda x: x[1], reverse=True))

# Correlazioni rimosse - ridondanti per selezione features

# Test ML rimossi - non necessari per selezione features

# Test Kolmogorov-Smirnov rimosso - ridondante con Mann-Whitney

# 6.6 Effect Size per features binarie (Cramér's V)
print("\n=== EFFECT SIZE (Cramér's V per features binarie) ===")
effect_sizes = {}
for feature in binary_features:
    contingency_table = pd.crosstab(df[feature], df['readmitted_NO'])
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    effect_sizes[feature] = cramers_v

# 7. COMBINARE TUTTI I RISULTATI
print("\n=== COMBINANDO TUTTI I RISULTATI ===")
combined_results = []

for feature in X.columns:
    result = {
        'feature': feature,
        'type': 'binary' if feature in binary_features else 'continuous',
        'mutual_info': mi_results.get(feature, 0)
    }

    if feature in binary_features:
        result['chi2_pvalue'] = binary_significance.get(feature, 1)
        result['cramers_v'] = effect_sizes.get(feature, 0)
        result['test_used'] = 'Chi-quadrato'
    else:
        result['statistical_pvalue'] = continuous_significance.get(feature, 1)
        result['normality_p'] = normality_results.get(feature, 1)
        result['test_used'] = 'T-test' if normality_results.get(feature, 0) > 0.05 else 'Mann-Whitney'

    combined_results.append(result)

# Converti in DataFrame per analisi più facile
results_df = pd.DataFrame(combined_results)

# 8. SALVATAGGIO RISULTATI E SELEZIONE AUTOMATICA
os.makedirs('outputs/dataset_pvalue', exist_ok=True)

# Salva analisi completa
results_df.to_csv('outputs/dataset_pvalue/analisi_significativita_completa.csv', index=False)

# 9. SELEZIONE AUTOMATICA FEATURES SIGNIFICATIVE (p < 0.05)
significant_features = []

# Features binarie significative
for feature, p_value in binary_significance.items():
    if p_value < 0.05:
        significant_features.append(feature)

# Features continue significative
for feature, p_value in continuous_significance.items():
    if p_value < 0.05:
        significant_features.append(feature)

# Ordina per significatività (p-value più basso = più significativo)
significant_with_pvalues = []
for feature in significant_features:
    if feature in binary_significance:
        significant_with_pvalues.append((feature, binary_significance[feature]))
    else:
        significant_with_pvalues.append((feature, continuous_significance[feature]))

significant_with_pvalues.sort(key=lambda x: x[1])  # Ordina per p-value
final_significant_features = [feature for feature, _ in significant_with_pvalues]

# Salva lista features significative per selezione_features_ml.py
with open('outputs/dataset_pvalue/selected_features.txt', 'w') as f:
    f.write("# FEATURES SIGNIFICATIVE (p < 0.05) per selezione_features_ml.py\n")
    f.write("# Ordinate per significatività (p-value crescente)\n\n")
    for i, (feature, p_value) in enumerate(significant_with_pvalues, 1):
        feat_type = 'binaria' if feature in binary_features else 'continua'
        f.write(f"'{feature}',  # {i}. p={p_value:.2e} ({feat_type})\n")

print(f"\nRisultati salvati in:")
print(f"- outputs/dataset_pvalue/analisi_significativita_completa.csv (analisi completa)")
print(f"- outputs/dataset_pvalue/selected_features.txt (features per ML)")
print(f"\nFeatures significative trovate: {len(final_significant_features)}")

print(f"\nTOP 5 FEATURES PIU' SIGNIFICATIVE:")
for i, (feature, p_value) in enumerate(list(all_significance.items())[:5], 1):
    feat_type = 'binaria' if feature in binary_features else 'continua'
    print(f"{i}. {feature} (p={p_value:.2e}, {feat_type})")
