import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Carica il dataset encodato e prepara X e y per la regressione logistica.
    """
    print("=== DIABETES REGRESSION ANALYSIS ===\n")
    
    # Carica il dataset con encoding robusto ottimizzato per ML
    df = pd.read_csv('outputs/encoded/diabetes_robust_target.csv')
    print(f"Dataset caricato: {df.shape}")
    
    # Definizione delle variabili per l'analisi
    target_col = 'readmitted_binary'
    exclude_cols = ['patient_nbr', 'readmitted', target_col]  # Identificatori e variabili ridondanti con il target
    
    # 3. Prepara X e y
    X = df.drop(columns=exclude_cols)
    y = df[target_col]
    
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Target distribution:")
    print(y.value_counts(normalize=True))
    
    # 4. Info sulle features
    print(f"\nFeatures disponibili:")
    feature_types = {
        'demographic': ['race', 'gender', 'age'],
        'clinical': ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications'],
        'history': ['number_outpatient', 'number_emergency', 'number_inpatient'],
        'medications': [col for col in X.columns if any(med in col.lower() for med in ['metformin', 'insulin', 'glyburide'])],
        'encoded': [col for col in X.columns if 'target_enc' in col]
    }
    
    for category, cols in feature_types.items():
        available_cols = [col for col in cols if col in X.columns]
        if available_cols:
            print(f"  {category}: {len(available_cols)} variabili")
    
    return X, y, df

def perform_regression_analysis(X, y):
    """
    Esegue analisi di regressione con diversi modelli.
    """
    print(f"\n=== MODELLI DI REGRESSIONE ===")
    
    # 1. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 2. Scaling per modelli lineari
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Definisci modelli
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Ridge Logistic': LogisticRegression(penalty='l2', C=0.1, random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    # 4. Training e valutazione
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Usa dati scalati per modelli lineari, originali per RF
        if 'Logistic' in name or 'Ridge' in name:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # Training
        model.fit(X_train_model, y_train)
        
        # Predizioni
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Metriche
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_model, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        results[name] = {
            'model': model,
            'auc_test': auc_score,
            'auc_cv_mean': cv_scores.mean(),
            'auc_cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"AUC Test: {auc_score:.4f}")
        print(f"AUC CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred))
    
    return results, X_test, y_test, scaler

def analyze_feature_importance(results, X, feature_names):
    """
    Analizza l'importanza delle features.
    """
    print(f"\n=== FEATURE IMPORTANCE ===")
    
    # Random Forest feature importance
    rf_model = results['Random Forest']['model']
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 features più importanti (Random Forest):")
    print(rf_importance.head(10))
    
    # Logistic Regression coefficients
    lr_model = results['Logistic Regression']['model']
    lr_coef = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTop 10 coefficienti più forti (Logistic Regression):")
    print(lr_coef.head(10)[['feature', 'coefficient']])
    
    return rf_importance, lr_coef

def generate_model_comparison_report(results):
    """
    Genera report di confronto tra modelli.
    """
    print(f"\n=== CONFRONTO MODELLI ===")
    
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC_Test': [results[name]['auc_test'] for name in results.keys()],
        'AUC_CV_Mean': [results[name]['auc_cv_mean'] for name in results.keys()],
        'AUC_CV_Std': [results[name]['auc_cv_std'] for name in results.keys()]
    }).sort_values('AUC_Test', ascending=False)
    
    print("Ranking modelli per performance:")
    print(comparison)
    
    # Migliore modello
    best_model_name = comparison.iloc[0]['Model']
    best_auc = comparison.iloc[0]['AUC_Test']
    
    print(f"\nMIGLIORE MODELLO: {best_model_name}")
    print(f"AUC Test: {best_auc:.4f}")
    
    return comparison, best_model_name

def save_results_and_recommendations(comparison, rf_importance, lr_coef):
    """
    Salva risultati e genera raccomandazioni.
    """
    print(f"\n=== SALVATAGGIO RISULTATI ===")
    
    # Salva confronto modelli
    comparison.to_csv('outputs/regression_results/model_comparison_report.csv', index=False)
    print("Model comparison salvato: outputs/regression_results/model_comparison_report.csv")
    
    # Salva feature importance
    rf_importance.to_csv('outputs/regression_results/feature_importance_report.csv', index=False)
    print("Feature importance salvato: outputs/regression_results/feature_importance_report.csv")
    
    # Genera raccomandazioni
    recommendations = f"""
=== RACCOMANDAZIONI FINALI ===

MIGLIORE MODELLO: {comparison.iloc[0]['Model']} (AUC: {comparison.iloc[0]['AUC_Test']:.4f})

TOP 5 FEATURES PIÙ PREDITTIVE:
{chr(10).join([f"- {row['feature']}: {row['importance']:.4f}" for _, row in rf_importance.head(5).iterrows()])}

INTERPRETAZIONE CLINICA:
- Il modello può predire riammissione con AUC {comparison.iloc[0]['AUC_Test']:.4f}
- Features cliniche più importanti identificate
- Modello pronto per validazione clinica

PROSSIMI PASSI:
1. Validazione su nuovi dati
2. Interpretazione clinica con medici
3. Implementazione in sistemi clinici
4. Monitoraggio performance nel tempo
"""
    
    with open('outputs/regression_results/regression_recommendations.txt', 'w') as f:
        f.write(recommendations)
    
    print("Raccomandazioni salvate: outputs/regression_results/regression_recommendations.txt")
    print(recommendations)

def main():
    """
    Funzione principale per l'analisi di regressione completa.
    """
    try:
        # Crea cartella output se non existe
        import os
        os.makedirs('outputs/regression_results', exist_ok=True)
        # 1. Caricamento dati
        X, y, df = load_and_prepare_data()
        
        # 2. Analisi di regressione
        results, X_test, y_test, scaler = perform_regression_analysis(X, y)
        
        # 3. Feature importance
        rf_importance, lr_coef = analyze_feature_importance(results, X, X.columns)
        
        # 4. Confronto modelli
        comparison, best_model = generate_model_comparison_report(results)
        
        # 5. Salvataggio risultati
        save_results_and_recommendations(comparison, rf_importance, lr_coef)
        
        print(f"\n*** ANALISI DI REGRESSIONE COMPLETATA CON SUCCESSO ***")
        print(f"Usa il modello '{best_model}' per predizioni future")
        
    except Exception as e:
        print(f"ERRORE nell'analisi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()