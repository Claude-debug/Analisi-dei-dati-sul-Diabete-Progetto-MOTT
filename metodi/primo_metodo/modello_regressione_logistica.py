"""
MODELLO DI REGRESSIONE LOGISTICA PER PREDIZIONE RIAMMISSIONE OSPEDALIERA

Questo script:
1. Carica il dataset ML-ready con le 15 features selezionate
2. Addestra un modello di regressione logistica
3. Valuta le performance del modello
4. Salva il modello e i risultati
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

def carica_e_prepara_dati():
    """Carica e prepara i dati per il ML"""
    print("Caricando dataset...")

    data_path = 'outputs/datasets_clean/third_clean/diabetes_ml_ready.csv'
    df = pd.read_csv(data_path, sep=';')

    # Separazione features e target
    X = df.drop(columns=['readmitted_NO'])
    y = df['readmitted_NO']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def addestra_modello(X_train, y_train):
    """Addestra il modello di regressione logistica"""
    print("Addestrando modello...")

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced'
    )

    model.fit(X_train, y_train)
    return model

def valuta_modello(model, X_train, X_test, y_train, y_test, feature_names):
    """Valuta le performance del modello"""
    print("Valutando performance...")

    # Predizioni
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metriche TRAIN SET
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred),
        'auc_roc': roc_auc_score(y_train, y_train_proba)
    }

    # Metriche TEST SET
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'auc_roc': roc_auc_score(y_test, y_test_proba)
    }

    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc'
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)

    return train_metrics, test_metrics, cv_scores, feature_importance, y_test, y_test_proba

def crea_visualizzazioni(metrics, feature_importance, y_test, y_test_proba, output_dir):
    """Crea visualizzazioni essenziali"""
    print("Creando visualizzazioni...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance Modello Regressione Logistica', fontsize=14)

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[0,0].plot(fpr, tpr, label=f'AUC = {metrics["auc_roc"]:.3f}')
    axes[0,0].plot([0, 1], [0, 1], 'k--')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend()

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, (y_test_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix')

    # Feature Importance
    top_features = feature_importance.head(10)
    axes[1,0].barh(range(len(top_features)), top_features['abs_coefficient'])
    axes[1,0].set_yticks(range(len(top_features)))
    axes[1,0].set_yticklabels(top_features['feature'])
    axes[1,0].set_title('Top 10 Feature Importance')
    axes[1,0].invert_yaxis()

    # Probability Distribution
    axes[1,1].hist(y_test_proba[y_test==0], alpha=0.7, label='Non Readmitted', bins=20)
    axes[1,1].hist(y_test_proba[y_test==1], alpha=0.7, label='Readmitted', bins=20)
    axes[1,1].set_title('Predicted Probabilities')
    axes[1,1].legend()

    plt.tight_layout()

    viz_path = os.path.join(output_dir, 'model_performance.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    return viz_path

def salva_risultati(model, scaler, train_metrics, test_metrics, cv_scores, feature_importance, output_dir):
    """Salva modello e risultati principali"""
    print("Salvando risultati...")

    # Salva modello e scaler
    joblib.dump(model, os.path.join(output_dir, 'logistic_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Salva metriche TRAIN e TEST
    train_df = pd.DataFrame([{
        'Set': 'Train',
        'Metric': metric,
        'Value': value
    } for metric, value in train_metrics.items()])

    test_df = pd.DataFrame([{
        'Set': 'Test',
        'Metric': metric,
        'Value': value
    } for metric, value in test_metrics.items()])

    results_df = pd.concat([train_df, test_df], ignore_index=True)
    results_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # Salva feature importance
    feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # Report semplice
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write("MODELLO REGRESSIONE LOGISTICA - SUMMARY\n")
        f.write("="*50 + "\n\n")

        f.write("PERFORMANCE TRAIN SET:\n")
        for metric, value in train_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")

        f.write("\nPERFORMANCE TEST SET:\n")
        for metric, value in test_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")

        # Calcola differenze per rilevare overfitting
        f.write("\nDIAGNOSI OVERFITTING:\n")
        for metric in train_metrics.keys():
            diff = train_metrics[metric] - test_metrics[metric]
            status = "OK" if abs(diff) < 0.05 else "OVERFITTING" if diff > 0.05 else "ANOMALIA"
            f.write(f"- {metric}: {diff:+.4f} ({status})\n")

        f.write(f"\nCROSS-VALIDATION AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

        f.write(f"\nTOP 5 FEATURES:\n")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
            impact = "RISCHIO" if row['coefficient'] > 0 else "PROTETTIVO"
            f.write(f"{i}. {row['feature']} ({impact}): {row['coefficient']:+.4f}\n")

def main():
    """Funzione principale"""
    print("MODELLO REGRESSIONE LOGISTICA")
    print("="*40)

    # Setup
    output_dir = 'outputs/ml_models'
    os.makedirs(output_dir, exist_ok=True)

    # Pipeline
    X_train, X_test, y_train, y_test, scaler, feature_names = carica_e_prepara_dati()
    model = addestra_modello(X_train, y_train)
    train_metrics, test_metrics, cv_scores, feature_importance, y_test_actual, y_test_proba = valuta_modello(
        model, X_train, X_test, y_train, y_test, feature_names
    )

    viz_path = crea_visualizzazioni(test_metrics, feature_importance, y_test_actual, y_test_proba, output_dir)
    salva_risultati(model, scaler, train_metrics, test_metrics, cv_scores, feature_importance, output_dir)

    # Summary finale
    print("\n" + "="*40)
    print("MODELLO COMPLETATO!")
    print("="*40)

    # Confronto Train vs Test
    print("PERFORMANCE CONFRONTO:")
    print(f"Train AUC: {train_metrics['auc_roc']:.4f}")
    print(f"Test AUC:  {test_metrics['auc_roc']:.4f}")
    print(f"CV AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Diagnosi overfitting
    auc_diff = train_metrics['auc_roc'] - test_metrics['auc_roc']
    if abs(auc_diff) < 0.02:
        print(f"DIAGNOSI: OTTIMO (diff AUC: {auc_diff:+.4f})")
    elif abs(auc_diff) < 0.05:
        print(f"DIAGNOSI: BUONO (diff AUC: {auc_diff:+.4f})")
    elif auc_diff > 0.05:
        print(f"DIAGNOSI: OVERFITTING (diff AUC: {auc_diff:+.4f})")
    else:
        print(f"DIAGNOSI: ANOMALIA (diff AUC: {auc_diff:+.4f})")

    print(f"\nTop 3 Features:")
    for i, (_, row) in enumerate(feature_importance.head(3).iterrows(), 1):
        impact = "+" if row['coefficient'] > 0 else "-"
        print(f"{i}. {row['feature']} ({impact}{row['abs_coefficient']:.3f})")

    print(f"\nFile salvati in: {output_dir}")

    return model, train_metrics, test_metrics

if __name__ == "__main__":
    model, train_results, test_results = main()