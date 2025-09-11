import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from category_encoders import BinaryEncoder, HashingEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedEncodingPipeline:
    """
    Pipeline avanzato per encoding ottimizzato con gestione memoria,
    cross-validation e strategie multiple intelligenti.
    """
    
    def __init__(self, input_file="outputs/cleaned_data/diabetes_clean_no_encoding.csv", 
                 output_dir="outputs/encoded/", target_col='readmitted_binary'):
        self.input_file = input_file
        self.output_dir = output_dir
        self.target_col = target_col
        self.df = None
        self.categorical_analysis = {}
        
    def load_and_validate_data(self):
        """Carica e valida il dataset con controlli robusti."""
        print("=== ADVANCED ENCODING PIPELINE ===\n")
        
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"[OK] Dataset caricato: {self.df.shape}")
        except FileNotFoundError:
            print(f"[ERROR] File {self.input_file} non trovato!")
            return False
        except Exception as e:
            print(f"[ERROR] nel caricamento: {e}")
            return False
        
        # Validazioni robuste
        if self.df.empty:
            print("[ERROR] Dataset vuoto!")
            return False
            
        if self.df.shape[0] < 100:
            print("[WARNING] Dataset molto piccolo (<100 righe)")
            
        # Prepara target se necessario
        if self.target_col == 'readmitted_binary':
            if self.target_col not in self.df.columns and 'readmitted' in self.df.columns:
                self.df[self.target_col] = self.df['readmitted'].map({'NO': 0, '<30': 1, '>30': 1})
                print(f"[OK] Target {self.target_col} creato automaticamente")
        
        return True
    
    def analyze_categorical_variables(self):
        """Analisi avanzata delle variabili categoriche."""
        print("\n=== ANALISI AVANZATA VARIABILI CATEGORICHE ===")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Rimuovi colonne non da encodare
        exclude_cols = ['patient_nbr'] if 'patient_nbr' in categorical_cols else []
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        print(f"Variabili categoriche trovate: {len(categorical_cols)}")
        
        # Analisi dettagliata per ogni variabile
        for col in categorical_cols:
            analysis = {
                'unique_count': self.df[col].nunique(),
                'missing_count': self.df[col].isnull().sum(),
                'missing_pct': self.df[col].isnull().mean() * 100,
                'most_frequent': self.df[col].value_counts().index[0] if len(self.df[col].value_counts()) > 0 else 'N/A',
                'most_frequent_pct': self.df[col].value_counts().iloc[0] / len(self.df) * 100 if len(self.df[col].value_counts()) > 0 else 0,
                'rare_categories_count': (self.df[col].value_counts() < 20).sum(),
                'memory_usage_mb': self.df[col].memory_usage(deep=True) / 1024 / 1024
            }
            
            # Classifica per strategia ottimale
            if analysis['unique_count'] <= 2:
                analysis['recommended_strategy'] = 'binary'
            elif analysis['unique_count'] <= 10 and analysis['rare_categories_count'] <= 2:
                analysis['recommended_strategy'] = 'onehot'
            elif analysis['unique_count'] <= 50:
                analysis['recommended_strategy'] = 'target_or_label'
            else:
                analysis['recommended_strategy'] = 'hashing_or_target'
            
            self.categorical_analysis[col] = analysis
            
            print(f"  {col}:")
            print(f"    - {analysis['unique_count']} categorie, {analysis['missing_pct']:.1f}% missing")
            print(f"    - Piu frequente: '{analysis['most_frequent']}' ({analysis['most_frequent_pct']:.1f}%)")
            print(f"    - Categorie rare (<20): {analysis['rare_categories_count']}")
            print(f"    - Strategia consigliata: {analysis['recommended_strategy']}")
        
        return categorical_cols
    
    def apply_smart_onehot_encoding(self, categorical_cols):
        """One-hot encoding intelligente con controllo esplosione dimensionale."""
        print(f"\n=== STRATEGIA 1: SMART ONE-HOT ENCODING ===")
        
        df_onehot = self.df.copy()
        
        # Seleziona solo variabili adatte per one-hot
        onehot_candidates = [col for col in categorical_cols 
                           if self.categorical_analysis[col]['recommended_strategy'] in ['binary', 'onehot']]
        
        print(f"Variabili selezionate per one-hot: {len(onehot_candidates)}")
        
        if not onehot_candidates:
            print("[WARNING] Nessuna variabile adatta per one-hot encoding")
            return None
            
        # Preprocessing intelligente delle categorie rare
        for col in onehot_candidates:
            rare_threshold = max(20, len(df_onehot) * 0.01)  # Almeno 1% o 20 osservazioni
            value_counts = df_onehot[col].value_counts()
            rare_categories = value_counts[value_counts < rare_threshold].index
            
            if len(rare_categories) > 0:
                df_onehot[col] = df_onehot[col].replace(rare_categories, f'{col}_Other')
                print(f"  {col}: {len(rare_categories)} categorie rare consolidate")
        
        # Applica one-hot con controllo memoria
        try:
            estimated_columns = sum([self.df[col].nunique() - 1 for col in onehot_candidates])
            if estimated_columns > 100:
                print(f"[WARNING] One-hot creera ~{estimated_columns} colonne")
                
            df_onehot_encoded = pd.get_dummies(
                df_onehot, 
                columns=onehot_candidates, 
                drop_first=True, 
                dtype=np.int8  # Risparmia memoria
            )
            
            output_file = f"{self.output_dir}diabetes_smart_onehot.csv"
            df_onehot_encoded.to_csv(output_file, index=False)
            
            print(f"[OK] Smart One-hot salvato: {output_file}")
            print(f"  Dimensioni: {df_onehot_encoded.shape}")
            print(f"  Memoria stimata: {df_onehot_encoded.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            return df_onehot_encoded
            
        except MemoryError:
            print("[ERROR] One-hot encoding fallito: memoria insufficiente")
            return None
        except Exception as e:
            print(f"[ERROR] One-hot encoding fallito: {e}")
            return None
    
    def apply_robust_target_encoding(self, categorical_cols):
        """Target encoding robusto con cross-validation per evitare overfitting."""
        print(f"\n=== STRATEGIA 2: ROBUST TARGET ENCODING ===")
        
        if self.target_col not in self.df.columns:
            print(f"[ERROR] Target column '{self.target_col}' non trovata")
            return None
            
        df_target = self.df.copy()
        
        # Seleziona variabili adatte per target encoding
        target_candidates = [col for col in categorical_cols 
                           if self.categorical_analysis[col]['unique_count'] > 10]
        
        print(f"Variabili per target encoding: {len(target_candidates)}")
        
        if not target_candidates:
            print("[WARNING] Nessuna variabile adatta per target encoding")
            return None
        
        # Target encoding con cross-validation (evita overfitting)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in target_candidates:
            print(f"  Encoding {col} con CV...")
            
            # Inizializza colonna encoded
            df_target[f'{col}_target_enc'] = 0.0
            
            # Cross-validation encoding
            for train_idx, val_idx in kf.split(df_target):
                # Media target sul train fold
                target_means = df_target.iloc[train_idx].groupby(col)[self.target_col].mean()
                
                # Applica al validation fold
                df_target.loc[val_idx, f'{col}_target_enc'] = (
                    df_target.loc[val_idx, col].map(target_means).fillna(df_target[self.target_col].mean())
                )
            
            # Rimuovi colonna originale
            df_target.drop(columns=[col], inplace=True)
        
        # Encoding semplice per variabili a bassa cardinalità
        remaining_categorical = [col for col in categorical_cols if col not in target_candidates]
        
        for col in remaining_categorical:
            if col in df_target.columns and df_target[col].dtype == 'object':
                le = LabelEncoder()
                df_target[col] = le.fit_transform(df_target[col].astype(str))
        
        output_file = f"{self.output_dir}diabetes_robust_target.csv"
        df_target.to_csv(output_file, index=False)
        
        print(f"[OK] Robust Target encoding salvato: {output_file}")
        print(f"  Dimensioni: {df_target.shape}")
        
        return df_target
    
    def apply_advanced_encoding_strategies(self, categorical_cols):
        """Strategie di encoding avanzate (Hashing, Binary)."""
        print(f"\n=== STRATEGIA 3: ADVANCED ENCODING METHODS ===")
        
        df_advanced = self.df.copy()
        
        # Strategia 3A: Binary Encoding per variabili medie
        medium_cardinality = [col for col in categorical_cols 
                            if 10 < self.categorical_analysis[col]['unique_count'] <= 50]
        
        if medium_cardinality:
            print(f"Binary encoding per {len(medium_cardinality)} variabili medie...")
            for col in medium_cardinality:
                try:
                    encoder = BinaryEncoder(cols=[col], drop_invariant=True)
                    encoded = encoder.fit_transform(df_advanced[[col]])
                    
                    # Rimuovi colonna originale e aggiungi encoded
                    df_advanced.drop(columns=[col], inplace=True)
                    df_advanced = pd.concat([df_advanced, encoded], axis=1)
                    print(f"  {col}: {encoded.shape[1]} colonne binarie create")
                except Exception as e:
                    print(f"  [ERROR] Binary encoding fallito per {col}: {e}")
        
        # Strategia 3B: Hashing per variabili ad alta cardinalità
        high_cardinality = [col for col in categorical_cols 
                          if self.categorical_analysis[col]['unique_count'] > 50]
        
        if high_cardinality:
            print(f"Hashing encoding per {len(high_cardinality)} variabili alte...")
            for col in high_cardinality:
                try:
                    # Hash size basato sulla cardinalità
                    hash_size = min(32, max(8, int(np.log2(self.categorical_analysis[col]['unique_count']))))
                    
                    encoder = HashingEncoder(cols=[col], n_components=hash_size, drop_invariant=True)
                    encoded = encoder.fit_transform(df_advanced[[col]])
                    
                    df_advanced.drop(columns=[col], inplace=True)
                    df_advanced = pd.concat([df_advanced, encoded], axis=1)
                    print(f"  {col}: ridotto a {hash_size} dimensioni hash")
                except Exception as e:
                    print(f"  [ERROR] Hashing encoding fallito per {col}: {e}")
        
        # Strategia 3C: Label encoding per variabili rimanenti
        remaining_cols = [col for col in categorical_cols 
                         if col in df_advanced.columns and df_advanced[col].dtype == 'object']
        
        if remaining_cols:
            print(f"Label encoding per {len(remaining_cols)} variabili rimanenti...")
            for col in remaining_cols:
                try:
                    le = LabelEncoder()
                    df_advanced[col] = le.fit_transform(df_advanced[col].astype(str))
                except Exception as e:
                    print(f"  [ERROR] Label encoding fallito per {col}: {e}")
        
        output_file = f"{self.output_dir}diabetes_advanced_encoding.csv"
        df_advanced.to_csv(output_file, index=False)
        
        print(f"[OK] Advanced encoding salvato: {output_file}")
        print(f"  Dimensioni: {df_advanced.shape}")
        
        return df_advanced
    
    def create_ensemble_encoding(self, categorical_cols):
        """Crea encoding ensemble combinando le migliori strategie."""
        print(f"\n=== STRATEGIA 4: ENSEMBLE ENCODING (OTTIMALE) ===")
        
        df_ensemble = self.df.copy()
        
        for col in categorical_cols:
            try:
                strategy = self.categorical_analysis[col]['recommended_strategy']
                unique_count = self.categorical_analysis[col]['unique_count']
                
                if strategy == 'binary' or unique_count <= 2:
                    # Binary: 0/1 encoding
                    unique_vals = df_ensemble[col].unique()
                    if len(unique_vals) == 2:
                        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                        df_ensemble[col] = df_ensemble[col].map(mapping)
                        print(f"  {col}: binary encoding applicato")
                        
                elif strategy == 'onehot' and unique_count <= 10:
                    # One-hot per bassa cardinalità
                    dummies = pd.get_dummies(df_ensemble[col], prefix=col, drop_first=True, dtype=np.int8)
                    df_ensemble = pd.concat([df_ensemble.drop(columns=[col]), dummies], axis=1)
                    print(f"  {col}: one-hot encoding ({dummies.shape[1]} colonne)")
                    
                elif strategy == 'target_or_label' and self.target_col in df_ensemble.columns:
                    # Target encoding per media cardinalità
                    target_means = df_ensemble.groupby(col)[self.target_col].mean()
                    df_ensemble[f'{col}_target'] = df_ensemble[col].map(target_means)
                    df_ensemble.drop(columns=[col], inplace=True)
                    print(f"  {col}: target encoding applicato")
                    
                else:
                    # Label encoding per alta cardinalità
                    le = LabelEncoder()
                    df_ensemble[col] = le.fit_transform(df_ensemble[col].astype(str))
                    print(f"  {col}: label encoding applicato")
                    
            except Exception as e:
                print(f"  [ERROR] Encoding fallito per {col}: {e}")
        
        output_file = f"{self.output_dir}diabetes_ensemble_encoding.csv"
        df_ensemble.to_csv(output_file, index=False)
        
        print(f"[OK] Ensemble encoding salvato: {output_file}")
        print(f"  Dimensioni: {df_ensemble.shape}")
        print(f"  Strategia adattiva per ogni variabile")
        
        return df_ensemble
    
    def generate_encoding_report(self, categorical_cols):
        """Genera report dettagliato delle strategie di encoding."""
        print(f"\n=== REPORT ENCODING STRATEGIES ===")
        
        report = []
        for col in categorical_cols:
            analysis = self.categorical_analysis[col]
            report.append({
                'Variable': col,
                'Unique_Values': analysis['unique_count'],
                'Missing_%': analysis['missing_pct'],
                'Most_Frequent_%': analysis['most_frequent_pct'],
                'Rare_Categories': analysis['rare_categories_count'],
                'Recommended_Strategy': analysis['recommended_strategy'],
                'Memory_MB': analysis['memory_usage_mb']
            })
        
        report_df = pd.DataFrame(report)
        report_file = f"{self.output_dir}encoding_analysis_report.csv"
        report_df.to_csv(report_file, index=False)
        
        print(f"[OK] Report salvato: {report_file}")
        print(f"\nSummary strategie consigliate:")
        strategy_counts = report_df['Recommended_Strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} variabili")
        
        return report_df
    
    def run_complete_pipeline(self):
        """Esegue l'intera pipeline di encoding avanzata."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Step 1: Caricamento e validazione
        if not self.load_and_validate_data():
            return None
        
        # Step 2: Analisi variabili
        categorical_cols = self.analyze_categorical_variables()
        if not categorical_cols:
            print("[ERROR] Nessuna variabile categorica trovata")
            return None
        
        # Step 3: Applica tutte le strategie
        strategies_results = {}
        
        strategies_results['smart_onehot'] = self.apply_smart_onehot_encoding(categorical_cols)
        strategies_results['robust_target'] = self.apply_robust_target_encoding(categorical_cols)  
        strategies_results['advanced'] = self.apply_advanced_encoding_strategies(categorical_cols)
        strategies_results['ensemble'] = self.create_ensemble_encoding(categorical_cols)
        
        # Step 4: Genera report
        report = self.generate_encoding_report(categorical_cols)
        
        # Step 5: Summary finale
        print(f"\n=== PIPELINE COMPLETATA ===")
        print(f"Dataset originale: {self.df.shape}")
        print(f"Strategie applicate: {len([k for k, v in strategies_results.items() if v is not None])}")
        print(f"Files generati in: {self.output_dir}")
        print(f"\nRACCOMANDAZIONI:")
        print(f"  - Per analisi statistica: usa diabetes_ensemble_encoding.csv")
        print(f"  - Per ML con memoria limitata: usa diabetes_advanced_encoding.csv")
        print(f"  - Per interpretabilita massima: usa diabetes_smart_onehot.csv")
        print(f"  - Per performance ML: usa diabetes_robust_target.csv")
        
        return strategies_results

def main():
    """Funzione principale per eseguire il pipeline avanzato."""
    try:
        pipeline = AdvancedEncodingPipeline()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"\n[SUCCESS] Pipeline di encoding completata. Verificare i file generati nella cartella outputs/encoded/")
        else:
            print(f"\n[ERROR] Pipeline di encoding fallita. Controllare i messaggi di errore precedenti.")
            
    except Exception as e:
        print(f"[ERROR] Errore durante l'esecuzione della pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()