import pandas as pd
import os

# Crea le cartelle di output se non esistono
os.makedirs('outputs/datasets_clean/first_clean', exist_ok=True)
os.makedirs('outputs/datasets_clean/second_clean', exist_ok=True)

# Input: carica il dataset pulito dalla cartella first_clean
input_path = 'outputs/datasets_clean/first_clean/diabetes_clean.csv'
df = pd.read_csv(input_path)

print(f"Caricato dataset da: {input_path}")
print(f"Righe iniziali: {len(df)}")

# Rimuovi righe con race_Other = 1
df = df[df['race_Other'] != 1]
print(f"Righe dopo rimozione race_Other = 1: {len(df)}")

# Rimuovi righe con gender_Unknown/Invalid = 1
df = df[df['gender_Unknown/Invalid'] != 1]
print(f"Righe dopo rimozione gender_Unknown/Invalid = 1: {len(df)}")

# Rimuovi le colonne race_Other, gender_Unknown/Invalid e readmitted_>30
df = df.drop(columns=['race_Other', 'gender_Unknown/Invalid', 'readmitted_>30'])
print(f"Colonne rimosse: race_Other, gender_Unknown/Invalid, readmitted_>30")
print(f"Colonne rimanenti: {len(df.columns)}")

# Output: salva il dataset filtrato nella cartella second_clean
output_path = 'outputs/datasets_clean/second_clean/diabetes_clean_filtered.csv'
df.to_csv(output_path, index=False)
print(f"Dataset filtrato salvato in: {output_path}")