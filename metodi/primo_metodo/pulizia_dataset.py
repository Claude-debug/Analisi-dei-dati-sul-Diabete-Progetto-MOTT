import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import json

# 1. Caricamento dataset
df = pd.read_csv("database/diabetic_data.csv", sep=';')

print(f"Colonne disponibili nel dataset: {list(df.columns)}")
print(f"Shape originale: {df.shape}")

# 2. Eliminare colonne inutili (solo quelle che esistono)
cols_to_drop = [
    "encounter_id", "weight", "admission_type_id",
    "admission_source_id", "discharge_disposition_id",
    "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3","max_glu_serum","A1Cresult"
]

# Filtra solo le colonne che esistono effettivamente
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
missing_cols = [col for col in cols_to_drop if col not in df.columns]

print(f"Colonne da eliminare presenti: {existing_cols_to_drop}")
print(f"Colonne da eliminare non trovate: {missing_cols}")

df.drop(columns=existing_cols_to_drop, inplace=True)

# 3. Eliminare record con valori mancanti o "?" che nel dataset rappresentano missing values
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# 4. Prendere i pazienti una sola volta (manteniamo la prima occorrenza per ogni patient_nbr)
df = df.drop_duplicates(subset="patient_nbr", keep="first")
df.drop(columns=["patient_nbr"], inplace=True)

# 5. Convertire la feature 'age' da range a valore numerico medio
def age_to_mean(age_range):
    age_range = age_range.strip("[]()")
    start, end = age_range.split("-")
    return (int(start) + int(end)) / 2

df["age"] = df["age"].apply(age_to_mean)

# 5.1. Convertire readmitted: <30 e >30 diventano "Si", NO rimane "No"
print(f"\nValori originali di readmitted: {df['readmitted'].unique()}")
df['readmitted'] = df['readmitted'].replace({'<30': 'Si', '>30': 'Si', 'NO': 'No'})
print(f"Valori trasformati di readmitted: {df['readmitted'].unique()}")

# 6. Identificare le colonne categoriche
categorical_cols = df.select_dtypes(include=["object"]).columns

# 7. Conversione features categoriche in interi usando Label Encoding
label_encoders = {}
encoding_mappings = {}

for col in categorical_cols:
    # Applicare Label Encoding
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

    # Memorizzare encoder e mapping per il report
    label_encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    encoding_mappings[col] = mapping


# 8. Salvataggio del dataset pulito
os.makedirs('outputs/datasets_clean/first_clean', exist_ok=True)
output_path = 'outputs/datasets_clean/first_clean/diabetes_clean.csv'
df.to_csv(output_path, index=False)

print(f"Dataset pulito salvato in: {output_path}")
