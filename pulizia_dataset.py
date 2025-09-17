import pandas as pd
import os

# 1. Caricamento dataset
df = pd.read_csv("database/diabetic_data.csv")

# 2. Eliminare colonne inutili
cols_to_drop = [
    "encounter_id", "weight", "admission_type_id",
    "admission_source_id", "discharge_disposition_id", "number_diagnoses",
    "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3","max_glu_serum","A1Cresult"
]
df.drop(columns=cols_to_drop, inplace=True)

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

# 6. Identificare le colonne categoriche
categorical_cols = df.select_dtypes(include=["object"]).columns

# 7. Applicare one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)  
# drop_first=True evita la collinearità (utile per regressione logistica)

# 8. Salvataggio del dataset pulito
os.makedirs('outputs/datasets_clean/first_clean', exist_ok=True)
output_path = 'outputs/datasets_clean/first_clean/diabetes_clean.csv'
df.to_csv(output_path, index=False)

print(f"Dataset pulito con one-hot encoding salvato in: {output_path}")
