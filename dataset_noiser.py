import numpy as np
import pandas as pd
import random 

df = pd.read_csv('dataset_initial.csv')

def add_missing(col, amount):
    X = col.copy()
    size = amount if amount >= 1 else int(len(X) * amount)
    indexes = np.random.choice(len(X), size, replace = False )
    X[indexes] = np.nan
    return X

def add_missing_rows(df, amount):
    X = df.copy()
    rows, cols = X.shape
    size = amount if amount >= 1 else int(rows * amount)
    indexes = np.random.choice(rows, size, replace = False ) + 0.5
    for i in indexes:
        X.loc[i] = np.full((cols,),np.nan)
    X = X.sort_index().reset_index(drop=True)
    return X

body_signals = [
    "height(cm)", "weight(kg)", "waist(cm)", "eyesight(left)", "eyesight(right)",
    "hearing(left)", "hearing(right)", "systolic", "relaxation", "fasting blood sugar",
    "Cholesterol", "triglyceride", "HDL", "LDL", "hemoglobin", "Urine protein",
    "serum creatinine", "AST", "ALT", "Gtp"
]
additional_cols = ["gender", "oral", "tartar"]
target_columns = body_signals + additional_cols



df.info()

for col in target_columns:
    if col in df.columns:
        df[col] = add_missing(df[col], random.randint(500,2000))
    
df.info()


target_column = 'smoking'
# Get the minimum count of samples per class to maintain balance
min_samples_per_class = 23658 // df[target_column].nunique()

# Perform stratified sampling
df_sampled = df.groupby(target_column).apply(lambda x: x.sample(n=min_samples_per_class, random_state=42))

# Drop the extra index added by groupby
df_sampled = df_sampled.reset_index(drop=True)

n=20000
# Save the balanced dataset
df_sampled.to_csv(f"dataset_{n}.csv", index=False)

print(df_sampled[target_column].value_counts())  # Check if balanced



n = 0
# df.to_csv(f"dataset_{n}.csv", index=False)