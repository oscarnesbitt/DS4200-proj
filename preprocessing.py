# import packages
import sys
import pandas as pd
import numpy as np

# usage for this file: python/python3 preprocessing.py input_file.csv output_file.csv

# assign arguments for files
inp, outp = sys.argv[1], sys.argv[2]

# load raw dataset
df = pd.read_csv(inp)

# clean data
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# drop mystery/blank values
df = df.replace("?", np.nan).dropna()

# drop fnlwgt feature (not relevant)
df = df.drop(columns=['fnlwgt'])

# set up mapping for 'education' categorical variable
edu_map = {
    "Preschool": "HS_dropout", "1st-4th": "HS_dropout", "5th-6th": "HS_dropout",
    "7th-8th": "HS_dropout", "9th": "HS_dropout", "10th": "HS_dropout",
    "11th": "HS_dropout", "12th": "HS_dropout",
    "HS-grad": "hs_grad",
    "Some-college": "college_dropout",
    "Assoc-acdm": "associate", "Assoc-voc": "associate",
    "Bachelors": "bachelor",
    "Masters": "masters",
    "Doctorate": "PhD",
    "Prof-school": "prof_school",
} 

# set up mapping for 'occupation' categorical variable
occ_map = {
    "Tech-support": "support",
    "Craft-repair": "blue_collar",
    "Other-service": "service",
    "Sales": "white_collar",
    "Exec-managerial": "white_collar",
    "Prof-specialty": "professional",
    "Handlers-cleaners": "labor",
    "Machine-op-inspct": "labor",
    "Adm-clerical": "clerical",
    "Farming-fishing": "labor",
    "Transport-moving": "labor",
    "Priv-house-serv": "service",
    "Protective-serv": "service",
    "Armed-Forces": "military",
}

# map 'native.country' to continent
continent_map = {
    # North America
    "United-States": "North America", "Canada": "North America", "Mexico": "North America",
    # Latin  America
    "Columbia": "Latin America", "Ecuador": "Latin America", "Peru": "Latin America",
    "Honduras": "Latin America", "Nicaragua": "Latin America", "El-Salvador": "Latin America",
    "Guatemala": "Latin America", "Trinadad&Tobago": "Latin America",
    # Europe
    "England": "Europe", "Germany": "Europe", "Italy": "Europe", "Poland": "Europe",
    "Portugal": "Europe", "Ireland": "Europe", "France": "Europe", "Scotland": "Europe",
    "Greece": "Europe", "Yugoslavia": "Europe", "Hungary": "Europe", "Holand-Netherlands": "Europe",
    # Asia
    "India": "Asia", "China": "Asia", "Japan": "Asia", "Iran": "Asia", "Philippines": "Asia",
    "Vietnam": "Asia", "Cambodia": "Asia", "Laos": "Asia", "Thailand": "Asia", "Taiwan": "Asia",
    # Africa
    "South-Africa": "Africa", "Egypt": "Africa", "Morocco": "Africa", "Kenya": "Africa",
    # Oceania
    "Hong": "Oceania", "Outlying-US(Guam-USVI-etc)": "Oceania",
}

# edit df with mappings
df["education"] = df["education"].map(edu_map).fillna("other")
df["occupation"] = df["occupation"].map(occ_map).fillna("other")
df["Continent"] = df["native.country"].map(continent_map).fillna("Other")

# drop country variable (not needed)
df = df.drop(columns=["native.country"])

# output cleaned csv
df.to_csv(outp, index=False)
print(f"✔ wrote: {outp} (rows: {len(df)})")
