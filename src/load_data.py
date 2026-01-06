import pandas as pd

file_path = "../data/raw/capture20110810.binetflow.2format"

df_head = pd.read_csv(
    file_path,
    sep=",",
    nrows=10,
    low_memory=False
)

print(df_head)
print("\nCOLUMNS:\n")
print(df_head.columns)

# Binary target creation
df_head["target"] = df_head["Label"].apply(
    lambda x: 0 if "Background" in x else 1
)

print("\nTARGET DISTRIBUTION (HEAD):")
print(df_head["target"].value_counts())


# ================================
# ADIM 4.1 - FULL DATA CLASS DISTRIBUTION
# ================================

chunk_iter = pd.read_csv(
    file_path,
    sep=",",
    chunksize=200_000,   # RAM-safe
    low_memory=False
)

class_counts = {0: 0, 1: 0}

for chunk in chunk_iter:
    # Binary target creation
    chunk["target"] = chunk["Label"].apply(
        lambda x: 0 if "Background" in x else 1
    )

    counts = chunk["target"].value_counts()
    for k, v in counts.items():
        class_counts[k] += v

print("\nFULL DATASET CLASS DISTRIBUTION:")
print(class_counts)


# ================================
# ADIM 4.3 - STRATIFIED SAMPLING
# ================================

sample_fraction = 0.05
sampled_chunks = []

chunk_iter = pd.read_csv(file_path, sep=",", chunksize=200_000, low_memory=False)

for chunk in chunk_iter:
    chunk["target"] = chunk["Label"].apply(lambda x: 0 if "Background" in x else 1)

    sampled_parts = []
    for cls, g in chunk.groupby("target"):
        sampled_parts.append(g.sample(frac=sample_fraction, random_state=42))

    sampled_chunk = pd.concat(sampled_parts, ignore_index=True)
    sampled_chunks.append(sampled_chunk)

sampled_df = pd.concat(sampled_chunks, ignore_index=True)

print("SAMPLED DATASET SHAPE:", sampled_df.shape)
print("SAMPLED CLASS DISTRIBUTION:")
print(sampled_df["target"].value_counts(normalize=True))


# ================================
# ADIM 5 - FEATURE SELECTION
# ================================

# Kolonları yazdır (kontrol için)
print("\nALL COLUMNS:")
print(sampled_df.columns.tolist())

# Silinecek kolonlar
drop_cols = [
    "SrcAddr", "DstAddr",
    "Sport", "Dport",
    "StartTime", "LastTime"
]

# Var olanları güvenli şekilde sil
drop_cols = [c for c in drop_cols if c in sampled_df.columns]

clean_df = sampled_df.drop(columns=drop_cols)

print("\nCOLUMNS AFTER DROPPING:")
print(clean_df.columns.tolist())
print("\nFINAL DATASET SHAPE:", clean_df.shape)



# ================================
# ADIM 6.1 - DATASET OVERVIEW
# ================================

print("\n--- DATASET OVERVIEW ---")
print(f"Number of rows (flows): {clean_df.shape[0]}")
print(f"Number of features: {clean_df.shape[1] - 1}")  # target hariç
print("\nClass distribution (counts):")
print(clean_df["target"].value_counts())

print("\nClass distribution (percentages):")
print(clean_df["target"].value_counts(normalize=True) * 100)



# ================================
# ADIM 6.2 - CLASS DISTRIBUTION PLOT
# ================================

import matplotlib.pyplot as plt

class_counts = clean_df["target"].value_counts().sort_index()

plt.figure()
class_counts.plot(kind="bar")
plt.xlabel("Class (0 = Normal, 1 = Botnet)")
plt.ylabel("Number of Flows")
plt.title("Class Distribution in Sampled Dataset")
plt.tight_layout()
plt.savefig("../figures/class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()



# ================================
# ADIM 6.3 - NUMERIC FEATURE DISTRIBUTIONS
# ================================

numeric_features = ["Dur", "TotBytes"]

for feature in numeric_features:
    plt.figure()
    clean_df[feature].hist(bins=50)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(f"../figures/{feature.lower()}_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.boxplot(clean_df[feature], vert=False)
    plt.xlabel(feature)
    plt.title(f"Boxplot of {feature}")
    plt.tight_layout()
    plt.savefig(f"../figures/{feature.lower()}_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()



# ================================
# ADIM 6.4 - CORRELATION HEATMAP
# ================================

import numpy as np

# Sadece numeric kolonları seç
numeric_df = clean_df.select_dtypes(include=[np.number])

# target'ı çıkar (etiketle korelasyon istemiyoruz)
if "target" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["target"])

# Korelasyon matrisi
corr_matrix = numeric_df.corr()

plt.figure()
plt.imshow(corr_matrix)
plt.colorbar()
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("../figures/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()



# ================================
# ADIM 7.1 - BASELINE MODEL (LOGISTIC REGRESSION)
# ================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Feature ve target ayır
X = clean_df.drop(columns=["Label", "target"])
y = clean_df["target"]

from sklearn.impute import SimpleImputer

categorical_features = ["Proto", "State"]
numeric_features = [c for c in X.columns if c not in categorical_features]

# Numeric NaN -> median
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical NaN -> most_frequent, sonra one-hot
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features),
    ]
)

logreg_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))

    ]
)


# Train-test split (stratified!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nMissing values per column (top):")
print(X.isna().sum().sort_values(ascending=False).head(10))


# Modeli eğit
logreg_model.fit(X_train, y_train)

# Tahmin
y_pred = logreg_model.predict(X_test)

print("\n--- LOGISTIC REGRESSION RESULTS ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))


# ================================
# ADIM 7.3 - RANDOM FOREST MODEL
# ================================

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]
)

# Modeli eğit
rf_model.fit(X_train, y_train)

# Tahmin
y_pred_rf = rf_model.predict(X_test)

print("\n--- RANDOM FOREST RESULTS ---")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))


# ================================
# Save a small stratified sample for GitHub
# ================================

import os
os.makedirs("../data/sample", exist_ok=True)

sample_for_github = clean_df.groupby("target", group_keys=False).apply(
    lambda x: x.sample(n=min(2500, len(x)), random_state=42)
)

sample_path = "../data/sample/ctu13_sample.csv"
sample_for_github.to_csv(sample_path, index=False)

print("\n✅ Sample dataset saved for GitHub")
print(f"Path: {sample_path}")
print("Sample class distribution:")
print(sample_for_github['target'].value_counts(normalize=True))

