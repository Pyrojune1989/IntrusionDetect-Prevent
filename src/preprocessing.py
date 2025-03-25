import pandas as pd
import numpy as np
import gzip
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Define file paths
RAW_DATA_PATH = "../data/raw/kddcup.data_10_percent.gz"
FEATURE_NAMES_PATH = "../data/features/kdd_feature_names.txt"
ATTACK_CATEGORIES_PATH = "../data/features/kdd_class_labels.txt"
PROCESSED_TRAIN_PATH = "../data/processed/kdd_train.csv"
PROCESSED_TEST_PATH = "../data/processed/kdd_test.csv"

def load_feature_names():
    """Loads feature names from metadata file."""
    with open(FEATURE_NAMES_PATH, "r") as f:
        feature_names = f.read().splitlines()
    return feature_names + ["label"]  # Add the attack label column

def load_attack_categories():
    """Loads attack categories from metadata file."""
    attack_mapping = {}
    with open(ATTACK_CATEGORIES_PATH, "r") as f:
        for line in f:
            attack, category = line.strip().split(",")
            attack_mapping[attack] = category
    return attack_mapping

def load_data():
    """Loads the KDD dataset from a .gz file and assigns feature names."""
    feature_names = load_feature_names()
    
    # Read gzipped file
    with gzip.open(RAW_DATA_PATH, "rt") as f:
        df = pd.read_csv(f, names=feature_names, header=None)
    
    return df

def preprocess_data():
    """Processes the KDD dataset: encodes categorical features, normalizes data, and splits train/test."""
    df = load_data()
    attack_mapping = load_attack_categories()

    # Map attack types to categories
    df["label"] = df["label"].map(attack_mapping)

    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag", "label"]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Separate features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save processed data
    train_df = pd.DataFrame(np.column_stack((X_train, y_train)))
    test_df = pd.DataFrame(np.column_stack((X_test, y_test)))
    
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    
    print(f"Preprocessed data saved:\n - {PROCESSED_TRAIN_PATH}\n - {PROCESSED_TEST_PATH}")

if __name__ == "__main__":
    preprocess_data()
