import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATASETS_DIR = "."
PREPROCESSED_DIR = "preprocessed_data"

def load_and_preprocess_full_data(dataset_idx):
    if dataset_idx == 1:
        path = os.path.join(DATASETS_DIR, "Data set 1", "Fraud.csv")
        df = pd.read_csv(path)
        df = df.drop(columns=['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], errors='ignore')
        target = 'isFraud'
    elif dataset_idx == 2:
        path = os.path.join(DATASETS_DIR, "data set 2", "credit_card_fraud_10k.csv")
        df = pd.read_csv(path)
        df = df.drop(columns=['transaction_id'], errors='ignore')
        target = 'is_fraud'
    elif dataset_idx == 3:
        path = os.path.join(DATASETS_DIR, "data set 3", "creditcard_2023.csv")
        df = pd.read_csv(path)
        df = df.drop(columns=['id'], errors='ignore')
        target = 'Class'
    elif dataset_idx == 4:
        path = os.path.join(DATASETS_DIR, "data set 4", "creditcard.csv")
        df = pd.read_csv(path)
        df = df.drop(columns=['Time'], errors='ignore')
        target = 'Class'
    elif dataset_idx == 5:
        path = os.path.join(DATASETS_DIR, "data set 5", "fraudTrain.csv")
        df = pd.read_csv(path)
        drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'trans_num', 'unix_time', 'dob']
        df = df.drop(columns=drop_cols, errors='ignore')
        target = 'is_fraud'
        
    print(f"Dataset {dataset_idx} loaded: {len(df)} rows.")
    df = df.dropna()
    y = df[target].values
    X = df.drop(columns=[target])
    
    # Label encode categorical columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    out_file = os.path.join(PREPROCESSED_DIR, f"dataset_{dataset_idx}.npz")
    np.savez_compressed(out_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"Saved dataset {dataset_idx} to {out_file}\n")

def main():
    for i in range(1, 6):
        try:
            print(f"Processing dataset {i}...")
            load_and_preprocess_full_data(i)
        except Exception as e:
            print(f"Error processing dataset {i}: {e}")

if __name__ == "__main__":
    main()
