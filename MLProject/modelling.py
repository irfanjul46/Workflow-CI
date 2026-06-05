import os
import shutil
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    # Load dataset yang ada di satu folder
    df = pd.read_csv('football_preprocessing.csv')
    
    y = df['goals']
    drop_cols = [c for c in df.columns if df[c].dtype == 'object'] + ['goals', 'rk', 'born']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Memulai pelatihan model di CI/CD...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Menyimpan model ke folder lokal agar mudah diambil oleh Docker
    model_path = "model_output"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    
    mlflow.sklearn.save_model(model, model_path)
    print("✓ Model berhasil dilatih dan disimpan ke folder model_output/")

if __name__ == "__main__":
    main()