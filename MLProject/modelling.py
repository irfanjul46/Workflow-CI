import os
import shutil
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    # 1. Pastikan jalur file konsisten
    # Jika di dalam CI/CD file ini berada di root, maka akses langsung ke CSV
    if not os.path.exists('football_preprocessing.csv'):
        # Fallback jika dijalankan dari dalam subfolder
        data_path = os.path.join('MLProject', 'preprocessing', 'football_preprocessing.csv')
    else:
        data_path = 'football_preprocessing.csv'
    
    df = pd.read_csv(data_path)
    
    y = df['goals']
    drop_cols = [c for c in df.columns if df[c].dtype == 'object'] + ['goals', 'rk', 'born']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Setup MLflow Tracking agar CI/CD mencatat hasil ke DagsHub
    # Pastikan environment variable MLFLOW_TRACKING_URI sudah diset di Workflow YAML kamu
    mlflow.set_experiment("Football_Goals_Prediction_CI_CD")
    mlflow.sklearn.autolog() # Wajib untuk sinkronisasi dengan level Basic
    
    print("Memulai pelatihan model di CI/CD...")
    
    with mlflow.start_run(run_name="CI_CD_Automated_Training"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Dapatkan lokasi absolut dari file modelling.py saat ini
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Gabungkan dengan nama folder output agar tersimpan persis di sebelah file modelling.py
        model_path = os.path.join(script_dir, "model_output")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        
        # Simpan dalam format MLflow model
        mlflow.sklearn.save_model(model, model_path)
        print(f"✓ Model berhasil dilatih, di-log ke MLflow, dan disimpan ke {model_path}/")

if __name__ == "__main__":
    main()