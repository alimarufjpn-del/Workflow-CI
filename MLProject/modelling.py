import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path="water_potability_preprocessing.csv"):
    # Setup MLflow - menggunakan local file system untuk CI
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("water-potability-ci")
    
    print('MLflow setup berhasil')
    
    # Membaca dataset
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Pisahkan fitur dan Target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Pisahkan data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standarisasi
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Auto-logging
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_CI"):
        # Buat dan latih model Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics tambahan
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="water_potability_preprocessing.csv",
        help="Path to dataset"
    )
    args = parser.parse_args()
    main(args.data_path)
