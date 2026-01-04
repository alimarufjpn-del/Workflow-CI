import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("file:///mlruns")  # Untuk local, atau bisa diset ke server MLflow
    mlflow.set_experiment("water-potability-ci")
    
    # Membaca dataset
    df = pd.read_csv("water_potability_preprocessing.csv")
    
    # Preprocessing dan modelling (sesuai kode Anda)
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    main()
