import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path):
    # Setup MLflow
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("water-potability-ci")
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 100)
        
        print(f"Accuracy: {accuracy}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="water_potability_preprocessing.csv")
    args = parser.parse_args()
    main(args.data_path)
