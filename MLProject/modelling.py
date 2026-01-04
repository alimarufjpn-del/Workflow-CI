import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_file, experiment_name, run_name):
    # Setup MLflow tracking - FIXED for CI
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment(experiment_name)
    
    print(f'MLflow Setup: experiment={experiment_name}, run={run_name}')
    print(f'Dataset: {data_file}')
    
    # Load dataset
    df = pd.read_csv(data_file)
    print(f"Dataset shape: {df.shape}")
    
    # Split features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name=run_name):
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    
    args = parser.parse_args()
    main(args.data_file, args.experiment_name, args.run_name)
