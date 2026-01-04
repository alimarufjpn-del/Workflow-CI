# modelling.py

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow

print("Import library berhasil")

# Set MLflow local tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("water_potability_ci")

# Load dataset (relative path)
df = pd.read_csv("water_potability_preprocessing.csv")

# Feature & target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Autolog
mlflow.autolog()

with mlflow.start_run(run_name="RandomForest_CI"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy
