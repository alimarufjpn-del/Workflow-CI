import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set tracking URI ke local (untuk CI, kita gunakan file store)
mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("water-potability-ci")

# Load data
df = pd.read_csv("water_potability_preprocessing.csv")

# Preprocessing
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Auto-logging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="ci-run"):
    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metric tambahan jika perlu
    mlflow.log_metric("accuracy", accuracy)

    print(f"Accuracy: {accuracy}")
