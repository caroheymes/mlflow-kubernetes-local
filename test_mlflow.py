import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-local")

# Charger les données
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Créer et entraîner un modèle
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Prédictions et accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Logger avec MLflow
with mlflow.start_run():
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"✅ Run enregistré ! Accuracy: {accuracy:.2f}")