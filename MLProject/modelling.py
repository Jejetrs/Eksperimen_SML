import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import uniform

# ========================
# ARGUMENT PARSER
# ========================
parser = argparse.ArgumentParser(description="Train Diabetes Prediction Model")
parser.add_argument("--data_path", type=str, default="MLProject/diabetes_preprocessing.csv")
parser.add_argument("--scaler_path", type=str, default="MLProject/scaler.pkl")
parser.add_argument("--n_iter", type=int, default=30)
parser.add_argument("--cv_folds", type=int, default=5)
args = parser.parse_args()

# ========================
# SETUP MLflow
# ========================
workspace_path = os.environ.get("GITHUB_WORKSPACE", ".")
mlruns_path = os.path.join(workspace_path, "MLProject/mlruns")
os.makedirs(mlruns_path, exist_ok=True)

mlflow.set_tracking_uri(f"file://{mlruns_path}")
mlflow.set_experiment("Diabetes Prediction - Workflow CI")

# ========================
# LOAD DATA
# ========================
data = pd.read_csv(args.data_path)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================
# LOAD SCALER
# ========================
scaler = joblib.load(args.scaler_path)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# PIPELINE & RANDOMIZED SEARCH
# ========================
pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("clf", LogisticRegression(max_iter=5000, random_state=42))
])

param_dist = {
    "poly__degree": [1, 2],
    "clf__penalty": ["elasticnet", "l2"],
    "clf__solver": ["saga"],
    "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7],
    "clf__C": uniform(0.001, 10),
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=args.n_iter,
    scoring="f1",
    cv=args.cv_folds,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train_scaled, y_train)

best_model = search.best_estimator_
best_params = search.best_params_

# ========================
# EVALUASI
# ========================
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ========================
# LOGGING MLflow
# ========================
with mlflow.start_run(run_name="LogReg_CI_Model") as run:
    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_artifact(args.scaler_path)

# ========================
# SAVE MODEL LOCAL
# ========================
joblib.dump(best_model, "MLProject/diabetes_model_ci.pkl")

print(f"Training completed. Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
