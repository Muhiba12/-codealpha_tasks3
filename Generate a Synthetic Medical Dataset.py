import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create Synthetic Dataset
np.random.seed(42)
n_samples = 500

data = {
    "age": np.random.randint(20, 80, n_samples),
    "blood_pressure": np.random.randint(90, 180, n_samples),
    "cholesterol": np.random.randint(150, 300, n_samples),
    "glucose": np.random.randint(70, 200, n_samples),
    "bmi": np.round(np.random.uniform(18, 40, n_samples), 1),
    "symptom_score": np.random.randint(0, 10, n_samples),
}

df = pd.DataFrame(data)


df["disease"] = (
    (df["blood_pressure"] > 140).astype(int)
    | (df["cholesterol"] > 240).astype(int)
    | (df["glucose"] > 160).astype(int)
    | (df["bmi"] > 30).astype(int)
    | (df["symptom_score"] > 6).astype(int)
)

print("Sample dataset:\n", df.head())


X = df.drop("disease", axis=1)
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", acc)


results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
print("\nModel Comparison:\n", results_df)

sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Disease Prediction Model Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()
